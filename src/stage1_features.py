"""Stage 1 -- feature extraction (the GPU-heavy, HDD-bound stage).

For every still image (``file_kind`` in jpg / jpg_motion) this computes:

* ``phash``            -- 64-bit perceptual hash (exact-dup detection)
* ``dinov2_embedding`` -- 768-d semantic vector (burst / scene similarity)
* ``quality_score``    -- MUSIQ 0..100 (which frame to keep)
* ``quality_meta``     -- JSON: sharpness, CLIP-IQA, face_quality, backend
* ``faces_json``       -- YuNet bbox + landmarks + score per face

Results stream into the ``features`` table, committing every
``scan.commit_every`` images, so a Ctrl+C (or crash) resumes exactly where it
stopped -- only rows without a ``status='done'`` feature row are reprocessed.

Two interchangeable backends
----------------------------
* **torch** -- real DINOv2 + pyiqa (MUSIQ/CLIP-IQA) + OpenCV YuNet. Needs the
  GPU stack from requirements.txt.
* **stub**  -- deterministic numpy-only features. Used by the test suite and
  for no-GPU dry runs. Same output schema, so stages 2/3 don't care which ran.

``features.backend: auto`` uses torch if importable, else stub.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import Config, load_config
from . import db
from . import quality as Q
from . import exposure
from . import decision
from . import eye_detection
from .scene_router import DecodeState, RoutingInput, SceneRouter

try:  # progress bar is optional
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):  # type: ignore
        return x


EMBED_DIM = 768


# ===========================================================================
# Backends
# ===========================================================================

class StubBackend:
    """Deterministic, numpy-only feature backend (no torch/CUDA required).

    * embedding -- L2-normalised 24x32 grayscale downsample (768-d), which is
      remarkably good at separating "same scene" from "different scene".
    * quality   -- variance-of-Laplacian sharpness mapped to 0..100.
    * faces     -- detects a bright-red block as a stand-in "face" so the
      check-in-photo split logic is exercisable end-to-end in tests.
    """

    name = "stub"

    def embed_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = np.zeros((len(images), EMBED_DIM), dtype=np.float32)
        for i, im in enumerate(images):
            g = im.convert("L").resize((32, 24), Image.BILINEAR)
            v = np.asarray(g, dtype=np.float32).flatten()
            v -= v.mean()
            n = np.linalg.norm(v)
            out[i] = v / n if n > 0 else v
        return out

    def quality(self, image: Image.Image) -> Tuple[float, Dict[str, Any]]:
        gray = np.asarray(image.convert("L"), dtype=np.float64)
        sharp = Q.variance_of_laplacian(gray)
        score = 100.0 * Q.normalize_sharpness(sharp)
        return score, {"sharpness": sharp, "clipiqa": None, "backend": "stub"}

    def faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        arr = np.asarray(image.convert("RGB"), dtype=np.int16)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mask = (r > 150) & (g < 100) & (b < 100)
        if int(mask.sum()) < 20:
            return []
        ys, xs = np.where(mask)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        return [{"bbox": bbox, "landmarks": [], "score": 0.95}]


class TorchBackend:
    """Real backend: DINOv2 embedding + pyiqa MUSIQ/CLIP-IQA + YuNet faces.

    Heavy imports happen in ``__init__`` so importing this module never pulls
    torch. Not exercised by the Linux test box; verified logically only.
    """

    name = "torch"

    def __init__(self, cfg: Config) -> None:
        import torch  # noqa: WPS433 (intentional lazy import)
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.cfg = cfg
        want_cuda = str(cfg.features.get("device", "cuda")) == "cuda"
        self.device = "cuda" if (want_cuda and torch.cuda.is_available()) else "cpu"

        model_name = cfg.features.get("dinov2_model", "facebook/dinov2-base")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

        self.musiq = None
        self.clipiqa = None
        if cfg.features.get("iqa_musiq", True) or cfg.features.get("iqa_clipiqa", True):
            import pyiqa  # noqa: WPS433

            if cfg.features.get("iqa_musiq", True):
                self.musiq = pyiqa.create_metric("musiq", device=self.device)
            if cfg.features.get("iqa_clipiqa", True):
                self.clipiqa = pyiqa.create_metric("clipiqa", device=self.device)

        self._face = _load_yunet(cfg)

    # -- embeddings ---------------------------------------------------------
    def embed_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        # pooler_output is the CLS-token summary (768-d for dinov2-base).
        emb = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        return emb.float().cpu().numpy()

    # -- quality ------------------------------------------------------------
    def quality(self, image: Image.Image) -> Tuple[float, Dict[str, Any]]:
        torch = self.torch
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = 0.0
        meta: Dict[str, Any] = {"backend": "torch"}
        if self.musiq is not None:
            score = float(self.musiq(tensor).item())  # MUSIQ ~0..100
        meta["musiq"] = score
        if self.clipiqa is not None:
            meta["clipiqa"] = float(self.clipiqa(tensor).item())  # 0..1
        gray = arr.mean(axis=2) * 255.0
        meta["sharpness"] = Q.variance_of_laplacian(gray)
        return score, meta

    # -- faces --------------------------------------------------------------
    def faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        if self._face is None:
            return []
        import cv2  # noqa: WPS433

        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        h, w = bgr.shape[:2]
        target = int(self.cfg.features.get("yunet_input_width", 640))
        scale = target / max(h, w) if max(h, w) > target else 1.0
        if scale != 1.0:
            bgr_small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        else:
            bgr_small = bgr
        sh, sw = bgr_small.shape[:2]
        self._face.setInputSize((sw, sh))
        _, dets = self._face.detect(bgr_small)
        faces: List[Dict[str, Any]] = []
        if dets is None:
            return faces
        for d in dets:
            x, y, fw, fh = (float(v) / scale for v in d[:4])
            landmarks = [float(v) / scale for v in d[4:14]]
            faces.append({"bbox": [x, y, fw, fh], "landmarks": landmarks, "score": float(d[14])})
        return faces


def _load_yunet(cfg: Config):
    """Download (if needed) and construct the OpenCV YuNet face detector."""
    try:
        import cv2  # noqa: WPS433
    except Exception:
        return None
    models_dir = cfg.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx = models_dir / "face_detection_yunet_2023mar.onnx"
    if not onnx.exists():
        url = cfg.features.get("yunet_url")
        try:
            import urllib.request

            print(f"[stage1] downloading YuNet model -> {onnx}")
            urllib.request.urlretrieve(url, onnx)  # noqa: S310
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"YuNet download failed: {exc}") from exc
    score_thr = float(cfg.features.get("yunet_score_threshold", 0.6))
    return cv2.FaceDetectorYN.create(str(onnx), "", (320, 320), score_thr, 0.3, 5000)


def resolve_backend(cfg: Config, override: Optional[str] = None):
    """Pick a backend per config/override. auto -> torch if importable else stub."""
    choice = (override or cfg.features.get("backend", "auto")).lower()
    if choice == "stub":
        return StubBackend()
    if choice == "torch":
        return TorchBackend(cfg)
    # auto
    try:
        import torch  # noqa: F401,WPS433

        return TorchBackend(cfg)
    except ImportError as exc:
        print(f"[stage1] torch unavailable ({exc}); using stub backend.")
        return StubBackend()


# ===========================================================================
# Main loop
# ===========================================================================

def _open_image_and_sha(path: Path) -> Tuple[Image.Image, str]:
    """Read compressed bytes once, hashing that same sequential read before decode."""
    digest = hashlib.sha256()
    data = io.BytesIO()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
            data.write(chunk)
    data.seek(0)
    img = Image.open(data)
    img = img.convert("RGB")
    return img, digest.hexdigest()


def _process_batch(
    backend: Any,
    rows: Sequence[Any],
    cfg: Config,
    eye_detector: Any = None,
    scene_router: Any = None,
) -> List[Dict[str, Any]]:
    """Compute features for one batch of file rows. Returns feature dicts.

    The router is constructed at batch scope when the caller did not retain a
    run-scoped instance.  It receives the same decoded RGB ``Image`` used by
    embeddings and technical detectors; it never opens the source path.
    """
    router = scene_router if scene_router is not None else _build_scene_router(cfg)
    images: List[Image.Image] = []
    hashes: List[str] = []
    valid_rows: List[Any] = []
    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        try:
            image, sha256 = _open_image_and_sha(Path(row["path"]))
            images.append(image)
            hashes.append(sha256)
            valid_rows.append(row)
        except Exception as exc:  # unreadable / corrupt file
            print(f"[stage1][WARN] cannot read {row['path']}: {exc}")
            routing = router.route(RoutingInput(
                metadata=_routing_metadata(row), decode_state=DecodeState.FAILED
            ))
            out_rows.append(_error_feature_row(int(row["id"]), routing=routing))

    if not valid_rows:
        return out_rows

    embeddings = backend.embed_batch(images)
    for row, image, sha256, emb in zip(valid_rows, images, hashes, embeddings):
        score, meta = backend.quality(image)
        faces = backend.faces(image)
        img_rgb = np.asarray(image)
        fq = Q.compute_face_quality(img_rgb, faces)
        meta["face_quality"] = fq
        meta["schema_version"] = 2
        meta["exposure"] = exposure.extract(
            image, faces, int(cfg.features.get("exposure_long_edge", 512))
        )
        meta["detectors"] = decision.detector_extensions()
        if eye_detector is not None:
            try:
                meta["eye_detection"] = eye_detector.analyze(
                    image, expected_face_count=len(faces)
                )
            except Exception:
                meta["eye_detection"] = eye_detection.unavailable_result(
                    "DETECTOR_INFERENCE_FAILED"
                )
        # Shadow-only, additive metadata.  The provider is intentionally None
        # until SigLIP is implemented; SceneRouter then emits MODEL_UNAVAILABLE
        # without model loading, downloads, or network access.
        meta["routing"] = router.route(RoutingInput(
            metadata=_routing_metadata(row), image=image
        ))
        out_rows.append(
            {
                "file_id": int(row["id"]),
                "phash": Q.phash_bytes(image),
                "content_sha256": sha256,
                "dinov2_embedding": Q.embedding_to_blob(emb),
                "quality_score": float(score),
                "quality_meta": json.dumps(meta),
                "face_count": len(faces),
                "faces_json": json.dumps(faces),
                "status": "done",
            }
        )
    return out_rows


def _build_scene_router(cfg: Config, provider: Any = None) -> SceneRouter:
    """Construct the optional offline provider only when explicitly enabled."""
    scene_cfg = cfg.features.get("scene_routing", {})
    if provider is None and isinstance(scene_cfg, dict) and scene_cfg.get("enabled") is True:
        try:
            from .siglip_runtime import build_siglip_provider

            provider = build_siglip_provider(cfg)
        except Exception:
            # Missing dependencies, artifacts, hashes, or CUDA all fail closed.
            provider = None
    return SceneRouter(cfg, provider=provider)


def _routing_metadata(row: Any) -> Dict[str, Any]:
    """Extract only present inventory fields needed by the metadata gate.

    ``sqlite3.Row`` supports subscription and ``keys()`` but not ``get``;
    tests and callers may use ordinary mappings.  Keeping this adapter narrow
    avoids adding invented values to routing metadata or changing DB schema.
    """
    fields = (
        "file_kind", "motion_partner_id", "motion_photo", "is_motion_photo",
        "motion_partner_path", "paired_asset_id",
    )
    if isinstance(row, dict):
        return {field: row[field] for field in fields if field in row}
    keys = row.keys() if hasattr(row, "keys") else ()
    return {field: row[field] for field in fields if field in keys}


def _error_feature_row(file_id: int, routing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    quality_meta: Dict[str, Any] = {"error": True}
    if routing is not None:
        quality_meta["routing"] = routing
    return {
        "file_id": file_id,
        "phash": None,
        "content_sha256": None,
        "dinov2_embedding": None,
        "quality_score": None,
        "quality_meta": json.dumps(quality_meta),
        "face_count": 0,
        "faces_json": "[]",
        "status": "done_error",
    }


def run(config_path: Optional[str] = None, backend_override: Optional[str] = None,
        limit: Optional[int] = None) -> Dict[str, int]:
    """Run stage 1. Returns a small stats dict."""
    cfg = load_config(config_path)
    conn = db.open_db(cfg.db_path)
    backend = resolve_backend(cfg, backend_override)
    eye_cfg = cfg.features.get("eye_detection", {})
    eye_detector = (
        eye_detection.MediaPipeEyeDetector(eye_cfg, cfg.models_dir)
        if eye_cfg.get("enabled", False)
        else None
    )
    # Construct once per Stage 1 run. Local runtime failures remain a
    # MODEL_UNAVAILABLE shadow record and never interrupt existing features.
    scene_router = _build_scene_router(cfg)
    batch_size = max(1, int(cfg.features.get("batch_size", 4)))
    max_pixels = int(float(cfg.features.get("max_inflight_megapixels", 80)) * 1_000_000)
    commit_every = int(cfg.scan.get("commit_every", 100))

    pending = list(db.iter_files_for_features(conn))
    if limit:
        pending = pending[:limit]
    total = len(pending)
    print(f"[stage1] backend={backend.name} pending={total} batch_size={batch_size}")

    done = 0
    since_commit = 0
    t0 = time.time()
    batches = _guarded_batches(pending, batch_size, max_pixels)
    for batch in tqdm(batches, desc="features", unit="batch"):
        feature_rows = _process_batch(
            backend, batch, cfg, eye_detector=eye_detector, scene_router=scene_router
        )
        db.batch_insert_features(conn, feature_rows)
        done += len(feature_rows)
        since_commit += len(feature_rows)
        if since_commit >= commit_every:
            conn.commit()
            since_commit = 0
    conn.commit()

    db.set_meta(conn, "stage1_backend", backend.name)
    db.set_meta(conn, "stage1_done_at", str(int(time.time())))
    dt = time.time() - t0
    rate = done / dt if dt > 0 else 0.0
    print(f"[stage1] processed {done} images in {dt:.1f}s ({rate:.1f}/s)")
    conn.close()
    return {"processed": done, "total": total}


def _guarded_batches(rows: Sequence[Any], batch_size: int, max_pixels: int) -> List[List[Any]]:
    """Bound full-resolution decoded images held concurrently."""
    batches: List[List[Any]] = []
    current: List[Any] = []
    pixels = 0
    for row in rows:
        item_pixels = max(1, int(row["width"] or 0) * int(row["height"] or 0))
        if current and (len(current) >= batch_size or pixels + item_pixels > max_pixels):
            batches.append(current)
            current, pixels = [], 0
        current.append(row)
        pixels += item_pixels
    if current:
        batches.append(current)
    return batches


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: feature extraction")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--backend", default=None, choices=["auto", "torch", "stub"])
    ap.add_argument("--limit", type=int, default=None, help="process at most N images")
    args = ap.parse_args(argv)
    run(config_path=args.config, backend_override=args.backend, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
