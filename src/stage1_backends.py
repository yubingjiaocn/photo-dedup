"""Stage 1 feature backends: the deterministic stub and the real torch stack.

Split out of :mod:`src.stage1_features` so the stage file stays about the
pipeline loop and this file stays about models. Both backends expose the same
three methods (``embed_batch`` / ``quality`` / ``faces``) and are always handed
an already-decoded ``PIL.Image``; neither ever opens a file, so Stage 1 keeps
its guarantee of exactly one read and one decode per photo.

Model failures are raised, never swallowed: ``resolve_backend`` will not quietly
fall back to the stub when torch is installed but its weights cannot load, so a
broken GPU setup can never masquerade as usable results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import Config
from . import quality as Q


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
