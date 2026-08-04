"""Stage 1 feature backends: the deterministic stub and the real torch stack.

Split out of :mod:`src.stage1_features` so the stage file stays about the
pipeline loop and this file stays about models. Both backends are always handed
an already-decoded ``PIL.Image``; neither ever opens a file, so Stage 1 keeps
its guarantee of exactly one read and one decode per photo.

Model failures are raised, never swallowed: ``resolve_backend`` will not quietly
fall back to the stub when torch is installed but its weights cannot load, so a
broken GPU setup can never masquerade as usable results.

The three-call shape
--------------------
``prepare_cpu(image, recorder)``
    Per-image, purely functional CPU work: the bounded IQA array, sharpness, the
    embedding preprocess, the face-detector input. Thread-safe by construction
    (it touches no instance state), so :mod:`src.stage1_pipeline` can run it on a
    worker thread — or on the main thread when ``cpu_workers: 0``.
``embed_prepared`` / ``quality_prepared``
    Main thread, GPU. One embedding call per batch, and for IQA **one call per
    distinct bounded input shape** per lane instead of one call per image.
``faces_prepared``
    Main thread, per image: ``cv2.FaceDetectorYN`` carries a mutable input size,
    so it is not thread-safe and is never called off the main thread.

Why IQA batching groups by shape (measured, not assumed)
-------------------------------------------------------
``docs/STAGE1_THROUGHPUT.md`` records the experiments on the real weights, and
``python -m scripts.verify_iqa_batching`` re-runs them on any CUDA machine.
Stacking equal-shaped tensors is score-preserving: across those runs MUSIQ agreed
to 1e-05 and CLIP-IQA to 5e-05 of the per-image result, the residual being
batch-dependent cuDNN kernel selection. Heterogeneous shapes cannot be batched
honestly: ``torch.cat`` is impossible, pyiqa rejects lists ("Unsupported source
type"), and zero-padding to a common size moved MUSIQ by **1.7 to 3.4 points**
and CLIP-IQA by **0.08 to 0.12**, depending on how much padding was needed —
three to five orders of magnitude more than the batching residual. Both models
resize internally from whatever tensor they are given, so the tensor's own height
and width are part of the feature definition, and padding would be a silent
scoring change disguised as an optimisation. Grouping by shape instead keeps
every per-image score comparable while collapsing N calls into one per distinct
bounded size: for a real library (one camera, two orientations) that is one or
two calls per batch rather than N.

Batching removes the ``.item()`` synchronisation per image, not kernel time: at
a 1920 px long edge these metrics are compute-bound (65 ms/image at batch 1, 64
ms/image at batch 8). The throughput comes from the CPU work moving off the main
thread; this file's job is to stop paying 2N host syncs and to keep VRAM bounded
(``iqa_batch_size``: 4 images measured at ~3.0 GiB peak, 8 at ~5.9 GiB).

Both backends carry an optional ``telemetry`` attribute (:mod:`src.stage1_telemetry`).
Stage 1 sets it; when it is unset every ``_phase``/``_gpu_phase`` call returns a
no-op span, so the timing hooks cost nothing and never change behaviour.
``_gpu_phase`` only *records* CUDA events (asynchronously) and only on a sampled
batch's first call per lane; the single synchronisation happens later, in the
batch span, outside every host-wall measurement.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import Config
from . import quality as Q
from . import stage1_telemetry as telemetry_mod


EMBED_DIM = 768
# Default cap on images per IQA model call. 4 was measured at ~3.0 GiB peak
# (MUSIQ, 1920x1440); 8 reaches ~5.9 GiB, which is imprudent next to DINOv2 on a
# 16 GiB card. Raising it buys no kernel time (64.2 vs 65.4 ms/image).
DEFAULT_IQA_BATCH_SIZE = 4


def _bounded_iqa_image(image: Image.Image, max_long_edge: int) -> tuple[Image.Image, float]:
    """Aspect-preserving IQA input; never enlarge the source."""
    width, height = image.size
    longest = max(width, height)
    if max_long_edge < 1:
        raise ValueError("features.iqa_max_long_edge must be at least 1")
    if longest <= max_long_edge:
        return image, 1.0
    scale = max_long_edge / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def _span(recorder: Any, key: str, count: int = 1) -> Any:
    """Phase span from a caller-supplied recorder (worker or main thread)."""
    return recorder.phase(key, count) if recorder is not None else telemetry_mod.NULL_SPAN


def _shape_groups(prepared: Sequence[Dict[str, Any]], cap: int) -> Iterator[List[int]]:
    """Indices grouped by identical IQA array shape, chunked to ``cap``.

    Order is deterministic: groups appear in first-appearance order and indices
    keep their batch order, so a run is reproducible and a score can always be
    traced back to its image.
    """
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for index, item in enumerate(prepared):
        groups.setdefault(tuple(np.shape(item["iqa_array"])), []).append(index)
    for indices in groups.values():
        for start in range(0, len(indices), max(1, cap)):
            yield indices[start:start + cap]


class _Instrumented:
    """Optional phase-timing hooks shared by both backends."""

    telemetry: Any = None

    def _phase(self, key: str, count: int = 1) -> Any:
        sink = self.telemetry
        return sink.phase(key, count) if sink is not None else telemetry_mod.NULL_SPAN

    def _gpu_phase(self, key: str) -> Any:
        """CUDA-event span: at most one sample per lane per sampled batch."""
        sink = self.telemetry
        return sink.gpu_event_phase(key) if sink is not None else telemetry_mod.NULL_SPAN

    def _count(self, key: str, value: int = 1) -> None:
        sink = self.telemetry
        if sink is not None:
            sink.count(key, value)

    # -- compatibility shims ------------------------------------------------
    # Existing callers (windows_benchmark, the SigLIP runner, tests) use the
    # single-image API. It is expressed in terms of the batched one so there is
    # exactly one implementation of the scoring semantics.
    def embed_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        prepared = [self.prepare_cpu(image, self.telemetry) for image in images]
        return self.embed_prepared(prepared)

    def quality(self, image: Image.Image) -> Tuple[float, Dict[str, Any]]:
        prepared = self.prepare_cpu(image, self.telemetry)
        return self.quality_prepared([prepared])[0]

    def faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        return self.faces_prepared(self.prepare_cpu(image, self.telemetry), image)


# ===========================================================================
# Backends
# ===========================================================================

class StubBackend(_Instrumented):
    """Deterministic, numpy-only feature backend (no torch/CUDA required).

    * embedding -- L2-normalised 24x32 grayscale downsample (768-d), which is
      remarkably good at separating "same scene" from "different scene".
    * quality   -- variance-of-Laplacian sharpness mapped to 0..100.
    * faces     -- detects a bright-red block as a stand-in "face" so the
      check-in-photo split logic is exercisable end-to-end in tests.
    """

    name = "stub"

    def __init__(self, iqa_max_long_edge: int = 1920) -> None:
        self.iqa_max_long_edge = int(iqa_max_long_edge)

    # -- per-image CPU preparation (thread-safe) ----------------------------
    def prepare_cpu(self, image: Image.Image, recorder: Any = None) -> Dict[str, Any]:
        with _span(recorder, "embed_preprocess"):
            gray = np.asarray(image.convert("L").resize((32, 24), Image.BILINEAR),
                              dtype=np.float32).flatten()
        with _span(recorder, "quality_preprocess"):
            bounded, scale = _bounded_iqa_image(image, self.iqa_max_long_edge)
            # The stub's IQA "array" is the grayscale plane its sharpness uses;
            # keeping the key name shared with the torch backend is what lets the
            # pipeline group by shape without knowing which backend ran.
            iqa_array = np.asarray(bounded.convert("L"), dtype=np.float32)
            size = list(bounded.size)
        with _span(recorder, "quality_sharpness"):
            sharpness = Q.variance_of_laplacian(iqa_array)
        return {"embed_gray": gray, "iqa_array": iqa_array, "iqa_input_size": size,
                "iqa_scale": scale, "sharpness": sharpness}

    # -- batched lanes (main thread) ----------------------------------------
    def embed_prepared(self, prepared: Sequence[Dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(prepared), EMBED_DIM), dtype=np.float32)
        with self._phase("embed_inference"):
            for index, item in enumerate(prepared):
                vector = item["embed_gray"]
                vector = vector - vector.mean()
                norm = np.linalg.norm(vector)
                out[index] = vector / norm if norm > 0 else vector
        self._count("embed_calls")
        self._count("embed_images", len(prepared))
        return out

    def quality_prepared(
        self, prepared: Sequence[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        results: List[Tuple[float, Dict[str, Any]]] = []
        for item in prepared:
            sharpness = item["sharpness"]
            results.append((100.0 * Q.normalize_sharpness(sharpness), {
                "sharpness": sharpness, "clipiqa": None, "backend": "stub",
                "iqa_input_size": item["iqa_input_size"], "iqa_scale": item["iqa_scale"],
            }))
        return results

    def faces_prepared(self, _prepared: Dict[str, Any],
                       image: Image.Image) -> List[Dict[str, Any]]:
        with self._phase("faces_yunet"):
            arr = np.asarray(image.convert("RGB"), dtype=np.int16)
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            mask = (r > 150) & (g < 100) & (b < 100)
            if int(mask.sum()) < 20:
                return []
            ys, xs = np.where(mask)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        return [{"bbox": bbox, "landmarks": [], "score": 0.95}]


class TorchBackend(_Instrumented):
    """Real backend: DINOv2 embedding + pyiqa MUSIQ/CLIP-IQA + YuNet faces.

    Heavy imports happen in ``__init__`` so importing this module never pulls
    torch.
    """

    name = "torch"

    def __init__(self, cfg: Config) -> None:
        import torch  # noqa: WPS433 (intentional lazy import)
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.cfg = cfg
        self.iqa_max_long_edge = int(cfg.features.get("iqa_max_long_edge", 1920))
        self.iqa_batch_size = int(cfg.features.get("iqa_batch_size", DEFAULT_IQA_BATCH_SIZE))
        if self.iqa_batch_size < 1:
            raise ValueError("features.iqa_batch_size must be at least 1")
        want_cuda = str(cfg.features.get("device", "cuda")) == "cuda"
        if want_cuda and not torch.cuda.is_available():
            raise RuntimeError("features.device=cuda but CUDA is unavailable; no CPU fallback")
        self.device = "cuda" if want_cuda else "cpu"

        model_name = cfg.features.get("dinov2_model", "facebook/dinov2-base")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self._verify_processor_is_per_image_separable()

        self.musiq = None
        self.clipiqa = None
        if cfg.features.get("iqa_musiq", True) or cfg.features.get("iqa_clipiqa", True):
            import pyiqa  # noqa: WPS433

            if cfg.features.get("iqa_musiq", True):
                self.musiq = pyiqa.create_metric("musiq", device=self.device)
            if cfg.features.get("iqa_clipiqa", True):
                self.clipiqa = pyiqa.create_metric("clipiqa", device=self.device)

        self._face = _load_yunet(cfg)

    def _verify_processor_is_per_image_separable(self) -> None:
        """Fail loudly if per-image preprocessing would change the embedding.

        Preprocessing moved off the main thread, which means the processor is now
        called with one image at a time instead of the whole batch. For
        ``facebook/dinov2-base`` (``BitImageProcessor``) that is bitwise identical
        — verified here rather than assumed, because a processor that padded or
        normalised *across* a batch would silently alter every embedding. Two
        tiny images make this a sub-millisecond check at model-load time.
        """
        torch = self.torch
        probes = [Image.new("RGB", (37, 23), (10, 200, 60)),
                  Image.new("RGB", (23, 37), (200, 10, 60))]
        together = self.processor(images=probes, return_tensors="pt")["pixel_values"]
        apart = torch.cat([self.processor(images=[image], return_tensors="pt")["pixel_values"]
                           for image in probes], dim=0)
        if not torch.equal(together, apart):
            raise RuntimeError(
                f"image processor for {self.cfg.features.get('dinov2_model')} is not "
                "per-image separable: preprocessing one image at a time would change "
                "the embedding. Set features.cpu_workers=0 to restore the batched "
                "preprocess, or use a processor with fixed-size output."
            )

    # -- per-image CPU preparation (thread-safe) ----------------------------
    def prepare_cpu(self, image: Image.Image, recorder: Any = None) -> Dict[str, Any]:
        """Everything the GPU lanes need, computed from one decoded frame.

        Touches no instance state beyond read-only config and the (verified
        thread-safe) processor, so N of these can run concurrently. The bounded
        RGB array is produced **once** and reused for the IQA tensor and for
        sharpness, which is exactly what the serial path did.
        """
        prepared: Dict[str, Any] = {}
        with _span(recorder, "embed_preprocess"):
            prepared["embed_pixels"] = self.processor(
                images=[image], return_tensors="pt")["pixel_values"]
        with _span(recorder, "quality_preprocess"):
            bounded, scale = _bounded_iqa_image(image, self.iqa_max_long_edge)
            array = np.asarray(bounded.convert("RGB"), dtype=np.float32) / 255.0
            prepared["iqa_array"] = array
            prepared["iqa_input_size"] = list(bounded.size)
            prepared["iqa_scale"] = scale
        with _span(recorder, "quality_sharpness"):
            prepared["sharpness"] = Q.variance_of_laplacian(array.mean(axis=2) * 255.0)
        with _span(recorder, "faces_preprocess"):
            prepared.update(self._face_input(image))
        return prepared

    def _face_input(self, image: Image.Image) -> Dict[str, Any]:
        """Bounded BGR array YuNet detects on, plus the scale to undo it."""
        try:
            import cv2  # noqa: WPS433
        except Exception:
            return {"face_bgr": None, "face_scale": 1.0}
        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        height, width = bgr.shape[:2]
        target = int(self.cfg.features.get("yunet_input_width", 640))
        scale = target / max(height, width) if max(height, width) > target else 1.0
        if scale != 1.0:
            bgr = cv2.resize(bgr, (int(width * scale), int(height * scale)))
        return {"face_bgr": bgr, "face_scale": scale}

    # -- embeddings ---------------------------------------------------------
    def embed_prepared(self, prepared: Sequence[Dict[str, Any]]) -> np.ndarray:
        """One model call per batch. Host-wall covers submit + the .cpu() sync."""
        torch = self.torch
        with self._phase("embed_inference"):
            with self._gpu_phase("embed_inference"):
                pixels = torch.cat([item["embed_pixels"] for item in prepared],
                                   dim=0).to(self.device)
                with torch.no_grad():
                    out = self.model(pixel_values=pixels)
            # pooler_output is the CLS-token summary (768-d for dinov2-base).
            embedding = (out.pooler_output if out.pooler_output is not None
                         else out.last_hidden_state[:, 0])
            result = embedding.float().cpu().numpy()
        self._count("embed_calls")
        self._count("embed_images", len(prepared))
        return result

    # -- quality ------------------------------------------------------------
    def quality_prepared(
        self, prepared: Sequence[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """MUSIQ + CLIP-IQA for a whole batch: one call per distinct shape.

        Per-image metadata (``iqa_input_size``, ``iqa_scale``, ``sharpness``) is
        carried through unchanged, and each lane's output crosses to the host
        exactly once per call — no ``.item()`` inside a per-image loop.
        """
        scores = [0.0] * len(prepared)
        clip: List[Optional[float]] = [None] * len(prepared)
        if self.musiq is not None or self.clipiqa is not None:
            groups = list(_shape_groups(prepared, self.iqa_batch_size))
            self._count("iqa_shape_groups", len(groups))
            for indices in groups:
                self._score_group(prepared, indices, scores, clip)
        return [self._meta(item, scores[index], clip[index])
                for index, item in enumerate(prepared)]

    def _score_group(self, prepared: Sequence[Dict[str, Any]], indices: Sequence[int],
                     scores: List[float], clip: List[Optional[float]]) -> None:
        torch = self.torch
        with self._phase("iqa_stack_upload"):
            # Per-image HWC->CHW exactly as the single-image path did, then one
            # host-side concat and one upload for the whole group.
            stacked = torch.cat(
                [torch.from_numpy(prepared[index]["iqa_array"]).permute(2, 0, 1).unsqueeze(0)
                 for index in indices], dim=0).to(self.device)
        try:
            if self.musiq is not None:
                with self._phase("quality_musiq"):
                    with self._gpu_phase("quality_musiq"):
                        values = self.musiq(stacked).reshape(-1).float().cpu().numpy()
                self._count("iqa_musiq_calls")
                self._count("iqa_musiq_images", len(indices))
                for position, index in enumerate(indices):
                    scores[index] = float(values[position])  # MUSIQ ~0..100
            if self.clipiqa is not None:
                with self._phase("quality_clipiqa"):
                    with self._gpu_phase("quality_clipiqa"):
                        values = self.clipiqa(stacked).reshape(-1).float().cpu().numpy()
                self._count("iqa_clipiqa_calls")
                self._count("iqa_clipiqa_images", len(indices))
                for position, index in enumerate(indices):
                    clip[index] = float(values[position])    # 0..1
        finally:
            del stacked        # release the group's VRAM before the next one

    @staticmethod
    def _meta(item: Dict[str, Any], score: float,
              clipiqa: Optional[float]) -> Tuple[float, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "backend": "torch", "iqa_input_size": item["iqa_input_size"],
            "iqa_scale": item["iqa_scale"], "musiq": score,
        }
        if clipiqa is not None:
            meta["clipiqa"] = clipiqa
        meta["sharpness"] = item["sharpness"]
        return score, meta

    # -- faces --------------------------------------------------------------
    def faces_prepared(self, prepared: Dict[str, Any],
                       _image: Image.Image = None) -> List[Dict[str, Any]]:
        """Main thread only: ``FaceDetectorYN`` carries a mutable input size."""
        bgr = prepared.get("face_bgr")
        if self._face is None or bgr is None:
            return []
        scale = float(prepared.get("face_scale", 1.0)) or 1.0
        with self._phase("faces_yunet"):
            height, width = bgr.shape[:2]
            self._face.setInputSize((width, height))
            _, dets = self._face.detect(bgr)
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
        return StubBackend(int(cfg.features.get("iqa_max_long_edge", 1920)))
    if choice == "torch":
        return TorchBackend(cfg)
    # auto
    try:
        import torch  # noqa: F401,WPS433

        return TorchBackend(cfg)
    except ImportError as exc:
        print(f"[stage1] torch unavailable ({exc}); using stub backend.")
        return StubBackend(int(cfg.features.get("iqa_max_long_edge", 1920)))
