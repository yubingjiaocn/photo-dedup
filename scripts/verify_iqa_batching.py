"""Verify the Stage 1 IQA batching claims against the real model weights.

Opt-in, read-only, and it never touches ``paths.root``: it generates its own
synthetic images, loads the real DINOv2 / MUSIQ / CLIP-IQA / YuNet stack, and
checks the three claims the batching design rests on. Run it on a machine with
CUDA before trusting the optimisation there:

```
python -m scripts.verify_iqa_batching                  # all checks
python -m scripts.verify_iqa_batching --check semantics
python -m scripts.verify_iqa_batching --check equivalence --images 12
```

The checks
----------
``semantics``
    Can pyiqa be batched at all, and how? Compares N single-image calls against
    one stacked call for equal shapes, shows that heterogeneous shapes cannot be
    concatenated or passed as a list, and measures what zero-padding to a common
    size does to the score. This is the evidence for grouping by shape and for
    refusing to pad; if a future pyiqa release changed it, this check is what
    would notice.
``equivalence``
    Does ``TorchBackend`` return, per image, what the per-image path returned?
    Runs the production ``prepare_cpu`` + ``quality_prepared`` + ``embed_prepared``
    and compares against calls made one image at a time, then prints the model
    call counts and peak VRAM.

Recorded results from an L40S run are in ``docs/STAGE1_THROUGHPUT.md``. Tolerances
below are deliberately loose enough for batch-dependent cuDNN kernel selection
(~1e-4) and tight enough that a real semantic change fails.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.config import Config
from src.stage1_backends import TorchBackend, _shape_groups

# A batched kernel may legitimately differ in the last few digits; a changed
# feature definition (e.g. padding) moves scores by whole points.
MUSIQ_TOLERANCE = 1e-3          # MUSIQ spans ~0..100
CLIPIQA_TOLERANCE = 1e-4        # CLIP-IQA spans 0..1
EMBED_COSINE_MINIMUM = 0.9999

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


class _CountingSink:
    """Telemetry stand-in that only counts, so call counts are observable."""

    def __init__(self) -> None:
        self.counters: Dict[str, int] = {}

    def phase(self, key: str, count: int = 1) -> "_CountingSink":
        return self

    def worker_phase(self, key: str, count: int = 1) -> "_CountingSink":
        return self

    def gpu_event_phase(self, key: str) -> "_CountingSink":
        return self

    def count(self, key: str, value: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + value

    def __enter__(self) -> "_CountingSink":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


def synthetic(width: int, height: int, seed: int) -> Image.Image:
    """A deterministic JPEG-round-tripped image, so decode artefacts are real."""
    rng = np.random.default_rng(seed)
    small = (rng.random((max(2, height // 16), max(2, width // 16), 3)) * 255)
    image = Image.fromarray(small.astype(np.uint8)).resize((width, height), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _tensor(array: np.ndarray, device: str) -> Any:
    import torch

    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).to(device)


def _report(label: str, left: Sequence[float], right: Sequence[float],
            tolerance: float) -> bool:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    worst = float(np.abs(left_array - right_array).max()) if left_array.size else 0.0
    ok = worst <= tolerance
    print(f"  {label:<34} max_abs_diff={worst:.3e}  tolerance={tolerance:.1e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_semantics(device: str, long_edge: int) -> bool:
    """Is stacking score-preserving, and is padding as unsafe as we claim?"""
    import pyiqa
    import torch

    passed = True
    for name, tolerance in (("musiq", MUSIQ_TOLERANCE), ("clipiqa", CLIPIQA_TOLERANCE)):
        print(f"\n[{name}]")
        metric = pyiqa.create_metric(name, device=device)
        square = int(long_edge * 0.75)
        same = [_tensor(np.asarray(synthetic(long_edge, square, seed),
                                   dtype=np.float32) / 255.0, device)
                for seed in range(4)]
        with torch.no_grad():
            singles = [float(metric(item).item()) for item in same]
            stacked = metric(torch.cat(same, dim=0)).reshape(-1).float().cpu().numpy()
        passed &= _report("stacked vs per-image (same shape)", singles,
                          [float(value) for value in stacked], tolerance)

        # Heterogeneous shapes: prove they cannot be batched, rather than assume it.
        mixed = [_tensor(np.asarray(synthetic(width, height, 50 + index),
                                    dtype=np.float32) / 255.0, device)
                 for index, (width, height) in enumerate(
                     ((long_edge, square), (square, long_edge)))]
        try:
            torch.cat(mixed, dim=0)
            print("  mixed shapes: torch.cat SUCCEEDED — re-examine the grouping design")
            passed = False
        except RuntimeError:
            print("  mixed shapes: torch.cat impossible (as designed for)")
        try:
            with torch.no_grad():
                metric(list(mixed))
            print("  mixed shapes: list input ACCEPTED — a list batch may now be possible")
        except Exception as exc:
            print(f"  mixed shapes: list input rejected ({type(exc).__name__})")

        # Padding: the thing we refuse to do. Show what it would cost.
        with torch.no_grad():
            mixed_singles = [float(metric(item).item()) for item in mixed]
            padded = []
            for item in mixed:
                canvas = torch.zeros((1, 3, long_edge, long_edge), device=device)
                canvas[:, :, :item.shape[2], :item.shape[3]] = item
                padded.append(canvas)
            padded_scores = metric(torch.cat(padded, dim=0)).reshape(-1).float().cpu().numpy()
        shift = float(np.abs(np.asarray(mixed_singles)
                             - np.asarray(padded_scores, dtype=np.float64)).max())
        print(f"  zero-padding would shift scores by {shift:.4f} "
              f"(> {tolerance:.1e} tolerance, hence never done)")
        if shift <= tolerance:
            print("  NOTE: padding no longer changes this metric; the design is still "
                  "correct but its justification would need re-stating")
        del metric, same, mixed, padded
        torch.cuda.empty_cache() if device == "cuda" else None
    return passed


def check_equivalence(device: str, long_edge: int, images: int,
                      iqa_batch_size: int) -> bool:
    """Does the production backend agree with the per-image path, per image?"""
    import torch

    cfg = Config({
        "features": {"backend": "torch", "device": device,
                     "iqa_max_long_edge": long_edge, "iqa_batch_size": iqa_batch_size,
                     "iqa_musiq": True, "iqa_clipiqa": True,
                     "yunet_input_width": 640, "yunet_score_threshold": 0.6,
                     "yunet_url": YUNET_URL},
        "paths": {"models_dir": "./models"},
    })
    started = time.perf_counter()
    backend = TorchBackend(cfg)
    print(f"\nbackend loaded in {time.perf_counter() - started:.1f}s "
          "(per-image processor separability verified at load)")

    # Mixed orientations plus one square: several bounded shapes in one batch.
    shapes: List[Tuple[int, int]] = [(4032, 3024), (3024, 4032), (2048, 2048)]
    subjects = [synthetic(*shapes[index % len(shapes)], seed=index) for index in range(images)]
    prepared = [backend.prepare_cpu(item) for item in subjects]
    groups = list(_shape_groups(prepared, iqa_batch_size))
    print(f"{images} images -> {len(set(tuple(np.shape(p['iqa_array'])) for p in prepared))} "
          f"distinct bounded shapes -> {len(groups)} model call(s) per lane")

    # The old path: one call per image per lane.
    with torch.no_grad():
        solo_musiq, solo_clip = [], []
        for item in prepared:
            tensor = _tensor(item["iqa_array"], backend.device)
            solo_musiq.append(float(backend.musiq(tensor).item()))
            solo_clip.append(float(backend.clipiqa(tensor).item()))

    sink = _CountingSink()
    backend.telemetry = sink
    results = backend.quality_prepared(prepared)
    passed = _report("MUSIQ batched vs per-image", solo_musiq,
                     [score for score, _ in results], MUSIQ_TOLERANCE)
    passed &= _report("CLIP-IQA batched vs per-image", solo_clip,
                      [meta["clipiqa"] for _, meta in results], CLIPIQA_TOLERANCE)

    # Per-image metadata must survive the batch untouched.
    metadata_ok = True
    for subject, (_, meta) in zip(subjects, results):
        width, height = subject.size
        expected = long_edge / max(width, height) if max(width, height) > long_edge else 1.0
        if abs(meta["iqa_scale"] - expected) > 1e-9 or not np.isfinite(meta["sharpness"]):
            metadata_ok = False
        if max(meta["iqa_input_size"]) > long_edge:
            metadata_ok = False
    print(f"  {'per-image iqa_scale/size/sharpness':<34} "
          f"{'OK' if metadata_ok else 'FAIL'}")
    passed &= metadata_ok

    # Embeddings: one batched call vs one call per image.
    batched = backend.embed_prepared(prepared)
    with torch.no_grad():
        solo = []
        for item in prepared:
            out = backend.model(pixel_values=item["embed_pixels"].to(backend.device))
            vector = (out.pooler_output if out.pooler_output is not None
                      else out.last_hidden_state[:, 0])
            solo.append(vector.float().cpu().numpy()[0])
    cosines = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
               for a, b in zip(solo, batched)]
    embed_ok = min(cosines) >= EMBED_COSINE_MINIMUM
    print(f"  {'embedding batched vs per-image':<34} min_cosine={min(cosines):.8f}  "
          f"minimum={EMBED_COSINE_MINIMUM}  {'OK' if embed_ok else 'FAIL'}")
    passed &= embed_ok

    faces = backend.faces_prepared(prepared[0])
    print(f"  YuNet on the main thread: {len(faces)} face(s) detected")
    print(f"\ncall counters: {sink.counters}")
    expected_calls = len(groups)
    counts_ok = (sink.counters.get("iqa_musiq_calls") == expected_calls
                 and sink.counters.get("iqa_musiq_images") == images
                 and sink.counters.get("embed_calls") == 1)
    print(f"  {'MUSIQ calls == shape groups':<34} "
          f"{sink.counters.get('iqa_musiq_calls')} == {expected_calls}  "
          f"{'OK' if counts_ok else 'FAIL'}")
    print(f"  per-image code would have made {images} MUSIQ + {images} CLIP-IQA calls")
    if device == "cuda":
        print(f"  peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB "
              f"at iqa_batch_size={iqa_batch_size}")
    return passed and counts_ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", default="all",
                        choices=["all", "semantics", "equivalence"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--images", type=int, default=6)
    parser.add_argument("--iqa-batch-size", type=int, default=4)
    parser.add_argument("--iqa-max-long-edge", type=int, default=1920)
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("torch is not installed; this tool needs the real model stack")
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; re-run with --device cpu (slow) or on a GPU box")
        return 2

    passed = True
    if args.check in ("all", "semantics"):
        print("=== pyiqa batching semantics ===")
        passed &= check_semantics(args.device, args.iqa_max_long_edge)
    if args.check in ("all", "equivalence"):
        print("\n=== TorchBackend equivalence ===")
        passed &= check_equivalence(args.device, args.iqa_max_long_edge,
                                    args.images, args.iqa_batch_size)
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
