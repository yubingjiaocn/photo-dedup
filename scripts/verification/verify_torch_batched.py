"""Real-weights verification of the batched IQA path against the old serial path.

Runs on this box (L40S) with the actual DINOv2 / MUSIQ / CLIP-IQA / YuNet stack.
It reconstructs the *old* per-image behaviour from the same prepared inputs and
compares it to the new batched path image by image, then counts model calls.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config                      # noqa: E402
from src.stage1_backends import TorchBackend, _shape_groups   # noqa: E402


class CountingTelemetry:
    """Minimal sink that only counts, so call counts are directly observable."""

    def __init__(self):
        self.counters = {}

    def phase(self, key, count=1):
        return _Null()

    def gpu_event_phase(self, key):
        return _Null()

    def count(self, key, value=1):
        self.counters[key] = self.counters.get(key, 0) + value


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def synth(seed, w, h):
    rng = np.random.default_rng(seed)
    small = (rng.random((max(2, h // 16), max(2, w // 16), 3)) * 255).astype(np.uint8)
    img = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


cfg = Config({
    "features": {"backend": "torch", "device": "cuda", "iqa_max_long_edge": 1920,
                 "iqa_batch_size": 4, "iqa_musiq": True, "iqa_clipiqa": True,
                 "yunet_input_width": 640, "yunet_score_threshold": 0.6,
                 "yunet_url": ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                               "face_detection_yunet/face_detection_yunet_2023mar.onnx")},
    "paths": {"models_dir": "./models"},
})
print("loading torch backend (real weights) ...")
t0 = time.perf_counter()
backend = TorchBackend(cfg)
print(f"loaded in {time.perf_counter() - t0:.1f}s; processor separability check passed")

# Mixed orientations + a small image => three distinct bounded IQA shapes.
images = [
    synth(0, 4032, 3024), synth(1, 3024, 4032), synth(2, 4032, 3024),
    synth(3, 1600, 1200), synth(4, 3024, 4032), synth(5, 4032, 3024),
]
prepared = [backend.prepare_cpu(image) for image in images]
shapes = [tuple(np.shape(item["iqa_array"])) for item in prepared]
print("bounded IQA shapes:", shapes)
print("shape groups (cap=4):", [g for g in _shape_groups(prepared, 4)])

# --- old behaviour: one MUSIQ + one CLIP-IQA call per image ---------------
serial_scores, serial_clip = [], []
with torch.no_grad():
    for item in prepared:
        tensor = (torch.from_numpy(item["iqa_array"]).permute(2, 0, 1)
                  .unsqueeze(0).to(backend.device))
        serial_scores.append(float(backend.musiq(tensor).item()))
        serial_clip.append(float(backend.clipiqa(tensor).item()))

# --- new behaviour: batched by shape --------------------------------------
telemetry = CountingTelemetry()
backend.telemetry = telemetry
results = backend.quality_prepared(prepared)
batched_scores = [score for score, _ in results]
batched_clip = [meta["clipiqa"] for _, meta in results]

musiq_diff = max(abs(a - b) for a, b in zip(serial_scores, batched_scores))
clip_diff = max(abs(a - b) for a, b in zip(serial_clip, batched_clip))
print(f"\nMUSIQ    serial={[round(v, 4) for v in serial_scores]}")
print(f"MUSIQ    batched={[round(v, 4) for v in batched_scores]}")
print(f"MUSIQ    max_abs_diff={musiq_diff:.3e}")
print(f"CLIP-IQA serial={[round(v, 5) for v in serial_clip]}")
print(f"CLIP-IQA batched={[round(v, 5) for v in batched_clip]}")
print(f"CLIP-IQA max_abs_diff={clip_diff:.3e}")
print(f"\ncall counters: {telemetry.counters}")
assert telemetry.counters["iqa_musiq_images"] == len(images)
assert telemetry.counters["iqa_musiq_calls"] == 3, telemetry.counters
assert telemetry.counters["iqa_clipiqa_calls"] == 3
print(f"=> 6 images scored with {telemetry.counters['iqa_musiq_calls']} MUSIQ calls "
      f"and {telemetry.counters['iqa_clipiqa_calls']} CLIP-IQA calls "
      f"(3 distinct shapes), instead of 6 + 6")

# --- per-image metadata preserved ----------------------------------------
for image, (_, meta) in zip(images, results):
    w, h = image.size
    expected_scale = 1920 / max(w, h) if max(w, h) > 1920 else 1.0
    assert abs(meta["iqa_scale"] - expected_scale) < 1e-9, meta
    assert meta["iqa_input_size"][0] <= 1920 and meta["iqa_input_size"][1] <= 1920
    assert np.isfinite(meta["sharpness"]) and meta["sharpness"] > 0
print("per-image iqa_scale / iqa_input_size / sharpness all preserved")

# --- embeddings: batched call vs per-image, and .faces on the main thread --
emb_batched = backend.embed_prepared(prepared)
with torch.no_grad():
    per_image = []
    for item in prepared:
        out = backend.model(pixel_values=item["embed_pixels"].to(backend.device))
        vector = (out.pooler_output if out.pooler_output is not None
                  else out.last_hidden_state[:, 0])
        per_image.append(vector.float().cpu().numpy()[0])
emb_diff = float(np.abs(np.asarray(per_image) - emb_batched).max())
cos = float(np.min([
    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    for a, b in zip(per_image, emb_batched)]))
print(f"embedding shape={emb_batched.shape} max_abs_diff={emb_diff:.3e} min_cosine={cos:.8f}")
print(f"embed counters: calls={telemetry.counters.get('embed_calls')} "
      f"images={telemetry.counters.get('embed_images')}")

faces = backend.faces_prepared(prepared[0])
print(f"faces_prepared -> {len(faces)} face(s) (YuNet ran on the main thread)")
print(f"peak VRAM = {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

# --- old single-image API still works (windows_benchmark / siglip runner) --
score, meta = backend.quality(images[0])
assert abs(score - serial_scores[0]) < 1e-2, (score, serial_scores[0])
single_emb = backend.embed_batch(images[:2])
assert single_emb.shape == (2, 768)
print(f"compat shims OK: quality()={score:.4f} embed_batch()={single_emb.shape}")
print("\nALL REAL-WEIGHT CHECKS PASSED")
