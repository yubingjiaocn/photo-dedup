"""End-to-end Stage 1 with the REAL torch stack, serial vs prefetched.

This is the check that cannot be done in CI: real DINOv2, real MUSIQ, real
CLIP-IQA, real YuNet, real CUDA, through the actual `stage1_features.run` -- twice
over the same synthetic library, once serial and once with prefetch + batched IQA.
The two must agree per image within the batch-kernel tolerance, and the optimised
run must show the reduced model-call counts.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db, stage0_inventory, stage1_features   # noqa: E402

WORK = Path("/tmp/e2e_torch_stage1_12mp")
COUNT = 40


def build(root):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    # 12 MP, matching the real library (phone photos), two orientations.
    sizes = [(4032, 3024), (3024, 4032), (4032, 3024), (4032, 3024), (3024, 4032), (4032, 3024)]
    for index in range(COUNT):
        w, h = sizes[index % len(sizes)]
        small = (rng.random((h // 12, w // 12, 3)) * 255).astype(np.uint8)
        Image.fromarray(small).resize((w, h), Image.BICUBIC).save(
            root / f"IMG_2026{index // 100 + 1:02d}{index % 100:02d}_120000.jpg",
            "JPEG", quality=90)


def config(directory, root, workers, prefetch):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"root": str(root), "db": str(directory / "inventory.sqlite"),
                  "output_dir": str(directory / "out"),
                  "models_dir": str(Path.cwd() / "models"),
                  "trash": str(directory / "trash")},
        "features": {"backend": "torch", "device": "cuda", "batch_size": 4,
                     "cpu_workers": workers, "prefetch_batches": prefetch,
                     "iqa_batch_size": 4, "iqa_max_long_edge": 1920,
                     "iqa_musiq": True, "iqa_clipiqa": True,
                     "thumbnails": {"enabled": True, "max_px": 320},
                     "telemetry": {"enabled": True, "gpu_event_every": 0}},
    }))
    return str(path)


def rows(db_path):
    conn = db.open_db(db_path)
    out = [dict(r) for r in conn.execute(
        "SELECT file_id, phash, content_sha256, dinov2_embedding, quality_score, "
        "quality_meta, face_count, faces_json, status FROM features ORDER BY file_id")]
    conn.close()
    return out


def run(name, workers, prefetch, root):
    directory = WORK / name
    shutil.rmtree(directory, ignore_errors=True)
    cfg = config(directory, root, workers, prefetch)
    stage0_inventory.run(config_path=cfg)
    t0 = time.perf_counter()
    stats = stage1_features.run(config_path=cfg, backend_override="torch")
    wall = time.perf_counter() - t0
    return stats, wall, rows(directory / "inventory.sqlite")


shutil.rmtree(WORK, ignore_errors=True)
root = WORK / "photos"
print(f"building {COUNT} synthetic ~3MP images ...")
build(root)

print("\n=== SERIAL (cpu_workers=0, prefetch=0) ===")
serial_stats, serial_wall, serial_rows = run("serial", 0, 0, root)
print("\n=== OPTIMISED (cpu_workers=4, prefetch=2, iqa_batch=4) ===")
fast_stats, fast_wall, fast_rows = run("fast", 4, 2, root)

print("\n\n################ COMPARISON ################")
print(f"processed serial={serial_stats['processed']} fast={fast_stats['processed']}")
assert serial_stats["processed"] == fast_stats["processed"] == COUNT
assert serial_stats["failed"] == fast_stats["failed"] == 0

max_q, max_clip, max_sharp, max_emb = 0.0, 0.0, 0.0, 0.0
for left, right in zip(serial_rows, fast_rows):
    assert left["file_id"] == right["file_id"]
    assert left["phash"] == right["phash"], f"phash differs for {left['file_id']}"
    assert left["content_sha256"] == right["content_sha256"]
    assert left["face_count"] == right["face_count"]
    assert left["status"] == right["status"] == "done"
    max_q = max(max_q, abs(left["quality_score"] - right["quality_score"]))
    lm, rm = json.loads(left["quality_meta"]), json.loads(right["quality_meta"])
    assert lm["iqa_input_size"] == rm["iqa_input_size"], (lm, rm)
    assert lm["iqa_scale"] == rm["iqa_scale"]
    max_clip = max(max_clip, abs(lm["clipiqa"] - rm["clipiqa"]))
    max_sharp = max(max_sharp, abs(lm["sharpness"] - rm["sharpness"]))
    assert lm["exposure"]["decode_long_edge"] == rm["exposure"]["decode_long_edge"]
    for key in ("clip_hi", "clip_lo", "ymean", "entropy_nonclip", "mass_usable"):
        assert abs(lm["exposure"][key] - rm["exposure"][key]) < 1e-9, key
    a = np.frombuffer(left["dinov2_embedding"], dtype=np.float16).astype(np.float32)
    b = np.frombuffer(right["dinov2_embedding"], dtype=np.float16).astype(np.float32)
    max_emb = max(max_emb, float(np.abs(a - b).max()))

print(f"max |MUSIQ diff|      = {max_q:.3e}")
print(f"max |CLIP-IQA diff|   = {max_clip:.3e}")
print(f"max |sharpness diff|  = {max_sharp:.3e}   (must be 0: same CPU code)")
print(f"max |embedding diff|  = {max_emb:.3e}   (float16 storage)")
print(f"phash / sha256 / faces / exposure : IDENTICAL")

sc = serial_stats["phase_telemetry"]["counters"]
fc = fast_stats["phase_telemetry"]["counters"]
print(f"\nserial    counters: {sc}")
print(f"optimised counters: {fc}")
print(f"\nMUSIQ calls: serial={sc['iqa_musiq_calls']} optimised={fc['iqa_musiq_calls']} "
      f"for {COUNT} images (old code would have been {COUNT})")
print(f"CLIP-IQA calls: serial={sc['iqa_clipiqa_calls']} optimised={fc['iqa_clipiqa_calls']}")
print(f"embed calls: serial={sc['embed_calls']} optimised={fc['embed_calls']}")

print(f"\nStage 1 wall: serial={serial_wall:.1f}s optimised={fast_wall:.1f}s "
      f"speedup={serial_wall / fast_wall:.2f}x")
sl, fl = serial_stats["phase_telemetry"], fast_stats["phase_telemetry"]
print(f"loop seconds: serial={sl['loop_seconds']:.1f}s optimised={fl['loop_seconds']:.1f}s "
      f"speedup={sl['loop_seconds'] / fl['loop_seconds']:.2f}x")
print(f"producer CPU seconds={fl['worker_seconds']:.1f}s "
      f"loop waited={fl['wait_accounted_seconds']:.1f}s "
      f"overlap={fl['overlap_seconds']:.1f}s")
print(f"images/s: serial={COUNT / sl['loop_seconds']:.2f} "
      f"optimised={COUNT / fl['loop_seconds']:.2f}")
print(f"\nthumbnails: serial={serial_stats['thumbnails']['created']} "
      f"fast={fast_stats['thumbnails']['created']}")

# resume must be a no-op
resume_cfg = config(WORK / "fast", root, 4, 2)
resume = stage1_features.run(config_path=resume_cfg, backend_override="torch")
print(f"resume run: processed={resume['processed']} total={resume['total']} "
      f"(must be 0/0)")
assert resume["processed"] == 0 and resume["total"] == 0
assert rows(WORK / "fast" / "inventory.sqlite") == fast_rows
print("resume left every row untouched")
print("\n############ REAL END-TO-END PASSED ############")
