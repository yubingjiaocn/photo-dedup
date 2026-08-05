"""Adversarial checks the test suite cannot express as unit tests.

1. Stage 1 must work with torch entirely unimportable (CPU-only user).
2. Prefetch must not deadlock when a batch is larger than the queue.
3. Ctrl+C (KeyboardInterrupt) mid-loop must commit what was done and leave no
   threads behind.
4. features.thumbnails disabled + no eye detector + no scene router.
"""
import builtins
import shutil
import sys
import threading
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORK = Path("/tmp/adversarial_stage1")
shutil.rmtree(WORK, ignore_errors=True)


def library(root, count):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (320 if index % 2 else 240, 240 if index % 2 else 320),
                  (10 + index * 9, 60, 120)).save(root / f"IMG_2026010{index}_1200.jpg")


def config(directory, root, **features):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"root": str(root), "db": str(directory / "inventory.sqlite"),
                  "output_dir": str(directory / "out"),
                  "models_dir": str(directory / "models"),
                  "trash": str(directory / "trash")},
        "features": {"backend": "stub", "batch_size": 3, **features},
    }))
    return str(path)


root = WORK / "photos"
library(root, 7)

# --- 1. torch entirely unimportable --------------------------------------
real_import = builtins.__import__
blocked = []


def blocking_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        blocked.append(name)
        raise ImportError("no module named 'torch' (simulated CPU-only machine)")
    return real_import(name, *args, **kwargs)


for module in [m for m in list(sys.modules) if m == "torch" or m.startswith("torch.")]:
    del sys.modules[module]
builtins.__import__ = blocking_import
try:
    from src import stage0_inventory, stage1_features
    from src.stage1_settings import resolve_loop_settings   # noqa: F401

    cfg = config(WORK / "cpu_only", root)
    stage0_inventory.run(config_path=cfg)
    stats = stage1_features.run(config_path=cfg, backend_override="stub")
    print(f"[1] CPU-only (torch import blocked): processed={stats['processed']} "
          f"failed={stats['failed']} prefetch={stats['loop_settings']['prefetch_enabled']}")
    assert stats["processed"] == 7 and stats["failed"] == 0
    snapshot = stats["phase_telemetry"]
    assert snapshot["worker_seconds"] > 0
    assert snapshot["gpu_event_synchronisations"] == 0
    print(f"[1] blocked torch imports: {sorted(set(blocked))[:3] or 'none needed'}")
finally:
    builtins.__import__ = real_import

# --- 2. batch larger than the queue --------------------------------------
cfg = config(WORK / "tiny_queue", root, batch_size=7, prefetch_batches=1, cpu_workers=2)
stage0_inventory.run(config_path=cfg)
stats = stage1_features.run(config_path=cfg, backend_override="stub")
print(f"[2] batch(7) > queue(1): processed={stats['processed']} "
      f"batches={stats['phase_telemetry']['batches']}")
assert stats["processed"] == 7

# --- 3. KeyboardInterrupt mid-loop ---------------------------------------
from src import db  # noqa: E402

cfg_dir = WORK / "interrupted"
cfg = config(cfg_dir, root, cpu_workers=4, prefetch_batches=2)
stage0_inventory.run(config_path=cfg)
original_consume = stage1_features._consume_batch
calls = {"n": 0}


def interrupting_consume(*args, **kwargs):
    calls["n"] += 1
    if calls["n"] == 2:
        raise KeyboardInterrupt("user pressed Ctrl+C")
    return original_consume(*args, **kwargs)


before_threads = threading.active_count()
stage1_features._consume_batch = interrupting_consume
try:
    stage1_features.run(config_path=cfg, backend_override="stub")
    raise AssertionError("KeyboardInterrupt did not propagate")
except KeyboardInterrupt:
    print("[3] KeyboardInterrupt propagated (not swallowed)")
finally:
    stage1_features._consume_batch = original_consume

import time  # noqa: E402

deadline = time.monotonic() + 10
while threading.active_count() > before_threads and time.monotonic() < deadline:
    time.sleep(0.05)
leaked = [t.name for t in threading.enumerate() if t.name.startswith("stage1-")]
print(f"[3] threads before={before_threads} after={threading.active_count()} leaked={leaked}")
assert not leaked, leaked

# Resume must pick up the remainder and finish cleanly.
resume = stage1_features.run(config_path=cfg, backend_override="stub")
conn = db.open_db(cfg_dir / "inventory.sqlite")
statuses = [r[0] for r in conn.execute("SELECT status FROM features ORDER BY file_id")]
conn.close()
print(f"[3] resume after Ctrl+C: processed={resume['processed']} "
      f"total_rows={len(statuses)} all_done={set(statuses) == {'done'}}")
assert len(statuses) == 7 and set(statuses) == {"done"}

# --- 4. no thumbnails, no eye detector, no router -----------------------
cfg = config(WORK / "minimal", root, cpu_workers=4, prefetch_batches=2,
             thumbnails={"enabled": False})
stage0_inventory.run(config_path=cfg)
stats = stage1_features.run(config_path=cfg, backend_override="stub")
print(f"[4] thumbnails off: processed={stats['processed']} "
      f"thumbnails_enabled={stats['thumbnails']['enabled']}")
assert stats["processed"] == 7 and stats["thumbnails"]["enabled"] is False

# --- 5. telemetry disabled entirely -------------------------------------
cfg = config(WORK / "no_telemetry", root, cpu_workers=4, prefetch_batches=2,
             telemetry={"enabled": False})
stage0_inventory.run(config_path=cfg)
stats = stage1_features.run(config_path=cfg, backend_override="stub")
print(f"[5] telemetry off: processed={stats['processed']} "
      f"enabled={stats['phase_telemetry']['enabled']}")
assert stats["processed"] == 7 and stats["phase_telemetry"]["enabled"] is False

print("\nALL ADVERSARIAL CHECKS PASSED")
