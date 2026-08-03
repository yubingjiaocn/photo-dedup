# Photo Dedup

Automatically de-duplicate a large local photo library (JPEG + motion photos +
video), then trash the same photos in Google Photos. Built for a Windows
desktop with an NVIDIA RTX 5070 Ti (16 GB) and photos on an HDD.

It is **selectively automatic** — only byte-identical duplicates become
`AUTO_REMOVE`; pHash-near duplicates, exposure extremes, and other boundary
cases go to `MAYBE`, and missing
features become `UNKNOWN`. Every group has a `KEEP`. Automatic items can only
move to recoverable trash (local undo / Google's trash), never permanent delete.

> Read `docs/ALGORITHM.md` for *why* every threshold is what it is. Read
> `DESIGN.md` for the original design.

---

## What it does (the hard cases it gets right)

- **Byte-identical duplicates** (same bytes copied twice) → keep one automatically.
  Re-encodes and pHash-near frames require review until face/eye evidence is calibrated.
- **Bursts / continuous shots** (many frames seconds apart) → keep the best.
- **Check-in photos** (a landmark is the subject, a person re-poses in front)
  → **kept separately**. A face-position guard refuses to merge frames where
  the person clearly moved, so your "same spot, different pose" shots survive.
- **Motion / live photos** (Xiaomi etc., embedded or paired `.mp4`/`.MP`) →
  the video lives and dies with its JPEG.

"Best" frame is chosen by image quality (MUSIQ), face sharpness, and
resolution — see `docs/ALGORITHM.md §4`.

---

## Install (Windows)

1. Install **Python 3.11+** (tick *"Add python.exe to PATH"*) and a current
   **NVIDIA driver** (CUDA 12.x runtime ships with the PyTorch wheels — no
   separate CUDA toolkit needed).
2. Double-click **`setup_windows.bat`**. It creates `.venv`, installs the
   CUDA PyTorch wheels, then the rest of `requirements.txt`, and prints whether
   your GPU is visible. Re-runnable.

No GPU / just trying it out? `setup_windows.bat` falls back to CPU wheels, and
you can also run everything with the built-in **stub** backend
(`features.backend: stub` in `config.yaml`) — deterministic, no models.

## Before a real GPU run: offline benchmark

Use the opt-in harness in [docs/WINDOWS_BENCHMARK.md](docs/WINDOWS_BENCHMARK.md)
on a small directory you deliberately selected, or on its checked-in synthetic
fixtures. It never scans `paths.root`, downloads a model, writes pipeline
manifests, changes decisions, or moves/deletes photos. It reports decode,
local SigLIP-provider, and Stage 1 timings with CUDA peak memory when available.

---

## Where to put things (SSD vs HDD)

Your photos (451 GB) stay on the HDD (`E:`). But the pipeline's working data is
small and benefits from the SSD:

| Thing | Size | Recommended location | Why |
|---|---|---|---|
| `inventory.sqlite` (`paths.db`) | a few hundred MB | **SSD** (system drive) | stages 2/3 do lots of random reads |
| `output/` (review + delete lists) | small | **SSD** | convenience |
| `models/` (YuNet ONNX + caches) | ~ hundreds MB | **SSD** | one-time |
| `_trash/` (`paths.trash`) | **can be 100+ GB** if you reclaim 20%+ | **usually HDD (`E:`)** | too big for most SSDs; move to SSD only if it fits |

The defaults already put the DB/output/models next to the repo (a relative
path), so **clone/unzip this project onto your SSD (e.g. `C:\tools\photo-dedup`)**
and you get the fast layout for free. `_trash` defaults to `E:/Photos/_trash`.

---

## Run (in order)

Activate the environment first: `\.venv\Scripts\activate`

1. **Edit `config.yaml`** → set `paths.root` to your library (e.g. `E:/Photos`).
   Optionally review the thresholds under `cluster:` (defaults are sensible).

2. **Stage 0 — inventory** (fast, one sequential pass, resumable):
   ```
   python -m src.stage0_inventory
   ```
   *~15–30 min for 66k files on an HDD (reads only headers).*

3. **Stage 1 — features** (GPU-heavy, resumable — Ctrl+C is safe):
   ```
   python -m src.stage1_features
   ```
   Stage 1 computes SHA-256 during its existing sequential file read—there is no
   extra whole-library hash pass. The default batch is 4 and an 80 MP inflight
   guard bounds RAM for 50 MP images; raise these only after measuring memory.
   *~1–4 h for 66k files; ~20–50 img/s on the 5070 Ti. Commits every 100 images,
   so re-running continues where it stopped.*

4. **Stage 2 — cluster** (CPU, minutes). Re-run freely after tweaking thresholds
   — it reads the cache, not the images:
   ```
   python -m src.stage2_cluster
   ```

### Offline scene/SigLIP shadow calibration (read-only)

After Stage 1 has accumulated `quality_meta.routing` records, create an
aggregate-only calibration report without opening photos, loading models, or
modifying decisions/manifests:

```
python -m src.scene_shadow_report --db inventory.sqlite --output-dir output/scene-shadow
# Or evaluate a portable JSONL stream of routing records:
python -m src.scene_shadow_report --jsonl routing-shadow.jsonl --output-dir output/scene-shadow
```

It writes atomic `scene-shadow-summary.json` and
`scene-shadow-tag-statistics.csv`, plus a resumable checkpoint. The summary
contains only anonymous record IDs for malformed inputs, hashes source/config
identity and prompt-bank/model provenance, and reports raw prompt statistics,
hard-negative gaps, UNKNOWN/reason distributions, and conflict/missing rates.
A mixed prompt-bank hash returns a non-zero CLI status and is explicitly marked
rejected; malformed records/NaN/schema mismatches are counted, never repaired.
This is **not** a threshold-setting or production-decision tool.

5. **Stage 3 — report + delete lists** (minutes):
   ```
   python -m src.stage3_report
   ```
   Produces in `output/`: `review.html` (separate risk-sorted Maybe/Unknown
   queues), `delete_local.txt`, `delete_cloud.json`, `summary.txt`. Manifests
   contain only `AUTO_REMOVE` decisions. Cloud entries without a timestamp and
   size are omitted because a filename is not unique.

6. **Review** — open `output/review.html` in a browser. Each group shows the
   keeper vs the delete candidates with scores. Spot-check a few dozen groups.
   Read `output/summary.txt` for totals (how many GB you'll reclaim).

7. **Delete locally** — first do a dry run (default), then apply:
   ```
   python -m src.execute_local                # dry run: writes output\execute_plan.txt, touches nothing
   python -m src.execute_local --no-dry-run   # verifies Stage-2 run/policy + manifest hash, then moves
   ```
   Changed your mind? Undo the whole session:
   ```
   python -m src.execute_local --undo E:/Photos/_trash/2026-07-23_101500
   ```
   Permanent deletion is intentionally unavailable for automatic decisions.

8. **Delete in Google Photos** — install **Tampermonkey** + the
   **Google Photos Toolkit** userscript, open `photos.google.com`, open the
   DevTools console (F12), and follow the header of
   **`scripts/gptk_delete.js`**: paste your `delete_cloud.json` into the script,
   run it once with `DRY_RUN = true` to see the match report, then set
   `DRY_RUN = false` to move the matches to Google's trash (60-day recovery).

---

## Tuning

Everything lives in `config.yaml`; `docs/ALGORITHM.md` explains each value.

### Optional closed-eye shadow metadata

`features.eye_detection` is disabled by default and only writes
`quality_meta.eye_detection`; it never changes Stage 2 or creates
`AUTO_REMOVE`. Install `requirements-eyes.txt`, place the configured
`face_landmarker.task` under `models_dir`, then enable it. No model is
downloaded automatically. Missing packages/assets and hash failures safely
record `UNKNOWN` without failing Stage 1.
Most common tweaks:

- **Automation coverage:** `conservative`, `balanced`, and `aggressive` use
  different quality-margin abstention thresholds. Exposure alone is never
  automatic: conservative marks it Unknown while balanced/aggressive queue it
  as Maybe. Byte identity is profile-independent.

- **Over-merging** (different photos grouped)? Raise `cluster.dinov2_threshold`
  (0.92 → 0.94) and/or lower `cluster.face_pose_shift_ratio` (0.30 → 0.20, more
  protective of check-in shots).
- **Under-merging** (bursts split)? Lower `dinov2_threshold` slightly, or raise
  `burst_window_seconds`.
- Re-run **only stage 2 + stage 3** after any threshold change — no image reads.

---

## FAQ / Troubleshooting

**CUDA / torch won't install.** Re-run `setup_windows.bat`; if the CUDA wheels
fail it falls back to CPU. You can also set `features.device: cpu` (slow) or
`features.backend: stub` (no models at all) in `config.yaml`. Confirm the GPU
with: `python -c "import torch; print(torch.cuda.is_available())"`.

**EXIF missing / weird timestamps.** Stage 0 falls back to a timestamp parsed
from the filename (`IMG_YYYYMMDD_HHMMSS`), then to file mtime, so clustering
always has a time. Wrong times only affect burst windowing, not exact-dup.

**Motion photo not recognized.** Embedded detection reads the JPEG header for
vendor XMP markers by default (cheap). If your phone appends the video without
a marker, set `scan.embedded_full_scan_max_bytes` to e.g. `20000000` to also
scan file tails (slower). Paired `.mp4`/`.MP` sidecars are always detected.

**A group looks wrong in review.html.** Nothing is deleted yet — adjust
thresholds and re-run stages 2–3. The SQLite cache means this is seconds, not
hours.

**Interrupted a stage.** Just run it again. Stages 0 and 1 are resumable
(commit every 100 files); stages 2 and 3 are idempotent.

**Google Photos script finds nothing.** Make sure the GPTK userscript is
loaded (`window.gptkApi` exists in the console) and that `photos.google.com`
finished loading before you pasted the script.

---

## Layout

```
config.yaml                 # all tunables
requirements.txt            # deps (torch installed separately, see setup)
setup_windows.bat           # one-shot environment setup
src/                        # pipeline (each stage is a `python -m src.<stage>`)
scripts/gptk_delete.js      # Google Photos cloud-delete console script
docs/ALGORITHM.md           # every threshold explained
tests/                      # pytest suite (`python -m pytest tests/`)
```
