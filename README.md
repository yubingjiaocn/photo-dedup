# Photo Dedup

## 5-minute Quick Start (Windows)

This first run only creates a review page. It **never calls `execute_local` and
never moves or deletes a photo**.

1. Install **Python 3.12** (recommended; 3.11+ is accepted), open Command Prompt
   or PowerShell in this folder, then set up the environment. Python 3.14 is
   allowed on a best-effort basis; setup will report the exact package if its
   Windows wheel is not available yet:

   ```bat
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
   .venv\Scripts\activate
   ```

2. Smoke-test 20 photos with the lightweight backend (no GPU/models needed).
   Put `--output` on the SSD, because it holds the DB and the thumbnail cache:

   ```bat
   python -m src.run_pipeline --root "E:\Photos" --output "C:\photo-review-smoke" --backend stub --limit 20
   ```

   **One `--output` belongs to exactly one `--root`.** The output directory
   records the root it was created for, so scanning `E:\Photos\2026` into the
   output of an earlier `E:\Photos\2015` run stops immediately with a
   `ParameterError` that names both roots — nothing is read, written, or
   deleted. Re-running with the same root always resumes (case and `\` vs `/`
   do not matter). Use one output directory per root.

3. Run the real local models. Start with a limit, then step it up; omit
   `--limit` when you are ready for the whole library:

   ```bat
   python -m src.run_pipeline --root "E:\Photos" --output "C:\photo-review" --backend torch --limit 100
   python -m src.run_pipeline --root "E:\Photos" --output "C:\photo-review" --backend torch --limit 1000
   python -m src.run_pipeline --root "E:\Photos" --output "C:\photo-review" --backend torch
   ```

   Each run is resumable: work already done (features **and** thumbnails) is
   skipped, so stepping the limit up costs only the new photos.

4. When every stage has finished, the command starts a local-only web server and
   opens the paged review. It prints a URL such as
   `http://127.0.0.1:54321/review.html`; press Ctrl+C when finished. If browser
   auto-open is unwanted, add `--no-open`. To use a fixed port, add
   `--port 8765`. To only generate files without serving, add `--no-serve`:

   ```bat
   start "" "C:\photo-review\review.html"
   ```

   With `--no-serve` the page shows a small static preview only; the review
   workbench and the paged ALL/MAYBE/UNKNOWN browsing need the local server.

   Every run also appends a diagnostic log to
   **`<output>\photo-dedup.log`**. It captures the console output, Python and
   platform details, Torch/CUDA/GPU detection, selected options, and complete
   error tracebacks. If something fails, send that file for diagnosis. The log
   includes local root/output paths and may include filenames from errors, but
   never image bytes or embeddings.

The server uses only Python's standard library, listens on `127.0.0.1`, and
serves the output folder plus a read-only paged API. To reopen an existing
review later **without rerunning any stage**, use the same output directory:

```bat
python -m src.serve_review --output "C:\photo-review"
```

This refreshes only the committed front-end build in that output directory and
then serves its existing `inventory.sqlite`, thumbnails and `review_state.json`.
It does not scan, cluster, regenerate features or touch an original. Add
`--port 8765` for a fixed port or `--no-open` to suppress browser launch.

The pipeline command stores its
resumable `inventory.sqlite` beside the review output and uses a temporary
runtime config, so it does not edit `config.yaml` — and you do not need to
either: `--root`/`--output` are the only paths you type. `config.yaml` supplies
thresholds and model settings; pass `--config my-thresholds.yaml` to use a
variant of it (its `paths:` are ignored). If a run fails, the final
`[pipeline][ERROR]` line states the cause. Stub results are for checking the
workflow; use `torch` before making real review decisions.

The run prints and writes `performance.txt` with each stage's wall time, Stage 0
files/second, Stage 1 images/second, and a **rough linear Stage 1** ETA for your
whole library. Even with `--limit`, Stage 0 cheaply counts all matching directory
entries without opening the unselected photos, so the projection uses the real
library item count rather than the sample count. It also reports the thumbnail cache: estimated full-library size,
actual current usage, and free space on the SSD. Below 20 GiB free it warns and
continues — the pipeline never demands tens of GB of headroom and never aborts
for space.

Browsing starts only after the pipeline finishes, on purpose: Stage 1 owns the
single sequential pass over the mechanical disk, and paging a UI at the same
time would turn that into random seeks.

Automatically de-duplicate a large local photo library (JPEG + motion photos +
video), then trash the same photos in Google Photos. Built for a Windows
desktop with an NVIDIA RTX 5070 Ti (16 GB), 32 GB RAM, and a library of roughly
**100,000+ JPEG / phone Motion-JPEG photos on a single mechanical HDD**.

The HDD is the bottleneck, so the design is deliberately single-stream: one
sequential read per eligible photo, one decode per eligible photo, and every derived artifact
(hash, embedding, quality, faces, **and the review thumbnail**) comes from that
same decode. By default, stills above **64 MP** or with a long-to-short-edge
ratio strictly above **3:1** are persistently marked `skipped_oversize` from
Stage 0's width/height metadata before full decode or GPU work. Ordinary 16:9,
21:9, and exactly 3:1 photos remain eligible; videos are never Stage-1
candidates. Re-runs do not retry skipped files, and older completed rows are
migrated out of clustering/review.

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
- **Motion / live photos** (Xiaomi etc., video embedded inside the JPEG) →
  handled as one intact file. Separate MP4/MOV files stay standalone.

"Best" frame is chosen by image quality (MUSIQ), face sharpness, and
resolution — see `docs/ALGORITHM.md §4`.

---

## Install (Windows)

1. Install **Python 3.12** (recommended; setup accepts 3.11+) and a current
   **NVIDIA driver** (the CUDA 12.8 runtime ships with the PyTorch wheels — no
   separate CUDA toolkit needed).
2. Run **`setup_windows.ps1`** from PowerShell. It creates `.venv`, installs the
   CUDA PyTorch wheels, then the rest of `requirements.txt`, and prints whether
   your GPU is visible. Re-runnable:

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
   ```

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

Your photos stay on the HDD (`E:`). The pipeline's working data is comparatively
small and belongs on the SSD:

| Thing | Size for a ~100k photo library | Recommended location | Why |
|---|---|---|---|
| `inventory.sqlite` (`paths.db`) | roughly 0.5–1 GB | **SSD** | stages 2/3 do lots of random reads |
| `output/thumbs/` (review thumbnails) | roughly **3–5 GB** (~30–45 KiB × photos) | **SSD** | the whole point: paged review without touching the HDD |
| `output/` (review page + delete lists) | a few MB | **SSD** | convenience |
| `models/` (YuNet ONNX + caches) | ~ hundreds MB | **SSD** | one-time |
| `_trash/` (`paths.trash`) | **can be 100+ GB** if you reclaim 20%+ | **usually HDD (`E:`)** | too big for most SSDs; move to SSD only if it fits |

With ~200 GB free on the SSD there is ample room: the thumbnail cache for 100k
photos is single-digit GB, and originals are never copied. The run prints the
estimated and actual cache size plus SSD free space; below 20 GiB free it warns
and keeps going.

The defaults already put the DB/output/models next to the repo (a relative
path), so **clone/unzip this project onto your SSD (e.g. `C:\tools\photo-dedup`)**
and you get the fast layout for free. `_trash` defaults to `E:/Photos/_trash`.

### The review UI (paged, SSD-only thumbnails)

Stage 1 writes one ~320 px JPEG per still image into `output/thumbs/`, named by
inventory file id, using the image it had already decoded for the models — no
extra read and no second decode. Stage 3 then builds a compact ordered index
inside SQLite, and the local server pages it.

The page opens on the **review workbench**: the pending group queue, one group at
a time. Beside it are two re-entry queues and three browse lists:

* **未审 / 稍后 / 已完成** — the group queues (no human decision yet / marked for
  later / accepted or picked). Each tab shows its own count.
* **全部（时间线）** — the complete timeline of every still image, ordered by
  capture time then directory then filename, with each photo's decision if it
  has one.
* **待确认** / **未知** — the risk-sorted review lists (`MAYBE` / `UNKNOWN`).

Queue membership is computed **on the server**: `/api/page?view=GROUPS&queue=…`
returns one page of that queue with the queue's own `total`, so the browser never
downloads the whole group list to filter it. Page size is 50, 100 (default), or
200, and every page comes from SQLite with `LIMIT/OFFSET`.

`/api/thumb/<id>.jpg` serves **only** files that already exist in
`output/thumbs`. A photo whose thumbnail could not be produced shows an explicit
"thumbnail unavailable" tile (the reason is recorded in `review_summary.json` and
`summary.txt`) — it is never silently blank and never triggers a fallback read of
the original.

Browsing and pagination remain SSD-thumbnail-only. Browse tiles scale with the
window (a responsive `auto-fill` grid collapsing to one column when narrow), and
**D** switches between 舒适 and 紧凑 density, which is remembered. A tile is one
click anywhere on the card.

Reading an original is the only HDD access, and only for what is on screen: the
loopback server streams exactly that inventoried JPEG/PNG after validating its
stored size and mtime, and never returns a source path. Leaving a photo releases
the image URL. Nothing is prefetched — JSON and thumbnails are cheap, originals
are not.

The page is in Chinese. Internal values (`MAYBE`, `UNKNOWN`, `KEEP`,
`AUTO_REMOVE`, group types, grouping reasons) are translated for display and kept
verbatim in small type next to the translation, so the UI still lines up with the
DB, the manifests and `docs/ALGORITHM.md`. Status colours are consistent: yellow
is focus, green is finished, blue/violet is the AI recommendation, orange is
"later", and red means destructive only. The server binds `127.0.0.1` only — it
is not reachable from the LAN.

Re-runs reuse thumbnails only when the recorded source size/mtime and pixel size
still match and the JPEG is really on disk; edit or replace an original and its
thumbnail is regenerated instead of served stale. Delete `output/thumbs/` to
rebuild the cache from scratch (that costs another Stage 1 pass over those
photos).

### The review workbench (keyboard-first)

The workbench is on screen as soon as the page loads — there is nothing to open
first. It has a group header (position in the queue, group type, member count,
current decision), one main photo, a filmstrip of the whole group, and a fixed
action bar.

**Review actions**

| Key | Action |
|---|---|
| `A` | 保留 AI 推荐 — the group is finished and leaves the pending queue |
| `P` | 保留当前照片 — the photo on stage becomes the keeper; group finished |
| `M` | 稍后处理 — the group moves to the 稍后 queue |
| `U` | 撤销上一步 — undo the most recent decision and jump back to that group and photo |

Each action shows a short toast and moves on by itself, so a whole queue can be
walked from the keyboard. One write is in flight at a time: a held key or a fast
double-tap decides the group once, not twice, and the buttons are disabled while
the request is out. `A` is unavailable for a group that has no AI recommendation
in this scope (which a group straddling two photo roots can produce) — use `P`.
After each decision the server is asked which group comes next in the queue, so
advancing is always forward even when the queue loses a whole page. `U` is a real undo: the server keeps a bounded history
in `review_state.json`, so it restores the group's exact previous state —
including "no decision at all" — survives a page reload or a server restart, and
refuses to reach outside the current photo root. It also comes back to the photo
that was on screen, not the group's first member: every decision sends a
`context_file_id` that the server validates as a member of that group and stores
beside the undo step, never inside the decision (`A`/`M` still choose no photo).
When the pending queue empties, the page says **本轮审阅完成** and offers the 稍后
and 已完成 counts to jump to, rather than sitting on the last group.

**Navigation**

| Key | Action |
|---|---|
| `J`/`↓`, `K`/`↑` | next / previous group |
| `L`/`→`, `H`/`←` | next / previous photo in the group (clicking a filmstrip frame is equivalent) |
| `Esc` | close the browse viewer / close help |
| `?`/`F1` | keyboard help |

**Looking at the photo**

| Key | Action |
|---|---|
| hold `C` | blink the main photo to the AI recommendation; release to come back |
| `Shift`+`C` | two-pane compare (current vs. AI recommendation), again to close |
| wheel / drag | zoom / pan; both panes stay in sync while comparing |
| `F` or double click | back to fit |
| `1` | 100% (actual pixels) |
| `E` | show/hide 推荐依据 |
| `D` | list density 舒适 / 紧凑 |

Scale is capped and pan is bounded to the scaled image, so a photo cannot be
zoomed or dragged out of sight. The shortcut hint is always visible in the action
bar.

**推荐依据** is the one place algorithm diagnostics appear: closed by default, and
when opened it shows the quality score, face count, system decision and reason
for the members of the group on screen, with Chinese labels and the internal enum
in small type. Cards, headers and the progress line carry none of it.

A reload restores the queue, the density, the 推荐依据 state and the group you
were on; if that group has meanwhile left the queue, `/api/locate` lands you on
the next one.

All human decisions are stored in `output/review_state.json` (atomic writes on
every change). **Original photos remain read-only**; AI decisions and deletion
manifests are unchanged. Human state is an independent overlay.

Only the compiled review UI is served statically — `review.html` plus the hashed
`assets/*` chunks named in `review_assets.json`, and nothing else: `/api/action`
requires a JSON content type and rejects a cross-site `Origin`/`Referer` (a
header-less local script still works), and every other file in the output
directory — the deletion manifests, the review state, the summary, the log — is
reachable only through the read-only API, or not at all. The server reads that
manifest as its allow-list rather than trusting the directory, so a stray file
dropped in the output folder is never web-served. `review_state.json` is written compactly and fsynced
before the rename, so a power loss cannot leave a truncated file behind. A file
this build cannot read is renamed to `review_state.corrupt-<timestamp>.json` and
the reviewer sees a banner at the top of the page saying so and naming the kept
file — review continues normally, and new decisions save. If that rename itself
cannot be done, the decision is **refused** rather than overwriting the file the
warning just promised to keep. A write that fails for any other reason is rolled
back in memory too, so the API never reports a decision that is not on disk and
the next successful write cannot smuggle a failed one along.

Because a Stage 2 rerun can hand the same `group_id` to a different set of
photos, every decision *and* every undo step records the fingerprint of the
members it was about, and all of them are re-checked against the group's current
members when the server starts. Anything that cannot be proven to still match is
ignored — including a step whose decision was already cleared, which is otherwise
exactly how a stale decision could come back through undo. Pruning happens in
memory only, and the original file is copied to `review_state.pre-prune.json`
first, so nothing the reviewer recorded is destroyed.

Delete manifests still contain only byte-identical `AUTO_REMOVE` items. Manual
decisions do not create new deletion entries — they guide which groups need
further attention.

The queues are classified in memory from `review_state.json` on each request,
which is comfortably fast at the size this tool targets:
`tests/test_review_scale.py` holds a synthetic 30k-group library under a loose
per-interaction ceiling as a regression fence. Much larger libraries are not a
tested configuration — the review index in SQLite scales, this in-memory
classification is what would need to move first.

`tests/test_review_queues.py` gates the queue totals, paging, state transitions,
undo and root scoping; `tests/test_review_hardening.py` gates the cross-site,
static-exposure and state-durability rules; `tests/test_review_frontend_build.py`
gates the committed build, its manifest and the served static allow-list. The
review UI itself is a Preact application under `frontend/` (see
[`frontend/README.md`](frontend/README.md) and [`FRONTEND_DECISION.md`](FRONTEND_DECISION.md));
its logic is unit-tested with Vitest (`frontend/test/`) and its real-browser
behaviour by `frontend/harness/` (run `python3 frontend/harness/run_all.py`),
which `scripts/verification/album_viewer_browser.py` forwards to.

---

## Run (in order)

Activate the environment first: `\.venv\Scripts\activate`

**The normal way to run everything is the one command from the Quick Start** — it
performs steps 2-5 below in order and then serves the review:

```bat
python -m src.run_pipeline --root "E:\Photos" --output "C:\photo-review" --backend torch
```

`--root` and `--output` are the only paths you ever type. The command derives
`paths.root`, `paths.db` (`<output>\inventory.sqlite`) and `paths.output_dir`
from them into a temporary runtime config, so **you never edit path settings in
`config.yaml`** and there is no per-run config file to maintain. `config.yaml`
holds *tunables only* (thresholds, quality weights, model settings); if you keep
a variant of it, pass it with `--config my-thresholds.yaml` — its `paths:` are
ignored, because the paths you just typed are the ones you mean. To re-run only
Stage 2+3 later, use `rebuild_review` with the same two arguments (step 4).

The individual stage commands below are the advanced path: run one at a time, and
inspect or repeat a single stage. **They take their paths from `config.yaml`**, so
this is the one case where you set `paths.root`/`paths.db`/`paths.output_dir`
yourself.

1. **Advanced path only — edit `config.yaml`** → set `paths.root` to your library
   (e.g. `E:/Photos`), and `paths.db`/`paths.output_dir` if the defaults beside
   the repo are not where you want them. Skip this entirely when using
   `run_pipeline`/`rebuild_review`. Optionally review the thresholds under
   `cluster:` (defaults are sensible).

2. **Stage 0 — inventory** (fast, one sequential pass, resumable):
   ```
   python -m src.stage0_inventory
   ```
   *Roughly 20–60 min for ~100k files on a 7200 RPM HDD (reads only headers);
   the actual number for your disk is printed as `stage0 throughput`.*

3. **Stage 1 — features** (GPU-heavy, resumable — Ctrl+C is safe):
   ```
   python -m src.stage1_features
   ```
   Stage 1 computes SHA-256 during its existing sequential file read—there is no
   extra whole-library hash pass, and the ~320 px review thumbnail is written
   from that same single decode. The default batch is 4 and an 80 MP inflight
   guard bounds RAM for 50 MP images; raise these only after measuring memory
   (32 GB RAM is not a reason to raise them blindly).

   While the GPU works on one batch, Stage 1 reads and prepares the next ones:
   `features.cpu_workers` (4) threads do the per-image CPU work and
   `features.prefetch_batches` (2) bounds how much may queue. **There is still
   exactly one reader thread** — a second one would turn the single sequential HDD
   stream into random seeks — and everything stateful (face detector, thumbnail
   cache, the database) stays on the main thread. The run prints the resulting
   memory ceiling. Set `cpu_workers: 0` for the original single-threaded pipeline.

   MUSIQ and CLIP-IQA are called **once per distinct IQA input size per batch**
   (`features.iqa_batch_size`, 4) instead of once per photo. Only images whose
   bounded IQA input has identical dimensions share a call: padding different
   sizes into one tensor would change the scores (measured: 1.7-3.4 MUSIQ points),
   so it is never done. `performance.txt` prints the call counts.

   *Expect hours for ~100k files; the run reports the measured images/s and a
   rough linear ETA instead of a guess. Commits every 100 images, so re-running
   continues where it stopped.*

4. **Stage 2 — cluster** (CPU, minutes). Re-run freely after tweaking thresholds
   — it reads the cached features, not the images:
   ```
   python -m src.stage2_cluster
   ```
   To re-run only Stage 2 and Stage 3 for a pipeline run's output directory
   (no image reads, no feature recomputation), use the rebuild command — it takes
   **the same `--root`/`--output` you gave `run_pipeline`**:
   ```
   python -m src.rebuild_review --root "E:\Photos" --output "C:\photo-review"
   ```
   It derives `paths.root`, `paths.db` (`<output>\inventory.sqlite`) and
   `paths.output_dir` from those two arguments, exactly like `run_pipeline` does,
   so a rebuild can never cluster a different database than the run it rebuilds.
   **There is no per-run config file** — paths are never read from a config file,
   only from `--root`/`--output`.

   The rebuild **fails closed before touching anything** if
   `<output>\inventory.sqlite` is missing, is not a photo-dedup inventory, is
   bound to a different `--root`, or has no completed Stage 1 with usable
   features. It never runs Stage 0/1, so the feature cache is never rebuilt by
   accident.

   Trying different thresholds? Put them in a copy of `config.yaml` and pass
   `--config`. It supplies **tunables only** (`cluster:`, `quality:`,
   `decision:`); any `paths:` inside it are ignored, because `--root`/`--output`
   decide the paths:
   ```
   python -m src.rebuild_review --root "E:\Photos" --output "C:\photo-review" --config my-thresholds.yaml
   ```

### Leakage-safe multi-scene shadow evaluation

Offline evaluation uses the versioned contract in `src/dataset_contract.py`.
A manifest contains only `dataset_id`, `split`, `owner_scope`, and event IDs;
it contains no source paths or labels. Splits are event-level: the same event
cannot appear in train/tune and held-out manifests.

- Willy's existing datasets are development data and must be marked `train` or
  `tune`.
- Lin's datasets must be marked `held_out`. Threshold search, training, prompt
  selection, and preset selection reject held-out manifests before reading
  labels. Held-out labels may be used only by `final_evaluation`, which returns
  a report-only contract with policy/config write-back and delete/trash
  authority disabled.
- `src/offline_evaluation.py` validates phase-level keeper annotations and
  compares baseline/candidate phase recall, keeper precision, and selected
  keepers per phase. It does not mutate runtime policy.

Visual groups now receive a deterministic cached-feature phase pass. Time gaps,
embedding changes, face-count changes, and face/subject-position changes create
phase boundaries. Every phase has at least one keeper; large phases and missing
features add keepers. Ranking combines IQA, face clarity, exposure, subject
completeness/occlusion proxies, then greedily adds visual diversity. The result
is stored in `groups.decision_json` with reason codes/evidence. Missing evidence
sets review-required and increases retention; it never grants removal authority.
`AUTO_REMOVE` remains byte-identical-only.

The checked-in tests use synthetic metadata only; no real held-out labels or
source images are included. Real held-out evaluation remains a separate final
step and must not feed results back into thresholds, prompts, or presets.

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
   Produces in `output/`: the paged review UI (`review.html`, the hashed
   `assets/*` chunks, and the `review_assets.json` allow-list — the committed
   `frontend/dist/` build copied in, no Node required), plus
   `review_summary.json`, `delete_local.txt`, `delete_cloud.json`, `summary.txt`,
   and the compact pagination index inside `inventory.sqlite`. Manifests contain
   only `AUTO_REMOVE` decisions. Cloud entries without a timestamp and size are
   omitted because a filename is not unique. Stage 3 reads no photos at all.

6. **Review** — open `output/review.html` (the one-command pipeline serves it
   for you). It lands on the 未审 group queue; walk it with `A`/`P`/`M`, or switch
   to the 全部 timeline / 待确认 / 未知 browse lists for a full pass over the
   library. Read `output/summary.txt` for totals.

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

Everything lives in `config.yaml` — **tunables only**, no paths to maintain;
`docs/ALGORITHM.md` explains each value. Edit it in place, or keep a variant and
pass it to either entry point with `--config`:

```bat
python -m src.run_pipeline    --root "E:\Photos" --output "C:\photo-review" --config my-thresholds.yaml
python -m src.rebuild_review  --root "E:\Photos" --output "C:\photo-review" --config my-thresholds.yaml
```

Both commands ignore any `paths:` in that file: `--root`/`--output` always decide
`paths.root`, `paths.db` and `paths.output_dir`. A missing `--config` file is an
error, never a silent fall back to defaults. Relative paths *inside* the file
(`features.scene_routing` model/prompt) keep resolving beside that file, not
beside the temporary runtime config. After a threshold change, re-run only
Stage 2+3 with `rebuild_review` — no photos are read.

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
vendor XMP markers by default (cheap), which covers phones that append the video
to the JPEG tail with a proper marker. If your phone appends the video without
any marker, set `scan.embedded_full_scan_max_bytes` to e.g. `20000000` to also
scan file tails (slower). Separate `.mp4`/`.MP` files are not paired by name.

**Thumbnails are missing in the review.** The review UI deliberately never reads
an original photo, so a missing thumbnail always means Stage 1 could not produce
one. The tile states the reason, and `review_summary.json` lists every failure.
Fix the source file (or accept it as unreadable) and re-run Stage 1 — only the
affected photos are reprocessed.

**A group looks wrong in review.html.** Nothing is deleted yet — adjust
thresholds and re-run stages 2–3. The SQLite cache means this is seconds, not
hours.

**Interrupted a stage.** Just run it again. Stages 0 and 1 are resumable
(commit every 100 files); stages 2 and 3 are idempotent. Stage 0 also notices
photos whose size or mtime changed since the last scan and invalidates their
cached features, hash, and thumbnail, so an edited or replaced original is
never judged (or displayed) from stale data.

**Google Photos script finds nothing.** Make sure the GPTK userscript is
loaded (`window.gptkApi` exists in the console) and that `photos.google.com`
finished loading before you pasted the script.

**"incompatible output directory" / reports showed photos from another year.**
An output directory is bound to one photo root. Earlier builds recorded no root
identity, so reusing one `--output` for `E:\Photos\2015` and then
`E:\Photos\2026` mixed both inventories and every count (review header, ETA,
manifests) described the union. Now the mismatch fails closed before anything is
touched, and the message lists the recorded root, the relation (unrelated /
parent / child) and your options. Fixes:

* use a separate `--output` per root (recommended), or
* re-run with the recorded root to resume that inventory, or
* for an *older* database that already holds several roots: point `--root` at
  their common parent to adopt the whole inventory, or start a fresh output
  directory. Nothing is deleted either way — the old database is left intact.

Every report now prints the scope line it describes, e.g.
`scope: root=E:\Photos\2026 run=<id> files_in_scope=61234 seen_this_run=61234`.

**Stage 1 feels slow / the GPU only pulses.** `performance.txt` and the console
log carry the Stage 1 phase breakdown: cumulative seconds, count, ms/image and
percentage for source read, decode, SHA-256, embedding preprocess vs inference,
MUSIQ, CLIP-IQA, sharpness, YuNet, exposure, thumbnail resize vs encode+write,
DB write/commit, plus batch p50/p95 and the leftover `unaccounted` time. Numbers
are host wall time (labelled `host-wall`) unless a line is under the CUDA-event
section, because with an async CUDA backend host timing includes submit + wait.
Turn it off or change the CUDA-event sampling with `features.telemetry`
(`enabled`, `gpu_event_every`; `0` disables event sampling).

With prefetch enabled the breakdown has three separate sections, on purpose:
**in-loop** (main-thread work inside a batch), **producer-thread** (the read,
decode and preparation that ran concurrently, measured against their own total and
never added to the loop, because summing concurrent threads into one wall clock
would invent time), and **between-batch** (`waiting for prepared batch` — what
prefetch actually cost the main thread). `overlap` is producer seconds minus that
wait: work genuinely hidden behind the GPU, stated as a measurement rather than a
claimed speedup.

If `waiting for prepared batch` dominates, the producer is the bottleneck — raise
`cpu_workers`. If it is near zero and the model lanes dominate, the GPU is now the
bottleneck, which is the intended outcome.

---

## Layout

```
config.yaml                 # all tunables (paths come from --root/--output)
requirements.txt            # deps (torch installed separately, see setup)
setup_windows.bat           # one-shot environment setup
src/                        # pipeline (each stage is a `python -m src.<stage>`)
src/thumbnails.py           # SSD thumbnail cache written from Stage 1's decode
src/review_server.py        # local-only paged review API (SSD thumbnails only)
src/review_assets.py        # locate/install the committed frontend build + manifest
src/review_page.py          # install the compiled review UI into the output dir
frontend/                   # Preact review UI (Vite + TypeScript; dist/ is committed)
frontend/dist/              # committed build the pipeline serves (no Node at runtime)
src/pipeline_report.py      # stage timings, throughput, rough ETA, disk report
src/root_scope.py           # durable root identity + per-run scope (fail closed)
src/runtime_config.py       # the one --root/--output -> paths mapping both CLIs use
src/rebuild_review.py       # re-run Stage 2+3 only for an existing --output (fail closed)
src/schema.py               # SQLite DDL + additive migrations
src/stage1_telemetry.py     # Stage 1 per-phase timing (host-wall vs CUDA events)
src/stage1_pipeline.py      # bounded, ordered CPU prefetch (one sequential reader)
src/stage1_backends.py      # DINOv2 + shape-grouped batched MUSIQ/CLIP-IQA + YuNet
src/stage1_setup.py         # admission, pending-work query, detectors, closing stats
src/stage1_settings.py      # validated loop knobs + the stated memory bound
scripts/gptk_delete.js      # Google Photos cloud-delete console script
scripts/verification/       # real-weight/real-GPU/real-browser checks CI cannot express
docs/ALGORITHM.md           # every threshold explained
docs/STAGE1_THROUGHPUT.md   # the measurements behind the throughput defaults
docs/WINDOWS_BENCHMARK.md   # opt-in Windows throughput harness
docs/archive/               # completed design research (why the thresholds are these)
research/                   # runtime data only: SigLIP prompt bank (config points here)
tests/                      # pytest suite (`python -m pytest tests/`)
```

### Developer test gates

Use `scripts/test-fast.sh` while iterating or before a local commit. It keeps the
core DB, inventory, decision-safety, Stage 1 and thumbnail regressions, while
skipping the repeated localhost-server and 100k-library integration fixtures.
Run `scripts/test-full.sh` before push/release, and whenever changing the review
server, pagination, end-to-end pipeline, SigLIP/routing, or benchmark/reporting.
The full suite remains authoritative; tests are tiered, not deleted.

The pytest gates for the review server stay in Python: the queue/undo semantics
by `tests/test_review_queues.py`, the write and static boundary by
`tests/test_review_hardening.py`, the committed build and served allow-list by
`tests/test_review_frontend_build.py`, and the per-interaction cost by
`tests/test_review_scale.py`.

**Changing the review UI needs Node** (only for development — the runtime never
does). The UI is a Preact app under `frontend/`; rebuild and commit `dist/` after
any change (see [`frontend/README.md`](frontend/README.md)):

```bash
cd frontend && npm ci && npm run typecheck && npm test && npm run build
git add frontend/src frontend/dist   # commit source and the rebuilt artifact together
```

`npm test` runs the Vitest reducer/keyboard/component/pool suite. For behaviour
that touches focus, zoom gestures, request races or which URLs the page requests,
run the real-browser pass: `python3 frontend/harness/run_all.py` (or
`scripts/verification/album_viewer_browser.py`, which forwards to it).
