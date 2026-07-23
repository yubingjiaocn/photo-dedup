# HANDOFF — Photo Dedup toolkit v0.1

Built by a sub-agent from `DESIGN.md`. This file is the "what/why/next" for the
main session and Willy.

---

## 1. What I built

A complete, resumable, HDD-friendly photo-dedup pipeline plus a Google Photos
cloud-delete script. Full toolkit per the DESIGN directory spec:

```
photo-dedup/
├── README.md            # Willy-facing install + run + FAQ (incl. SSD/HDD layout)
├── HANDOFF.md           # this file
├── DESIGN.md            # original design (unchanged)
├── requirements.txt     # deps; torch installed separately from CUDA index
├── setup_windows.bat    # one-shot venv + CUDA torch + requirements
├── config.yaml          # all tunables (paths, thresholds, weights, safety)
├── conftest.py          # makes `src` importable in tests
├── src/
│   ├── config.py            # YAML load + defaults + resolved paths
│   ├── db.py                # SQLite schema (WAL) + all helpers
│   ├── motion_photo.py      # embedded + paired live-photo detection
│   ├── quality.py           # pHash(+numpy DCT fallback), sharpness, face-quality, keep-score
│   ├── stage0_inventory.py  # os.walk → files table (EXIF/filename/mtime time)
│   ├── stage1_features.py   # DINOv2 + MUSIQ/CLIP-IQA + YuNet  |  torch/stub backends
│   ├── stage2_cluster.py    # exact-dup + burst + check-in split + keep-select
│   ├── stage3_report.py     # review.html + delete_local.txt + delete_cloud.json + summary.txt
│   └── execute_local.py     # move-to-trash (default) / delete / --undo
├── scripts/gptk_delete.js   # GPTK console script (dry-run first, batched trash)
├── docs/ALGORITHM.md        # every threshold/formula justified
└── tests/                   # pytest: motion, db, cluster, + 10-image mini pipeline
```

### Design decisions worth knowing
- **Two feature backends behind one interface.** `torch` = real DINOv2-base +
  pyiqa (MUSIQ/CLIP-IQA) + OpenCV YuNet. `stub` = deterministic numpy features
  (grayscale-downsample embedding, Laplacian-variance quality, red-block "face"
  detector). Stages 2/3 don't care which ran. `backend: auto` uses torch if
  importable, else stub. This is what lets the whole pipeline run + be tested on
  a machine with no GPU/torch.
- **Everything caches in one SQLite file** (WAL + `synchronous=NORMAL`). Stage 1
  is the only expensive stage; stages 2/3 re-read the cache, so tuning
  thresholds is seconds, not hours. Stages 0/1 commit every 100 files → Ctrl+C
  resumes.
- **Check-in guard** is the crux (DESIGN's hard case). In the burst layer, if a
  candidate pair *both* have a usable face and the dominant face center moves
  more than `face_pose_shift_ratio` (30%), we refuse the union. Exact-dup layer
  (pHash) intentionally does **not** apply the guard (identical pixels ⇒ face
  didn't move). See `docs/ALGORITHM.md §3`.
- **Motion photos are bound to their JPEG.** `motion_partner_id` links pairs;
  stage 3 auto-appends sidecar videos to the local delete list. Embedded videos
  need no extra handling (deleting the one JPEG removes them).
- **Safety defaults:** `dry_run: true`, `mode: move`. Local deletes are
  reversible via `--undo <trash_session>`; cloud deletes hit Google's 60-day
  trash.

### Patch folded in (hardware note from Willy, mid-build)
- `paths.db`/`output`/`models` default to **relative paths next to the repo** →
  put the repo on the **SSD** and the DB is on SSD automatically. Config
  comments + README "Where to put things" table spell out the SSD/HDD tradeoff;
  `_trash` stays on HDD by default (can be 100+ GB). No DB write-strategy change
  (WAL only), as requested.

### On BuIQA (parallel research)
A sibling sub-agent's `research/BUIQA_ASSESSMENT.md` concludes **do not** replace
MUSIQ with BuIQA (no open weights, burst-sequence-specific architecture,
marginal gain on subjective "pick best" tasks, no MUSIQ comparison in the
paper). My design already matches that call: **MUSIQ is primary, CLIP-IQA is a
stored auxiliary, and the backend `quality()` interface makes any future metric
a drop-in** without touching stages 2/3.

---

## 2. What's verified

- `pytest tests/` → **26 passed** (unit: motion detection, DB schema/resume,
  DSU, face-split, keep-select, pHash/hamming; plus the full mini pipeline).
- **Mini pipeline** (10 synthetic images: 3-frame burst + 3-frame check-in +
  4 independent) → stage 2 yields **exactly 1 group** (the burst), stage 3
  emits **exactly 2** local deletes + 2 cloud items. This directly exercises the
  check-in guard (the check-in trio is correctly NOT merged).
- **Real artifact generation**: `review.html` (16 KB, base64 thumbnails inline),
  `delete_cloud.json` (correct `YYYY-MM-DDTHH:MM:SS` format), `summary.txt`,
  `delete_local.txt`.
- **Safety cycle**: `execute_local` dry-run (touches nothing) → `--no-dry-run`
  move (relocates + writes `_undo_manifest.json`) → `--undo` (restores all
  files). Verified byte-for-byte file restoration.
- Every source file is **< 500 lines** (largest 349). All stage CLIs
  (`python -m src.<stage> --help`) import and parse cleanly with no torch.

## 3. What's NOT done / caveats (be honest)

- **The torch backend was written but not executed** — this box has no GPU/torch.
  DINOv2 (`AutoModel.pooler_output`), pyiqa MUSIQ/CLIP-IQA tensor calls, and
  OpenCV `FaceDetectorYN` usage are coded to the documented APIs but should be
  smoke-tested on the Windows machine (a 100-image `--limit 100` run). Any API
  drift is isolated to `TorchBackend` in `src/stage1_features.py`.
- **GPTK API field names are version-dependent.** `scripts/gptk_delete.js`
  probes several field-name variants (`dedupKey`/`mediaKey`, `items`/`mediaItems`,
  `nextPageId`/`nextPageTimestamp`/`cursor`) and defaults to `DRY_RUN=true`.
  The pagination cursor + `getItemsByUploadedDate`/`moveItemsToTrash` shapes are
  marked `TODO: confirm against your GPTK build`. Verify with a dry run before
  trusting it.
- **Never run on the real 66k library** — impossible here. Correctness is proven
  on synthetic data; performance numbers in README are DESIGN estimates.
- **Check-in guard uses face *position*, not identity.** Two *different* people
  standing in the same spot wouldn't be split by position alone. Face-embedding
  identity is a noted future upgrade; position covers the stated "same person
  re-poses" case and is cheap.
- **pHash uses `imagehash` when present, else a self-contained numpy DCT.** Both
  paths tested to agree on identical images; the numpy fallback keeps the core
  from hard-depending on scipy.

## 4. Willy's next steps

1. Copy the repo onto the **SSD** (e.g. `C:\tools\photo-dedup`). Run
   `setup_windows.bat`. Confirm `torch.cuda.is_available()` is `True`.
2. Edit `config.yaml` → `paths.root: E:/Photos`.
3. **Smoke test the GPU stage on a subset first:**
   ```
   python -m src.stage0_inventory
   python -m src.stage1_features --limit 200
   python -m src.stage2_cluster
   python -m src.stage3_report
   ```
   Open `output/review.html`. If groups look sane, run stage 1 without `--limit`
   for the full library (it resumes from the 200 already done).
4. Review → `execute_local` dry-run → `--no-dry-run` (move). Keep `_trash` until
   you're happy; `--undo` if needed.
5. Cloud: Tampermonkey + GPTK, run `scripts/gptk_delete.js` with `DRY_RUN=true`,
   check the match report, then `DRY_RUN=false`.

## 5. Known risks / to-verify checklist

- [ ] `TorchBackend` runs end-to-end on Windows/CUDA (DINOv2 pooler_output dim
      = 768; pyiqa metric return shapes; YuNet ONNX auto-download).
- [ ] YuNet model URL still valid; downloads to `models/`.
- [ ] GPTK API method/field names match the installed userscript version.
- [ ] On real data, sanity-check `cluster.dinov2_threshold` (0.92) and
      `face_pose_shift_ratio` (0.30) against a review.html sample; tune per
      `docs/ALGORITHM.md`.
- [ ] Motion-photo detection: if any phone appends video with no XMP marker,
      bump `scan.embedded_full_scan_max_bytes`.

---

*v0.1 — clustering is fully re-runnable from the SQLite cache, so threshold
tuning after seeing real results is cheap. Start with a `--limit` smoke test.*
