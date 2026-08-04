# ALGORITHM.md — thresholds, formulas, and the reasoning behind each number

This document explains **every tunable** in `config.yaml` and **why** its
default is what it is. If a run over- or under-merges, this is the file to
read before changing anything. All clustering is re-runnable from the SQLite
cache (stage 2 reads no images), so experimentation is cheap.

---

## 0. Mental model

The library is a mix of:

1. **True duplicates** — same file uploaded twice, or an original + a
   re-compressed copy. Pixels are ~identical.
2. **Bursts / continuous shots** — many frames of the same scene, seconds
   apart. We want to keep the best one.
3. **Check-in photos** *(the hard case)* — a landmark/building is the real
   subject and a person re-poses in front of it. These look near-identical to
   a naïve similarity metric but are **not** duplicates and must be kept.
4. **Unrelated photos** — everything else.

We find candidates for (1) with a perceptual hash but prove automatic byte
identity with SHA-256, (2) with semantic embeddings gated by a
time window, and (3) with a face-position guard that *refuses* to merge frames
where the person clearly moved.

---

## 1. Features (stage 1)

### Resource admission and bounded IQA
- `max_process_megapixels: 64` is checked from inventory `width × height`
  **before** full decode, hashing, thumbnail generation, or model/GPU calls.
  `max_process_aspect_ratio: 3.0` applies the same exclusion when
  `max(width/height, height/width)` is strictly greater than 3.0. Thus normal
  16:9 and 21:9 photos, plus the exact 3:1 boundary, remain eligible while
  extreme panoramas do not consume model work. Excluded stills are persisted
  as `features.status='skipped_oversize'` with reason `PIXEL_LIMIT` or
  `ASPECT_RATIO`; this also migrates formerly `done` rows out of Stage 2 and
  review. MP4/MOV rows are not still candidates and are never decoded by Stage
  1. The original is untouched.
- `iqa_max_long_edge: 1920` resizes eligible IQA/sharpness input with preserved
  aspect ratio and no upscaling. `quality_meta.iqa_input_size` and `iqa_scale`
  record the exact scoring scale. Laplacian sharpness uses float32 rather than
  full-resolution float64 working arrays.
- A requested CUDA backend fails clearly when CUDA is unavailable; it does not
  silently run expensive inference on CPU.

### pHash (perceptual hash), 64-bit
- **What:** DCT-based hash of a 32×32 grayscale reduction (`imagehash.phash`,
  with a self-contained numpy-DCT fallback so the core has no scipy hard-dep).
- **Why 64-bit:** standard; the Hamming distance between two 64-bit hashes is a
  robust "are these visually the same picture" signal that ignores JPEG
  recompression, minor resizing, and metadata changes.

### SHA-256 content hash
- Computed while Stage 1 performs its existing sequential compressed-file read;
  it does **not** add another library scan.
- This is the only P0 automatic duplicate proof. pHash Hamming 0–2 is only a
  candidate signal: blinks and tiny local edits can retain the same pHash.

### DINOv2 embedding (`facebook/dinov2-base`, 768-d)
- **What:** the CLS/pooler vector of a self-supervised ViT. Stored as float16
  (1536 bytes/image → ~150 MB for 100k images, fits in RAM for stage 2 on 32 GB).
- **Why DINOv2-base (not -small / CLIP):**
  - DINOv2 captures **scene structure and layout**, which is exactly what
    separates "same viewpoint of the same place" from "different place". CLIP
    leans more on semantic category ("a beach") and would happily call two
    *different* beaches similar — bad for de-dup.
  - `-base` (86M) is the accuracy/VRAM sweet spot on a 16 GB RTX 5070 Ti;
    `-small` trades away discrimination we actually need, `-large`/`-giant`
    give marginal gains for much more VRAM and time.
- **Cosine similarity** is the comparison metric (embeddings are L2-normalised).

### MUSIQ (via `pyiqa`) — primary image-quality score, 0..100
- **Why:** no-reference IQA that correlates well with human "is this a good
  photo" judgement (sharpness, exposure, noise). It decides *which* frame in a
  group to keep. Higher = better.

### CLIP-IQA (via `pyiqa`) — auxiliary quality signal, 0..1
- Stored in `quality_meta` as a secondary reference. Not weighted into the keep
  score by default (MUSIQ is the workhorse); kept so you can compare or swap.
- **BuIQA** is under separate evaluation; the backend interface (`quality()`
  returning `(score, meta)`) is designed so a better metric can be dropped in
  without touching stages 2/3.

### YuNet face detection (OpenCV, ONNX)
- **What:** fast, accurate face detector → bbox + 5 landmarks + confidence.
- **Why we need faces at all:** two reasons — (a) the check-in split (§3), and
  (b) face quality in keep-selection (a group with a sharp, confident face
  beats one where the face is blurry).
- `yunet_score_threshold: 0.6` — ignore low-confidence detections that are
  often texture false-positives.

### Explainable exposure features (numpy/OpenCV)

During the same Stage 1 decode, the image is reduced to a 512 px long edge and
raw metrics are stored under `quality_meta.exposure`: all-channel highlight and
shadow clipping ratios, largest connected clipping regions, usable-tone mass,
mid-tone anchor mass, non-clipped luminance entropy, mean luminance, plus the
same metrics for each already-detected face ROI. Labels are derived in Stage 2,
so changing decision thresholds does not re-read images. Reject requires
multiple conditions; silhouettes and local highlights are protected by anchor,
connected-region, entropy, and face-ROI gates. Until calibrated on real labels,
exposure classifications are review evidence only and never cause AUTO_REMOVE.

---

## 2. Clustering layers (stage 2)

A file lands in **exactly one** group. Layers run in priority order and a
union-find (DSU) merges members; each final group is labelled by its strongest
edge type (`exact_dup` > `burst` > `similar_scene`).

### Layer 1 — pHash near-duplicate candidates
- **Rule:** pHash Hamming distance ≤ `phash_hamming_threshold` (**default 2**).
- **Why 2:** 0 is only hash-identical, not byte-identical; 1–2 tolerates JPEG re-encode / a resave
  without letting genuinely different photos in. Above ~4 you start merging
  merely-similar images, which is what Layer 2 is *for* (with time gating).
- **Performance:** naïve all-pairs is O(N²) ≈ 10¹⁰ for 100k — too slow. We use
  **multi-index hashing**: split the 64-bit hash into four 16-bit bands. By the
  pigeonhole principle, two hashes within Hamming 2 must share **≥ 2** of the 4
  bands, so they will collide in at least one band bucket. We only compare
  within buckets. (Guaranteed correct for threshold ≤ 3; if you set it higher,
  increase the band count or accept recall loss — noted in code.)
- DSU connectivity never grants deletion trust. Every candidate is compared
  directly with the selected keeper, and only matching non-empty SHA-256 values
  use the automatic duplicate lane. Thus A≈B≈C chaining cannot remove C.

### Layer 2 — Burst / continuous shot
- **Rule:** within `burst_window_seconds` (**default 30 s**) *and* DINOv2 cosine
  ≥ `dinov2_threshold` (**default 0.92**) → same group, **unless** the check-in
  face guard fires (§3).
- **Why 30 s:** phone burst mode and "take three to be safe" behaviour cluster
  within seconds; 30 s is generous enough to catch human-paced re-shoots but
  short enough that unrelated photos rarely share a window. Chains are allowed
  (A~B, B~C ⇒ A,B,C) so a longer burst still forms one group.
- **Why cosine 0.92:** empirically the band where "same shot, tiny changes"
  lives. Lower (0.85) starts pulling in "same room, different subject"; higher
  (0.96) misses bursts where someone waved an arm. 0.92 is the conservative
  middle. Raise it if you see over-merging; lower it if bursts are split.
- **Time source:** EXIF `DateTimeOriginal`, falling back to a timestamp parsed
  from the filename (`IMG_YYYYMMDD_HHMMSS`), falling back to file mtime — so
  windowing always has *a* value.

### Layer 3 — Similar scene (loose) — **OFF by default**
- **Rule:** longer window (`loose_window_seconds`, default 300 s) + stricter
  cosine (`loose_dinov2_threshold`, default 0.96), same face guard.
- **Why off:** this is precisely where check-in photos get wrongly merged (same
  spot, minutes apart, person moved). Enable only if you specifically want to
  collapse "shot the same object repeatedly" and have reviewed the face guard.

---

## 3. The check-in guard (the whole reason this is hard)

When a candidate burst pair **both contain a usable face**
(confidence ≥ `min_face_score`, default 0.6), we compute the **normalised
displacement of the dominant (highest-confidence) face center**:

```
shift = max( |cx_i/W_i − cx_j/W_j| ,  |cy_i/H_i − cy_j/H_j| )
```

If `shift > face_pose_shift_ratio` (**default 0.30**, i.e. the face moved more
than 30 % of the frame), we **refuse to union** the pair. A run of check-in
shots therefore stays as separate keepers instead of collapsing to one.

- **Why "both have a face":** if neither frame has a face, it's a scenery burst
  → normal merge. The guard only engages when a person is present in both.
- **Why 30 %:** small pose wobble (turning a head, a step) stays under 30 % and
  still merges as a burst; deliberately re-composing ("now you stand there")
  crosses it. Lower the ratio to be more protective of check-in photos (fewer
  merges), raise it to merge more aggressively.
- **Limitation / TODO:** this uses face *position*, not identity. Two different
  people standing in the same spot would not be split by position alone. Face
  embedding/identity is a future upgrade; position covers the stated case
  ("same person changes pose") well and is cheap.

---

## 4. Keep-selection score

Within a group we keep the member with the highest:

```
score = weight_iqa       * (MUSIQ / 100)      # default 0.6, normalised to 0..1
      + weight_face       * face_quality       # default 0.3, 0..1 (0 if no face)
      + weight_resolution * resolution_bonus   # default 0.1, 0..1
```

- **MUSIQ normalised to 0..1** so the three terms share a scale (raw MUSIQ is
  0..100; mixing it un-normalised would let it dominate ~100:1).
- **face_quality** = `clip(confidence,0,1) * (0.5 + 0.5 * sharpness_of_face_crop)`
  — a confidently-detected, sharp face scores ~1; a soft or absent face scores
  low. Rationale: for people-photos the *face* being in focus matters more than
  global sharpness, hence its own term.
- **resolution_bonus** = `min(1, megapixels / resolution_ref_mp)` with
  `resolution_ref_mp` default 12 — rewards keeping the full-res original over a
  downscaled copy, capped so a 48 MP file doesn't massively outweigh quality.
- **Weights 0.6 / 0.3 / 0.1:** quality is the main driver; faces are a strong
  secondary (people photos); resolution is a tie-breaker. Ties break toward the
  **larger file** (original vs recompressed).

---

## 5. Motion photos (live photos)

A JPEG with an embedded MP4 after EOI is `jpg_motion`. It remains one intact
file throughout inventory, analysis and review; no extra deletion expansion is
needed. Separate `.mp4`/`.MP`/`.mov` files are inventoried as standalone video
and are never paired to a JPEG by filename. See `src/motion_photo.py`.

Embedded detection defaults to a **cheap header-only XMP-marker scan**
(`GCamera:MicroVideo` / `MotionPhoto`), which is what Xiaomi/Samsung/Google
write. A full trailing-MP4 scan is available (`scan.embedded_full_scan_max_bytes`)
but off by default to preserve stage 0's small-read budget on the HDD.

---

## 6. Safety defaults (stage 4)

- `execute.dry_run: true` — first run writes a plan and touches nothing.
- `execute.mode: move` — files go to `paths.trash/<timestamp>/` preserving the
  tree, with an `_undo_manifest.json` enabling `--undo`.
- Cloud deletes go to the Google Photos **trash** (60-day recovery).
- Cloud manifest items require filename + capture timestamp + size. The JS
  executor requires exactly one size-and-time match; ambiguity is skipped.
- Local execution verifies a Stage-2 run id, policy version, count and SHA-256
  binding before moving anything. Dry run remains available without mutation.

Stage 2 emits `KEEP/AUTO_REMOVE/MAYBE/UNKNOWN` with a mandatory keeper floor.
`similar_scene` never auto-removes by default. Stage 3 manifests contain only
`AUTO_REMOVE`; Maybe and Unknown are shown separately in risk order. Stage 4
accepts only recoverable `move`, so automatic decisions cannot hard-delete.

Profiles (`conservative`, default `balanced`, `aggressive`) tune abstention
margin (0.14 / 0.08 / 0.04). Uncalibrated exposure is `UNKNOWN` under
conservative and `MAYBE` under balanced/aggressive, but never automatic; only
byte identity takes the duplicate automatic lane. Closed-eye and OFIQ hooks exist but remain disabled/unknown
until their models are validated.

Stage 2 validates every embedding as exactly 768 finite, non-zero values before
cosine clustering; malformed legacy BLOBs are excluded instead of crashing.
Unreadable images become terminal `done_error` rows, so resumable runs do not
retry them forever. `scan.extensions` is an actual allow-list; unsupported HEIC
is not silently inventoried.
