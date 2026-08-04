Stage 1 throughput measurements (L40S, torch 2.6.0+cu124, pyiqa 0.1.16)
========================================================================
Raw probe logs behind every design decision in the Stage 1 batching/prefetch
change.

Reproduce the semantics and equivalence claims on any CUDA box with the checked-in
tool (it generates its own images and never reads ``paths.root``):

    python -m scripts.verify_iqa_batching

Reproduce the end-to-end comparison anywhere, including CPU-only machines:

    python -m src.stage1_benchmark --images 48

Sections 3 and 4 came from throwaway probes of Pillow/numpy/transformers
behaviour rather than of this package's code; their numbers are recorded here
because they are what set the defaults, and the scaling can be re-measured with
the benchmark above on the machine that matters.

1. pyiqa batching semantics (probe_pyiqa_batch.py)
--------------------------------------------------
MUSIQ, six 1440x1920 inputs:
  singles  [22.484501, 22.528511, 22.272861, 22.304062, 22.207373, 22.264860]
  batched  [22.484501, 22.528515, 22.272854, 22.304052, 22.207376, 22.264854]
  max_abs_diff 9.537e-06   max_rel_diff 4.276e-07
CLIP-IQA, same inputs:
  max_abs_diff 7.594e-05   max_rel_diff 1.259e-04
Single path re-run against itself: max_abs_diff 0.000e+00 (deterministic).
=> stacking equal-shaped tensors is score-preserving; the residual is cuDNN
   batch-dependent kernel selection, not a semantic change.

Heterogeneous shapes:
  torch.cat -> RuntimeError "Sizes of tensors must match except in dimension 0"
  metric([t1, t2, t3]) -> Exception "Unsupported source type" (lists rejected)
Zero-padding to a common size, MUSIQ:
  singles [22.243563, 22.248837, 23.141171]
  padded  [23.944439, 22.809065, 21.733555]   max_abs_diff 1.701 MUSIQ points
Zero-padding, CLIP-IQA: max_abs_diff 0.1164 (19% relative)
=> padding is NOT semantics-preserving. Shape grouping is the only legal
   batching strategy. Both models resize internally from the given tensor, so
   the tensor's own H/W is part of the feature definition.

2. Batching is throughput-neutral on the GPU (probe_vram.py)
------------------------------------------------------------
MUSIQ, 1920x1440:                    CLIP-IQA, 1920x1440:
  batch  1   65.4 ms   65.4 ms/img     batch  1   37.7 ms   37.7 ms/img
  batch  2  129.6 ms   64.8 ms/img     batch  2   76.6 ms   38.3 ms/img
  batch  4  259.0 ms   64.8 ms/img     batch  4  155.5 ms   38.9 ms/img
  batch  8  513.4 ms   64.2 ms/img     batch  8  313.8 ms   39.2 ms/img
peak VRAM: MUSIQ 0.83 / 1.55 / 2.99 / 5.87 GiB; CLIP-IQA 1.06 / 1.73 / 3.07 / 5.75
8 per-image .item() calls vs one batch of 8: 822 ms vs 828 ms (0.99x)
=> at a 1920 px long edge these metrics are compute-bound, not launch-bound.
   Batching removes 2N host syncs but buys no kernel time, and VRAM grows
   linearly. Hence iqa_batch_size default 4 (2.99 GiB peak, matches the
   measured 4.5/16 GiB Windows headroom); 8 would peak near 6 GiB per lane.

3. The real headroom is CPU parallelism + overlap (probe_pipeline.py)
--------------------------------------------------------------------
Per-image CPU preprocessing of 12 MP JPEGs (decode + 1920 px IQA array +
sharpness + 512 px exposure array), 16 images:
  serial      4.99 s   312 ms/img
  2 threads   2.63 s   164 ms/img   1.90x
  4 threads   1.57 s    98 ms/img   3.18x
  6 threads   1.33 s    83 ms/img   3.74x
  8 threads   1.17 s    73 ms/img   4.28x
=> Pillow's decoder, numpy and the resize all release the GIL, so threads (not
   processes) are enough. No pickling of decoded frames, no extra copies.

Overlap with a blocking GPU IQA batch:
  GPU IQA for 8 images alone                       0.83 s
  same GPU call with 8-image CPU prefetch running  0.86 s total
  serial equivalent 3.33 s -> 3.87x
=> worker CPU work is essentially free while the GPU lane is busy.

4. DINOv2 preprocessing is per-image separable and thread-safe (probe_embed.py)
------------------------------------------------------------------------------
BitImageProcessor, 8 mixed-orientation 12 MP images:
  processor(images=[all]) vs cat(processor(images=[one]) ...): bitwise identical
  (torch.equal -> True, max_abs_diff 0.000e+00)
  serial per-image 0.808 s; 4 threads 0.372 s (2.17x), identical to serial
  48 concurrent calls all identical to the single-threaded reference
=> per-image preprocessing in workers + one torch.cat on the main thread is
   exact, so embedding inputs are unchanged.

5. What this predicts for the Windows baseline
----------------------------------------------
Measured Windows loop 504.80 s / 1000 images. Parallelisable, order-preserving
CPU work: IQA preprocess 82.82 + embedding preprocess 51.50 + sharpness 49.39 +
decode 40.18 + exposure 31.14 + read 24.99 = 280.02 s. Main-thread work that
must stay serial (YuNet is stateful, SQLite is single-threaded, the IQA models
are the GPU lane): MUSIQ 70.93 + CLIP-IQA 54.43 + YuNet 56.52 + embedding
inference 3.61 + face quality/routing/thumbnail/DB.
At the measured 3.18x for 4 workers the producer needs ~88 s of wall time and
overlaps the ~190-200 s serial remainder, so the loop should land near
200-260 s: roughly 1.9-2.5x. The IQA batching itself is worth the removed
syncs and O(shapes) calls, not kernel time -- proven in section 2.

6. What the design does with all of this
----------------------------------------
* **Batched IQA by shape group.** One MUSIQ call and one CLIP-IQA call per
  distinct bounded input shape per batch, capped by ``iqa_batch_size``. Never
  padded, never a list, never reordered. For a real library (one camera, two
  orientations) that is one or two calls per batch instead of ``N``.
* **One host transfer per call.** ``.item()`` inside the per-image loop is gone;
  each lane's whole output crosses once via ``reshape(-1).float().cpu()``.
* **One reader thread, a small worker pool.** The sequential HDD pass is
  preserved exactly (one open, one decode per file, in inventory order); only the
  per-image CPU preparation fans out.
* **Bounded queue, deterministic order.** ``prefetch_batches`` batches queued
  plus one being assembled, resolved in submission order.
* **Stateful work stays on the main thread.** YuNet (mutable input size), the
  thumbnail cache, every SQLite statement, the eye detector and the scene router.

7. Verified on this box with the real stack
-------------------------------------------
``python -m scripts.verify_iqa_batching`` (real weights, unit level):

    [musiq]    stacked vs per-image (same shape)  max_abs_diff 1.144e-05
               zero-padding would shift scores by 3.4033
    [clipiqa]  stacked vs per-image (same shape)  max_abs_diff 3.251e-05
               zero-padding would shift scores by 0.0781
    mixed shapes: torch.cat impossible; list input rejected

    6 images -> 3 distinct bounded shapes -> 3 model calls per lane
    MUSIQ batched vs per-image      max_abs_diff 7.629e-06
    CLIP-IQA batched vs per-image   max_abs_diff 4.721e-05
    embedding batched vs per-image  min_cosine 1.00000000
    per-image iqa_scale/size/sharpness preserved
    peak VRAM 3.14 GiB at iqa_batch_size=4

The padding shift is larger here than in section 1 (3.40 vs 1.70 MUSIQ points)
because the padded fraction is larger; either way it is thousands of times the
batching residual, which is the point.

A full ``stage1_features.run`` was also executed twice over the same synthetic
library — once with ``cpu_workers=0``/``prefetch_batches=0`` and once with the
shipped defaults — through real DINOv2 + MUSIQ + CLIP-IQA + YuNet on CUDA, with
40 images at 12 MP (4032x3024 / 3024x4032, the real library's shape).
``python -m src.stage1_benchmark --backend torch --images 40`` performs the same
comparison and asserts the equality part:

    max |MUSIQ diff|       0.000e+00
    max |CLIP-IQA diff|    0.000e+00
    max |sharpness diff|   0.000e+00
    max |embedding diff|   0.000e+00
    phash / sha256 / face_count / exposure / faces_json : identical
    MUSIQ calls 20 and CLIP-IQA calls 20 for 40 images (per-image code: 40 each)
    Stage 1 wall  serial 33.9s -> optimised 11.6s   = 2.92x
    producer CPU 31.2s, loop waited 0.8s, overlap 30.4s

Two things in that table are worth stating plainly rather than glossing over:

* **``loop_seconds`` barely moves (7.2s -> 7.8s) and that is expected.** The loop
  no longer *contains* the read/decode/prepare work, so its total stops being a
  measure of the stage. The wall clock is the honest figure, and the reason it
  falls is the 30.4 s of producer work that overlapped it. Quoting "loop
  images/s" after this change would understate the gain in one direction and
  misdescribe it in the other; that is why the report prints three separate
  sections and an explicit ``overlap``.
* **20 calls for 40 images is 2 per call, not 4.** ``batch_size`` is 4 and the
  library has two orientations, so a batch of 4 usually splits into two shape
  groups. That is the honest ceiling of shape grouping on mixed-orientation
  input; a single-orientation batch collapses to one call. The alternative —
  padding to a common shape — is what section 1 rules out.

The stub-backend equivalent, runnable anywhere including CI, is
``python -m src.stage1_benchmark``. It asserts identical features and reports
timing without asserting a speedup: the stub's "inference" costs microseconds, so
there is no GPU work for the producer to hide behind and any speedup there would
be an artefact rather than a measurement.

8. Expected effect on the Windows baseline, and the residual risks
-----------------------------------------------------------------
This box has an L40S and an NVMe scratch disk, so the *ratio* transfers but the
absolute numbers do not. On the 8C/16T Ryzen with a single HDD, the parallelisable
280 s of the measured 504.80 s loop should compress to roughly 90-110 s of
producer wall time and overlap the ~200 s that must stay serial, i.e. a loop
nearer 210-260 s. Expect **1.9-2.4x**, not the 2.92x measured here, because:

* the HDD read (24.99 s measured) cannot be parallelised and stays on the critical
  path — it is the one lane the design deliberately refuses to speed up;
* 8 physical cores shared with the GPU lanes scale less well than this box's;
* Windows thread scheduling and Pillow's Windows JPEG decoder differ from Linux.

Residual risks, stated rather than discovered later: peak RSS rises by the stated
memory bound (~1.2 GiB with the defaults); a library of uniformly *identical*
dimensions gets fewer IQA calls than a mixed one, so per-run call counts vary
legitimately; and ``iqa_batch_size`` above 4 grows VRAM linearly (5.87 GiB at 8)
for no measured kernel-time gain.
