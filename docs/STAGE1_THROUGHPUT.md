Stage 1 throughput
==================

Measured on Ryzen 7 9700X, RTX 5070 Ti 16GB, single HDD.

## IQA batching (by shape, never padded)

Same-shape stacking preserves scores (max diff ~1-5e-05). Zero-padding moves
scores by 1-3 points, so we group by identical bounded shape and never pad.

MUSIQ/CLIP-IQA at 1920x1440: batching removes 2N host syncs but kernel time is
unchanged (~65ms/img). `iqa_batch_size: 4` peaks at ~3.0 GiB VRAM.

## CPU prefetch throughput

Per-image decode + IQA prep + sharpness + exposure (12 MP JPEGs):
  - serial: 312 ms/img
  - 2 threads: 164 ms/img (1.90x)
  - 4 threads: 98 ms/img (3.18x)
  - 8 threads: 73 ms/img (4.28x)

Overlapped against GPU: 0.83s GPU alone, 0.86s with 8 images prep concurrent
(3.87x vs serial). `cpu_workers: 4` is the default.

## End-to-end (40 images @ 12 MP, real weights)

Serial vs optimized, same synthetic library twice:
  - Feature diffs: all 0.000e+00 (MUSIQ, CLIP-IQA, sharpness, embedding, phash, sha256, faces)
  - Stage 1 wall: 33.9s -> 11.6s (2.92x)
  - MUSIQ/CLIP-IQA calls: 40 each -> 20 each (2 orientations = 2 shape groups)
  - Producer CPU: 31.2s total, loop waited 0.8s, overlap 30.4s

Mixed-orientation input peaks at 2 images/call (batch_size 4, 2 orientations).
