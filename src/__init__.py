"""Photo Dedup toolkit.

A staged, HDD-friendly, resumable pipeline for automatically de-duplicating a
large local photo library (and producing a matching cloud-delete list for
Google Photos via GPTK).

Stages
------
0. inventory  -> scan filesystem, EXIF, motion-photo pairing  (files table)
1. features   -> DINOv2 embedding + IQA + faces + pHash        (features table)
2. cluster    -> exact-dup / burst grouping, keep-selection     (groups tables)
3. report     -> review.html + delete lists + summary
execute_local -> move/delete locally (with undo)

See DESIGN.md and docs/ALGORITHM.md for the full rationale.
"""

__version__ = "0.1.0"
