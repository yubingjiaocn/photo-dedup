# Community reuse review and build-vs-reuse decision

Review date: 2026-09-08. Scope is deliberately bounded to unresolved multi-scene shadow-mode problems: logical action-phase protection, diverse keeper selection, technical/face quality, and near-duplicate review. No weights or new models were downloaded. This review does not expand `AUTO_REMOVE`, which remains `BYTE_IDENTICAL` only.

## Decision

| Area | Reuse candidate | License / activity checked | Compatibility and cost | Decision |
|---|---|---|---|---|
| Still-burst change points | `ruptures` | BSD-2-Clause; active repository (pushed 2026-07-06 when checked) | Offline Windows Python; NumPy/SciPy; accepts arbitrary cached vectors | **Defer adapter.** Cost/penalty/min-segment selection needs train/tune labels. Adding it now would hide untuned thresholds behind a dependency. Keep explicit adjacent evidence and bounded hysteresis for v1. |
| Edited-video shots | PySceneDetect | BSD-3-Clause; active, Windows packages/docs | CPU/OpenCV, no model; designed for decoded video streams | **Reuse later for ordinary video**, not sparse still bursts. |
| Edited-video shots | TransNetV2 | MIT; repository last pushed 2023-12-04 when checked | TensorFlow/PyTorch plus pretrained weights | **Reject for still v1.** It adds a model/runtime and solves edited-video transitions, not fixed-background pose/action changes. |
| Diverse subset | MMR | Established algorithm (Carbonell & Goldstein, 1998) | A few deterministic NumPy operations over existing utility and DINO cosine | **Reuse the algorithm now.** Local implementation avoids dependency/model risk and exposes quality/redundancy evidence. |
| Diverse subset | k-DPP / facility location; `submodlib`, `apricot` | `submodlib` MIT (pushed 2025-05-14); `apricot` MIT (pushed 2025-11-17) | DPP/kernel/numerical choices; submodlib C++ engine; facility location may require O(n²) similarities | **Defer.** Current phases are small and bounded MMR is sufficient. Benchmark only if labelled train/tune data shows MMR failure. |
| Burst best-shot | Wang et al., *Real-time Burst Photo Selection Using a Light-Head Adversarial Network* | IEEE TIP / arXiv:1803.07212; no maintained official drop-in checkpoint/package found | New learned weights/training data; unclear Windows packaging | **Ideas only.** It supports relative burst ranking but violates the no-new-large-model gate as an implementation choice. |
| General IQA | MUSIQ / IQA-PyTorch (`pyiqa`) | Existing project dependency; toolbox active when checked | Existing CPU/GPU path and cached `quality_score`; weight licenses still require artifact-specific review | **Keep existing integration.** Do not add NIMA/TOPIQ without labelled failure evidence. |
| Eyes/expression | MediaPipe Face Landmarker | Apache-2.0 code; active Google AI Edge repo | Local task model, Python/Windows; explicitly provisioned and hash-pinned in this project | **Keep optional shadow evidence.** Detection failure/profile/mask/occlusion remain `UNKNOWN`; never an automatic rejection. |
| Face/identity | YuNet + SFace | Existing OpenCV Zoo ONNX assets | Small, offline, Windows CPU friendly | **Keep.** Face position/quality and anonymous same-subject gate remain explainable and fail closed. |
| Near duplicates | `ImageHash` | BSD-2-Clause; active repository; already a dependency | Small offline Python package, Windows compatible | **Keep for candidate generation only.** SHA-256 remains the only automatic duplicate proof. |

## Open-source product practice

- **Immich** (AGPL-3.0, highly active) uses file hashes for upload duplicate checks and CLIP embeddings for likely-duplicate review. Its duplicate UI supports keeping multiple assets/stacking. Its machine-learning documentation also warns that its permission to use InsightFace models does not transfer to third parties. Reuse the review workflow, not service code or restricted weights.
- **digiKam** (GPL application; current upstream manual reviewed) caches fingerprints, scopes expensive similarity search by albums/tags, offers a similarity range, and lets users choose a reference-image rule. Reuse the cached/scoped/reference rationale; do not embed GPL application code.
- **Czkawka** (MIT core/CLI, some GUI components GPL-3.0-only; very active) clearly separates exact hashes from similar-image scans, caches results, works offline and supports Windows. Reuse that operational separation.
- **PhotoPrism** documents SHA1+size exact duplicate handling and explicitly says its perceptual hash is suitable for visual sorting but not precise enough for automatic stacking/duplicate decisions. This directly supports the current exact-vs-visual authority boundary.

## What was built locally, and why

The retained local code is project-specific glue, not a reinvention of a general library:

1. event-level train/tune/held-out guards;
2. logical phases inside an existing DB group, using cached time, DINO, face count and detected position evidence;
3. variation-aware, bounded keeper budget;
4. deterministic MMR over one authoritative utility score map;
5. reason/evidence records and missing-feature monotonic fallback;
6. immutable cached-feature A/B reporting.

`ruptures` remains the only plausible small future dependency. It must beat the explicit baseline on train/tune events before adoption. PySceneDetect is the preferred future library for real video keyframes. DPP/facility-location and additional IQA/face models remain research candidates, not runtime dependencies.

## Acceptance gate for any later reuse

A dependency/model enters runtime only if code and weight licenses are recorded; assets are explicit and hash-pinned with no implicit download; Windows CPU works and GPU is optional; outputs map to current reason/evidence schema; missing capability only increases retention/review; all selection/tuning uses train/tune manifests; held-out stays report-only; and removal authority remains byte-identical-only.

## Primary sources

- Burst selection paper: https://arxiv.org/abs/1803.07212
- PySceneDetect: https://github.com/Breakthrough/PySceneDetect and https://www.scenedetect.com/docs/
- TransNetV2: https://github.com/soCzech/TransNetV2
- ruptures: https://github.com/deepcharles/ruptures , https://centre-borelli.github.io/ruptures-docs/ , https://arxiv.org/abs/1801.00826
- MMR: https://aclanthology.org/X98-1025.pdf
- k-DPP: https://www.alexkulesza.com/pubs/kdpps_icml11.pdf
- submodlib: https://github.com/decile-team/submodlib and https://submodlib.readthedocs.io/en/latest/functions/facilityLocation.html
- apricot: https://github.com/jmschrei/apricot and https://www.jmlr.org/papers/v21/19-467.html
- MUSIQ: https://research.google/blog/musiq-assessing-image-aesthetic-and-technical-quality-with-multi-scale-transformers/ and https://github.com/google-research/google-research/tree/master/musiq
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
- ImageHash: https://github.com/JohannesBuchner/imagehash
- Immich duplicate/settings/model docs: https://docs.immich.app/features/duplicates-utility , https://docs.immich.app/administration/system-settings , https://github.com/immich-app/immich/blob/main/machine-learning/README.md
- digiKam similarity/duplicates: https://docs.digikam.org/en/left_sidebar/similarity_view.html and https://docs.digikam.org/en/maintenance_tools/maintenance_duplicates.html
- Czkawka: https://github.com/qarmin/czkawka
- PhotoPrism duplicates/perceptual hashes: https://docs.photoprism.app/user-guide/library/duplicates/ and https://docs.photoprism.app/developer-guide/metadata/perceptual-hashes/
