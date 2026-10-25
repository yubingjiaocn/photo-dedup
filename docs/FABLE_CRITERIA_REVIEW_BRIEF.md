# Fable review brief — photo-dedup hard-group judgment

Date: 2026-09-30

## Decision requested

Independently critique and redesign the evidence package and evaluation for difficult/high-similarity photo groups. Do not edit code. The goal is not automated deletion; it is to reduce human review fatigue by showing 1–3 concrete, trustworthy differences and the exact photos/regions to inspect.

## Owner intent

- Easy groups with decisive CV evidence should be explained directly without a VLM.
- Groups sent to a VLM are inherently difficult or extremely similar. For those groups, no real criterion family may be silently discarded merely because it was weak, default-off, or shadow-only in the old selector.
- The VLM should make an independent group-level judgment from the images plus all useful CV criteria. It must not rationalize the persisted keeper.
- Different scenes need different priorities: action timing, intentional motion, stage lighting, multi-person interaction, and fireworks cannot share one fixed quality ranking.
- The human-facing result should contain at most three distinctions. Full evidence may remain available for audit.
- All output remains review-only: no keeper/sort/delete mutation.

## Current flow

1. Only byte-identical `sha_exact` + `AUTO_READY` groups bypass VLM.
2. Other selected groups receive high-resolution labelled contact sheets and up to four face crops.
3. The current sidecar has two contexts:
   - `audit_context`: lossless persisted fields after path/privacy redaction.
   - `model_context`: intended to be compact and model-usable.
4. VLM recommendation and reasons are sanitized against frame aliases, visual regions, and exact CV refs. At most three medium/high-confidence distinctions may ground a recommendation. Low-confidence grounded observations are optional attention pointers only.
5. Sidecar input/context/prompt hashes invalidate stale results; review API ignores invalid caches.

## Criterion families currently available

- Exact identity and near-duplicate evidence: content SHA, pHash/Hamming.
- Global visual similarity: DINO embedding/cosine.
- Global quality: aggregate quality score, MUSIQ/CLIP-IQA and associated metadata.
- Face/person evidence: face count, boxes/landmarks, face-local quality, detected-face crops.
- Eye evidence: open/closed/maybe/unknown, per-face reasons and detector disagreement.
- Exposure evidence: clip-high/clip-low, usable mass, entropy, luminance/anchor measures and classification.
- Local/region evidence: detector catalog, shared regions, region quality and local-quality embeddings.
- Pose/action/phase evidence: pose variation, protected phase/variant signals and shadow findings.
- Subject completeness/occlusion evidence: current interfaces and shadow-only observations where present.
- Selection diagnostics: utility, pair margin, contradiction/unknown/applicability/authority fields. Existing keeper/decision authority itself is deliberately withheld from the VLM to avoid post-hoc justification.
- Scene/regime clues from pixels and persisted routing metadata.

## First frozen-set result

Datasets:
- development Apr-04: 8 hard groups;
- regime holdout Jun-23: 6 hard groups;
- local-quality holdout Jul-25: 8 hard groups.

Initial Sonnet 5.5 run:
- 22 groups, about 1,058,615 input tokens in total;
- after fixing an over-narrow visual-region enum, 7/22 produced medium-confidence grounded recommendations and 15/22 abstained;
- all 7 recommendations included the persisted keeper, but persisted keeper is not human truth;
- the sealed Jun-23 action group G002 contains non-substitutable grounded and airborne phases. The model recommended only the airborne frame at low confidence; the gate correctly abstained, but the prompt/representation had failed to preserve both phases.

## Representation failures found

1. Raw DINO/local-quality embeddings and pHash bytes were initially sent as base64. This was invalid: auditable storage was confused with model-usable evidence.
2. That was replaced with pairwise DINO/local-quality cosine and pHash Hamming. Raw values remain only in `audit_context`.
3. Preflight across the same 22 groups still measured:
   - audit JSON: 1,704,000 bytes;
   - model JSON: 1,582,259 bytes;
   - reduction: only 7.14%.
4. The remaining bulk is flattened `quality_meta.local_region_set` catalogs, raw boxes, per-region arrays, repeated detector interfaces/constants and runtime/thumbnail telemetry. Sending these verbatim would again confuse full storage with usable criteria.
5. No second paid-model pass was run after this preflight.

## Representative evidence to inspect

- Pipeline and prompt: `src/vlm_review.py`
- Sanitizer: `src/vlm_semantic.py`
- Context serializer: `src/semantic_observer.py`
- Current workflow: `docs/VLM_REVIEW_WORKFLOW.md`
- Integration tests: `tests/test_vlm_review.py`, `tests/test_vlm_semantic.py`
- Sealed holdout labels: `/home/ubuntu/photo-dedup-eval/astra-regime-validation-20260910/labels/holdout-jun23-sealed/labels-01.json`
- Holdout mapping: `/home/ubuntu/photo-dedup-eval/astra-regime-validation-20260910/panels/2026-06-23/mapping.json`
- First-run sidecars: `/home/ubuntu/photo-dedup-eval/astra-regime-validation-20260910/local-holdout/2026-06-23/vlm-review/groups/`
- Prepared visuals are under each dataset's `vlm-review/assets/<group_id>/`.

## Questions for Fable

1. What is the minimal sufficient, model-usable representation for each criterion family above? Distinguish raw audit data, derived evidence, and prompt-visible evidence.
2. How should large detector/region catalogs be summarized without silently dropping a criterion that matters in combination?
3. Should the VLM see scalar values, pairwise deltas/rankings, natural-language findings, or a mix? Define an exact bounded schema.
4. How should non-substitutable phases be represented so the model can recommend several keepers rather than force one winner?
5. Which fields are measurement evidence, provenance/authority, runtime telemetry, or post-decision leakage?
6. How should low-confidence observations be surfaced as attention pointers without becoming recommendation evidence?
7. Design a bounded evaluation using the existing development and sealed holdout assets. Separate coverage, grounding, stability, phase recall, human attention reduction, and preference correctness. Do not treat old keepers or model-provisional labels as unquestionable truth.
8. Give explicit stop/go gates before any Google Photos or new personal-photo validation.

## Constraints

- Read-only analysis only; do not change code/config/services or call paid image models.
- Do not access Google Photos.
- Do not recommend automatic deletion.
- Prefer a smaller coherent design over another large schema dump.
