# Offline CV/VLM review judgments

This workflow adds independent, group-level review advice without changing the
photo-dedup decision pipeline. It opens `inventory.sqlite` read-only and writes
only sidecars under `<output>/vlm-review/`. It never updates `groups`,
`group_members`, review actions, keeper selection, sorting, quarantine, or
deletion state.

The experiment history, rejected approaches, and evidence behind the current
design are recorded separately in
[`archive/VLM_EXPERIMENT_AUDIT_TRAIL.md`](archive/VLM_EXPERIMENT_AUDIT_TRAIL.md).

## Routing

- A byte-identical `sha_exact` group in `AUTO_READY`, with one persisted SHA-256
  shared by every member, uses a direct CV-only explanation and makes zero model
  calls.
- Every other selected group is treated as hard/high-similarity. Its scene and
  visual plan must pass fail-closed gates before any model call.
- `audit_context` is L0: path-redacted persisted CV leaves, including lossless
  binary storage. `derived_context` is L1: measurements, scalar ranks/deltas,
  source-family coverage, applicability, authority and conflict summaries.
  Both layers use `file_id + content_sha256` identities, never display aliases.
- `model_context` is L2, compiled only from L1. It contains discriminating
  scalar evidence, shared applicability/conflicts, binary-derived pairwise
  distances, and explicit checked-family summaries. Large groups use pHash
  nearest neighbors, cosine extremes and complete exact-equality sets; complete
  measurements remain in L1. Registered shared local regions are not available,
  so local-region evidence is currently aggregate-only.
- The sidecar explicitly records `alias_map` (display alias to file ID and SHA).
  Historical panel and sidecar aliases must be joined through both identity
  fields. `vlm_identity.join_maps(..., by="alias")` always rejects alias joins.
- L2 forbids raw binary/base64, raw boxes/landmarks, selection utility/pair margin,
  runtime telemetry and large arrays. Schema/guard failures raise before invocation.
  The text gate uses a UTF-8 byte/3 estimate, including prompt-template
  overhead, capped at 15,000 per group. Batch diagnostics also gate the median at
  8,000. The ideal 1,500–2,000-token context remains a target, not a passed gate.
- Visuals use one contact sheet with actual image short edges at least 400px,
  and at most two shared-window cross-file strips with native and displayed short
  edges at least 256px. Action requires explicitly authorized primary-subject
  localization; person/face detections alone cannot identify the main actor.
  Unknown scenes and unavailable group/landscape shared-region registration stop
  rendering. Bystander face crops are not a fallback.

## Prepare without a model call

Generate bounded sheets/strips and all three contexts for up to eight hard review
groups whose scene/ROI evidence passes the visual gate:

```bash
python -m src.vlm_review --output /path/to/review-output --prepare-only
```

Prepared artifacts have `status: prepared`; the review server ignores them
until a validated judgment exists.

## Run explicit groups

A model is never selected implicitly. Supply exactly one allow-listed
Sonnet/Terra-class model and one or more review group IDs:

```bash
python -m src.vlm_review \
  --output /path/to/review-output \
  --group-id 17 --group-id 42 \
  --model-id global.anthropic.claude-sonnet-5-5
```

Terra alternative:

```bash
python -m src.vlm_review \
  --output /path/to/review-output \
  --group-ids-file review-groups.txt \
  --model-id global.openai.gpt-5.6-terra
```

`review-groups.txt` may be a JSON list or comma/whitespace-separated integer
IDs. With no IDs, `--select hard --limit 8` is the default. `--select all`
includes easy exact groups; `--limit` is hard-capped at 20 and `--max-members`
at 20. A current matching cache is reused unless `--force` is explicit.

## Validation and UI behavior

New review calls require all human-readable reasons in concise Simplified Chinese;
JSON keys, enum values, frame aliases and region IDs remain schema-defined English.
Historical frozen batches keep their original English text so their request/response
hashes remain verifiable.

Each accepted recommendation must be grounded by a discriminator naming the
same frame and citing that frame/region and/or an exact CV reference. Unsupported
aliases, unknown CV refs, sensitive inferences, and ungrounded recommendations
fail closed. Valid grounded low-confidence observations are retained only as
explicit optional attention pointers; they never ground a recommendation. At most
three recommendation distinctions and three attention pointers reach the UI, each
with exact photo IDs and regions; full deterministic metrics remain in the
collapsed details section.

Distinct visible action, expression, interaction, and scene phases are
non-substitutable. The model is instructed not to force one winner and may recommend
multiple frames, but every recommended frame must independently pass grounding.

The server recomputes cache identity from the transformation version, exact
`model_context`, prompt hash, and source identity. It separately checks the stored
audit/derived/model context hashes and the complete alias map. Missing, malformed, `prepared`, old-schema, stale-prompt,
stale-transformation, or hash-mismatched artifacts are ignored and the existing
deterministic CV explanation remains available.
Sidecars and generated images are not added to the static-file allow-list.

## Deterministic frozen-set diagnostics

```bash
python3 scripts/report_vlm_rebuild.py --output artifacts/fable-rebuild
```

This reads the frozen 22 groups and original image hashes without model calls or
inventory writes. It verifies panel/inventory/sidecar joins, records all layer
sizes and coverage dispositions, and emits a G002 detector-overlay diagnostic.
The report's ROI failures are blockers; diagnostic sheets are not model-ready
visual assets. Exact reproduction details and unmet gates are in
`docs/FABLE_REBUILD_STATUS.md`.

## Safety boundary

- The command may read selected original images to create review visuals.
- A non-`--prepare-only` hard-group run may incur model cost; there is no retry
  or automatic escalation.
- Output is `shadow_review_only` and has `deletion_authority: none`.
- Human review actions remain separate in `review_state.json`.
- Removing `<output>/vlm-review/` removes only this optional cache.
