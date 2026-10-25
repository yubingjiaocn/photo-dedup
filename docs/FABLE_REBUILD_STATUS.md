# Fable evidence-chain rebuild status

Date: 2026-09-30

## Current stop condition

**G002 has a review-only moving foreground candidate; reliability/independent
acceptance remains open.** A class-agnostic registered color-component proposal
localizes the central action region without using person/face detections as truth.
The original-source strip includes the body, hands and feet; each actual-image
cell is 512×838 from a native 1158×1896 crop, with no bystander face crops.

- Candidate strip: `artifacts/fable-roi-narrow/G002-comparison-strip.jpg`.
- Source-backed measurements, SHA, masks, constraints and refusals:
  `artifacts/fable-roi-narrow/G002-candidate-report.json`.
- Method, existing-capability review and limits: `docs/G002_ACTION_ROI_CANDIDATE.md`.

Repeat and reversed-frame controls give identical windows; half-resolution gives
window IoU 0.99656. JPEG90 re-encoding abstains because the foreground component is
not unique/available at every segmentation granularity. The static-frame control
also correctly abstains. No full reliability PASS is claimed. Willy confirmed the single-subject
comparison-strip direction; production/robustness acceptance remains open. Production `vlm_visual_budget.plan(..., regime="action")`
still requires independently authorized primary-subject localization; candidate
packets do not satisfy that authority. Production comparison strips remain zero.
The candidate strip is a separate manual-review artifact, not a ready sidecar.

A separate three-group multi-subject/stage review-only slice is recorded in
`docs/MULTI_SUBJECT_REVIEW_CANDIDATE.md`. It produces source-verified full-ensemble
sheets and subject-set evidence, but all three groups stop on coverage/extent or
performer-detection gaps. Production group/landscape registration remains
unsupported. No expansion to 22-group ROI planning or new decision/multi-select/
irreplaceable schema is authorized. Existing legacy grounded multi-select and
attention behavior does not meet the new P1-2 contract.

## Implemented gates

- The raw-binary guard accepts only the exact typed measurement-interface
  descriptors at the interface location; arbitrary pHash/embedding payloads still
  raise. Raw bytes/base64, oversized numeric arrays, raw geometry fields and
  selection-anchor payloads fail closed at the L2 boundary.
- L0 is complete path-redacted supplied inventory/review data. L1 contains
  compiled measurements, stable-identity pairwise evidence, numeric ranks/deltas,
  applicability/conflict state and coverage ledgers. Only the L1-derived L2 is
  eligible for model invocation. Source images must match content SHA before
  rendering, even when size/mtime match.
- Internal frame identity is `file_id + content_sha256`. Display aliases appear
  in L2 and the explicit sidecar `alias_map`. Cache schema/version/hash checks
  cover all three layers and the alias map. Old sidecars cannot be reused.
- Deterministic historical joins verify original source SHA for both panel and
  inventory paths and compare sidecar identities. All 22 groups join bijectively.
  G002 panel F01=107/F02=99, old sidecar F01=99/F02=107; identity join gives
  F01→F02/F02→F01. Direct alias join is rejected.
- All source-family ledger dispositions are exact/derived/summarized/non-evidence;
  all replacement refs resolve. Required-family coverage explicitly distinguishes
  missing or uncompiled evidence. No old keeper/provisional labels are used as
  preference truth.
- Known portrait visual fixtures pass a single contact sheet with actual-image
  short edge ≥400px and a cross-file strip with native/displayed short edge
  ≥256px. Action requires independently authorized primary-subject localization;
  unknown scenes and unsupported shared-region registration fail closed.
- No decision-table mutation. Frozen-set before/after digests are identical for
  all 22 groups; ready-sidecar and review API tests also exercise inertness and
  stale cache handling.

## Incomplete evidence gates

P0-3 is **partial**, not complete:

1. Spatially registered top-3 cross-file local regions are unavailable; L2 uses
   aggregate region statistics and explicitly records that limitation.
2. Group `decision_json.phase_selection` mixes indexed measurements, logical
   phase protection and selector authority. It is retained in L0 and explicitly
   marked uncompiled/non-evidence in L1 rather than applying an unsafe index join
   or treating old phase/keeper decisions as semantic truth. Safe extraction of
   those shadow measurements is still required.
3. Real semantic action phases and primary-subject completeness/occlusion are not
   established by these persisted catalogs. Missing/unknown does not mean absent.
4. The ideal 1.5–2K L2 token target is not generally achieved. The hard text gates
   pass **the documented deterministic estimate**, not a provider tokenizer or a
   paid model's billed token count.

P1-1 is **partial/blocked**, not complete: the frozen production planner report
remains 0/22. G002 has the separate review candidate and stability limit above;
other frozen groups lack a unique persisted supported scene regime.
Group/landscape shared-region planning also remains unsupported. P1-2 is deferred at the requested stop condition.

## Frozen 22-group deterministic report

Current layer measurement report: `artifacts/fable-roi-narrow/context-layers-22.json`.
Frozen baseline compiler/join/ROI report: `artifacts/fable-rebuild/report-22.json`.
The current measurement-only rerun makes zero ROI planning attempts.

| Layer | Total canonical bytes | Median/group | Maximum/group |
|---|---:|---:|---:|
| L0 audit | 3,600,281 | 105,092 | 768,341 |
| L1 derived | 6,402,964 | 218,748.5 | 1,406,130 |
| L2 prompt | 281,028 | 10,939.5 | 27,386 |

L1 deliberately retains complete derived measurements and verbose stable refs;
it is not a prompt budget. L2 byte/3 token estimates: median **3,647**, maximum
**9,129**. Including the unchanged prompt template in the combined byte/3 estimate: median
**4,479.5**, maximum **9,962**. Thus estimated median≤8K and maximum≤15K pass.
Actual provider tokenization and image token cost have not been measured.

Frozen baseline (not rerun as a 22-group ROI batch):

- L2 compiler/guard: 22/22.
- Verified source/panel/sidecar identity joins: 22/22.
- Missing coverage replacement refs: 0.
- Decision-table digest changes: 0.
- ROI plan pass: 0/22 (blocked, not warnings).
- External image-model calls, Google Photos access: 0.

Reproduce current layer measurements without ROI expansion, a model call or
evaluation-tree writes:

```bash
python3 scripts/report_context_layers.py
```

The measurement script requires the frozen 22-sidecar set, opens SQLite with
`mode=ro`, and restricts writes to the repository. It reads no sealed preference
labels. Per-group and aggregate outputs use `audit_bytes`, `derived_bytes`,
`prompt_context_bytes`, and explicitly named ratios. L1 has no prompt budget.
L2 context and total-text (context+template) estimates/limits are separate;
aggregate ratios use summed bytes and aggregate gates apply per-group limits,
not limits on the sum. Empty aggregates have null gate results. Provider/image
tokenization is not measured. Original SHA/join evidence remains in the frozen
baseline report; G002 originals are reverified by the narrow candidate probe.

## Verification artifacts

- `artifacts/fable-roi-narrow/focused-tests.txt`
- `artifacts/fable-roi-narrow/broad-python-tests.txt`
- `artifacts/fable-roi-narrow/ruff-focused.txt`
- `artifacts/fable-roi-narrow/ruff.txt` (whole-repository lint findings outside this change)
- `artifacts/fable-roi-narrow/frontend-tests-build.txt`
- `artifacts/fable-roi-narrow/diff-check.txt`

Current bounded-follow-up verification and provenance are recorded in
`artifacts/fable-roi-narrow/run-log.md`; baseline verification remains in
`artifacts/fable-rebuild/run-log.md`. Existing unrelated dirty changes are preserved.
No commit, publication, Gateway/config change, automatic deletion, or external
image-model call was made. Independent parent acceptance remains outstanding.
