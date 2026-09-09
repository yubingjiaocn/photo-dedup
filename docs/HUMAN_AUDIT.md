# Blinded phase/group human audit (v1)

This is an offline annotation product, not a deletion interface. It freezes the
v4 probability sample, collects independent human phase/keeper judgments, then
compares them with the frozen candidate. It does not run inference, read a DB,
open originals, change the review workbench state, or write policy/manifests.
`AUTO_REMOVE` remains `BYTE_IDENTICAL` only.

## Why a separate export

The existing Preact workbench is an operational decision overlay: accept the AI
keeper, select a photo, defer, undo. It is not a probability-labeling instrument.
The v4 `review-prototype.html` and contact sheets expose candidate keepers, scores,
reasons and expected-value ordering. Its four-button labels cannot express a
complete human phase partition. `offline_evaluation.py` validates phase labels
but does not provide a blind collection/replay workflow or survey-risk report.

This slice therefore uses a separate static reviewer bundle and JSONL snapshot,
leaving both existing interfaces and deletion authority untouched. No model
score or existing provisional tune label is imported as human truth.

## Freeze and export

Use explicit development dataset manifests and explicit **thumbnail cache**
roots. Scope validation precedes reading group inputs or images. Only owner
`willy` or `synthetic`, split `train`/`tune`, is accepted. Held-out, other owners,
unknown datasets and mismatched manifests fail closed. This is a trusted local
operator contract, not a mechanism for recognizing mislabeled private files.
Never point `--cache` at originals; it reads only `<integer-id>.jpg`, rejects
symlinks, oversized/non-JPEG cache entries, and has no original-image fallback.
Missing/corrupt thumbnails make a task unassessable. JPEGs are re-encoded from
pixels so EXIF, comments and appended data do not enter the reviewer bundle.

Example for the existing immutable v4 artifacts (no pipeline rerun):

```bash
python -m src.human_audit_bundle export \
  --groups /home/ubuntu/photo-dedup-eval/risk-coverage-v4/risk-groups.json \
  --audit /home/ubuntu/photo-dedup-eval/risk-coverage-v4/audit-manifest.json \
  --provenance /home/ubuntu/photo-dedup-eval/risk-coverage-v4/provenance.json \
  --scopes fixtures/willy-audit-development-scopes-v1.json \
  --cache disney /home/ubuntu/photo-dedup-eval/disney-conservative/thumbs \
  --cache jx3 /home/ubuntu/photo-dedup-eval/jx3-identity/thumbs \
  --fixture fixtures/disney-g10-narrative-review-v1.json \
  --seed human-phase-audit-v1 \
  --output /home/ubuntu/photo-dedup-eval/human-audit-v1-20260909-r2
```

Output must not exist and must be separate from cache roots. No input is modified.
The output contains:

- `reviewer/index.html` + `reviewer/thumbs/`: the **only** files needed by the
  human reviewer. Open `index.html` directly in a local browser. No server,
  network requests, authentication or original-photo endpoint is involved.
- `manifest.json`: private analyst metadata, source DB hashes inherited from
  v4 provenance, frozen candidate/member mapping, sampling strata, media hashes
  and bundle fingerprint. **Do not give this to a blinded reviewer** or serve
  the parent directory. Hashes detect accidental mutation; they are not signatures.
- `empty-labels.jsonl`: an empty snapshot, never a generated human conclusion.

Same inputs, seed and cache pixels in the same Pillow/JPEG environment produce
the same bundle and display order. A change of input, candidate, membership,
cache image or presentation seed produces a new bundle ID; old labels cannot be
silently reused. Codec changes may also change media bytes/identity. Member order
is inherited from the v4 timestamp-ordered export; anonymous labels M1, M2, …
retain that order for narrative review. Group presentation order is hashed and
reproducible, not sorted by risk. The reviewer payload has no scores, candidate
keepers, model phases, reasons, dataset/source paths, strata or G10 hint.

The sampler is replayed against the entire pre-audit silent population:
**72 SAFE_SILENT + 30 DIAGNOSTIC_SAMPLE = 102 groups**, not just the 72 remaining
silent groups. IDs, counts, strata, weights, inclusion probabilities and pending
status must agree exactly with the replayed sample. No resampling after labels.
The report targets this frozen operating point, not the primary queue.

## Human workflow and replay

1. Use a reviewer who has not seen this run's candidate predictions. Enter a
   pseudonymous reviewer code and explicitly attest that judgments are human.
2. Give every member a phase number. Equal numbers mean the same narrative or
   action phase. Select at least one **acceptable** keeper per phase (several
   are allowed), then answer whether unrelated content has been falsely merged.
   No phase, keeper or impurity answer is preselected.
3. Click **记录完整判断**. If thumbnails cannot resolve expression/action/quality,
   choose **不确定** or **无法判断**. Those records have no phases or correctness
   conclusion and cannot count as safety evidence.
4. Export JSONL before leaving. Nothing writes to the source or pipeline state.
   Reload starts blank; import the downloaded JSONL to restore the snapshot and
   controls. Import validates the whole snapshot before replacing any state.
   Editing a recorded task invalidates its saved judgment until recorded again.
   Undo removes that task's record; a subsequent export reflects the removal.
5. Keep exported snapshots separately (version them by filename); do not append
   duplicate rows or merge reviewer files blindly. This v1 has one judgment per
   task, not an adjudication/multi-reviewer system. Both browser import and analyst CLI reject duplicate JSON keys, non-integer
   version tokens, NaN and invalid schema/partitions. Snapshot replacement asks
   before discarding unexported labels or drafts; exporting recorded labels does
   not dismiss the loss warning for unrecorded form edits.

A JSONL row has exactly these keys:

```json
{"schema_version":1,"bundle_id":"<frozen bundle hash>","task_id":"<opaque task id>","status":"reviewed","annotator":"reviewer-01","human_attested":true,"phases":[{"members":["M1","M2"],"acceptable_keepers":["M1"]}],"group_impure":false,"note":"optional visual observation"}
```

The example is schema documentation, **not a label for any real task**. Status
`uncertain`/`unassessable` requires `phases: []`, `group_impure: null`. Missing rows
are pending. Exact membership partition, keeper subsets, explicit booleans,
bundle/task scope, duplicate tasks and unknown fields are validated. A human
attestation records provenance; software cannot prove that a human really looked.

Replay to a new report file:

```bash
python -m src.human_audit_bundle report \
  --bundle /home/ubuntu/photo-dedup-eval/human-audit-v1-20260909-r2 \
  --labels /path/to/downloaded-audit.jsonl \
  --output /path/to/new-audit-report.json
```

The CLI checks bundle, reviewer HTML (including member-to-image mapping), and
exported thumbnail fingerprints before reporting. The HTML fingerprint normalizes
only its single bundle-ID marker, avoiding a circular hash dependency.
Changed/missing exported media or stale labels fail, rather than quietly
changing the evidence. A report output already present is never overwritten.

## Group error and statistical contract

For each reviewed group, compare frozen candidate keeper aliases to the human
partition only **after** annotation:

- error if the human marked a false merge;
- error if any human phase has no acceptable candidate keeper;
- error if any selected candidate keeper is not acceptable in its human phase.

No judgment about the model's phase count is required from the blinded human.
The report includes `phase_miss`, `keeper_bad` and the union group-error flag.

For stratum h, N_h is the frozen silent population and n_h is the drawn sample.
The v4 sampler fixes allocation by stratum sizes and takes simple random draws
without replacement within each stratum. It stores inclusion n_h/N_h and weight
N_h/n_h. The seed must have been fixed without inspecting labels; replay alone
cannot prove this protocol was followed.

Only when **all drawn probability samples are reviewed and assessable**, and
every nonempty population stratum has n_h > 0, report:

- weighted error estimate: sum_h (N_h/N) × (errors_h/n_h);
- one-sided simultaneous 95% upper bound: sum_h (N_h/N) × U_h;
- census stratum: U_h = errors_h/N_h exactly;
- non-census stratum: U_h = min(1, errors_h/n_h +
  sqrt(log(H/0.05)/(2 n_h))), where H is the number of non-census strata.

Hoeffding's bounded-sample bound is conservative for sampling without replacement;
Bonferroni allocates 0.05/H across non-census strata. No independence across
strata is needed for that union bound. We deliberately omit a finite-population
correction; sparse strata can make the bound very loose. This is not pooled
Wilson/binomial inference and not a claim of calibrated model probabilities.
Partial reports show observed counts and any complete stratum summaries, but
**overall risk and bound stay null** on missing/uncertain/unassessable samples or
zero-selection strata. Do not drop missing tasks, substitute model labels or
selectively audit until a desired bound is reached.

The estimand is this frozen **development-event silent group** population, not
photo-level errors, future events or held-out generalization. Human uncertainty,
prior model exposure and correlated annotation mistakes remain external validity
limits. The report always says `safety_validated: false`: no accepted safety
threshold or deletion authority is created here. Fit a new policy only after
recording this report, and validate it on a separately frozen evaluation design.

## Disney G10: recorded review request, not ground truth

`fixtures/disney-g10-narrative-review-v1.json` pins G10 membership
430/431/432/433 and the v4 Disney source DB fingerprint. It records **pending**
`narrative_recheck`, no expected phase count, keeper or verdict. Changed membership
or source identity rejects the fixture. In the current v4 operating point G10 is
primary, so it adds one purposive task hidden among the 30 probability tasks.
Its eventual human result is reported separately and excluded from prevalence
and confidence bounds. If a future frozen probability design genuinely samples
that same task, it appears once and retains its actual probability provenance.
The local fixture does not complete the previously requested visual review.

## Separate provisional AI experiment

`python scripts/prepare_model_audit.py --bundle <bundle-dir> --output <new-dir>`
prepares G10 plus one probability-sample task from each of three deterministically
chosen strata. It reads only verified exported thumbnails. `blind-input/` contains
anonymous cases in randomized order; `analyst-unblind.json` stays separate until
blind model judgments are frozen. The script performs **no inference** and makes
no visual conclusion. The packet is `model_provisional`,
`human_annotation: false`, `selective_risk_eligible: false`; human label validation
rejects it. These purposive AI samples cannot estimate population safety.

Under a no-network-exfiltration constraint, only an already available local
vision model with networking disabled may consume the images. Sending cached
pixels to a remote Astra/API still constitutes an upload even though the originals
stay on disk. If no local model is available, retain the prepared packet with a
concrete blocker instead of fabricating visual findings. Record model/version,
prompt/settings, packet ID and blind output before revealing the candidate
mapping; then document discrepancies and research hypotheses separately from
human judgments. This optional experiment never blocks the annotation product.

## Verification

```bash
python -m pytest -q tests/test_human_audit.py
python scripts/verification/human_audit_browser.py  # Playwright + installed Chrome
bash scripts/test-fast.sh
bash scripts/test-full.sh
```

The browser harness uses only generated synthetic cache images and synthetic
labels. It checks blank defaults, phase entry, download → report, reload/import,
edit invalidation, atomic stale-import rejection and local-only asset requests.
Focused tests cover determinism, source/sample/label tampering, pending fixture,
held-out preflight, missing media, symlinks, weighted/census results and
incomplete-sample abstention. No real human labels are synthesized by a gate.
