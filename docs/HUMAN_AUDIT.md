# Blinded phase/group human audit (v2)

This standalone offline reviewer collects human evidence, not deletion decisions.
It reads only explicit development metadata and verified cached JPEGs. It does
not read originals, DBs, embeddings or held-out data; run inference; change the
operational workbench; or write back policy. `AUTO_REMOVE` stays `BYTE_IDENTICAL`
only. Reviewer data has no candidate keepers, model scores/phases, dataset paths,
strata, or purposive-case hints. Human annotation and model-provisional packets
remain separate. An attestation records provenance, not proof of human identity.

## Human workflow

1. Enter a pseudonymous reviewer code; attest that reviewed judgments are human.
2. Give every member a phase number: equal numbers mean the same action/narrative
   phase. Answer the separate false-merge question. No judgment is preselected.
3. For **each phase**, explicitly choose:
   - **质量已判断** (`assessed`): select one or more acceptable keeper members.
   - **质量无法判断** (`quality_abstain`): the phase is distinguishable, but image
     quality/expression/sharpness cannot be judged. No acceptable keeper is
     required or stored. Different phases may have different quality statuses.
4. **保存阶段判断与备注** saves the complete partition, per-phase quality status,
   group impurity and note. If even the partition cannot be judged, use
   **保存为不确定（含备注）** or **保存为无法审阅（含备注）** instead. Both save
   the note, but intentionally store no phase/quality/impurity judgments. Never
   force `reviewed` just to keep a note.
5. Saved feedback explicitly confirms the note and export/recovery availability.
   The page attempts a bundle-scoped `localStorage` backup of **saved records**;
   it reports storage failure without pretending persistence succeeded. Reload
   restores that snapshot when available. Browser storage can be cleared, blocked,
   quota-limited or scoped differently between local-file/host/port URLs: **export
   JSONL before leaving and confirm that the download exists**.
6. Editing invalidates the task's saved record. Unsaved form edits are drafts,
   are not exported or automatically restored, and trigger a leave warning even
   after other saved records were exported. Click any appropriate save button
   to retain the note. Undo removes the saved record but leaves an unsaved form.
7. Import validates the entire file before replacing state. It is snapshot
   replacement, not append/merge, and asks before discarding unexported records
   or drafts. Import restores notes, phases and quality controls. A new judgment
   requires the reviewer code and human attestation again. There is no
   multi-reviewer adjudication or implicit note interpretation. The separately
   authorized `migrate-notes` operation below is explicit, deterministic and audited.

### Image viewer

Click a thumbnail or focus its button and press Enter/Space. The modal supports
previous/next buttons and Left/Right keys within the same group; Escape closes
and returns focus to the opener. **适屏** fits the cached image to the viewport;
**原像素（缓存 1:1）** displays its intrinsic pixels. The UI states dimensions
and **“放大不增加细节”**. Neither mode is an original-image endpoint.

The r3 release reuses only r2's verified 320×240 thumbnail bytes. No larger
trusted derivative has been supplied to this migration; none is discovered,
generated, interpolated into evidence or read from originals. The generic fresh
exporter can accept a bounded JPEG from an explicitly supplied trusted thumbnail
cache root (maximum dimension 2048), but cannot recognize a mislabeled original
cache. Operators must never point it at originals. Any future larger derivative
requires an explicit trusted source and a new sealed bundle.

## Strict label schema and v1 compatibility

Label/report semantics are version 2; the frozen sampling/bundle envelope remains
version 1. Ordinary labels have exactly `schema_version`, `bundle_id`, `task_id`,
`status`, `annotator`, `human_attested`, `phases`, `group_impure`, `note`.
Explicit authorized-migration v2 labels additionally carry `provenance`, whose
entire record must match the sealed migration registry. Arbitrary provenance or
extra fields are rejected; this is not an unrestricted metadata extension.
The following is a **synthetic schema example**, not a real annotation:

```json
{"schema_version":2,"bundle_id":"<hash>","task_id":"<opaque id>","status":"reviewed","annotator":"reviewer-01","human_attested":true,"phases":[{"members":["M1","M2"],"keeper_status":"assessed","acceptable_keepers":["M1"]},{"members":["M3"],"keeper_status":"quality_abstain"}],"group_impure":false,"note":"Action is clear; M3 expression is not assessable."}
```

- `reviewed` requires complete disjoint membership coverage and explicit boolean
  impurity/human attestation. Assessed phases require a nonempty unique keeper
  subset. An abstaining phase has **exactly** `members` and `keeper_status`;
  `acceptable_keepers` is forbidden (including an empty list).
- `uncertain` / `unassessable` require `phases: []`, `group_impure: null` and
  preserve `note`. Missing task rows are pending, not successful judgments.
- Duplicate JSON keys/tasks/members/keepers, unknown fields, non-integer version
  tokens (including `2.0`), NaN, incomplete partitions and scope mismatches fail.
  Browser and analyst CLI enforce the same label contract.
- Old v1 phase objects have only `members` and `acceptable_keepers`. A v1
  `reviewed` record retains **fully assessed** semantics, even if its note says
  the image is unclear. Imports show it as an editable old v1 label; software
  never implicitly guesses abstention, changes phase granularity or rewrites the
  note. Explicitly authorized note migration produces separately attributed v2
  records; ordinary v1 imports retain their original semantics.
- Export always emits strict v2. Converting v1 adds `keeper_status: assessed`
  and binds to the current bundle, preserving all human content and statistical
  meaning. Quality abstention requires an explicit human re-edit or the separately
  authorized deterministic note-migration rules below.

## Freeze/export and verified r2 → r3 migration

Fresh export uses the existing frozen v4 metadata and explicitly authorized cache
roots; source data is immutable:

```bash
python -m src.human_audit_bundle export \
  --groups /home/ubuntu/photo-dedup-eval/risk-coverage-v4/risk-groups.json \
  --audit /home/ubuntu/photo-dedup-eval/risk-coverage-v4/audit-manifest.json \
  --provenance /home/ubuntu/photo-dedup-eval/risk-coverage-v4/provenance.json \
  --scopes fixtures/willy-audit-development-scopes-v1.json \
  --cache disney /home/ubuntu/photo-dedup-eval/disney-conservative/thumbs \
  --cache jx3 /home/ubuntu/photo-dedup-eval/jx3-identity/thumbs \
  --fixture fixtures/disney-g10-narrative-review-v1.json \
  --seed human-phase-audit-v1 --output /path/to/new-bundle
```

Only owner `willy`/`synthetic`, split `train`/`tune` is authorized. Scope checks
precede data reads. Cache paths are explicit numeric JPEGs; symlinks, oversized
images and invalid scope fail closed; absent/corrupt thumbnails block reviewed
labels. Fresh exports re-encode pixels to remove metadata, never fall back to
originals, and require an output separate from caches that does not already exist.

For r3, **do not re-export/resample**: reuse the verified r2 bundle instead:

```bash
python -m src.human_audit_bundle upgrade \
  --bundle /home/ubuntu/photo-dedup-eval/human-audit-v1-20260909-r2 \
  --labels /home/ubuntu/photo-dedup-eval/human-audit-v1-20260909-r2/willy-partial-15.jsonl \
  --output /home/ubuntu/photo-dedup-eval/human-audit-v2-20260909-r3
```

Upgrade verifies the old manifest, sealed HTML and media first. It preserves
sampling strata/weights, source identity, task IDs/order, blind member mapping,
candidates and **exact cached media bytes**. It creates a newly sealed UI with
initial editable v1 labels, a sealed `legacy_bundle_id` and source-label hash.
Only v1 labels from that exact source ID are importable; v2 labels
must use the new ID. Fresh v2 bundles without legacy lineage reject v1 imports.
There is no arbitrary stale-bundle bypass. The old bundle
and input labels are not modified. An already-migrated bundle is not an upgrade
source. Output must be a new separate directory.

Outputs:
- `reviewer/index.html`, `reviewer/thumbs/`: the only reviewer-facing files. The
  sealed page embeds the imported human labels, not analyst metadata. Open it
  locally, or serve **only this subdirectory**, bound to localhost.
- `manifest.json`: private analyst mapping/provenance and fingerprints; do not
  expose or serve the bundle parent directory.
- `imported-v1-labels.jsonl`: byte-preserved source snapshot.
- `labels-v2.jsonl`: explicit structural conversion of that snapshot, not re-labeling.
- `imported-report.json`: replay of original label semantics, not a fresh review.
- `empty-labels.jsonl`: empty snapshot, never manufactured human ground truth.

The UI fingerprint normalizes only the single main bundle-ID marker, avoiding a
circular hash. Embedded initial labels retain the source ID and are covered by
the page hash. Replay rejects modified media, initial notes, HTML mapping or
manifest. Hashes detect mutation; they are not signatures against an attacker
who can replace the entire trusted bundle and recompute every hash.

## Explicit authorized note migration

Use this operation only after the reviewer explicitly authorizes note-based
conversion. It is not a default import behavior or a new human visual judgment:

```bash
python -m src.human_audit_bundle migrate-notes \
  --bundle /home/ubuntu/photo-dedup-eval/human-audit-v1-20260909-r2 \
  --labels /home/ubuntu/photo-dedup-eval/human-audit-v1-20260909-r2/willy-partial-15.jsonl \
  --authorization-id willy-20260909-explicit-note-migration \
  --output /home/ubuntu/photo-dedup-eval/human-audit-v2-20260909-r3-note-migration
```

The existing r2 and r3 are immutable. This new r3 variant uses the same verified
source mapping and media; there is no cache discovery, original-image read,
resampling or phase inference. All source notes and source JSONL bytes stay exact.

The versioned, deterministic rules in `human_audit_migration.py` prioritize:

1. A non-reviewed source, any question mark, or explicit phase uncertainty:
   `needs_review`; do not infer/repair a phase.
2. Both unclear-quality and affirmative quality phrases: `needs_review`.
3. Explicit unclear-thumbnail/cannot-judge/need-sharpness remarks:
   preserve the phase partition, make every phase `quality_abstain`, and remove
   `acceptable_keepers` (the strict v2 representation of no keeper judgment).
4. Explicit composition/completeness/water occlusion/horizon/frontal preference
   with existing keepers: preserve `assessed` and the original keeper subset.
5. No matching explicit quality statement: `needs_review`, not a guessed pass.
   Phase-only descriptions do not imply an affirmative quality evaluation.

`needs_review` preserves the old partition/keepers **as editable drafts**, not
statistical evidence. Its report status is `needs_review`; all coverage, quality
and joint values are null until a new human confirmation. Nothing silently
changes the source partition. A normal explicit UI save creates a strict
`human_reassessment` marker with the same `migration_id`, `confirmed: true`, and
`human_attestation_scope: current_judgment`; reviewed judgments also require
current human attestation. Simply removing provenance or reimporting the old v1
row for a migrated task is rejected in this variant. Normal v1 compatibility
remains available in the base r3 bundle.

`migration-manifest.json` is separate analyst-side evidence. Per task it records
source line number and raw line SHA256 (UTF-8 bytes excluding the line terminator),
verbatim note, rule ID, matched phrases, needs-review flag and complete v2 result.
The manifest records the authorization ID, exact rule patterns/version, input
hash, source/target bundle IDs and automatic/abstain/assessed/needs-review counts.
It is hashed with current-bundle IDs normalized to `@current` to avoid a circular
seal; replay validates the actual target IDs before normalizing and checking its
hash. The UI payload uses the same relative marker only for initial v2 rows,
materializing it to the sealed current ID before validation/export.

Each derived v2 record carries `provenance.kind=authorized_note_migration`,
`migration_id`, `source_row_sha256`, `needs_review`, and
`human_attestation_scope=original_visual_observation`. The preserved original
`human_attested` does **not** attest to the conversion or a new human operation.
The full derived row is bound to the sealed registry, so changing its note,
judgment, provenance or review flag without a new explicit human confirmation
marker is rejected by both browser and CLI. Removing the marker also fails
closed. Export/import/recovery preserves this distinction. A confirmation marker
records an explicit assertion; software cannot prove that a human actually
performed it. The retained `imported-v1-labels.jsonl` is checked against its sealed
hash as well as the migration manifest, including missing/symlink rejection.

`note_migration_counts` in reports distinguishes loaded automatic records,
needs-review records, explicit human reassessments and unmarked labels. The migration manifest retains the
initial counts even after a reviewer confirms/edits some tasks.

## Reporting: separate coverage, quality and legacy joint risk

```bash
python -m src.human_audit_bundle report \
  --bundle /home/ubuntu/photo-dedup-eval/human-audit-v2-20260909-r3 \
  --labels /path/to/downloaded-v2.jsonl --output /path/to/new-report.json
```

Reports never overwrite an existing output. Compare frozen candidate aliases to
human evidence only after annotation:

| Field | Meaning / eligibility |
|---|---|
| `phase_coverage_error` | Any human phase contains no selected candidate **member**, regardless of quality. Eligible for every complete reviewed partition, including quality abstention. |
| `keeper_quality_error` | Any selected candidate is unacceptable in its phase. Group-level value only if **every** phase is quality-assessed; otherwise null. Coverage omission is not itself a quality error. |
| `quality_assessed_phase_count/errors` | Descriptive phase-level counts on assessed phases only, including assessed phases of mixed groups. An assessed phase with no selected candidate has no bad selection; its omission is counted in coverage instead. |
| `quality_abstain_phase_count` | Explicit quality abstentions: excluded, never counted as pass or error. |
| `phase_miss`, `keeper_bad`, `error` | Legacy acceptable-phase-coverage, bad-keeper and false-merge union. All null if any phase quality abstains, even when another error is known. With full assessment, old v1 semantics are unchanged. |

`evidence_counts` separately lists probability and purposive coverage/quality
eligible/error counts, joint eligible count, assessed-phase and abstention
counts. `strata.reviewed` is not synonymous with `joint_eligible_count`.

The estimand remains the frozen **102-group development silent population**
(72 SAFE_SILENT + 30 DIAGNOSTIC_SAMPLE), excluding primary review. Replay fixes
the 30 probability draws and stratum allocation; no post-label resampling.
Disney G10 is one additional purposive narrative recheck, pinned to members
430/431/432/433 and the source fingerprint, not a supplied expected conclusion.
Its result never enters probability prevalence or confidence bounds.

The old joint weighted error and upper bound require all drawn probability
labels fully assessed and every nonempty stratum sampled. Missing, uncertain,
unassessable, needs-review or any probability keeper abstention keeps overall values **null**.
Do not drop abstentions, substitute model labels or selectively label until a
desired bound appears. A purposive abstention does not invalidate the separate
probability estimand. Complete stratum summaries may still be reported.

For stratum h, weighted risk is Σ(N_h/N)·errors_h/n_h. Census bounds are exact;
otherwise U_h=min(1, errors_h/n_h + sqrt(log(H/0.05)/(2n_h))), where H is the
number of non-census strata. Weight U_h by N_h/N for the simultaneous 95% bound.
This conservative SRS-without-replacement Hoeffding + Bonferroni bound omits a
finite-population correction and is not pooled Wilson/binomial inference.

If all probability phase partitions are available, `phase_only` can separately
show a weighted **descriptive** phase-coverage error rate, including quality
abstentions. It has no confidence bound and **is not safety certification**.
All reports say `safety_validated: false`; no threshold or deletion authority is
created. Correlated human mistakes, low-resolution evidence, prior exposure and
development-event generalization limits remain. Forced legacy judgments require explicit human correction or authorized,
provenance-marked note migration before substantive conclusions. Ambiguous
migration records still require human confirmation.

## Reproducible checks

```bash
.venv/bin/python -m pytest -q tests/test_human_audit.py tests/test_risk_coverage.py
bash scripts/test-fast.sh
bash scripts/test-full.sh
# Use an environment with Playwright, pytest, Pillow and numpy already installed:
python scripts/verification/human_audit_browser.py
```

The Chrome test creates synthetic caches and labels only. It covers seeded v1
editing, v2 conversion, uncertain/unassessable notes, mixed phase quality,
export/import/local recovery, strict malformed inputs, atomic rejection, draft
warnings, failed storage, lightbox keyboard/focus/zoom, and reviewer-only requests.
