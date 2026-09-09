# Human audit v2 acceptance — 2026-09-09

## Delivered product

Standalone blinded annotation: saved notes for uncertain/unassessable tasks;
complete phase partitions with per-phase assessed/quality-abstain states; strict
v1 import and v2 export; separate coverage and quality reporting; cached-image
lightbox with keyboard navigation, fit and native-pixel views; verified bundle
migration without resampling or rewriting human labels.

No operational Preact/pipeline change, no inference, no original/DB/embedding/
held-out access, no deletion authorization change, and no remote push. Untracked
`docs/images/` is outside the read/write/staging scope.

## Authoritative r3 artifact

`/home/ubuntu/photo-dedup-eval/human-audit-v2-20260909-r3/`

- Reviewer: `reviewer/index.html`; only `reviewer/` is served locally.
- Bundle ID: `842e52453556a611c5c86520491ee3e4e880033e120118fad02cab89c5b6d9dc`.
- Source r2 ID: `63a9e5d109d79ec8601a49b27d00f7c4f43fee51d4f4cb71841382a35da99e85`.
- 30 frozen probability tasks from 102 silent groups; one purposive G10 task.
- **15 editable legacy reviewed records imported; 16 tasks remain pending.**
- `imported-v1-labels.jsonl` exactly preserves the source bytes, including notes,
  phases, keepers, statuses, reviewer provenance and old bundle ID.
- Source-label SHA256:
  `93d4ab38710900f49f8ef25ed329fd447d8bfd1e5438eb703b21cf0e63a79deb`.
- `labels-v2.jsonl` is explicit structural conversion: assessed status added to
  every v1 phase and current bundle binding; no human judgment inferred/changed.
- `imported-report.json` preserves every old task's joint error, phase miss and
  bad-keeper result. Overall joint risk and upper bound remain **null**.
- All 74 media files are byte-identical to r2: 52 are 320×240, 22 are 240×320.
  No larger derivative was supplied or generated. All 31 tasks have media.
- Sampling, order, source identity, task IDs, blind alias/member mapping and
  candidate mapping are unchanged. All **80 r2 file hashes** match the snapshot
  taken before this work, including the user's JSONL and existing report.
- `acceptance-proof.json` records these comparisons. This is metadata/byte
  verification, not a new visual judgment of any real photo.

## Real reviewer path

The server was started bound to **127.0.0.1:8773**, serving only the r3 reviewer
subdirectory. It is a local test process, not an installed persistent service.
Open the file directly if the process has stopped; the bundle is independent
of the server. No public bind, original-photo endpoint or analyst-data endpoint
was added.

Chrome/Playwright verified the real r3 page:

- 31 sections, 15 visible editable v1 records with notes, 16 pending.
- 74 cached images loaded; no page JavaScript error.
- Keyboard image opening, native pixels, group next image, Escape/focus return.
- Export of the initial 15 records matches the validated v2 structural conversion;
  reloading the page retains the seeded source labels. **No human labels edited.**
- Requests remain localhost-only. `manifest.json`, `imported-v1-labels.jsonl` and
  `acceptance-proof.json` return HTTP 404 through the reviewer service.
- Proof: `live-browser-proof.json` in r3.

## Final verification results

| Gate | Result |
|---|---|
| Focused human audit + risk coverage | **65 passed** |
| `scripts/test-fast.sh` | **337 passed** |
| `scripts/test-full.sh` | **756 passed** |
| Ruff, compileall, `git diff --check` | **PASS** |
| Synthetic Chrome save/export/replay/partial/lightbox workflow | **PASS** |
| Real r3 localhost seed/export/lightbox/isolation workflow | **PASS** |
| r2 file hashes and r3 sampling/mapping/media parity | **PASS** |

Final focused/fast/full logs and the synthetic browser log are preserved under
r3 `verification/`. This host's browser command uses the already-installed
Playwright package without downloading dependencies:

```bash
PYTHONPATH=/home/ubuntu/.local/lib/python3.12/site-packages \
  .venv/bin/python scripts/verification/human_audit_browser.py
```

## Verification coverage

Focused tests cover uncertainty-note serialization/replay; partial/full quality
abstention; assessed-phase errors without a fabricated group verdict; phase-only
coverage; null joint risk under probability abstention; purposive exclusion;
legacy semantics regardless of note text; strict duplicate/unknown-field/type/
partition rejection; exact source snapshot migration; sealed lineage; source,
media and embedded-label tamper gates; and migration CLI completion without
rereading the source labels.

The reproducible synthetic Chrome gate exercises actual save/download/import,
local recovery, storage failure, draft-loss and canceled-import behavior; legacy
editing and conversion; mixed phase quality; raw malformed JSON parity; lightbox
mouse/keyboard/zoom/focus; and local-only reviewer requests. It uses synthetic
labels/images only and never manufactures labels for the real dataset.

Independent read-only review checked the schema, statistical eligibility,
migration and UI. Its lineage and source-snapshot findings were repaired and
covered by regression tests. Follow-up review confirmed those core fixes and
found one completion-message source reread; that final CLI read was removed and
separately regression-tested by the parent. The reviewer did not execute Chrome;
the parent performed the synthetic and real-bundle browser gates.

## Remaining limits

1. The 15 v1 `reviewed` judgments keep their original fully assessed meaning.
   The user's report that some were forced to save notes is not a machine-readable
   replacement annotation. Human re-edit is needed; do not use those legacy
   quality claims as newly verified truth.
2. Enlarging these cached pixels cannot establish expression/sharpness details.
   Use per-phase `quality_abstain` where needed. A larger source requires a new,
   explicitly trusted safe derivative workflow, never an original-image fallback.
3. Phase-only results are descriptive, not safety certification. Missing or
   uncertain probability labels/quality abstentions prevent joint risk estimates.
   `safety_validated` remains false; no deletion authority is granted.
4. Browser storage is convenience recovery, not a durable file backup. Export
   JSONL and confirm the download; unsaved form drafts are not recovered.
5. Fingerprints detect mutation relative to a trusted bundle, not replacement
   by an attacker who can recompute the complete bundle and its hashes.
