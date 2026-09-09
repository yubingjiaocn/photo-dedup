# Authorized note migration — 2026-09-09

## Authorization and scope

The reviewer explicitly authorized deterministic conversion of the 15 existing
v1 notes. This is a separate operation from normal v1 import, which continues to
preserve old reviewed semantics without interpreting notes. The conversion is
`authorized_note_migration`, not a new human visual review.

Source: `human-audit-v1-20260909-r2/willy-partial-15.jsonl` under
`/home/ubuntu/photo-dedup-eval/`. Source SHA256:
`93d4ab38710900f49f8ef25ed329fd447d8bfd1e5438eb703b21cf0e63a79deb`.

The new immutable variant is named
`/home/ubuntu/photo-dedup-eval/human-audit-v2-20260909-r3-note-migration/`.
The original r2 and already-published r3 are retained unchanged; their reviewer
states are not silently replaced. No originals, DBs, embeddings or held-out data
are inputs. Only existing verified thumbnails are copied, byte for byte.

Released bundle ID:
`f2c7ee2b71c67def791e5cb098c202a543ee2530448921ec0b9f0887033f9655`.
The reviewer is served on this host at `http://127.0.0.1:8774/`, from the new
variant's `reviewer/` only. The previous r3 and its 8773 service are unchanged.
Both servers are temporary local processes, not installed persistent services.

## Deterministic disposition of the 15 source rows

| Source line numbers | Disposition | Reason |
|---|---|---|
| 1, 5, 7, 8, 11, 12, 14 | 7 quality-abstain records | Explicit unclear thumbnail / cannot choose / need sharpness statements. Preserve all phases; no keeper acceptability claim. |
| 2, 4, 9 | 3 assessed records | Explicit composition completeness, water occlusion, or frontal-face preference with selected keepers. |
| 3, 6, 15 | 3 needs-review records | Notes contain question marks; no automatic interpretation of uncertainty. |
| 10, 13 | 2 needs-review records | Stage/pose/similarity descriptions without an explicit quality judgment. No guessed assessed pass or guessed abstention. |

Thus **10 automatic migrations + 5 needs-review**. All 15 retain the exact
original note and phase partition. The other 16 bundle tasks remain pending.
The two unmatched notes are conservative review fallbacks, not evidence that
the existing phase partition is incorrect.

For quality abstention, `acceptable_keepers` is absent rather than an empty array:
this is the existing strict v2 representation of **no keeper judgment**. Neither
selected nor acceptable keepers are inferred from note text. Assessed records
keep their original keeper subset. Needs-review records retain original values
only as editable drafts; they contribute no coverage, quality or joint evidence.

## Provenance and audit trail

`migration-manifest.json` separately records:

- Authorization ID and versioned rule definitions/priority.
- Source file hash, source/target bundle IDs and migration ID.
- Per-task source line number/hash, verbatim note, matched phrases, rule ID,
  needs-review flag and complete v2 result.
- Initial automatic/assessed/abstain/needs-review counts.

Each derived JSONL row carries a strictly validated `provenance` object with
`kind=authorized_note_migration`, migration/source-row identifiers,
`needs_review`, and `human_attestation_scope=original_visual_observation`.
The original `human_attested` bit is retained only in that stated scope. It does
not attest that the reviewer personally performed or checked the conversion.

The full derived row is bound to the sealed bundle registry. Altering its note,
partition, keeper state, provenance or review flag is rejected on import. The
manifest itself is also sealed, with current-bundle placeholders normalized to
avoid circular hashing. Replaying with changed manifest content or target IDs
fails closed. The byte-preserved `imported-v1-labels.jsonl` is also required to
match its sealed hash; changing, removing or replacing it with a symlink fails.

The UI visibly distinguishes authorized migrations from new human judgments.
Export/import and local recovery preserve provenance. For a needs-review task,
a human can confirm or edit the restored controls, provide current attestation
and explicitly save; that produces a v2 record carrying explicit
`human_reassessment` provenance, the same migration ID, `confirmed: true`, and
current-judgment attestation scope. Deleting provenance alone or downgrading a
migrated task to its old v1 row cannot promote it to evidence. This is an explicit
attestation marker, not proof of human identity or physical interaction.
No additional source observation is manufactured by the migration.

## Statistical interpretation

- Seven quality-abstain records can support phase coverage, but not keeper
  quality or the old union group-risk verdict.
- Five needs-review records are excluded from **all** coverage/quality/joint
  calculations, until explicitly confirmed by a human.
- Reports list automatic versus needs-review counts separately, rather than
  counting all 15 as directly completed human judgments.
- The probability sample is still incomplete, so overall joint risk and upper
  95% bound remain null. Phase-only descriptions are not safety certification.
- `safety_validated: false`, annotation-only authority and byte-identical-only
  automatic removal policy are unchanged.

## Acceptance results

- Focused: **88 passed**; fast: **360 passed**; full: **779 passed**.
- Ruff, compileall and diff-check: **PASS**.
- Independent repair recheck: **PASS**, both provenance-removal and retained
  source-artifact integrity findings closed.
- Synthetic Chrome: **PASS**, including provenance round trips, removal
  rejection, explicit reassessment markers and needs-review statistics.
- Real new-variant Chrome: **PASS** — 31 tasks, all 15 notes exact, 10 automatic
  + 5 needs-review + 16 pending, 74 loaded cache images, original-pixel/lightbox
  keyboard flow, exact provenance-preserving export, and reload.
- No real human label was edited during browser testing. Export was compared
  exactly with the validated migration snapshot.
- Analyst manifest, migration manifest and original-label copy return HTTP 404
  through the reviewer service. No page errors or non-local page requests.
- Original r2's **80** and previous r3's **91** file hashes remain unchanged.
  Frozen sample, blind mapping and all **74** thumbnail bytes are unchanged.

Proof and final logs are retained in the new variant directory. This acceptance
is for the annotation product and migration contract, not a photo-quality or
algorithm safety certification.

## Reproducible verification

The focused and browser gates cover deterministic classification, question and
conflict precedence, unmatched-note fallback, raw-line hashes, exact notes and
partitions, original attestation scope, provenance binding, manifest tampering,
needs-review statistical exclusion, source immutability, and explicit new human
confirmation. Synthetic tests use only synthetic images and labels.

Commands and the explicit `migrate-notes` CLI invocation are documented in
[HUMAN_AUDIT.md](HUMAN_AUDIT.md). Real-bundle acceptance evidence belongs in the
new variant's `acceptance-proof.json`, `live-browser-proof.json`, `release.json`
and `verification/`, separately from the original source files.

No external product download, APK execution, algorithm benchmarking or deletion
semantics change is part of this increment. Low-resolution zoom remains a UX
aid, not additional quality evidence.
