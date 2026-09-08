# Multi-scene shadow-mode evaluation — 2026-09-08

## Safety contract

- Source media stayed read-only. No Google Photos write API or trash/delete action was used.
- `AUTO_REMOVE` remains restricted to byte-identical assets; all visual groups require review.
- The event-policy model contract accepts only event kind, confidence, valuable-state hints, mixed-theme, and identity sensitivity. Arbitrary thresholds and delete/action vocabulary are rejected.
- Capability fallback is monotonic-safe: missing VLM/LLM/network reduces maximum visual-group size, raises keeper count, requires identity evidence, and never expands removal authority.
- Every compiled preset protects singleton and rare states. Large visual groups must be split by action phase or retain multiple keepers.

## Full-set regression

| Dataset | Scope | Before | After | Largest visual group | Result |
|---|---:|---:|---:|---:|---|
| Disney 2026-09-06 | 1,706 items / 1,667 stills | 192 groups, 957 MAYBE | 68 groups, 102 MAYBE | 46 → 5 | Conservative stage profile sharply reduces fixed-background/action over-grouping; 0 auto-delete |
| JX3 2026-08-29 | 508 items / 508 stills | 40 groups, 66 MAYBE | 45 groups, 92 MAYBE | 6 → 9 | Local anonymous SFace gate ran fail-closed; more review is expected because identity uncertainty no longer permits aggressive automatic trust; 0 auto-delete |

The JX3 “after” count is not a deletion-rate improvement metric. It is a safety regression: face-bearing visual edges are accepted only when both sides contain usable anonymous same-subject evidence. No names or cross-event identity index are produced.

Review entry points:

- `/home/ubuntu/photo-dedup-eval/disney-conservative/review.html`
- `/home/ubuntu/photo-dedup-eval/jx3-identity/review.html`

## Read-only Google Photos smoke matrix

Sampler evidence: `/home/ubuntu/photo-dedup-eval/multiscene-smoke/inventory.json` and three position screenshots where the Photos grid loaded in time.

| Date / scene | Read-only observation | Policy exercised | Status |
|---|---|---|---|
| 2026-07-31 ChinaJoy | 600 unique items enumerated; contact sheets show dense halls, booths, crowds, statues/products, mascots and repeated angles | convention; identity gate; static-detail and mascot-state protection | PASS (inventory + visual sample) |
| 2026-08-02 Arknights carnival | date search rendered only a partial 41-item “most relevant” subset during this run; known day total is 2,107 | convention; mixed-theme fallback | PARTIAL — do not treat 41 as the day total |
| 2026-04-04 Xuhui Riverside dog park | 556 unique items enumerated; sheets show many dog identities, pose/action sequences, occasional multi-dog frames and video | pet; multi-keeper action phase; no human-face quality assumption | PASS (inventory + visual sample) |
| 2026-06-14 Noa Lolita/design show | date search did not render results during this run; previously read-only indexed as 1,148 items | design_reference; multi-angle/detail/back-view/singleton protection | PARTIAL — policy regression only, fresh visual sample unavailable |
| 2025-11-23 Disney tree lighting | 1,415 unique items enumerated; sheets show parade/stage/float sequences and dense action states; previously observed 1,036 Live Photos | stage; action-phase multi-keeper; Live Photo binding | PASS (inventory + visual sample) |

## Scene policy matrix

| Preset | Max non-exact visual group | Keepers per action phase | Identity gate | Extra protection |
|---|---:|---:|---|---|
| general | 8 | 1 | required | singleton + rare state |
| stage | 5 | 2 | optional | action/character-count/position changes |
| convention | 8 | 1 | required | different subject, static detail |
| pet | 6 | 2 | optional | animal pose/action and multi-subject states |
| design_reference | 4 | 2 | optional | front/back/side, texture/detail, opportunistic reference |
| any degraded capability | ≤4 | ≥2 | required | review-only; singleton + rare state |

## Verification

- Focused safety/regression suite: **83 passed**.
- `python -m compileall`: passed.
- `git diff --check`: passed.
- Full-set source counts matched prior inventories; all reports show **0 files to delete / 0 cloud items to trash**.

## v1 evaluation protocol (framework only)

- Dataset manifests are versioned and split by whole event. Existing Willy data
  is development-only (`train`/`tune`); Lin data must carry a distinct
  `dataset_id` and `held_out` split.
- Held-out manifests fail closed for training, threshold search, prompt
  selection, and preset selection. Final evaluation is report-only and cannot
  write policy/config or gain delete/trash authority.
- Non-exact visual groups now have deterministic cached-feature phase
  segmentation and at least one keeper per phase. Large/uncertain phases retain
  more; evidence includes quality, face clarity, exposure, subject
  completeness/occlusion proxies, and embedding diversity.
- Offline annotation/metrics compare baseline and candidate at phase level.
  Checked-in validation is synthetic. **No Lin labels or real held-out results
  were read or claimed for this framework change.**

## Known limits

1. This is a conservative shadow-mode prototype, not a learned universal aesthetic model and not permission to delete.
2. The Disney profile controls group size through short windows/high similarity thresholds; explicit visual action-phase segmentation is represented in the constrained policy and multi-keeper contract but is not yet a trained pose/temporal model.
3. SFace uses the largest usable face per frame. Multi-person reordering, masks, profile faces, small faces and pets fail closed to review; they do not enable grouping.
4. Google Photos search virtualization/network loading made the Arknights and Noa fresh samples incomplete. Their policies are exercised by deterministic tests, but this run does not claim visual coverage of the full days.
5. Downloaded Google Photos copies are suitable for relative grouping regressions, not absolute raw-image quality calibration.
