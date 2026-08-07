# Frontend production decision

The review workbench is now a Preact application under `frontend/`, built by
Vite, with its compiled `frontend/dist/` committed and served by the Python
review server. This document records the two decisions the migration required —
**component framework** and **zoom viewer** — and the exact test evidence behind
them.

## Summary

| Decision | Choice | Why |
|---|---|---|
| Component/state framework | **Preact 10.29.8** + Vite 8.2.1 + TypeScript | Small, no VDOM-runtime surprises, no router/global-state/UI-lib needed; the whole queue/undo/blink/HDD logic became pure and unit-testable. |
| Zoom viewer | **@panzoom/panzoom 4.6.2** (OpenSeadragon 6.1.0 retained as a lazy-loaded comparison co-finalist) | Keeps the `<img>` + CSS-transform model, so one decoded `<img>` per active-group URL *is* the HDD policy; less application glue; identical behavioural gates pass. |

Exact versions are locked in `frontend/package-lock.json` (81 lockfile packages,
3 direct runtime deps). No Node.js is required to *run* the pipeline — the build
is committed and Stage 3 copies it into the output directory.

## Viewer decision: Panzoom vs OpenSeadragon

The brief required both viewers normalised to one state contract — `scale = 1`
is contain-fit, `x = y = 0` is the fitted centre — and then judged on
behavioural stability and application glue, **not** bundle size.

### Normalisation done before deciding

OpenSeadragon's native unit is *viewport zoom* (1 = image width fills the
container), and its raw open state is not the stage's "fit". The spike left this
un-normalised (`b-osd` reported a non-zero centre immediately after open). The
production `OsdZoom` adapter now:

- calls `goHome` on open and reports `scale = zoom / homeZoom`, so a freshly
  opened image reads exactly `scale = 1`;
- reports pan as a screen-px offset from the home-bounds centre (derived from
  `getHomeBounds()`, not a captured snapshot, so it is layout-timing
  independent), so fit reads exactly `x = 0, y = 0`.

With that, **both** viewers pass the identical centred-fit assertion
(`scale≈1, x≈0, y≈0`, rendered frame/image centres within 2 px) at every phase:
initial, after switching photo, after F reset, after closing compare, and after
releasing blink. Evidence: `frontend/harness/browser_check.py` runs the whole
gate against `prod-panzoom` **and** `prod-osd`; both pass with the same exact
request sequence and zero page errors.

### What decided it: glue, not bytes

Both are behaviourally viable. Panzoom wins on **application glue** and **model
fit**, which is where the brief said to decide:

- **HDD policy falls out of the model.** Panzoom drives a plain `<img>`, so the
  project's HDD rule — one decoded original per active-group URL, fetched once
  under `no-store`, moved between panes without re-`src`, released on group
  change — is implemented by a small image pool that owns those `<img>`
  elements (`viewer/panzoom-adapter.ts`, `ImageStore`). OSD renders to a canvas
  world; it never reuses an `<img>`, so the same guarantees would need a
  parallel, viewer-specific mechanism. The pool is application policy we own
  under any viewer; with Panzoom it is *also* the rendering path, so there is no
  second thing to keep correct.
- **Fewer unit translations.** Panzoom's adapter subtracts one flex-centred
  contain offset. OSD's adapter continuously translates among home/viewport/
  image zoom units and screen-px pan, plus explicit `close()`/`addSimpleImage`
  world lifecycle and wheel capture — more surface to keep aligned with the
  shared contract.
- **Bundle size is a tiebreaker only, and it agrees.** Not a gate, but recorded:

  | Chunk (committed build) | raw | gzip | brotli |
  |---|---:|---:|---:|
  | `review-*.js` (Preact + app + Panzoom, production initial graph) | 57,826 B | 21,045 B | 18,351 B |
  | `review-*.css` | 8,086 B | 2,513 B | 2,180 B |
  | `osd-adapter-*.js` (lazy; only on `?viewer=osd`) | 351,916 B | 88,156 B | 72,724 B |

  OSD is ~88 KB gzip vs Panzoom folded into a 21 KB gzip initial bundle. Per the
  brief this is not a negative on its own — but it is code-split behind the
  `?viewer=osd` query, so it never loads for a normal reviewer, and nothing about
  the production experience carries it.

**Decision: ship Panzoom.** OSD stays in the tree, lazy-imported, exercised by
the same harness, so the co-finalist remains real and the decision stays
auditable.

## Test evidence

### Frontend unit tests — Vitest (`frontend/test/`, 44 tests)

- `store.test.ts` — reducer + selectors: page load, keepIndex, photo/group
  navigation, compare/blink pane logic (pane A never swaps its source),
  **HDD retention** (retains exactly the current group's members, releases on
  group change, does not grow on compare/blink), preference restore validation.
- `keyboard.test.ts` — the full key map: A/P/M/U, auto-repeat suppression,
  hold-C blink vs Shift+C compare, H/L vs J/K by mode, F/1/E/D, help/escape.
- `image-pool.test.ts` — the HDD pool: one decoded `<img>` per URL, reuse on
  re-display, release outside the allowed set.
- `components.test.tsx` — tabs counts + active state, ActionBar enable/disable
  (no-AI-keeper, busy), filmstrip `/api/thumb`-only + AI/current badges,
  evidence panel diagnostics + internal enum, state-warning banner (present /
  hidden / no diagnostics), browse cards, stats without diagnostics.

### Production browser harness — real Chromium (`frontend/harness/`)

Drives the **built** app served by the **real** `src.review_server`
(`fixture_server.py` / `run_all.py`). Run: `python3 frontend/harness/run_all.py`.

Gates enforced (all pass, both viewers, **0 page errors**):

- Landing on the PENDING queue with the workbench visible, three filmstrip
  thumbnails, and exactly **one** `/api/original` request (no prefetch).
- Centred contain-fit asserted at initial / switch-photo / reset / compare-close
  / blink-release (`scale≈1, x≈0, y≈0`, centres within 2 px).
- Wheel zoom in, drag pan, `F` reset, `1` → 100%, zoomed-frame cursor class.
- Dual-pane compare with synchronised zoom across both panes.
- **Exact full request sequence `/2 → /1 → /5`** (initial keeper, move off
  keeper with `L`, next group with the header button).
- **Three hold-C blink cycles add exactly zero** `/api/original` requests
  (blink overlays the already-pooled AI keeper in pane B; pane A's source is
  never reassigned).
- Opening compare fetches nothing (the AI keeper was already retained).
- On group change, exactly one new original, and the **previous group's decoded
  `<img>` pool is released** (Panzoom).

Queue-workflow gate (`run_queue_workflow`, production build): A keeps the AI
pick and advances; P records the shown photo; M → later, then **U undoes back to
that exact group and photo**; a double-tapped decision posts **once** (mutation
guard); reload restores the queue and group; a form-encoded write is **403**
(CSRF); only `review.html` + hashed assets are served (working files 404);
emptying the queue shows the completion panel.

State-warning gate: an unreadable `review_state.json` raises the Chinese banner,
the workbench stays usable, a decision still records, and the banner carries no
diagnostics.

Screenshots: `frontend/artifacts/screenshots/` (`prod-panzoom-*`, `prod-osd-*`,
`prod-workflow-complete`, `prod-state-warning`). Machine-readable request log:
`frontend/artifacts/browser-results.json`.

### Python suite

`652 passed` (`.venv/bin/python -m pytest tests/`). Server API and safety tests
(CSRF/origin, root scope, read-only, static allow-list, thumbnails, originals,
pagination, queues, undo, state durability) are unchanged. The brittle
inline-JS/node-stub tests were removed; their behaviour now lives in Vitest and
the browser harness. New Python tests gate the build/manifest/served-allow-list
(`tests/test_review_frontend_build.py`) and the UI install path
(`tests/test_stage3_performance.py`), and the front-end source boundary
(`tests/test_review_workbench_ui.py`). `ruff` clean; `compileall` clean.

## Boundaries preserved

Server queues; A/P/M with true `U` undo; cross-page next; reload restore;
completion panel; density; recommendation/evidence panel; unreadable-state
warning; CSRF + loopback-only + read-only + root scope; delete manifests;
thumbnails-only browsing with demand-only originals and no prefetch. The browser
only ever receives `file_id`/`basename` — no source path — and the UI uses only
the read-only API plus the single `/api/action` mutation.

## Explicit non-goals (kept)

No router, no global-state library, no design system, no UI component library.
No DZI/tile pyramids, service worker, speculative preload or adjacent-original
prefetch. Preact is presentation/state only; deletion/move safety stays in
Python.
