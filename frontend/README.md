# Review workbench frontend

The review UI is a [Preact](https://preactjs.com/) application built with
[Vite](https://vite.dev/) and TypeScript. Its compiled output, `dist/`, is
**committed** to the repository and is the runtime source of truth: Stage 3
copies it into the review output directory and the Python server serves it.

**Running the pipeline needs no Node.js.** You only need Node to *change* the UI
and rebuild `dist/`.

## Layout

```
frontend/
  review.html              # Vite entry (mounts #app, loads the bundle)
  src/
    main.tsx               # entry: mounts <App>, picks the viewer
    app.tsx                # async orchestration (fetch/decide/undo, viewer, keyboard)
    store.ts               # pure reducer + selectors (queue/compare/blink/HDD)
    keyboard.ts            # pure key -> intent map
    api.ts                 # read-only /api/* + single /api/action
    persistence.ts         # localStorage restore (v2 shape)
    types.ts               # server JSON shapes + Chinese display vocabulary
    review.css             # the workbench stylesheet
    zoom-controller.ts     # imperative bridge to the zoom viewer + pane sync
    components/chrome.tsx   # presentational Preact components
    viewer/
      zoom-adapter.ts      # the fit=1 / pan=0 contract both viewers implement
      panzoom-adapter.ts   # production viewer + the decoded-<img> HDD pool
      osd-adapter.ts       # OpenSeadragon co-finalist (lazy; ?viewer=osd only)
  test/                    # Vitest unit tests (reducer/keyboard/pool/components)
  harness/                 # real-Chromium production acceptance test
  dist/                    # COMMITTED build artifact (do not hand-edit)
```

## Commands

```bash
cd frontend
npm ci            # install exact locked deps (Preact 10.29.8, Vite 8.2.1, ...)
npm run typecheck # tsc --noEmit
npm test          # Vitest unit tests
npm run build     # tsc --noEmit && vite build  -> writes dist/
npm run dev       # Vite dev server on :18930, proxying /api -> :18931
```

For `npm run dev`, start a review server for the API first, e.g.:

```bash
python3 frontend/harness/fixture_server.py --port 18931   # synthetic fixture
# then open http://127.0.0.1:18930/review.html
```

### Rebuilding after a UI change

```bash
cd frontend
npm ci
npm run build
# commit the regenerated dist/ alongside your source change
git add src dist
```

`dist/` must be committed with any change to `src/`, `review.html`, or the
dependency versions — CI/tests assert the committed build matches the source and
that `review.html` references only manifest-listed assets.

## Real-browser acceptance test

```bash
python3 frontend/harness/run_all.py
```

Starts the real `src.review_server` over a synthetic fixture and drives Chromium
through the production gates for both viewers (Panzoom and, via `?viewer=osd`,
OpenSeadragon): centred fit, wheel/drag/F/1, dual-pane sync, hold-C blink, the
exact `/api/original` request sequence, three-blink zero-request release,
old-group release, the A/P/M/U queue workflow, the mutation guard, reload
restore, CSRF and the static allow-list. Screenshots and a machine-readable
request log land in `frontend/artifacts/`.

## Design notes

See [`../FRONTEND_DECISION.md`](../FRONTEND_DECISION.md) for the framework and
viewer choices and the test evidence. Non-goals, kept: no router, no global-state
library, no design system, no UI component library; Preact is presentation/state
only, and deletion/move safety stays in Python.
