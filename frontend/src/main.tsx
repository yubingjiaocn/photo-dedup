/**
 * Production entry. Panzoom is the shipped viewer (see FRONTEND_DECISION.md); it
 * is imported directly. OpenSeadragon is only reachable via `?viewer=osd`, and is
 * lazy-imported so it never enters the production initial bundle graph — the
 * page still renders and works if that dynamic chunk is absent.
 */
import { render } from 'preact';
import './review.css';
import { App, type AppDeps } from './app';
import { PanzoomZoom, retainImages } from './viewer/panzoom-adapter';
import type { ZoomAdapter } from './viewer/zoom-adapter';

const wantOsd = new URLSearchParams(location.search).get('viewer') === 'osd';

async function makeDeps(): Promise<AppDeps> {
  if (wantOsd) {
    // Comparison harness only: pull OSD in on demand.
    const { OsdZoom } = await import('./viewer/osd-adapter');
    return {
      osd: true,
      makeAdapter: (host: HTMLElement): ZoomAdapter => new OsdZoom(host),
      // OSD owns its own canvas world; the <img> pool does not apply.
      retainImages: () => {},
    };
  }
  return {
    osd: false,
    makeAdapter: (host: HTMLElement): ZoomAdapter => new PanzoomZoom(host),
    retainImages,
  };
}

const root = document.getElementById('app');
if (root) {
  makeDeps().then((deps) => render(<App deps={deps} />, root));
}
