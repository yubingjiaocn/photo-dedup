import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { App } from '../src/app';
import type { AppDeps } from '../src/app';
import type { ZoomAdapter, ZoomState } from '../src/viewer/zoom-adapter';

class NoopZoom implements ZoomAdapter {
  readonly kind = 'panzoom' as const;
  setImage(): void {}
  zoomBy(): void {}
  zoomToScale(): void {}
  reset(): void {}
  actualPixelScale(): number { return 2; }
  state(): ZoomState { return { scale: 1, x: 0, y: 0 }; }
  applyState(): void {}
  onChange(): void {}
  destroy(): void {}
}

const deps: AppDeps = {
  makeAdapter: () => new NoopZoom(),
  retainImages: () => {},
  osd: false,
};

const groupPage = {
  view: 'GROUPS', queue: 'PENDING', page: 1, pages: 1, page_size: 100, total: 1,
  items: [{ group_id: 7, group_type: 'burst', member_count: 1, members: [{
    file_id: 70, group_id: 7, decision: 'KEEP', basename: 'group.jpg', width: 10,
    height: 10, quality_score: 1, face_count: 0, reason: 'keep', thumb: 'ok',
    is_keep: true,
  }] }],
  review_state: {}, queue_counts: { PENDING: 1, LATER: 0, DONE: 0 },
  undo_depth: 0, state_warning: null,
};

function browsePage(view: string, page: number, pageSize: number) {
  return {
    view, page, pages: 3, page_size: pageSize, total: view === 'ALL' ? 101 : 22,
    shown: 1, remaining_after_page: 0,
    items: [{ file_id: view === 'ALL' ? 1 : 2, group_id: null, decision: 'MAYBE',
      basename: `${view}.jpg`, width: 4000, height: 3000, size_bytes: 1,
      exif_datetime: '2026-01-01', file_kind: 'jpg', quality_score: 1,
      face_count: 0, reason: null, thumb: 'ok', thumb_error: null, is_keep: null }],
  };
}

async function settle() {
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });
}

async function reviewApi() {
  for (let i = 0; i < 20; i += 1) {
    const review = (globalThis as Record<string, unknown>).__review;
    if (review) return review as { goView: (view: 'MAYBE' | 'ALL') => Promise<void> };
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('__review test hook did not boot');
}

afterEach(() => {
  for (const child of [...document.body.children]) render(null, child);
  document.body.innerHTML = '';
  delete (globalThis as Record<string, unknown>).__review;
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe('browse loading uses the requested target, not stale reducer state', () => {
  it('first click from GROUPS to ALL requests ALL immediately', async () => {
    const urls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input); urls.push(url);
      if (url.includes('view=GROUPS')) return new Response(JSON.stringify(groupPage), { status: 200 });
      if (url === '/api/status') return new Response(JSON.stringify({ queues: groupPage.queue_counts, undo_depth: 0 }), { status: 200 });
      const parsed = new URL(url, 'http://local');
      const view = parsed.searchParams.get('view')!;
      return new Response(JSON.stringify(browsePage(view, Number(parsed.searchParams.get('page')), Number(parsed.searchParams.get('page_size')))), { status: 200 });
    }));
    const host = document.createElement('div'); document.body.appendChild(host);
    render(<App deps={deps} />, host); await settle();

    urls.length = 0;
    await act(async () => { (host.querySelector('[data-view="ALL"]') as HTMLButtonElement).click(); await settle(); });

    expect(urls[0]).toContain('view=ALL');
    expect(urls[0]).not.toContain('view=GROUPS');
    expect(host.textContent).toContain('ALL.jpg');
    expect(host.textContent).not.toContain('group.jpg');
  });

  it('rapid MAYBE then ALL ignores the older response', async () => {
    let releaseMaybe!: () => void;
    const maybeGate = new Promise<void>((resolve) => { releaseMaybe = resolve; });
    const urls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input); urls.push(url);
      if (url.includes('view=GROUPS')) return new Response(JSON.stringify(groupPage), { status: 200 });
      if (url === '/api/status') return new Response(JSON.stringify({ queues: groupPage.queue_counts, undo_depth: 0 }), { status: 200 });
      const parsed = new URL(url, 'http://local');
      const view = parsed.searchParams.get('view')!;
      if (view === 'MAYBE') await maybeGate;
      return new Response(JSON.stringify(browsePage(view, 1, 100)), { status: 200 });
    }));
    const host = document.createElement('div'); document.body.appendChild(host);
    render(<App deps={deps} />, host); await settle();

    const review = await reviewApi();
    let maybe!: Promise<void>;
    await act(async () => {
      maybe = review.goView('MAYBE');
      await review.goView('ALL');
    });
    releaseMaybe();
    await act(async () => { await maybe; await settle(); });

    expect(urls.some((url) => url.includes('view=MAYBE'))).toBe(true);
    expect(urls.some((url) => url.includes('view=ALL'))).toBe(true);
    expect(host.textContent).toContain('ALL.jpg');
    expect(host.textContent).not.toContain('MAYBE.jpg');
  });
});
