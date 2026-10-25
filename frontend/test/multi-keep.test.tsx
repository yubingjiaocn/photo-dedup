import { describe, expect, it } from 'vitest';
import { render } from 'preact';
import { ActionBar, Filmstrip, Header } from '../src/components/chrome';
import { keyIntent, isHandledKey, type KeyContext } from '../src/keyboard';
import { humanKeeperId, humanKeptIds, initialState, reducer, type State } from '../src/store';
import type { Group, GroupPage, Item } from '../src/types';

function mount(node: preact.ComponentChild): HTMLElement {
  const host = document.createElement('div');
  document.body.appendChild(host);
  render(node as never, host);
  return host;
}

const item = (id: number, over: Partial<Item> = {}): Item => ({
  file_id: id, group_id: 1, decision: 'MAYBE', basename: `IMG_${id}.jpg`, width: 10, height: 10,
  quality_score: 1, face_count: 0, reason: 'R', thumb: 'ok', is_keep: id === 2, ...over,
});
const grp: Group = { group_id: 1, group_type: 'phash_near', member_count: 3, members: [item(1), item(2), item(3)] };
const page: GroupPage = {
  view: 'GROUPS', queue: 'PENDING', page: 1, pages: 1, page_size: 100, total: 1, items: [grp],
  review_state: {}, queue_counts: { PENDING: 1, LATER: 0, DONE: 0 }, undo_depth: 0, state_warning: null,
};
const loaded: State = reducer(initialState, { type: 'groupPageLoaded', data: page, keepIndex: false });
const noop = () => {};
const ctx: KeyContext = { mode: 'queue', helpOpen: false, browseOpen: false, phase: 'down', repeat: false, shift: false };

describe('multi-keep', () => {
  it('maps X/S/G to select, keep-selected, keep-all in the queue only', () => {
    expect(keyIntent('x', ctx)).toEqual({ kind: 'toggleSelect' });
    expect(keyIntent('S', ctx)).toEqual({ kind: 'keep', all: false });
    expect(keyIntent('g', ctx)).toEqual({ kind: 'keep', all: true });
    expect(keyIntent('g', { ...ctx, repeat: true })).toEqual({ kind: 'none' });
    expect(keyIntent('x', { ...ctx, mode: 'browse' as KeyContext['mode'] })).toEqual({ kind: 'none' });
    expect(isHandledKey('X')).toBe(true);
  });

  it('humanKeptIds reads keep lists and legacy picks; humanKeeperId stays pick-only', () => {
    const keep = { ...loaded, reviewState: { '1': { action: 'keep', file_ids: [1, 3], keep_all: false } } };
    expect(humanKeptIds(keep, 1)).toEqual([1, 3]);
    expect(humanKeeperId(keep, 1)).toBeNull();
    const pick = { ...loaded, reviewState: { '1': { action: 'pick', file_id: 3 } } };
    expect(humanKeptIds(pick, 1)).toEqual([3]);
    expect(humanKeptIds(loaded, 1)).toEqual([]);
  });

  it('renders kept badges, tick marks, keep buttons and the done header', () => {
    const keep = { ...loaded, reviewState: { '1': { action: 'keep', file_ids: [1, 3], keep_all: false } } };
    const strip = mount(<Filmstrip state={keep} onShowIndex={noop} selected={[2]} />);
    expect(strip.querySelectorAll('.badge.human').length).toBe(3);
    const bar = mount(
      <ActionBar state={loaded} onAccept={noop} onPick={noop} onMark={noop} onUndo={noop}
        onToggle={noop} onKeep={noop} selectedCount={0} />,
    );
    expect((bar.querySelector('#bKeepSel') as HTMLButtonElement).disabled).toBe(true);
    expect((bar.querySelector('#bKeepAll') as HTMLButtonElement).disabled).toBe(false);
    const head = mount(<Header state={keep} onPrev={noop} onNext={noop} onClose={noop} />);
    expect(head.textContent).toContain('保留多张（2 张）');
  });
});
