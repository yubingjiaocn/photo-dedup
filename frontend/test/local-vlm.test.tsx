import { describe, expect, it } from 'vitest';
import { render } from 'preact';
import { ActionBar, Filmstrip, LocalVlmPanel } from '../src/components/chrome';
import { initialState, reducer } from '../src/store';
import type { Group, GroupPage, Item, LocalVlm } from '../src/types';

function mount(node: preact.ComponentChild): HTMLElement {
  const host = document.createElement('div');
  document.body.appendChild(host);
  render(node as never, host);
  return host;
}

const item = (id: number): Item => ({
  file_id: id, group_id: 1, decision: 'MAYBE', basename: `IMG_${id}.jpg`, width: 10, height: 10,
  quality_score: 1, face_count: 0, reason: 'R', thumb: 'ok', is_keep: id === 2,
});
const ok: LocalVlm = {
  schema: 'local-trial-vlm-v1', status: 'ok', model: 'my-alias', served_model: 'upstream-x',
  verdict: 'recommend', confidence: 'medium', keep_file_ids: [1, 3],
  reasons: [{ text: '<b>不同表情</b>', file_ids: [1, 3] }], human_confirmation: '仅供参考',
};
const items = [item(1), item(2), item(3)];

describe('local VLM panel', () => {
  it('renders reasons as text and accepts the exact suggested ids only on click', () => {
    let accepted: number[] | null = null;
    const host = mount(<LocalVlmPanel vlm={ok} items={items} busy={false} onAccept={(ids) => { accepted = ids; }} />);
    expect(host.textContent).toContain('建议保留：第1张、第3张');
    expect(host.textContent).toContain('接口返回 upstream-x');
    expect(host.querySelector('b')).toBeNull();
    expect(host.textContent).toContain('<b>不同表情</b>');
    expect(accepted).toBeNull();
    (host.querySelector('#bVlmAccept') as HTMLButtonElement).click();
    expect(accepted).toEqual([1, 3]);
  });

  it('shows failures visibly with no accept button, and nothing without a sidecar', () => {
    const failed: LocalVlm = { ...ok, status: 'failed', error: 'invalid_answer', detail: 'keep names a photo that is not in this group' };
    const host = mount(<LocalVlmPanel vlm={failed} items={items} busy={false} onAccept={() => {}} />);
    expect(host.textContent).toContain('VLM 建议失败：模型回复不符合格式');
    expect(host.querySelector('#bVlmAccept')).toBeNull();
    expect(mount(<LocalVlmPanel vlm={undefined} items={items} busy={false} onAccept={() => {}} />).textContent).toBe('');
  });

  it('badges VLM picks in the filmstrip and keeps the A button named CV', () => {
    const grp: Group = { group_id: 1, group_type: 'phash_near', member_count: 3, members: items, local_vlm: ok };
    const page: GroupPage = {
      view: 'GROUPS', queue: 'PENDING', page: 1, pages: 1, page_size: 100, total: 1, items: [grp],
      review_state: {}, queue_counts: { PENDING: 1, LATER: 0, DONE: 0 }, undo_depth: 0, state_warning: null,
    };
    const s = reducer(initialState, { type: 'groupPageLoaded', data: page, keepIndex: false });
    const strip = mount(<Filmstrip state={s} onShowIndex={() => {}} />);
    const badged = [...strip.querySelectorAll('.fs-item')].filter((el) => el.querySelector('.badge.vlm'));
    expect(badged.map((el) => el.getAttribute('data-file-id'))).toEqual(['1', '3']);
    const bar = mount(<ActionBar state={s} onAccept={() => {}} onPick={() => {}} onMark={() => {}} onUndo={() => {}} />);
    expect(bar.querySelector('#bAccept')!.textContent).toBe('A 保留 CV 推荐');
  });
});
