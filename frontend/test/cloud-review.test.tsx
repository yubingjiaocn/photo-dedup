import { expect, it, vi } from 'vitest';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { CloudGroupReview, type CloudGroup } from '../src/cloud-review';

const a = { file_id: 248, content_sha256: 'a'.repeat(64) };
const b = { file_id: 284, content_sha256: 'b'.repeat(64) };
function group(status: CloudGroup['ai']['status'] = 'select'): CloudGroup {
  return { group_id: 14, alias: '2026-04-04-G014', member_count: 2,
    members: [a, b].map(f => ({ ...f, basename: `${f.file_id}.jpg`, image_url: '/x.jpg', width: 4000, height: 3000 })),
    ai: { status, recommended: status === 'select' ? [b] : [], multi_select_reason: null,
      failure: status === 'manual' ? 'unexpected content' : null,
      discriminators: [{ index: 0, text: 'A potentially inaccurate observation.', confidence: 'low', evidence: [{ frame: b, confidence: 'low' }] }] }, review: null };
}
const button = (host: HTMLElement, text: string) => [...host.querySelectorAll('button')].find(b => b.textContent === text)!;

it('keeps every frame visible and allows one-click expansion from single to multi-select', async () => {
  const host = document.createElement('div'), action = vi.fn(async () => {});
  await act(() => render(<CloudGroupReview group={group()} busy={false} onAction={action} />, host));
  expect(host.querySelectorAll('.cloud-card')).toHaveLength(2);
  expect(host.querySelectorAll('.recommended')).toHaveLength(1);
  expect(host.textContent).toContain('弱观察（不支持推荐）');
  expect(host.textContent).toContain('结构校验不等于判断正确');
  const picks = host.querySelectorAll<HTMLInputElement>('input');
  expect(picks[0].checked).toBe(false); expect(picks[1].checked).toBe(true);
  await act(() => { picks[0].dispatchEvent(new Event('change', { bubbles: true })); });
  await act(() => button(host, '保存我的选择').click());
  expect(action).toHaveBeenLastCalledWith('modify', { selected: [a, b] });
  await act(() => button(host, '全部保留').click());
  expect(action).toHaveBeenLastCalledWith('keep_all');
  await act(() => button(host, '稍后处理').click());
  expect(action).toHaveBeenLastCalledWith('later');
});

it('can accept recommendation while flagging or ignoring incorrect explanation', async () => {
  const host = document.createElement('div'), action = vi.fn(async () => {});
  await act(() => render(<CloudGroupReview group={group()} busy={false} onAction={action} />, host));
  await act(() => button(host, '接受AI建议').click());
  expect(action).toHaveBeenLastCalledWith('accept');
  await act(() => button(host, '忽略此理由').click());
  expect(action).toHaveBeenLastCalledWith('feedback', { inaccurate: true, ignored_reasons: [0] });
  await act(() => button(host, '建议不准确').click());
  expect(action).toHaveBeenLastCalledWith('feedback', { inaccurate: true });
});

it.each(['manual', 'abstain'] as const)('shows %s with all photos and disables AI accept', async status => {
  const host = document.createElement('div');
  await act(() => render(<CloudGroupReview group={group(status)} busy={false} onAction={async () => {}} />, host));
  expect(host.querySelectorAll('.cloud-card')).toHaveLength(2);
  expect(button(host, '接受AI建议').disabled).toBe(true);
  expect(host.textContent).toContain(status === 'manual' ? 'AI结果不可用，手工检查' : 'AI 已弃权，请手工选择');
});

it('restores human multi-selection independently from AI single selection', async () => {
  const host = document.createElement('div'), g = group();
  g.review = { status: 'modified', selected: [a, b], feedback: true, ignored_reasons: [0] };
  await act(() => render(<CloudGroupReview group={g} busy={false} onAction={async () => {}} />, host));
  expect([...host.querySelectorAll<HTMLInputElement>('input')].every(x => x.checked)).toBe(true);
  expect(host.querySelectorAll('.recommended')).toHaveLength(1);
  expect(host.textContent).toContain('已忽略，不影响保留选择');
});

it('does not discard unsaved extra keep when feedback refreshes the same group', async () => {
  const host = document.createElement('div'), action = vi.fn(async () => {}), g = group();
  await act(() => render(<CloudGroupReview group={g} busy={false} onAction={action} />, host));
  await act(() => { host.querySelector('input')!.dispatchEvent(new Event('change', { bubbles: true })); });
  const refreshed = { ...g, review: { status: 'pending', selected: [], feedback: true, ignored_reasons: [0] } };
  await act(() => render(<CloudGroupReview group={refreshed} busy={false} onAction={action} />, host));
  expect(host.querySelector<HTMLInputElement>('input')!.checked).toBe(true);
});

it('reuses a failed unchanged save ID, changes it with intent, and clears after success', async () => {
  const { CloudReview } = await import('../src/cloud-review');
  const host = document.createElement('div');
  const payloads: Record<string, unknown>[] = [];
  let fail = true;
  const data = { items: [group()], page: 1, pages: 1, total: 1,
    status: { batch_id: 'fixture', total: 1, processed: 0, remaining: 1, counts: {}, state_warning: null } };
  vi.stubGlobal('fetch', vi.fn(async (_url, init) => {
    if (init) {
      payloads.push(JSON.parse(init.body));
      return { ok: !fail, json: async () => fail ? { error: 'simulated HTTP failure' } : { ok: true } };
    }
    return { ok: true, json: async () => data };
  }));
  try {
    await act(async () => { render(<CloudReview />, host); });
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
    const click = async (text: string) => { await act(async () => { button(host, text).click(); await new Promise(resolve => setTimeout(resolve, 0)); }); };
    await click('保存我的选择');
    await click('保存我的选择');
    expect(payloads[0].request_id).toBe(payloads[1].request_id);
    await act(() => { host.querySelector('input')!.dispatchEvent(new Event('change', { bubbles: true })); });
    await click('保存我的选择');
    expect(payloads[2].request_id).not.toBe(payloads[1].request_id);
    await click('建议不准确');
    expect(payloads[3].request_id).not.toBe(payloads[2].request_id);
    await click('忽略此理由');
    expect(payloads[4].request_id).not.toBe(payloads[3].request_id);
    await click('忽略此理由');
    expect(payloads[5].request_id).toBe(payloads[4].request_id);
    fail = false;
    await click('忽略此理由');
    expect(payloads[6].request_id).toBe(payloads[5].request_id);
    await click('忽略此理由');
    expect(payloads[7].request_id).not.toBe(payloads[6].request_id);
  } finally {
    await act(() => render(null, host));
    vi.unstubAllGlobals();
  }
});
