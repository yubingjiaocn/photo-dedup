import { useEffect, useRef, useState } from 'preact/hooks';
import { ReviewText } from './components/chrome';

export interface Frame { file_id: number; content_sha256: string }
export interface CloudMember extends Frame { basename: string; image_url: string; width: number; height: number }
export interface CloudGroup {
  group_id: number; alias: string; member_count: number; members: CloudMember[];
  ai: { status: 'select' | 'abstain' | 'manual' | 'attention_only'; recommended: Frame[];
    discriminators: { index: number; text: string; confidence: string; evidence: { frame: Frame; confidence: string }[] }[];
    multi_select_reason: { text: string } | null; failure: string | null };
  review: { status: string; selected: Frame[]; feedback: boolean; ignored_reasons: number[] } | null;
}
interface Status { batch_id: string; total: number; processed: number; remaining: number; counts: Record<string, number>; state_warning: string | null; estimated_cost_usd?: number }
interface Page { items: CloudGroup[]; page: number; pages: number; total: number; status: Status }
const filters = [['ALL', '全部'], ['PENDING', '待处理'], ['ACCEPTED', '已接受AI'], ['MODIFIED', '已修改'], ['AI_ISSUES', 'AI失败/弃权']] as const;
export const frameKey = (f: Frame) => `${f.file_id}@${f.content_sha256}`;
const frames = (members: Frame[]) => members.map(({ file_id, content_sha256 }) => ({ file_id, content_sha256 }));
async function request<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || '请求失败');
  return data;
}

export function CloudReview() {
  const [data, setData] = useState<Page | null>(null);
  const [filter, setFilter] = useState('ALL');
  const [index, setIndex] = useState(0);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const generation = useRef(0);
  const pendingIntent = useRef<{ key: string; id: string } | null>(null);
  async function load(queue: string, page = 1, position = 0) {
    const gen = ++generation.current;
    try {
      const result = await request<Page>(`/api/page?view=GROUPS&queue=${queue}&page=${page}&page_size=50`);
      if (gen !== generation.current) return;
      setData(result); setIndex(Math.min(position, Math.max(0, result.items.length - 1))); setFilter(queue); setError('');
    } catch (e) { if (gen === generation.current) setError((e as Error).message); }
  }
  useEffect(() => { void load('ALL'); }, []);
  const group = data?.items[index];
  async function action(name: string, extra: Record<string, unknown> = {}) {
    if (!group || !data || busy) return;
    setBusy(true); setError(''); setMessage('');
    try {
      const intent = { action: name, group_id: group.group_id, batch_id: data.status.batch_id, ...extra };
      const key = JSON.stringify(intent);
      if (pendingIntent.current?.key !== key) pendingIntent.current = { key, id: crypto.randomUUID() };
      await request('/api/action', { ...intent, request_id: pendingIntent.current.id });
      pendingIntent.current = null;
      await load(filter, data.page, index);
      setMessage(name === 'feedback' ? '反馈已保存，保留选择未改变' : '已保存到本地审阅状态，原片未改动');
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return <main class="cloud-review">
    <header class="cloud-top"><div><div class="cloud-eyebrow">PRIVATE REVIEW · 只读原片</div><h1>把值得留下的瞬间，交给你决定</h1><p>AI 提供起点，你保留最终选择。所有照片始终可见，未勾选不代表删除。</p></div><span class="cloud-safe">仅保存审阅记录</span></header>
    {data && <><section class="cloud-progress" aria-label="审阅进度"><strong>{data.status.total}<small>组照片</small></strong><strong class="cloud-done">{data.status.processed}<small>已处理</small></strong><strong>{data.status.remaining}<small>剩余</small></strong><progress max={data.status.total} value={data.status.processed} /><details><summary>批次摘要</summary><p>建议只通过结构校验，不代表内容正确。请检查独特动作与表情。</p>{data.status.estimated_cost_usd != null && <p>本批次模型成本估算 ${data.status.estimated_cost_usd.toFixed(2)}，非实际账单。</p>}</details></section>
      <nav class="cloud-filters" aria-label="筛选">{filters.map(([key, label]) => <button disabled={busy} class={filter === key ? 'on' : ''} onClick={() => void load(key)}>{label}<span>{data.status.counts[key]}</span></button>)}</nav></>}
    {data?.status.state_warning && <div class="statewarn" role="alert">{data.status.state_warning}</div>}
    {error && <div class="statewarn" role="alert">{error}<button onClick={() => void load(filter)}>重新加载</button></div>}
    <div class="cloud-message" role="status" aria-live="polite">{message}</div>
    {group && data && <><div class="wbhead"><b class="pos">{group.alias}</b><span class="meta">{group.member_count} 张 · {(group.review?.status === 'accepted' ? '已接受AI' : group.review?.status === 'modified' ? '已修改' : group.review?.status === 'later' ? '稍后处理' : '待处理')}</span><span>{(data.page - 1) * 50 + index + 1} / {data.total}</span><button disabled={busy || (index === 0 && data.page === 1)} onClick={() => index > 0 ? setIndex(index - 1) : void load(filter, data.page - 1, 49)}>上一组</button><button disabled={busy || (index === data.items.length - 1 && data.page === data.pages)} onClick={() => index + 1 < data.items.length ? setIndex(index + 1) : void load(filter, data.page + 1)}>下一组</button></div>
      <CloudGroupReview key={`${data.status.batch_id}:${group.group_id}`} group={group} busy={busy || !!data.status.state_warning} onAction={action} onIntentChange={() => { pendingIntent.current = null; }} />
    </>}
    {data && !group && <section class="cloud-empty"><h2>这个筛选下没有分组</h2><p>已保存的选择可以随时在“全部”中重新调整。</p><button onClick={() => void load('ALL')}>查看全部照片</button></section>}
    {!data && !error && <p role="status">正在核对批次与照片身份…</p>}
  </main>;
}

export function CloudGroupReview({ group, busy, onAction, onIntentChange }: { onIntentChange?: () => void; group: CloudGroup; busy: boolean; onAction: (name: string, extra?: Record<string, unknown>) => Promise<void> }) {
  const initial = () => new Set((group.review?.selected.length ? group.review.selected : group.ai.recommended).map(frameKey));
  const [selected, setSelected] = useState(initial);
  const [zoom, setZoom] = useState<CloudMember | null>(null);
  const [dirty, setDirty] = useState(false);
  useEffect(() => { setSelected(initial()); setDirty(false); }, [group.group_id, JSON.stringify(group.review?.selected ?? [])]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') setZoom(null); };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, []);
  const recommended = new Set(group.ai.recommended.map(frameKey));
  const ignored = group.review?.ignored_reasons ?? [];
  function toggle(member: Frame) {
    onIntentChange?.();
    const next = new Set(selected), key = frameKey(member);
    if (next.has(key)) next.delete(key); else next.add(key);
    setSelected(next); setDirty(true);
  }
  return <>
    <div class="cloud-layout"><section class="cloud-photos" aria-label="完整组照片"><div class="cloud-section-title"><h2>完整组照片</h2><span>点击勾选可直接改为多选 · 点击照片放大</span></div><div class="cloud-grid">{group.members.map((m) => <article class={`cloud-card ${recommended.has(frameKey(m)) ? 'recommended' : ''} ${selected.has(frameKey(m)) ? 'selected' : ''}`} key={frameKey(m)}>
      <div class="cloud-card-head"><span>#{m.file_id}</span>{recommended.has(frameKey(m)) && <b class="cloud-ai-badge">AI 建议</b>}</div>
      <button class="cloud-photo-button" onClick={() => setZoom(m)} aria-label={`放大照片 ${m.file_id}`}><img src={m.image_url} alt={`照片 ${m.file_id} ${m.basename}`} loading="lazy" /></button>
      <label class="cloud-pick"><input type="checkbox" disabled={busy} checked={selected.has(frameKey(m))} onChange={() => toggle(m)} /><span>{selected.has(frameKey(m)) ? '保留' : '未选中'}</span><small>{m.width} × {m.height}</small></label><div class="cloud-filename" title={m.basename}>{m.basename}</div>
    </article>)}</div></section>
    <aside class="cloud-advice"><div class="cloud-eyebrow">AI 建议 · {group.ai.status === 'select' ? 'SELECT' : group.ai.status === 'abstain' ? 'ABSTAIN' : 'MANUAL'}</div><h2>{group.ai.status === 'select' ? `建议保留 ${group.ai.recommended.length} 张` : group.ai.status === 'abstain' ? 'AI 已弃权，请手工选择' : 'AI结果不可用，手工检查'}</h2><p class="cloud-caveat">结构校验不等于判断正确。不同动作、表情和互动，可能都值得留下。</p>
      {group.ai.discriminators.map(d => <section class={`cloud-reason ${ignored.includes(d.index) ? 'ignored' : ''}`} key={d.index}><div class="cloud-reason-head"><b>{d.confidence === 'low' ? '弱观察（不支持推荐）' : `区分点 ${d.index + 1}`}</b><button disabled={busy} aria-pressed={ignored.includes(d.index)} onClick={() => void onAction('feedback', { inaccurate: true, ignored_reasons: ignored.includes(d.index) ? ignored.filter(i => i !== d.index) : [...ignored, d.index] })}>{ignored.includes(d.index) ? '恢复理由' : '忽略此理由'}</button></div><ReviewText text={d.text} /><div class="cloud-evidence-ids">涉及照片 {Array.from(new Set(d.evidence.map(e => e.frame.file_id))).map(id => `#${id}`).join('、')}{ignored.includes(d.index) && ' · 已忽略，不影响保留选择'}</div></section>)}
      {group.ai.multi_select_reason && <details><summary>为什么建议多选</summary><ReviewText text={group.ai.multi_select_reason.text} /></details>}
      {group.ai.failure && <details><summary>结果不可用的原因</summary><p>{group.ai.failure}</p></details>}
      <button class="cloud-feedback" disabled={busy} aria-pressed={group.review?.feedback ?? false} onClick={() => void onAction('feedback', { inaccurate: !group.review?.feedback })}>{group.review?.feedback ? '已标记建议不准确 · 撤回' : '建议不准确'}</button><p class="cloud-footnote">反馈和忽略理由不改变你的保留选择。</p>
    </aside></div>
    <footer class="cloud-actions"><div><b>已勾选 {selected.size} / {group.members.length} 张</b><small>{dirty ? '选择尚未保存' : '可增删保留，也可全部保留'}</small></div><button disabled={busy || group.ai.status !== 'select'} class="cloud-accept" onClick={() => void onAction('accept')}>接受AI建议</button><button disabled={busy || selected.size === 0} class="cloud-save" onClick={() => void onAction('modify', { selected: frames(group.members.filter(m => selected.has(frameKey(m)))) })}>保存我的选择</button><button disabled={busy} onClick={() => void onAction('keep_all')}>全部保留</button><button disabled={busy} onClick={() => void onAction('later')}>稍后处理</button><span>不移动、不删除原片</span></footer>
    {zoom && <div class="cloud-lightbox" role="dialog" aria-modal="true" aria-label={`照片 ${zoom.file_id} 大图`} onClick={() => setZoom(null)}><button autoFocus onClick={() => setZoom(null)}>关闭大图 · Esc</button><img src={zoom.image_url} alt={zoom.basename} onClick={e => e.stopPropagation()} /></div>}
  </>;
}
