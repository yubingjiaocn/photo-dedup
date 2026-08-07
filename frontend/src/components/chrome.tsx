/** Presentational chrome: tabs, header, filmstrip, action bar, toast, completion,
 *  evidence, warning, help, browse list. Pure functions of state + handlers, so
 *  the store stays the single source of truth and these stay trivially testable. */
import type { ComponentChildren } from 'preact';
import {
  BROWSE_LABEL,
  BROWSE_VIEWS,
  QUEUES,
  QUEUE_LABEL,
  decisionText,
  groupTypeText,
  reasonText,
  thumbSrc,
  type BrowseView,
  type Group,
  type Item,
  type Queue,
  type View,
} from '../types';
import {
  aiKeeperId,
  currentGroup,
  currentItem,
  groupStateOf,
  humanKeeperId,
  mode,
  positionInQueue,
  stageItems,
  type State,
} from '../store';

function badges(item: Item, aiId: number | null, humanId: number | null, short = false) {
  const fid = Number(item.file_id);
  return (
    <>
      {aiId != null && Number(aiId) === fid && (
        <span class="badge ai">🤖AI{short ? '' : '推荐'}</span>
      )}
      {humanId != null && Number(humanId) === fid && (
        <span class="badge human">👤已选{short ? '' : '保留'}</span>
      )}
    </>
  );
}

export function QueueTabs({ state, onQueue, onView, onDensity, onEvidence, onHelp }: {
  state: State;
  onQueue: (q: Queue) => void;
  onView: (v: View) => void;
  onDensity: () => void;
  onEvidence: () => void;
  onHelp: () => void;
}) {
  return (
    <nav class="tabs" aria-label="审阅队列">
      {QUEUES.map((q) => (
        <button
          key={q}
          data-queue={q}
          class={mode(state) === 'queue' && q === state.queue ? 'on' : ''}
          type="button"
          onClick={() => onQueue(q)}
        >
          {QUEUE_LABEL[q]}
          <span class="n">{state.queueCounts[q] || 0}</span>
        </button>
      ))}
      <span class="sep">|</span>
      <span class="grouplabel">浏览</span>
      {BROWSE_VIEWS.map((v) => (
        <button key={v} data-view={v} class={state.view === v ? 'on' : ''} type="button" onClick={() => onView(v)}>
          {v === 'ALL' ? '全部（时间线）' : BROWSE_LABEL[v]}
        </button>
      ))}
      <span class="sep">|</span>
      <button id="densityBtn" type="button" onClick={onDensity}>
        密度：{state.density === 'dense' ? '紧凑' : '舒适'}
      </button>
      <button id="evidenceBtn" type="button" class={state.evidenceOpen ? 'on' : ''} onClick={onEvidence}>
        推荐依据 E
      </button>
      <button id="helpBtn" type="button" onClick={onHelp}>快捷键 ?</button>
    </nav>
  );
}

export function Stats({ state }: { state: State }) {
  if (mode(state) === 'queue') {
    const c = state.queueCounts;
    return (
      <div class="stats" id="stats">
        未审 {c.PENDING || 0} · 稍后 {c.LATER || 0} · 已完成 {c.DONE || 0}
        {state.total ? ` · 当前 ${positionInQueue(state)}/${state.total}` : ''}
      </div>
    );
  }
  return (
    <div class="stats" id="stats">
      {BROWSE_LABEL[state.view as BrowseView] || state.view}：共 {state.total} 项 · 第 {state.page}/{state.pages} 页
    </div>
  );
}

export function StateWarning({ warning }: { warning: string | null }) {
  if (!warning) return <div class="statewarn" id="stateWarn" role="alert" hidden />;
  const kept = /\(saved as ([^)]+)\)/.exec(String(warning));
  return (
    <div class="statewarn" id="stateWarn" role="alert">
      <b>审阅状态文件无法读取</b>：本次审阅从空白开始，之前的记录没有丢失。
      {kept ? (
        <>
          原文件已另存为 <code>{kept[1]}</code>。
        </>
      ) : (
        '原文件已保留。'
      )}
      <br />
      如需恢复，请退出后检查该文件，或直接继续审阅（新的决定会正常保存）。
    </div>
  );
}

export function Header({ state, onPrev, onNext, onClose }: {
  state: State;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
}) {
  const group = currentGroup(state);
  const items = stageItems(state);
  const gid = group ? Number(group.group_id) : null;
  const gstate = groupStateOf(state, gid);
  const isQueue = mode(state) === 'queue';
  const headClass =
    gstate && (gstate.action === 'accept' || gstate.action === 'pick')
      ? 'wbhead done'
      : gstate && gstate.action === 'mark'
        ? 'wbhead later'
        : 'wbhead';
  const stateText =
    gstate == null
      ? '待审'
      : gstate.action === 'accept'
        ? '已完成 · 保留AI推荐'
        : gstate.action === 'pick'
          ? '已完成 · 保留人工所选'
          : '稍后处理';
  const aiId = aiKeeperId(items);
  return (
    <div class={headClass} id="wbHead">
      <span class="pos" id="wbPos">
        {isQueue ? `${QUEUE_LABEL[state.queue]} ${positionInQueue(state)}/${state.total}` : '查看器'}
      </span>
      <span class="meta" id="wbMeta">
        {group && (
          <>
            <span class="gtype">{groupTypeText(group.group_type)}</span>
            <span class="enum"> {group.group_type || ''}</span>
            {` · ${group.member_count || items.length} 张 · 第 ${Math.min(state.mIndex, Math.max(0, items.length - 1)) + 1}/${items.length} 张`}
            {isQueue ? ` · ${stateText}` : ''}
            {aiId == null ? ' · 本组无AI推荐' : ''}
          </>
        )}
      </span>
      {isQueue ? (
        <>
          <button id="wbPrev" type="button" onClick={onPrev}>‹ 上一组 K</button>
          <button id="wbNext" type="button" onClick={onNext}>下一组 J ›</button>
        </>
      ) : (
        <button id="closeStage" type="button" onClick={onClose}>关闭 Esc</button>
      )}
    </div>
  );
}

export function Evidence({ state }: { state: State }) {
  if (!state.evidenceOpen) return null;
  const group = currentGroup(state);
  const members = (group && group.members) || [];
  const aiId = aiKeeperId(members);
  if (!members.length) {
    return (
      <div class="evidence open" id="evidence">
        本组没有可显示的推荐依据。
      </div>
    );
  }
  return (
    <div class="evidence open" id="evidence">
      <table>
        <tr>
          <th>照片</th>
          <th>清晰度评分</th>
          <th>人脸数</th>
          <th>系统判断</th>
          <th>依据</th>
        </tr>
        {members.map((m) => {
          const fid = Number(m.file_id);
          return (
            <tr key={fid}>
              <td class="name">
                {m.basename || ''}
                {aiId != null && aiId === fid && <span class="badge ai">🤖AI</span>}
              </td>
              <td>{m.quality_score == null ? '—' : Number(m.quality_score).toFixed(1)}</td>
              <td>{m.face_count == null ? '—' : Number(m.face_count)}</td>
              <td>
                {decisionText(m.decision)}
                <span class="enum">{m.decision || ''}</span>
              </td>
              <td>
                {m.reason ? (
                  <>
                    {reasonText(m.reason)}
                    <span class="enum">{m.reason}</span>
                  </>
                ) : (
                  '—'
                )}
              </td>
            </tr>
          );
        })}
      </table>
    </div>
  );
}

export function Filmstrip({ state, onShowIndex }: { state: State; onShowIndex: (i: number) => void }) {
  const items = stageItems(state);
  const item = currentItem(state);
  const aiId = aiKeeperId(items);
  const gid = currentGroup(state) ? Number(currentGroup(state)!.group_id) : null;
  const humanId = humanKeeperId(state, gid);
  const currentId = item ? Number(item.file_id) : null;
  return (
    <div class="filmstrip" id="strip" aria-label="组内照片">
      {items.map((m, i) => {
        const fid = Number(m.file_id);
        const cur = currentId === fid;
        return (
          <div
            key={fid}
            class={`fs-item${cur ? ' current' : ''}`}
            data-file-id={fid}
            title={m.basename || ''}
            aria-current={cur ? 'true' : 'false'}
            onClick={() => onShowIndex(i)}
          >
            {m.thumb === 'ok' ? (
              <img loading="lazy" src={thumbSrc(fid)} alt="" />
            ) : (
              <div class="fs-miss">无缩略图</div>
            )}
            <div class="fs-meta">
              <span class="fs-idx">
                {i + 1}/{items.length}
              </span>
              {badges(m, aiId, humanId, true)}
              {cur && <span class="badge cur">当前</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function ZoomBar({ zoomText, onCompare, onFit, onHundred }: {
  zoomText: string;
  onCompare: () => void;
  onFit: () => void;
  onHundred: () => void;
}) {
  return (
    <div class="zoombar">
      <button id="compareBtn" type="button" onClick={onCompare}>双栏对比 Shift+C</button>
      <button id="fitBtn" type="button" onClick={onFit}>适应窗口 F</button>
      <button id="hundredBtn" type="button" onClick={onHundred}>100% 键 1</button>
      <span>
        缩放 <span class="val" id="zoomVal">{zoomText}</span>
      </span>
      <span>滚轮缩放 · 拖动平移 · 双击回到适应窗口 · 1 = 100%</span>
    </div>
  );
}

export function ActionBar({ state, onAccept, onPick, onMark, onUndo }: {
  state: State;
  onAccept: () => void;
  onPick: () => void;
  onMark: () => void;
  onUndo: () => void;
}) {
  const items = stageItems(state);
  const noAi = aiKeeperId(items) == null;
  const busy = state.mutationBusy;
  return (
    <div class="actionbar" id="actionbar">
      <button id="bAccept" type="button" disabled={busy || noAi} onClick={onAccept}>A 保留 AI 推荐</button>
      <button id="bPick" type="button" disabled={busy} onClick={onPick}>P 保留当前照片</button>
      <button id="bMark" type="button" disabled={busy} onClick={onMark}>M 稍后处理</button>
      <button id="bUndo" type="button" disabled={busy || state.undoDepth <= 0} onClick={onUndo}>U 撤销上一步</button>
      <span class="hint">
        <kbd>J</kbd>/<kbd>K</kbd> 换组 · <kbd>H</kbd>/<kbd>L</kbd> 换照片 · 按住 <kbd>C</kbd> 闪切 AI 推荐 ·{' '}
        <kbd>Shift</kbd>+<kbd>C</kbd> 双栏 · <kbd>E</kbd> 推荐依据 · <kbd>D</kbd> 密度 · <kbd>?</kbd> 帮助
      </span>
    </div>
  );
}

export function DonePanel({ state, onGoLater, onGoDone }: {
  state: State;
  onGoLater: () => void;
  onGoDone: () => void;
}) {
  const c = state.queueCounts;
  return (
    <div class="donepanel" id="donePanel">
      <h2>本轮审阅完成</h2>
      <p id="doneText">
        已完成 {c.DONE || 0} 组 · 稍后 {c.LATER || 0} 组。
      </p>
      <button id="goLater" type="button" onClick={onGoLater}>查看稍后（{c.LATER || 0}）</button>
      <button id="goDone" type="button" onClick={onGoDone}>查看已完成（{c.DONE || 0}）</button>
    </div>
  );
}

function tileDecisionClass(item: Item): string {
  return String(item.decision || 'ungrouped').toLowerCase();
}

export function BrowseList({ state, onOpen, onPage, onSize }: {
  state: State;
  onOpen: (fileId: number) => void;
  onPage: (page: number) => void;
  onSize: (size: number) => void;
}) {
  return (
    <div id="listWrap">
      <nav id="pager">
        <button id="first" type="button" disabled={state.page <= 1} onClick={() => onPage(1)}>« 首页</button>
        <button id="prev" type="button" disabled={state.page <= 1} onClick={() => onPage(state.page - 1)}>‹ 上一页</button>
        <button id="next" type="button" disabled={state.page >= state.pages} onClick={() => onPage(state.page + 1)}>下一页 ›</button>
        <button id="last" type="button" disabled={state.page >= state.pages} onClick={() => onPage(state.pages)}>尾页 »</button>
        <label>
          每页{' '}
          <select id="size" value={String(state.size)} onChange={(e) => onSize(Number((e.currentTarget as HTMLSelectElement).value))}>
            <option value="50">50</option>
            <option value="100">100</option>
            <option value="200">200</option>
          </select>{' '}
          张
        </label>
      </nav>
      <div class="row" id="content">
        {state.browseItems.length === 0 ? (
          <div>本视图没有条目。</div>
        ) : (
          state.browseItems.map((r) => {
            const fid = Number(r.file_id);
            const aiId = r.is_keep === true ? fid : null;
            return (
              <div
                key={fid}
                class={`card ${tileDecisionClass(r)}`}
                data-file-id={fid}
                tabIndex={0}
                title="点击查看大图"
                onClick={() => onOpen(fid)}
              >
                {r.thumb === 'ok' ? (
                  <img loading="lazy" src={thumbSrc(fid)} alt="" />
                ) : (
                  <div class="miss">缩略图不可用</div>
                )}
                <div class="cap">
                  <span class="tag">{decisionText(r.decision)}</span>
                  {badges(r, aiId, null)}
                  <br />
                  {r.basename}
                  <br />
                  {r.width || '?'}x{r.height || '?'}
                  <br />
                  {r.exif_datetime || '无拍摄时间'}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export function Help({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <div class="help open" id="help">
      <div class="help-box">
        <h2>快捷键说明</h2>
        <table>
          <tr><td>? 或 F1</td><td>显示/隐藏此帮助</td></tr>
          <tr><td colspan={2} class="grouphead">审阅动作</td></tr>
          <tr><td>A</td><td>保留 AI 推荐，本组完成并进入下一组</td></tr>
          <tr><td>P</td><td>保留当前照片，本组完成并进入下一组</td></tr>
          <tr><td>M</td><td>稍后处理，本组移入「稍后」队列</td></tr>
          <tr><td>U</td><td>撤销上一步，回到刚才操作的那一组与那张照片</td></tr>
          <tr><td colspan={2} class="grouphead">导航</td></tr>
          <tr><td>J / ↓&nbsp;&nbsp;K / ↑</td><td>下一组 / 上一组</td></tr>
          <tr><td>L / →&nbsp;&nbsp;H / ←</td><td>组内下一张 / 上一张（点击底部缩略图同效）</td></tr>
          <tr><td>Esc</td><td>关闭浏览查看器 / 关闭帮助</td></tr>
          <tr><td colspan={2} class="grouphead">看图</td></tr>
          <tr><td>按住 C</td><td>闪切到 AI 推荐照片，松开回到当前照片</td></tr>
          <tr><td>Shift + C</td><td>双栏对比（当前照片 vs. AI 推荐），再按恢复单张</td></tr>
          <tr><td>滚轮 / 拖动</td><td>缩放 / 平移；双栏时两侧同步</td></tr>
          <tr><td>F 或双击</td><td>回到适应窗口</td></tr>
          <tr><td>1</td><td>100%（原始像素）</td></tr>
          <tr><td colspan={2} class="grouphead">界面</td></tr>
          <tr><td>E</td><td>展开/收起「推荐依据」</td></tr>
          <tr><td>D</td><td>列表密度：舒适 / 紧凑</td></tr>
        </table>
        <p style="color:#999;font-size:12px;margin-top:12px">
          未审队列为默认入口：A/P 决定后该组退出未审队列，M 移入稍后队列，队列清空后显示「本轮审阅完成」。所有状态仅写入
          output 目录，原图永远只读。
        </p>
        <button type="button" style="width:100%;margin-top:12px" onClick={onClose}>关闭 Esc</button>
      </div>
    </div>
  );
}

/** The stage panes. Host elements carry stable ids so the imperative zoom
 *  controller can mount adapters into them; Preact never re-creates them. */
export function Stage({ osd, labelA, labelB, aiA, compare, blink, secondaryReady, frameARef, frameBRef }: {
  osd: boolean;
  labelA: ComponentChildren;
  labelB: ComponentChildren;
  aiA: boolean;
  compare: boolean;
  blink: boolean;
  secondaryReady: boolean;
  frameARef: (el: HTMLElement | null) => void;
  frameBRef: (el: HTMLElement | null) => void;
}) {
  const showB = compare || (blink && secondaryReady);
  const stageClass = `stage${compare ? ' compare' : ''}${blink && secondaryReady ? ' blink-mode' : ''}`;
  return (
    <div class={stageClass} id="stage">
      <div class={`pane${aiA ? ' ai' : ''}`} id="paneA">
        <div class="label" id="labelA">{labelA}</div>
        {osd ? (
          <div class="osd" id="hostA" ref={frameARef} />
        ) : (
          <div class="frame" id="hostA" ref={frameARef} />
        )}
      </div>
      <div class={`pane${showB ? '' : ' hide'}`} id="paneB">
        <div class="label" id="labelB">{labelB}</div>
        {osd ? (
          <div class="osd" id="hostB" ref={frameBRef} />
        ) : (
          <div class="frame" id="hostB" ref={frameBRef} />
        )}
      </div>
    </div>
  );
}

export function Toast({ text, show }: { text: string; show: boolean }) {
  return (
    <div id="toast" class={show ? 'show' : ''} role="status">
      {text}
    </div>
  );
}

export function itemLabel(item: Item | null, aiId: number | null, humanId: number | null) {
  if (!item) return null;
  return (
    <>
      {decisionText(item.decision)} · {item.basename || ''}
      {badges(item, aiId, humanId)}{' '}
      <span class="enum">
        {item.width || '?'}×{item.height || '?'}
      </span>
    </>
  );
}

export type { Group, Item };
