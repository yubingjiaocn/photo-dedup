/**
 * The review workbench, wired together.
 *
 * State transitions live in the pure reducer (`store.ts`); the key map lives in
 * `keyboard.ts`; the zoom viewer lives behind `ZoomController`. This component is
 * the async orchestration: fetch pages, post decisions and follow the server to
 * the next/undone group, drive the image pool, persist to localStorage, and show
 * a toast. It reproduces the exact server contract the hand-written page used
 * (`/api/page|status|locate|next|group|action`) so the Python side is unchanged.
 */
import { useCallback, useEffect, useReducer, useRef, useState } from 'preact/hooks';
import * as api from './api';
import {
  ActionBar,
  BrowseList,
  DonePanel,
  Evidence,
  Filmstrip,
  Header,
  Help,
  QueueTabs,
  Stage,
  StateWarning,
  Stats,
  Toast,
  ZoomBar,
  itemLabel,
} from './components/chrome';
import { keyIntent, type KeyContext } from './keyboard';
import { LS_KEY, readLocal, saveLocal } from './persistence';
import {
  aiKeeperId,
  currentGroup,
  currentItem,
  humanKeeperId,
  initialState,
  mode,
  paneAItem,
  paneBActive,
  paneBItem,
  reducer,
  retainedOriginals,
  showCompare,
  stageItems,
  type Action,
  type State,
} from './store';
import { originalSrc, QUEUES, type BrowseView, type Queue, type View } from './types';
import { ZoomController, type AdapterFactory } from './zoom-controller';

export interface AppDeps {
  makeAdapter: AdapterFactory;
  retainImages: (urls: string[]) => void;
  osd: boolean;
}

export function App({ deps }: { deps: AppDeps }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [zoomText, setZoomText] = useState('适应窗口');
  const [toast, setToast] = useState<{ text: string; show: boolean }>({ text: '', show: false });

  // Latest state, so async callbacks and the keyboard listener never close over
  // a stale snapshot (they are registered once).
  const stateRef = useRef(state);
  stateRef.current = state;

  const controllerRef = useRef<ZoomController | null>(null);
  const loadGenRef = useRef(0);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const restoreRef = useRef<{ groupId: number | null; fileId: number | null } | null>(null);
  const bootedRef = useRef(false);

  const showToast = useCallback((text: string) => {
    setToast({ text, show: true });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast((t) => ({ ...t, show: false })), 1900);
  }, []);

  if (!controllerRef.current) {
    controllerRef.current = new ZoomController(deps.makeAdapter, setZoomText);
  }

  // --- density reflected on <body> (CSS var switch) ------------------------
  useEffect(() => {
    document.body.classList.toggle('dense', state.density === 'dense');
  }, [state.density]);

  // --- persist on any meaningful change ------------------------------------
  useEffect(() => {
    if (bootedRef.current) saveLocal(state);
  });

  // --- drive the viewer + image pool from state ----------------------------
  useEffect(() => {
    const controller = controllerRef.current!;
    controller.setCompare(showCompare(state));
    // Release everything outside the current group first, then set panes. Both
    // panes pull from the shared pool, so this order guarantees the current
    // group's decoded originals survive while stale groups are dropped.
    deps.retainImages(retainedOriginals(state).map(originalSrc));
    const a = paneAItem(state);
    controller.setPaneImage('A', a ? originalSrc(a.file_id) : null);
    const b = paneBItem(state);
    controller.setPaneImage('B', b ? originalSrc(b.file_id) : null);
  });

  // === async orchestration ================================================

  const loadQueue = useCallback(async (opts: {
    keepIndex?: boolean;
    queue?: Queue;
    page?: number;
    size?: number;
  } = {}) => {
    const generation = ++loadGenRef.current;
    const s = stateRef.current;
    const targetQueue = opts.queue ?? s.queue;
    const targetPage = opts.page ?? s.page;
    const targetSize = opts.size ?? s.size;
    try {
      const data = await api.fetchGroupPage(targetQueue, targetPage, targetSize);
      if (generation !== loadGenRef.current) return;
      dispatch({ type: 'groupPageLoaded', data, keepIndex: opts.keepIndex === true });
      controllerRef.current!.reset();
    } catch (error) {
      showToast('分页浏览需要本地服务器（不要加 --no-serve）。' + (error as Error).message);
    }
  }, [showToast]);

  const loadBrowse = useCallback(async (opts: {
    view?: BrowseView;
    page?: number;
    size?: number;
  } = {}) => {
    const generation = ++loadGenRef.current;
    const s = stateRef.current;
    const targetView = opts.view ?? (s.view as BrowseView);
    const targetPage = opts.page ?? s.page;
    const targetSize = opts.size ?? s.size;
    try {
      const data = await api.fetchBrowsePage(targetView, targetPage, targetSize);
      if (generation !== loadGenRef.current) return;
      dispatch({ type: 'browsePageLoaded', items: data.items || [], page: data.page, pages: data.pages, total: data.total });
      const status = await api.fetchStatus();
      if (generation !== loadGenRef.current) return;
      dispatch({
        type: 'statusLoaded',
        queueCounts: status.queues,
        undoDepth: Number(status.undo_depth || 0),
        stateWarning: status.state_warning ?? null,
      });
    } catch (error) {
      showToast('分页浏览需要本地服务器。' + (error as Error).message);
    }
  }, [showToast]);

  const goQueue = useCallback(
    async (name: Queue) => {
      if (QUEUES.indexOf(name) < 0) return;
      dispatch({ type: 'setQueue', queue: name });
      await loadQueue({ queue: name, page: 1 });
    },
    [loadQueue],
  );

  const goView = useCallback(
    async (name: View) => {
      dispatch({ type: 'setView', view: name });
      if (name === 'GROUPS') await loadQueue({ page: 1 });
      else await loadBrowse({ view: name, page: 1 });
    },
    [loadQueue, loadBrowse],
  );

  const focusGroup = useCallback(
    async (groupId: number, fileId: number | null, target?: {
      queue?: Queue;
      size?: number;
    }) => {
      const s = stateRef.current;
      const targetQueue = target?.queue ?? s.queue;
      const targetSize = target?.size ?? s.size;
      try {
        const found = await api.locate(targetQueue, groupId, targetSize);
        const targetPage = found.page || 1;
        dispatch({ type: 'setPage', page: targetPage });
        await loadQueue({ queue: targetQueue, page: targetPage, size: targetSize });
        const after = stateRef.current;
        const wanted = found.group_id != null ? Number(found.group_id) : null;
        const at = after.groups.findIndex((g) => Number(g.group_id) === wanted);
        dispatch({ type: 'setGroupIndex', gIndex: at >= 0 ? at : 0 });
        if (fileId != null) {
          const items = after.groups[at >= 0 ? at : 0]?.members ?? [];
          const hit = items.findIndex((m) => Number(m.file_id) === Number(fileId));
          if (hit >= 0) dispatch({ type: 'showIndex', index: hit });
        }
      } catch {
        await loadQueue({ queue: targetQueue, page: 1, size: targetSize });
      }
    },
    [loadQueue],
  );

  const advanceAfter = useCallback(
    async (decidedGroupId: number) => {
      const s = stateRef.current;
      try {
        const next = await api.nextInQueue(s.queue, decidedGroupId, s.size);
        dispatch({ type: 'setQueueCounts', queueCounts: next.queue_counts || s.queueCounts });
        const targetPage = next.page || 1;
        dispatch({ type: 'setPage', page: targetPage });
        await loadQueue({ page: targetPage });
        const after = stateRef.current;
        const wanted = next.group_id != null ? Number(next.group_id) : null;
        const at = after.groups.findIndex((g) => Number(g.group_id) === wanted);
        dispatch({ type: 'setGroupIndex', gIndex: at >= 0 ? at : 0 });
      } catch {
        await loadQueue({ keepIndex: true });
      }
    },
    [loadQueue],
  );

  const decide = useCallback(
    async (action: 'accept' | 'pick' | 'mark') => {
      const s = stateRef.current;
      if (mode(s) !== 'queue' || s.mutationBusy) return;
      const group = currentGroup(s);
      if (!group) {
        showToast('本队列已经没有待处理的分组');
        return;
      }
      const gid = Number(group.group_id);
      const item = currentItem(s);
      if (action === 'accept' && aiKeeperId(stageItems(s)) == null) {
        showToast('本组没有 AI 推荐，请用 P 选一张');
        return;
      }
      const payload: api.ActionPayload = { group_id: gid, action };
      if (item) payload.context_file_id = Number(item.file_id);
      if (action === 'pick') {
        if (!item) {
          showToast('请先选择一张照片');
          return;
        }
        payload.file_id = Number(item.file_id);
      }
      dispatch({ type: 'setBusy', on: true });
      try {
        const data = await api.postAction(payload);
        if (data.queues) dispatch({ type: 'setQueueCounts', queueCounts: data.queues, undoDepth: Number(data.undo_depth || 0) });
        await advanceAfter(gid);
        if (mode(stateRef.current) === 'queue' && stateRef.current.groups.length === 0) {
          showToast('本轮审阅完成');
          return;
        }
        showToast(
          action === 'accept'
            ? '已保留 AI 推荐，进入下一组'
            : action === 'pick'
              ? '已保留当前照片，进入下一组'
              : '已标记稍后处理',
        );
      } catch (error) {
        showToast('操作失败：' + (error as Error).message);
      } finally {
        dispatch({ type: 'setBusy', on: false });
      }
    },
    [advanceAfter, showToast],
  );

  const undoLast = useCallback(async () => {
    const s = stateRef.current;
    if (mode(s) !== 'queue' || s.mutationBusy) return;
    dispatch({ type: 'setBusy', on: true });
    try {
      const data = await api.postAction({ action: 'undo' });
      if (data.queues) dispatch({ type: 'setQueueCounts', queueCounts: data.queues, undoDepth: Number(data.undo_depth || 0) });
      const undone = data.undo;
      if (undone) {
        if (QUEUES.indexOf(undone.queue) >= 0) dispatch({ type: 'setQueue', queue: undone.queue });
        await focusGroup(undone.group_id, undone.focus_file_id, { queue: undone.queue });
        showToast(`已撤销上一步，回到分组 #${undone.group_id}`);
      }
    } catch (error) {
      const message = (error as Error).message;
      showToast(message === 'nothing to undo' ? '没有可撤销的操作' : '撤销失败：' + message);
    } finally {
      dispatch({ type: 'setBusy', on: false });
    }
  }, [focusGroup, showToast]);

  const stepGroup = useCallback(
    async (delta: number) => {
      const s = stateRef.current;
      if (mode(s) !== 'queue') return;
      const next = s.gIndex + delta;
      if (next >= 0 && next < s.groups.length) {
        dispatch({ type: 'stepGroupLocal', delta });
        controllerRef.current!.reset();
        return;
      }
      if (delta < 0 && s.page > 1) {
        const targetPage = s.page - 1;
        dispatch({ type: 'setPage', page: targetPage });
        await loadQueue({ page: targetPage });
        dispatch({ type: 'setGroupIndex', gIndex: Math.max(0, stateRef.current.groups.length - 1) });
        return;
      }
      if (delta > 0 && s.page < s.pages) {
        const targetPage = s.page + 1;
        dispatch({ type: 'setPage', page: targetPage });
        await loadQueue({ page: targetPage });
        dispatch({ type: 'setGroupIndex', gIndex: 0 });
        return;
      }
      showToast(delta > 0 ? '已经是本队列最后一组' : '已经是本队列第一组');
    },
    [loadQueue, showToast],
  );

  const openBrowsePhoto = useCallback(async (fileId: number) => {
    const s = stateRef.current;
    const chosen = s.browseItems.find((r) => Number(r.file_id) === Number(fileId));
    if (!chosen) return;
    const gid = chosen.group_id != null ? Number(chosen.group_id) : null;
    let members = [] as typeof s.browseItems;
    if (gid != null) {
      try {
        const group = await api.fetchGroup(gid);
        members = group.members || [];
      } catch {
        members = [];
      }
    }
    if (!members.length) members = [chosen];
    const mIndex = Math.max(0, members.findIndex((r) => Number(r.file_id) === Number(fileId)));
    dispatch({
      type: 'openBrowse',
      group: { group_id: gid ?? -1, group_type: '', member_count: members.length, members },
      mIndex,
    });
  }, []);

  const closeBrowse = useCallback(() => {
    dispatch({ type: 'closeBrowse' });
  }, []);

  const toggleCompare = useCallback(() => {
    const s = stateRef.current;
    const items = stageItems(s);
    const item = currentItem(s);
    const aiId = aiKeeperId(items);
    if (!s.compareMode) {
      if (aiId == null) {
        showToast('本组没有 AI 推荐，无法对比');
        return;
      }
      if (item && Number(item.file_id) === aiId) {
        showToast('当前就是 AI 推荐');
        return;
      }
    }
    dispatch({ type: 'toggleCompare' });
    controllerRef.current!.reset();
  }, [showToast]);

  const setBlink = useCallback(
    (on: boolean) => {
      const s = stateRef.current;
      if (on && aiKeeperId(stageItems(s)) == null) {
        if (!s.blinkOn) showToast('本组没有 AI 推荐');
        return;
      }
      dispatch({ type: 'setBlink', on });
    },
    [showToast],
  );

  // --- keyboard (registered once; reads latest state via refs) -------------
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const s = stateRef.current;
      const ctx: KeyContext = {
        mode: mode(s),
        helpOpen: s.helpOpen,
        browseOpen: s.browseOpen,
        phase: 'down',
        repeat: event.repeat,
        shift: event.shiftKey,
      };
      const intent = keyIntent(event.key, ctx);
      if (intent.kind === 'none') return;
      event.preventDefault();
      switch (intent.kind) {
        case 'help':
          dispatch({ type: 'setHelp', open: intent.open });
          break;
        case 'escape':
          closeBrowse();
          break;
        case 'evidence':
          dispatch({ type: 'toggleEvidence' });
          break;
        case 'density':
          dispatch({ type: 'toggleDensity' });
          break;
        case 'fit':
          controllerRef.current!.reset();
          break;
        case 'hundred':
          controllerRef.current!.hundred();
          break;
        case 'compare':
          toggleCompare();
          break;
        case 'blink':
          setBlink(intent.on);
          break;
        case 'photo':
          dispatch({ type: 'stepPhoto', delta: intent.delta });
          controllerRef.current!.reset();
          break;
        case 'group':
          void stepGroup(intent.delta);
          break;
        case 'decide':
          if (!s.mutationBusy) void decide(intent.action);
          break;
        case 'undo':
          if (!s.mutationBusy) void undoLast();
          break;
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      const intent = keyIntent(event.key, {
        mode: mode(stateRef.current),
        helpOpen: stateRef.current.helpOpen,
        browseOpen: stateRef.current.browseOpen,
        phase: 'up',
        repeat: false,
        shift: event.shiftKey,
      });
      if (intent.kind === 'blink') setBlink(false);
    };
    const onBlur = () => setBlink(false);
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, [closeBrowse, decide, setBlink, stepGroup, toggleCompare, undoLast]);

  // --- boot ----------------------------------------------------------------
  useEffect(() => {
    const restored = readLocal();
    if (restored) {
      dispatch({ type: 'restorePrefs', prefs: restored.prefs });
      restoreRef.current = { groupId: restored.restoreGroupId, fileId: restored.restoreFileId };
    }
    (async () => {
      const targetView = restored?.prefs.view ?? stateRef.current.view;
      const targetQueue = restored?.prefs.queue ?? stateRef.current.queue;
      const targetSize = restored?.prefs.size ?? stateRef.current.size;
      if (targetView === 'GROUPS' && restoreRef.current?.groupId != null) {
        await focusGroup(restoreRef.current.groupId, restoreRef.current.fileId, {
          queue: targetQueue,
          size: targetSize,
        });
      } else if (targetView === 'GROUPS') {
        await loadQueue({ queue: targetQueue, page: 1, size: targetSize });
      } else {
        await loadBrowse({ view: targetView as BrowseView, page: 1, size: targetSize });
      }
      bootedRef.current = true;
    })();
    // Test hook: expose the same public surface the harness drove on the old page.
    (globalThis as Record<string, unknown>).__review = {
      state: () => stateRef.current,
      states: () => controllerRef.current!.states(),
      zoomText: () => controllerRef.current!.zoomText(),
      kind: () => controllerRef.current!.kind(),
      showIndex: (i: number) => {
        dispatch({ type: 'showIndex', index: i });
        controllerRef.current!.reset();
      },
      goQueue,
      goView,
      loadBrowse: (view: BrowseView, page = 1, size = stateRef.current.size) => loadBrowse({ view, page, size }),
      focusGroup,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // === render ==============================================================
  const inQueue = mode(state) === 'queue';
  const showStage = inQueue ? state.total > 0 : state.browseOpen;
  const done = inQueue && state.total === 0;
  const items = stageItems(state);
  const gid = currentGroup(state) ? Number(currentGroup(state)!.group_id) : null;
  const aiId = aiKeeperId(items);
  const humanId = humanKeeperId(state, gid);
  const paneA = paneAItem(state);
  const paneB = paneBItem(state);

  const labelA = paneA ? itemLabel(paneA, aiId, humanId) : null;
  const labelB = paneB ? (
    <>
      <span class="badge ai">🤖AI推荐</span> {itemLabel(paneB, aiId, humanId)}
    </>
  ) : null;

  return (
    <>
      <h1>
        照片去重 - 审阅<span class="sub">连续审片工作台 · 原图只读</span>
      </h1>
      <StateWarning warning={state.stateWarning} />
      <QueueTabs
        state={state}
        onQueue={(q) => void goQueue(q)}
        onView={(v) => void goView(v)}
        onDensity={() => dispatch({ type: 'toggleDensity' })}
        onEvidence={() => dispatch({ type: 'toggleEvidence' })}
        onHelp={() => dispatch({ type: 'setHelp', open: true })}
      />
      <Stats state={state} />
      {showStage && (
        <div id="workbench" class={!inQueue && state.browseOpen ? 'overlay' : ''}>
          <Header
            state={state}
            onPrev={() => void stepGroup(-1)}
            onNext={() => void stepGroup(1)}
            onClose={closeBrowse}
          />
          <Evidence state={state} />
          <Stage
            osd={deps.osd}
            labelA={labelA}
            labelB={labelB}
            aiA={false}
            compare={showCompare(state)}
            blink={state.blinkOn}
            secondaryReady={paneBActive(state)}
            frameARef={(el) => controllerRef.current!.attach(el, null)}
            frameBRef={(el) => controllerRef.current!.attach(null, el)}
          />
          <Filmstrip
            state={state}
            onShowIndex={(i) => {
              dispatch({ type: 'showIndex', index: i });
              controllerRef.current!.reset();
            }}
          />
          <ZoomBar
            zoomText={zoomText}
            onCompare={toggleCompare}
            onFit={() => controllerRef.current!.reset()}
            onHundred={() => controllerRef.current!.hundred()}
          />
          {inQueue && (
            <ActionBar
              state={state}
              onAccept={() => void decide('accept')}
              onPick={() => void decide('pick')}
              onMark={() => void decide('mark')}
              onUndo={() => void undoLast()}
            />
          )}
        </div>
      )}
      {done && <DonePanel state={state} onGoLater={() => void goQueue('LATER')} onGoDone={() => void goQueue('DONE')} />}
      {!inQueue && <BrowseList state={state} onOpen={(id) => void openBrowsePhoto(id)} onPage={(p) => { dispatch({ type: 'setPage', page: p }); void loadBrowse({ page: p }); }} onSize={(sz) => { dispatch({ type: 'setSize', size: sz }); void loadBrowse({ page: 1, size: sz }); }} />}
      <Help open={state.helpOpen} onClose={() => dispatch({ type: 'setHelp', open: false })} />
      <Toast text={toast.text} show={toast.show} />
    </>
  );
}

export { LS_KEY };
export type { State, Action };
