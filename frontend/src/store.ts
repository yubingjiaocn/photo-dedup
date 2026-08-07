/**
 * The review workbench state, as a pure reducer plus selectors.
 *
 * Everything that decides *what the UI shows* lives here and is synchronous and
 * side-effect free, so it can be unit-tested without a browser. The async
 * orchestration (fetching pages, posting decisions, driving the zoom viewer,
 * localStorage) lives in `app.tsx` and only ever calls `dispatch` with the
 * results. That split is deliberate: the queue/undo/blink/compare logic that was
 * the risk in the hand-written page is now testable in isolation.
 */
import {
  BROWSE_VIEWS,
  PAGE_SIZES,
  QUEUES,
  type BrowseView,
  type Group,
  type GroupPage,
  type GroupState,
  type Item,
  type Queue,
  type QueueCounts,
  type View,
} from './types';

export type Density = 'comfy' | 'dense';
export type Mode = 'queue' | 'browse';

export interface State {
  view: View;
  queue: Queue;
  page: number;
  pages: number;
  size: number;
  total: number;
  queueCounts: QueueCounts;
  undoDepth: number;
  // queue mode
  groups: Group[];
  gIndex: number;
  mIndex: number;
  reviewState: Record<string, GroupState>;
  // browse mode
  browseItems: Item[];
  browseGroup: Group | null;
  browseOpen: boolean;
  // view toggles
  compareMode: boolean;
  blinkOn: boolean;
  evidenceOpen: boolean;
  density: Density;
  helpOpen: boolean;
  // guards / messages
  mutationBusy: boolean;
  stateWarning: string | null;
}

export const initialState: State = {
  view: 'GROUPS',
  queue: 'PENDING',
  page: 1,
  pages: 1,
  size: 100,
  total: 0,
  queueCounts: { PENDING: 0, LATER: 0, DONE: 0 },
  undoDepth: 0,
  groups: [],
  gIndex: 0,
  mIndex: 0,
  reviewState: {},
  browseItems: [],
  browseGroup: null,
  browseOpen: false,
  compareMode: false,
  blinkOn: false,
  evidenceOpen: false,
  density: 'comfy',
  helpOpen: false,
  mutationBusy: false,
  stateWarning: null,
};

// --- selectors (pure) -------------------------------------------------------

export const mode = (s: State): Mode => (s.view === 'GROUPS' ? 'queue' : 'browse');

export function currentGroup(s: State): Group | null {
  return mode(s) === 'queue' ? s.groups[s.gIndex] ?? null : s.browseGroup;
}

export function stageItems(s: State): Item[] {
  const group = currentGroup(s);
  return group && group.members ? group.members : [];
}

export function currentItem(s: State): Item | null {
  const items = stageItems(s);
  if (!items.length) return null;
  const index = Math.max(0, Math.min(s.mIndex, items.length - 1));
  return items[index] ?? null;
}

export function aiKeeperId(items: Item[]): number | null {
  const found = (items || []).find((m) => m.is_keep === true);
  return found ? Number(found.file_id) : null;
}

export function humanKeeperId(s: State, gid: number | null): number | null {
  const st = (gid == null ? null : s.reviewState[String(gid)]) || null;
  return st && st.action === 'pick' && st.file_id != null ? Number(st.file_id) : null;
}

export function groupStateOf(s: State, gid: number | null): GroupState | null {
  return (gid == null ? null : s.reviewState[String(gid)]) || null;
}

/** The AI keeper item currently on stage, if the group has one distinct from focus. */
export function aiItem(s: State): Item | null {
  const items = stageItems(s);
  const id = aiKeeperId(items);
  return id == null ? null : items.find((m) => Number(m.file_id) === id) ?? null;
}

/** Whether the split compare pane should be visible for the current selection. */
export function showCompare(s: State): boolean {
  const ai = aiItem(s);
  const item = currentItem(s);
  return s.compareMode && ai != null && item != null && Number(ai.file_id) !== Number(item.file_id);
}

/** Pane A always shows the focused photo. Blink does NOT swap pane A's source
 *  (that would re-fetch a `no-store` original); it overlays pane B instead. */
export function paneAItem(s: State): Item | null {
  return currentItem(s);
}

/** Whether the AI-keeper overlay/split (pane B) is needed for this selection. */
export function paneBActive(s: State): boolean {
  const ai = aiItem(s);
  const item = currentItem(s);
  if (ai == null || item == null || Number(ai.file_id) === Number(item.file_id)) return false;
  return s.compareMode || s.blinkOn;
}

/** The photo pane B holds when active (the AI keeper), else null. */
export function paneBItem(s: State): Item | null {
  return paneBActive(s) ? aiItem(s) : null;
}

/**
 * The URLs the image pool may keep decoded: every member of the current group.
 *
 * This is the whole HDD policy, and it drives *pruning* only — it never triggers
 * a fetch (a member's `<img>` is acquired solely when the viewer is asked to
 * display it). Listing all current-group members means any original displayed
 * while on this group stays pooled (so H/L/blink/compare re-display and the three
 * blink cycles add zero `no-store` requests), while moving to another group
 * replaces the set and releases the entire previous group's decoded originals.
 * Members never displayed are simply never fetched, so nothing is prefetched.
 */
export function retainedOriginals(s: State): number[] {
  return stageItems(s).map((m) => Number(m.file_id));
}

export function positionInQueue(s: State): number {
  return s.total ? (s.page - 1) * s.size + s.gIndex + 1 : 0;
}

// --- actions ----------------------------------------------------------------

export type Action =
  | { type: 'groupPageLoaded'; data: GroupPage; keepIndex: boolean }
  | { type: 'browsePageLoaded'; items: Item[]; page: number; pages: number; total: number }
  | { type: 'statusLoaded'; queueCounts: QueueCounts; undoDepth: number; stateWarning: string | null }
  | { type: 'setQueue'; queue: Queue }
  | { type: 'setView'; view: View }
  | { type: 'setPage'; page: number }
  | { type: 'setSize'; size: number }
  | { type: 'setGroupIndex'; gIndex: number }
  | { type: 'showIndex'; index: number }
  | { type: 'stepPhoto'; delta: number }
  | { type: 'stepGroupLocal'; delta: number }
  | { type: 'setCompare'; on: boolean }
  | { type: 'toggleCompare' }
  | { type: 'setBlink'; on: boolean }
  | { type: 'toggleEvidence' }
  | { type: 'toggleDensity' }
  | { type: 'setHelp'; open: boolean }
  | { type: 'openBrowse'; group: Group; mIndex: number }
  | { type: 'closeBrowse' }
  | { type: 'setBusy'; on: boolean }
  | { type: 'setStateWarning'; value: string | null }
  | { type: 'setQueueCounts'; queueCounts: QueueCounts; undoDepth?: number }
  | { type: 'restorePrefs'; prefs: Partial<Pick<State, 'view' | 'queue' | 'size' | 'density' | 'evidenceOpen'>> };

function clampIndex(index: number, length: number): number {
  if (length <= 0) return 0;
  return ((index % length) + length) % length;
}

export function reducer(s: State, action: Action): State {
  switch (action.type) {
    case 'groupPageLoaded': {
      const { data, keepIndex } = action;
      const groups = data.items || [];
      const gIndex = keepIndex ? Math.min(s.gIndex, Math.max(0, groups.length - 1)) : 0;
      return {
        ...s,
        view: 'GROUPS',
        queue: data.queue,
        page: data.page,
        pages: data.pages,
        total: data.total,
        size: data.page_size,
        groups,
        reviewState: data.review_state || {},
        queueCounts: data.queue_counts || s.queueCounts,
        undoDepth: Number(data.undo_depth || 0),
        gIndex,
        mIndex: keepIndex ? s.mIndex : 0,
        compareMode: false,
        blinkOn: false,
        stateWarning: data.state_warning !== undefined ? data.state_warning : s.stateWarning,
      };
    }
    case 'browsePageLoaded':
      return {
        ...s,
        browseItems: action.items,
        page: action.page,
        pages: action.pages,
        total: action.total,
      };
    case 'statusLoaded':
      return {
        ...s,
        queueCounts: action.queueCounts,
        undoDepth: action.undoDepth,
        stateWarning: action.stateWarning,
      };
    case 'setQueue':
      return { ...s, queue: action.queue, view: 'GROUPS', page: 1, gIndex: 0, mIndex: 0, browseOpen: false, compareMode: false, blinkOn: false };
    case 'setView':
      return {
        ...s,
        view: action.view,
        page: 1,
        browseOpen: false,
        gIndex: action.view === 'GROUPS' ? 0 : s.gIndex,
        mIndex: action.view === 'GROUPS' ? 0 : s.mIndex,
      };
    case 'setPage':
      return { ...s, page: action.page };
    case 'setSize':
      return { ...s, size: action.size, page: 1 };
    case 'setGroupIndex':
      return { ...s, gIndex: action.gIndex, mIndex: 0 };
    case 'showIndex': {
      const items = stageItems(s);
      if (!items.length) return s;
      return { ...s, mIndex: clampIndex(action.index, items.length), compareMode: false };
    }
    case 'stepPhoto': {
      const items = stageItems(s);
      if (!items.length) return s;
      return { ...s, mIndex: clampIndex(s.mIndex + action.delta, items.length), compareMode: false };
    }
    case 'stepGroupLocal': {
      const next = s.gIndex + action.delta;
      if (next < 0 || next >= s.groups.length) return s;
      return { ...s, gIndex: next, mIndex: 0, compareMode: false, blinkOn: false };
    }
    case 'setCompare':
      return { ...s, compareMode: action.on };
    case 'toggleCompare':
      return { ...s, compareMode: !s.compareMode };
    case 'setBlink':
      return s.blinkOn === action.on ? s : { ...s, blinkOn: action.on };
    case 'toggleEvidence':
      return { ...s, evidenceOpen: !s.evidenceOpen };
    case 'toggleDensity':
      return { ...s, density: s.density === 'dense' ? 'comfy' : 'dense' };
    case 'setHelp':
      return { ...s, helpOpen: action.open };
    case 'openBrowse':
      return {
        ...s,
        browseGroup: action.group,
        mIndex: Math.max(0, action.mIndex),
        browseOpen: true,
        compareMode: false,
        blinkOn: false,
      };
    case 'closeBrowse':
      return { ...s, browseOpen: false, browseGroup: null, compareMode: false, blinkOn: false };
    case 'setBusy':
      return { ...s, mutationBusy: action.on };
    case 'setStateWarning':
      return action.value === s.stateWarning ? s : { ...s, stateWarning: action.value };
    case 'setQueueCounts':
      return { ...s, queueCounts: action.queueCounts, undoDepth: action.undoDepth ?? s.undoDepth };
    case 'restorePrefs': {
      const p = action.prefs;
      const next = { ...s };
      if (p.view && (p.view === 'GROUPS' || BROWSE_VIEWS.indexOf(p.view as BrowseView) >= 0)) next.view = p.view;
      if (p.queue && QUEUES.indexOf(p.queue) >= 0) next.queue = p.queue;
      if (p.size && (PAGE_SIZES as readonly number[]).indexOf(Number(p.size)) >= 0) next.size = Number(p.size);
      if (p.density === 'dense') next.density = 'dense';
      next.evidenceOpen = p.evidenceOpen === true;
      return next;
    }
    default:
      return s;
  }
}
