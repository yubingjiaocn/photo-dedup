/**
 * localStorage restore, to the exact v2 shape the hand-written page used, so an
 * in-flight reviewer keeps their queue, page size, density, evidence panel and
 * the group+photo they were on across a reload. Only display preferences and two
 * ids are stored — never a source path, never algorithm data.
 */
import type { Density, State } from './store';
import { currentGroup, currentItem } from './store';
import type { BrowseView, Queue, View } from './types';

export const LS_KEY = 'photo-dedup-review-v2';

export interface SavedPrefs {
  view: View;
  queue: Queue;
  size: number;
  density: Density;
  evidenceOpen: boolean;
  group_id: number | null;
  file_id: number | null;
}

export function saveLocal(state: State): void {
  const group = currentGroup(state);
  const item = currentItem(state);
  const payload: SavedPrefs = {
    view: state.view,
    queue: state.queue,
    size: state.size,
    density: state.density,
    evidenceOpen: state.evidenceOpen,
    group_id: group ? Number(group.group_id) : null,
    file_id: item ? Number(item.file_id) : null,
  };
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(payload));
  } catch {
    /* private mode / quota — persistence is best-effort */
  }
}

export interface RestoreResult {
  prefs: Partial<Pick<State, 'view' | 'queue' | 'size' | 'density' | 'evidenceOpen'>>;
  restoreGroupId: number | null;
  restoreFileId: number | null;
}

const BROWSE_VIEWS: BrowseView[] = ['ALL', 'MAYBE', 'UNKNOWN'];
const QUEUES: Queue[] = ['PENDING', 'LATER', 'DONE'];

export function readLocal(): RestoreResult | null {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(LS_KEY);
  } catch {
    return null;
  }
  if (!raw) return null;
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!value || typeof value !== 'object') return null;
  const saved = value as Partial<SavedPrefs>;
  const prefs: RestoreResult['prefs'] = {};
  if (saved.view === 'GROUPS' || BROWSE_VIEWS.indexOf(saved.view as BrowseView) >= 0) prefs.view = saved.view as View;
  if (QUEUES.indexOf(saved.queue as Queue) >= 0) prefs.queue = saved.queue as Queue;
  if ([50, 100, 200].indexOf(Number(saved.size)) >= 0) prefs.size = Number(saved.size);
  if (saved.density === 'dense') prefs.density = 'dense';
  prefs.evidenceOpen = saved.evidenceOpen === true;
  return {
    prefs,
    restoreGroupId: saved.group_id != null ? Number(saved.group_id) : null,
    restoreFileId: saved.file_id != null ? Number(saved.file_id) : null,
  };
}
