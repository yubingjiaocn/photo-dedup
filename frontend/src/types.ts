/**
 * The shapes the read-only `/api/*` endpoints already return, and the display
 * vocabulary the page renders them with.
 *
 * The server projects every DB row through `review_server._public_item`, which
 * is path-free on purpose: the browser only ever sees `file_id`/`basename`, and
 * this file is written to that contract so nothing here can leak a source path.
 */

export type Queue = 'PENDING' | 'LATER' | 'DONE';
export type BrowseView = 'ALL' | 'MAYBE' | 'UNKNOWN';
export type View = 'GROUPS' | BrowseView;
export type ActionName = 'accept' | 'pick' | 'mark' | 'undo' | 'clear';

/** One photo, exactly as the server exposes it (no path, ever). */
export interface Item {
  file_id: number;
  group_id: number | null;
  decision: string;
  basename: string | null;
  width: number | null;
  height: number | null;
  size_bytes?: number | null;
  exif_datetime?: string | null;
  quality_score: number | null;
  face_count: number | null;
  reason: string | null;
  thumb: string;
  is_keep: boolean | null;
}

export interface Group {
  group_id: number;
  group_type: string;
  member_count: number;
  members: Item[];
}

/** The one recorded human decision per group (subset the UI reads). */
export interface GroupState {
  action?: string;
  file_id?: number;
}

export type QueueCounts = Record<Queue, number>;

/** `/api/page?view=GROUPS` envelope. */
export interface GroupPage {
  view: 'GROUPS';
  queue: Queue;
  page: number;
  pages: number;
  page_size: number;
  total: number;
  items: Group[];
  review_state: Record<string, GroupState>;
  queue_counts: QueueCounts;
  undo_depth: number;
  state_warning: string | null;
}

/** `/api/page?view=ALL|MAYBE|UNKNOWN` envelope. */
export interface BrowsePage {
  view: BrowseView;
  page: number;
  pages: number;
  page_size: number;
  total: number;
  items: Item[];
}

export interface StatusResponse {
  queues: QueueCounts;
  undo_depth: number;
  state_warning?: string | null;
  review_state?: { total: number; reviewed: number; marked: number };
}

export interface LocateResponse {
  queue: Queue;
  total: number;
  found: boolean;
  page: number;
  index: number | null;
  group_id: number | null;
  queue_counts: QueueCounts;
  page_size: number;
}

export interface NextResponse {
  queue: Queue;
  total: number;
  found: boolean;
  group_id: number | null;
  index: number | null;
  page: number;
  wrapped: boolean;
  queue_counts: QueueCounts;
  page_size: number;
}

export interface ActionResponse {
  ok?: boolean;
  error?: string;
  group_id?: number;
  summary?: { total: number; reviewed: number; marked: number };
  queues?: QueueCounts;
  undo_depth?: number;
  undo?: {
    group_id: number;
    queue: Queue;
    focus_file_id: number | null;
  };
}

// --- display vocabulary (Chinese UI, internal enum kept beside it) ----------

export const QUEUES: Queue[] = ['PENDING', 'LATER', 'DONE'];
export const BROWSE_VIEWS: BrowseView[] = ['ALL', 'MAYBE', 'UNKNOWN'];
export const PAGE_SIZES = [50, 100, 200] as const;

export const QUEUE_LABEL: Record<Queue, string> = {
  PENDING: '未审',
  LATER: '稍后',
  DONE: '已完成',
};

export const BROWSE_LABEL: Record<BrowseView, string> = {
  ALL: '全部时间线',
  MAYBE: '待确认',
  UNKNOWN: '未知',
};

export const GROUP_TYPE_LABEL: Record<string, string> = {
  sha_exact: '完全相同',
  phash_near: '高度相似',
  burst: '连拍',
  similar_scene: '相似场景',
};

export const DECISION_LABEL: Record<string, string> = {
  KEEP: '保留',
  AUTO_REMOVE: '建议删除',
  MAYBE: '待确认',
  UNKNOWN: '未知',
  UNGROUPED: '未分组',
};

export const REASON_LABEL: Record<string, string> = {
  GROUP_KEEPER: '组内最佳',
  BYTE_IDENTICAL: '字节完全相同',
  LOW_MARGIN: '与最佳差距很小',
  PHASH_NEAR_UNTRUSTED: '相似判断不够可靠',
  SIMILAR_SCENE_UNTRUSTED: '仅场景相似',
  GROUP_IMPURE: '同组内容不一致',
  SHA_GROUP_NO_MATCH: '哈希组内无匹配',
  PHASH_NEAR_DUP_ONLY: '仅像素级近重复',
  FACE_COUNT_MISMATCH: '人脸数量不一致',
  RELATIVE_ONLY: '仅相对比较',
  FEATURE_MISSING: '缺少分析特征',
  SUBJECTIVE_ONLY: '主观取舍',
  EXPOSURE_BORDERLINE: '曝光处于临界',
  dup: '近重复',
  keep: '保留',
  best: '组内最佳',
};

export const groupTypeText = (value: string | null | undefined): string => {
  const key = String(value ?? '');
  return GROUP_TYPE_LABEL[key] || key || '未知类型';
};

export const decisionText = (value: string | null | undefined): string => {
  const key = String(value ?? 'UNGROUPED').toUpperCase();
  return DECISION_LABEL[key] || key;
};

export const reasonText = (value: string | null | undefined): string => {
  const raw = String(value ?? '');
  if (!raw) return '';
  const head = raw.split(':')[0].trim();
  return REASON_LABEL[head] || REASON_LABEL[raw] || raw;
};

export const queueLabel = (name: Queue): string => QUEUE_LABEL[name] || name;

/** Keyboard letter → mutating action, matching the production key map. */
export const actionKey = (key: string): ActionName | null => {
  const k = String(key ?? '').toLowerCase();
  return k === 'a' ? 'accept' : k === 'p' ? 'pick' : k === 'm' ? 'mark' : k === 'u' ? 'undo' : null;
};

export const originalSrc = (fileId: number): string => `/api/original/${Number(fileId)}`;
export const thumbSrc = (fileId: number): string => `/api/thumb/${Number(fileId)}.jpg`;
