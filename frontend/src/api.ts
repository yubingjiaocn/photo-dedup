/**
 * The only way this app talks to the server: the read-only paged endpoints plus
 * the single `/api/action` mutation. Every list comes from the server with the
 * queue/view already applied, so the browser holds one page and never the whole
 * library. Nothing here reads `/api/original` — that is the viewer's job, and it
 * is driven entirely by what is on screen.
 */
import type {
  ActionResponse,
  BrowsePage,
  BrowseView,
  Group,
  GroupPage,
  LocateResponse,
  NextResponse,
  Queue,
  StatusResponse,
} from './types';

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error('HTTP ' + response.status);
  const value = await response.json();
  if (value && typeof value === 'object' && 'error' in value && value.error) {
    throw new Error(String(value.error));
  }
  return value as T;
}

export function fetchGroupPage(queue: Queue, page: number, pageSize: number): Promise<GroupPage> {
  return getJson<GroupPage>(
    `/api/page?view=GROUPS&queue=${queue}&page=${page}&page_size=${pageSize}`,
  );
}

export function fetchBrowsePage(view: BrowseView, page: number, pageSize: number): Promise<BrowsePage> {
  return getJson<BrowsePage>(`/api/page?view=${view}&page=${page}&page_size=${pageSize}`);
}

export function fetchStatus(): Promise<StatusResponse> {
  return getJson<StatusResponse>('/api/status');
}

export function fetchGroup(groupId: number): Promise<Group> {
  return getJson<Group>(`/api/group/${Number(groupId)}`);
}

export function locate(queue: Queue, groupId: number, pageSize: number): Promise<LocateResponse> {
  return getJson<LocateResponse>(
    `/api/locate?queue=${queue}&group_id=${Number(groupId)}&page_size=${pageSize}`,
  );
}

export function nextInQueue(queue: Queue, after: number, pageSize: number): Promise<NextResponse> {
  return getJson<NextResponse>(
    `/api/next?queue=${queue}&after=${Number(after)}&page_size=${pageSize}`,
  );
}

export interface ActionPayload {
  action: string;
  group_id?: number;
  file_id?: number;
  context_file_id?: number;
}

export async function postAction(payload: ActionPayload): Promise<ActionResponse> {
  const response = await fetch('/api/action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = (await response.json()) as ActionResponse;
  if (!response.ok || data.error) {
    throw new Error(data.error || 'HTTP ' + response.status);
  }
  return data;
}
