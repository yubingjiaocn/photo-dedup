import { describe, expect, it } from 'vitest';
import { render } from 'preact';
import {
  ActionBar,
  BrowseList,
  Evidence,
  Filmstrip,
  QueueTabs,
  StateWarning,
  Stats,
} from '../src/components/chrome';
import { initialState, reducer, type State } from '../src/store';
import type { Group, GroupPage, Item } from '../src/types';

function mount(node: preact.ComponentChild): HTMLElement {
  const host = document.createElement('div');
  document.body.appendChild(host);
  render(node as never, host);
  return host;
}

function item(fileId: number, over: Partial<Item> = {}): Item {
  return {
    file_id: fileId,
    group_id: 1,
    decision: 'MAYBE',
    basename: `IMG_${fileId}.jpg`,
    width: 2400,
    height: 1800,
    quality_score: 42.5,
    face_count: 2,
    reason: 'LOW_MARGIN',
    thumb: 'ok',
    is_keep: false,
    ...over,
  };
}

const grp: Group = {
  group_id: 1,
  group_type: 'phash_near',
  member_count: 3,
  members: [
    item(1),
    item(2, { is_keep: true, decision: 'KEEP', reason: 'GROUP_KEEPER' }),
    item(3, { thumb: 'missing' }),
  ],
};

const page: GroupPage = {
  view: 'GROUPS',
  queue: 'PENDING',
  page: 1,
  pages: 1,
  page_size: 100,
  total: 1,
  items: [grp],
  review_state: {},
  queue_counts: { PENDING: 4, LATER: 1, DONE: 2 },
  undo_depth: 3,
  state_warning: null,
};

const loaded: State = reducer(initialState, { type: 'groupPageLoaded', data: page, keepIndex: false });
const noop = () => {};

describe('QueueTabs', () => {
  it('shows each queue count and marks the active one', () => {
    const host = mount(
      <QueueTabs state={loaded} onQueue={noop} onView={noop} onDensity={noop} onEvidence={noop} onHelp={noop} />,
    );
    const text = host.textContent!.replace(/\s+/g, '');
    expect(text).toContain('未审4');
    expect(text).toContain('稍后1');
    expect(text).toContain('已完成2');
    expect(host.querySelector('[data-queue=PENDING]')!.classList.contains('on')).toBe(true);
  });
});

describe('ActionBar', () => {
  it('enables A when the group has an AI keeper', () => {
    const host = mount(<ActionBar state={loaded} onAccept={noop} onPick={noop} onMark={noop} onUndo={noop} />);
    expect((host.querySelector('#bAccept') as HTMLButtonElement).disabled).toBe(false);
    expect((host.querySelector('#bUndo') as HTMLButtonElement).disabled).toBe(false); // undo_depth 3
  });

  it('disables A when there is no AI keeper, and disables all while busy', () => {
    const noKeeper: State = { ...loaded, groups: [{ ...grp, members: grp.members.map((m) => ({ ...m, is_keep: false })) }] };
    let host = mount(<ActionBar state={noKeeper} onAccept={noop} onPick={noop} onMark={noop} onUndo={noop} />);
    expect((host.querySelector('#bAccept') as HTMLButtonElement).disabled).toBe(true);

    const busy: State = { ...loaded, mutationBusy: true };
    host = mount(<ActionBar state={busy} onAccept={noop} onPick={noop} onMark={noop} onUndo={noop} />);
    for (const id of ['#bAccept', '#bPick', '#bMark', '#bUndo']) {
      expect((host.querySelector(id) as HTMLButtonElement).disabled).toBe(true);
    }
  });

  it('uses the agreed Chinese wording', () => {
    const host = mount(<ActionBar state={loaded} onAccept={noop} onPick={noop} onMark={noop} onUndo={noop} />);
    expect(host.querySelector('#bAccept')!.textContent).toContain('保留 AI 推荐');
    expect(host.querySelector('#bPick')!.textContent).toContain('保留当前照片');
    expect(host.querySelector('#bMark')!.textContent).toContain('稍后处理');
    expect(host.querySelector('#bUndo')!.textContent).toContain('撤销上一步');
  });
});

describe('Filmstrip', () => {
  it('renders one frame per member, marks current and AI, and uses /api/thumb only', () => {
    const host = mount(<Filmstrip state={loaded} onShowIndex={noop} />);
    const frames = host.querySelectorAll('.fs-item');
    expect(frames.length).toBe(3);
    expect(host.querySelectorAll('.fs-item.current').length).toBe(1);
    const imgs = [...host.querySelectorAll('img')] as HTMLImageElement[];
    for (const img of imgs) expect(img.getAttribute('src')).toMatch(/^\/api\/thumb\/\d+\.jpg$/);
    // the missing-thumb member shows a placeholder, not an <img>
    expect(host.querySelector('.fs-miss')).not.toBeNull();
    expect(host.textContent).toContain('🤖AI');
  });
});

describe('Evidence', () => {
  it('is null when closed and shows diagnostics + enum when open', () => {
    expect(mount(<Evidence state={loaded} />).textContent).toBe('');
    const open: State = { ...loaded, evidenceOpen: true };
    const host = mount(<Evidence state={open} />);
    expect(host.textContent).toContain('清晰度评分');
    expect(host.textContent).toContain('GROUP_KEEPER'); // internal enum kept in small type
    expect(host.textContent).toContain('42.5');
  });
});

describe('StateWarning', () => {
  it('is hidden with no warning and names the kept file when present', () => {
    expect((mount(<StateWarning warning={null} />).querySelector('#stateWarn') as HTMLElement).hidden).toBe(true);
    const host = mount(<StateWarning warning={'unreadable (saved as review_state.corrupt-1)'} />);
    expect(host.textContent).toContain('审阅状态文件无法读取');
    expect(host.textContent).toContain('review_state.corrupt-1');
  });

  it('carries no algorithm or runtime diagnostics', () => {
    const host = mount(<StateWarning warning={'unreadable (saved as x)'} />);
    for (const token of ['quality', 'face', '缩略图缓存', 'GiB', 'ms']) {
      expect(host.textContent).not.toContain(token);
    }
  });
});

describe('BrowseList', () => {
  it('renders one clickable card per item with no duplicate open button', () => {
    const browseState: State = { ...loaded, view: 'ALL', browseItems: [item(1), item(2), item(3)] };
    const host = mount(<BrowseList state={browseState} onOpen={noop} onPage={noop} onSize={noop} />);
    expect(host.querySelectorAll('.card').length).toBe(3);
    expect(host.querySelectorAll('.card button').length).toBe(0);
    expect(host.textContent).not.toContain('查看高清大图');
  });
});

describe('Stats', () => {
  it('reports queues and position, no diagnostics', () => {
    const host = mount(<Stats state={loaded} />);
    expect(host.textContent).toContain('未审 4');
    expect(host.textContent).toContain('已完成 2');
    expect(host.textContent).not.toContain('43.0');
  });
});
