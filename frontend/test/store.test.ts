import { describe, expect, it } from 'vitest';
import {
  aiItem,
  aiKeeperId,
  currentGroup,
  currentItem,
  humanKeeperId,
  initialState,
  mode,
  paneAItem,
  paneBActive,
  paneBItem,
  positionInQueue,
  reducer,
  retainedOriginals,
  showCompare,
  stageItems,
  type State,
} from '../src/store';
import type { Group, GroupPage, Item } from '../src/types';

function item(fileId: number, over: Partial<Item> = {}): Item {
  return {
    file_id: fileId,
    group_id: 1,
    decision: 'MAYBE',
    basename: `IMG_${fileId}.jpg`,
    width: 2400,
    height: 1800,
    quality_score: 40,
    face_count: 1,
    reason: 'LOW_MARGIN',
    thumb: 'ok',
    is_keep: false,
    ...over,
  };
}

// group with keeper at members[1], matching the album fixture
function group(gid: number, base: number): Group {
  return {
    group_id: gid,
    group_type: 'phash_near',
    member_count: 3,
    members: [
      item(base, { group_id: gid }),
      item(base + 1, { group_id: gid, is_keep: true, decision: 'KEEP', reason: 'GROUP_KEEPER' }),
      item(base + 2, { group_id: gid }),
    ],
  };
}

function pageOf(groups: Group[], over: Partial<GroupPage> = {}): GroupPage {
  return {
    view: 'GROUPS',
    queue: 'PENDING',
    page: 1,
    pages: 1,
    page_size: 100,
    total: groups.length,
    items: groups,
    review_state: {},
    queue_counts: { PENDING: groups.length, LATER: 0, DONE: 0 },
    undo_depth: 0,
    state_warning: null,
    ...over,
  };
}

function loaded(): State {
  return reducer(initialState, {
    type: 'groupPageLoaded',
    data: pageOf([group(1, 1), group(2, 4)]),
    keepIndex: false,
  });
}

describe('groupPageLoaded', () => {
  it('lands on the first group and its first photo', () => {
    const s = loaded();
    expect(s.groups.length).toBe(2);
    expect(s.gIndex).toBe(0);
    expect(s.mIndex).toBe(0);
    expect(currentGroup(s)?.group_id).toBe(1);
    expect(currentItem(s)?.file_id).toBe(1);
    expect(mode(s)).toBe('queue');
  });

  it('keepIndex preserves the group index but resets compare/blink', () => {
    let s = loaded();
    s = reducer(s, { type: 'setGroupIndex', gIndex: 1 });
    s = { ...s, compareMode: true, blinkOn: true };
    s = reducer(s, { type: 'groupPageLoaded', data: pageOf([group(1, 1), group(2, 4)]), keepIndex: true });
    expect(s.gIndex).toBe(1);
    expect(s.compareMode).toBe(false);
    expect(s.blinkOn).toBe(false);
  });

  it('carries state_warning through', () => {
    const s = reducer(initialState, {
      type: 'groupPageLoaded',
      data: pageOf([group(1, 1)], { state_warning: 'unreadable (saved as x)' }),
      keepIndex: false,
    });
    expect(s.stateWarning).toContain('unreadable');
  });
});

describe('selectors', () => {
  it('aiKeeperId / aiItem find the keeper (members[1])', () => {
    const s = loaded();
    expect(aiKeeperId(stageItems(s))).toBe(2);
    expect(aiItem(s)?.file_id).toBe(2);
  });

  it('humanKeeperId reads a pick decision from review state', () => {
    let s = loaded();
    s = { ...s, reviewState: { '1': { action: 'pick', file_id: 3 } } };
    expect(humanKeeperId(s, 1)).toBe(3);
    expect(humanKeeperId(s, 2)).toBeNull();
  });

  it('positionInQueue accounts for page offset', () => {
    let s = loaded();
    s = { ...s, page: 2, size: 100, gIndex: 4, total: 250 };
    expect(positionInQueue(s)).toBe(105);
  });
});

describe('photo / group navigation', () => {
  it('showIndex wraps and cancels compare', () => {
    let s = loaded();
    s = { ...s, compareMode: true };
    s = reducer(s, { type: 'showIndex', index: 3 });
    expect(s.mIndex).toBe(0); // 3 % 3
    expect(s.compareMode).toBe(false);
    s = reducer(s, { type: 'showIndex', index: -1 });
    expect(s.mIndex).toBe(2);
  });

  it('stepGroupLocal stays within the loaded page', () => {
    let s = loaded();
    s = reducer(s, { type: 'stepGroupLocal', delta: 1 });
    expect(s.gIndex).toBe(1);
    s = reducer(s, { type: 'stepGroupLocal', delta: 1 }); // past end -> no change
    expect(s.gIndex).toBe(1);
    s = reducer(s, { type: 'stepGroupLocal', delta: -1 });
    expect(s.gIndex).toBe(0);
  });
});

describe('compare / blink pane logic', () => {
  it('no compare when focus IS the ai keeper', () => {
    let s = loaded();
    s = reducer(s, { type: 'showIndex', index: 1 }); // keeper
    s = { ...s, compareMode: true };
    expect(showCompare(s)).toBe(false);
    expect(paneBActive(s)).toBe(false);
  });

  it('compare shows pane B (the ai keeper) when focus differs', () => {
    let s = loaded(); // focus = member 0
    s = { ...s, compareMode: true };
    expect(showCompare(s)).toBe(true);
    expect(paneBActive(s)).toBe(true);
    expect(paneBItem(s)?.file_id).toBe(2);
    // pane A never swaps away from focus (would re-fetch a no-store original)
    expect(paneAItem(s)?.file_id).toBe(1);
  });

  it('blink activates pane B without swapping pane A', () => {
    let s = loaded();
    s = reducer(s, { type: 'setBlink', on: true });
    expect(paneAItem(s)?.file_id).toBe(1);
    expect(paneBActive(s)).toBe(true);
    expect(paneBItem(s)?.file_id).toBe(2);
  });
});

describe('HDD retention policy', () => {
  it('retains exactly the current group members, releasing others on group change', () => {
    let s = loaded();
    expect(retainedOriginals(s).sort()).toEqual([1, 2, 3]);
    s = reducer(s, { type: 'stepGroupLocal', delta: 1 }); // group 2
    expect(retainedOriginals(s).sort()).toEqual([4, 5, 6]);
  });

  it('retention does not grow when toggling compare or blink', () => {
    let s = loaded();
    const base = retainedOriginals(s).sort();
    s = { ...s, compareMode: true };
    expect(retainedOriginals(s).sort()).toEqual(base);
    s = { ...s, compareMode: false, blinkOn: true };
    expect(retainedOriginals(s).sort()).toEqual(base);
  });
});

describe('preferences restore', () => {
  it('accepts only known views, queues and sizes', () => {
    const s = reducer(initialState, {
      type: 'restorePrefs',
      prefs: { view: 'ALL', queue: 'LATER', size: 200, density: 'dense', evidenceOpen: true },
    });
    expect(s.view).toBe('ALL');
    expect(s.queue).toBe('LATER');
    expect(s.size).toBe(200);
    expect(s.density).toBe('dense');
    expect(s.evidenceOpen).toBe(true);
  });

  it('rejects a bogus size and view', () => {
    const s = reducer(initialState, {
      type: 'restorePrefs',
      prefs: { view: 'HACK' as unknown as State['view'], size: 999, evidenceOpen: false },
    });
    expect(s.view).toBe('GROUPS');
    expect(s.size).toBe(100);
  });
});

describe('browse mode', () => {
  it('setView to a browse view switches mode and resets page', () => {
    let s = loaded();
    s = reducer(s, { type: 'setView', view: 'ALL' });
    expect(mode(s)).toBe('browse');
    expect(s.page).toBe(1);
  });

  it('openBrowse / closeBrowse manage the overlay group', () => {
    let s = reducer(initialState, { type: 'setView', view: 'ALL' });
    const g = group(9, 20);
    s = reducer(s, { type: 'openBrowse', group: g, mIndex: 1 });
    expect(s.browseOpen).toBe(true);
    expect(currentItem(s)?.file_id).toBe(21);
    s = reducer(s, { type: 'closeBrowse' });
    expect(s.browseOpen).toBe(false);
    expect(s.browseGroup).toBeNull();
  });
});
