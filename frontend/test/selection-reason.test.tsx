import { expect, it } from 'vitest';
import { render } from 'preact';
import { Evidence } from '../src/components/chrome';
import { initialState, reducer, type State } from '../src/store';
import type { GroupPage } from '../src/types';

it('renders evidence-linked shadow reasons and the human confirmation prompt', () => {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const page: GroupPage = {
    view: 'GROUPS', queue: 'PENDING', page: 1, pages: 1, page_size: 100, total: 1,
    review_state: {}, queue_counts: { PENDING: 1, LATER: 0, DONE: 0 }, undo_depth: 0,
    state_warning: null,
    items: [{
      group_id: 7, group_type: 'phash_near', member_count: 2,
      members: [
        { file_id: 1, group_id: 7, decision: 'MAYBE', basename: 'rival.jpg', width: 3000, height: 2000, quality_score: 40, face_count: 1, reason: 'LOW_MARGIN', thumb: 'ok', is_keep: false },
        { file_id: 2, group_id: 7, decision: 'KEEP', basename: 'keeper.jpg', width: 2400, height: 1800, quality_score: 43, face_count: 1, reason: 'GROUP_KEEPER', thumb: 'ok', is_keep: true },
      ],
      selection_reason: {
        schema_version: 1, generator: 'deterministic-selection-reason-v1', mode: 'shadow_review_only', status: 'explained',
        recommendation: { code: 'RETAIN_PERSISTED_KEEPER', text: '建议保留 keeper.jpg。', evidence_codes: ['E001'] },
        support: [{ code: 'QUALITY_SCORE_ADVANTAGE', text: '缓存质量分更高。', evidence_codes: ['E002'] }],
        tradeoffs: [{ code: 'RIVAL_RESOLUTION_ADVANTAGE', text: '对手分辨率更高。', evidence_codes: ['E003'] }],
        confidence: { level: 'medium', score: 0.5, code: 'EXPLANATION_CONFIDENCE_MEDIUM', text: '结构化证据覆盖中等。', evidence_codes: ['E001', 'E002'] },
        human_confirmation: { required: true, code: 'HUMAN_CONFIRM_SUBJECTIVE_QUALITY', text: '请人工确认表情、遮挡、构图和叙事。' },
        evidence: [
          { id: 'E001', code: 'PERSISTED_KEEPER', source: 'group_members.is_keep', subject_file_id: 2, value: true },
          { id: 'E002', code: 'QUALITY_SCORE_ADVANTAGE', source: 'features.quality_score', subject_file_id: 2 },
          { id: 'E003', code: 'RIVAL_RESOLUTION_ADVANTAGE', source: 'files.width_height', subject_file_id: 1 },
        ],
        invariants: { keeper_file_ids: [2], changes_keeper: false, changes_sorting: false, changes_routing: false, deletion_authority: 'none', model_calls: 0 },
      },
    }],
  };
  const loaded = reducer(initialState, { type: 'groupPageLoaded', data: page, keepIndex: false });
  const open: State = { ...loaded, evidenceOpen: true };
  render(<Evidence state={open} />, host);
  expect(host.textContent).toContain('人类可读选片理由（shadow）');
  expect(host.textContent).toContain('RETAIN_PERSISTED_KEEPER · E001');
  expect(host.textContent).toContain('缓存质量分更高');
  expect(host.textContent).toContain('对手分辨率更高');
  expect(host.textContent).toContain('请人工确认表情、遮挡、构图和叙事');
});
