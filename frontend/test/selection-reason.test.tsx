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
  expect(host.querySelector('details')).not.toBeNull();
});


it('shows an independent cached VLM judgment with exact targets and keeps metrics collapsed', () => {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const page: GroupPage = {
    view: 'GROUPS', queue: 'PENDING', page: 1, pages: 1, page_size: 100, total: 1,
    review_state: {}, queue_counts: { PENDING: 1, LATER: 0, DONE: 0 }, undo_depth: 0,
    state_warning: null,
    items: [{
      group_id: 9, group_type: 'phash_near', member_count: 2,
      members: [
        { file_id: 10, group_id: 9, decision: 'KEEP', basename: 'a.jpg', width: 100, height: 100, quality_score: 80, face_count: 1, reason: 'GROUP_KEEPER', thumb: 'ok', is_keep: true },
        { file_id: 11, group_id: 9, decision: 'MAYBE', basename: 'b.jpg', width: 100, height: 100, quality_score: 79, face_count: 1, reason: 'LOW_MARGIN', thumb: 'ok', is_keep: false },
      ],
      review_judgment: {
        schema_version: 2, mode: 'shadow_review_only', source: 'vlm',
        model_id: 'global.anthropic.claude-sonnet-5-5', status: 'grounded',
        recommended_file_ids: [11], confidence: 'high', max_distinctions: 3,
        distinctions: [{
          code: 'EXPRESSION_DIFFERENCE', text: 'The eyes and mouth are more clearly visible.', confidence: 'high',
          targets: [{ file_id: 11, region: 'face' }], compare_to_file_ids: [10],
          cv_refs: ['F02:quality_meta.eye_detection.status'],
        }],
        attention_pointers: [{
          code: 'FRAMING_DIFFERENCE', text: 'A subtle edge crop may merit a glance.', confidence: 'low',
          targets: [{ file_id: 10, region: 'full_frame' }], compare_to_file_ids: [11],
          cv_refs: [], authority: 'optional_attention_pointer_only',
        }],
        max_attention_pointers: 3,
        human_confirmation: '仅供人工审阅；不会改变既有 keeper、排序或删除状态。',
        invariants: { changes_keeper: false, changes_sorting: false, changes_routing: false, changes_decision_tables: false, deletion_authority: 'none', model_calls: 1 },
      },
    }],
  };
  const loaded = reducer(initialState, { type: 'groupPageLoaded', data: page, keepIndex: false });
  render(<Evidence state={{ ...loaded, evidenceOpen: true }} />, host);
  expect(host.textContent).toContain('独立 VLM 审阅建议');
  expect(host.textContent).toContain('优先查看照片 11');
  expect(host.textContent).toContain('照片 11 · 脸部 ↔ 照片 10');
  expect(host.textContent).toContain('F02:quality_meta.eye_detection.status');
  expect(host.textContent).toContain('低置信度注意点（不支持推荐）');
  expect(host.textContent).toContain('A subtle edge crop may merit a glance.');
  expect(host.querySelectorAll('.review-judgment .review-focus-list > li')).toHaveLength(1);
  expect(host.querySelector('details')).toBeNull();
});
