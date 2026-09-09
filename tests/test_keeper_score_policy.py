"""Candidate scoring compares members on common, observed evidence only."""
import copy
import json

import pytest

from src import phase_selection as PS
from src.config import load_config


def member(fid, quality=80, face=None, exposure=True):
    meta = {}
    if face is not None:
        meta['face_quality'] = face
    if exposure:
        meta['exposure'] = {'clip_hi': 0.01, 'clip_lo': 0.01}
    return {'id': fid, 'quality_score': quality, 'quality_meta': json.dumps(meta),
            'face_count': int(face is not None), 'size_bytes': 100, 'width': 100,
            'height': 100, 'exif_timestamp': fid, 'faces_json': '[]'}


def choose(members, policy='per_member', requirements=None):
    return PS.select_phase_keepers(
        members, [PS.Phase(0, tuple(range(len(members))), (), False)] if members else [],
        score_policy=policy, phase_requirements=requirements,
    )


def test_default_score_policy_preserves_legacy_output():
    rows = [member(1, 80, 0.1), member(2, 75)]
    phases = [PS.Phase(0, (0, 1), (), False)]
    assert PS.select_phase_keepers(rows, phases) == choose(rows)
    assert PS.utility_scores(rows) == PS.utility_scores(rows, score_policy='per_member')
    assert choose(rows)['keepers'] == [1]


def test_common_evidence_removes_missing_face_availability_bonus():
    rows = [member(1, 80, 0.1), member(2, 75)]
    result = choose(rows, 'common_evidence')
    assert result['keepers'] == [0]
    assert result['scoring_context']['shared_components'] == ['quality', 'exposure']
    assert result['scoring_context']['excluded_noncommon_components'] == ['face_clarity']
    assert result['scoring_context']['fallback'] is None
    assert result['utility_scores'][0] == pytest.approx((0.4 * .8 + .15 * .98) / .55)
    assert result['group_keeper_budget'] == choose(rows)['group_keeper_budget'] == 1


def test_homogeneous_availability_preserves_scores():
    rows = [member(1, 75, .4), member(2, 76, .6)]
    assert PS.utility_scores(rows, score_policy='common_evidence') == PS.utility_scores(rows)
    assert choose(rows, 'common_evidence')['keepers'] == choose(rows)['keepers']


def test_common_evidence_is_not_a_frontality_or_missing_face_penalty():
    rows = [member(1, 70, .9), member(2, 80)]
    assert choose(rows, 'common_evidence')['keepers'] == [1]
    context = choose(rows, 'common_evidence')['scoring_context']
    assert context['semantics'] == 'shared_observed_components_not_subject_or_face_truth'


def test_no_shared_components_falls_back_explicitly_without_imputed_values():
    rows = [member(1, 80, exposure=False), member(2, None, .5, exposure=False)]
    result = choose(rows, 'common_evidence')
    assert result['utility_scores'] == PS.utility_scores(rows)
    assert result['keepers'] == choose(rows)['keepers']
    assert result['scoring_context']['fallback'] == 'NO_SHARED_COMPONENTS_LEGACY_UNCHANGED'


def test_input_objects_unchanged_and_no_deletion_authority():
    rows = [member(1, 80, .1), member(2, 75)]
    original = copy.deepcopy(rows)
    result = choose(rows, 'common_evidence')
    assert rows == original
    assert result['authority'] == 'shadow_review_only'
    assert result['physical_group_split'] is False
    assert all(d['auto_remove_authority'] == 'BYTE_IDENTICAL_ONLY'
               for d in result['phase_coverage_diagnostics'])


def test_external_phase_labels_still_cannot_steer_selection():
    rows = [member(1, 80, .1), member(2, 75)]
    requirements = {'source': 'human', 'confidence': 'high', 'evidence_ref': 'synthetic',
                    'phases': [{'phase_id': 'a', 'member_ids': [1]},
                               {'phase_id': 'b', 'member_ids': [2]}]}
    assert choose(rows, 'common_evidence', requirements)['keepers'] == choose(rows, 'common_evidence')['keepers']
    assert choose(rows, 'common_evidence', requirements)['mandatory_review'] is True


@pytest.mark.parametrize('policy', ['unknown', '', None, [], 1])
def test_invalid_policy_fails_closed(policy):
    with pytest.raises(ValueError, match='score_policy'):
        PS.utility_scores([member(1)], score_policy=policy)
    with pytest.raises(ValueError, match='score_policy'):
        choose([member(1)], policy)


def test_empty_common_evidence_input_is_supported():
    assert PS.utility_scores([], score_policy='common_evidence') == {}
    assert choose([], 'common_evidence')['keepers'] == []


def test_config_policy_defaults_legacy_and_validates_opt_in(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('cluster:\n  keeper_score_policy: common_evidence\n')
    assert load_config(config).cluster.get('keeper_score_policy') == 'common_evidence'
    config.write_text('cluster:\n  keeper_score_policy: bad\n')
    with pytest.raises(ValueError, match='keeper_score_policy'):
        load_config(config).validate()
