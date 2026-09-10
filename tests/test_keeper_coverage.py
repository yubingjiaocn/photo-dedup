import json

import pytest

from src import phase_selection as ps
from src.config import load_config
from tests.test_phase_selection import member
from tests.test_pose_evidence import body, record


def test_quality_banded_budget_changes_secondary_not_primary_or_count():
    members = [member(1, 0, [1, 0], q=95), member(2, 1, [1, .01], q=94),
               member(3, 2, [.98, .2], q=86)]
    phases = [ps.Phase(0, (0, 1, 2), (), True, ('TIME_MISSING',))]
    baseline = ps.select_phase_keepers(members, phases, diversity_similarity=.9999)
    candidate = ps.select_phase_keepers(members, phases, diversity_similarity=.9999,
                                       diversity_policy='quality_banded', diversity_quality_slack=.08)
    assert baseline['keepers'] == [0, 1]
    assert candidate['keepers'] == [0, 2]
    assert candidate['group_keeper_budget'] == baseline['group_keeper_budget'] == 2
    assert candidate['review_required']
    assert 'QUALITY_BANDED_BUDGET_ALLOCATION' in candidate['phases'][0]['reason_codes']


def test_quality_band_cannot_trade_unbounded_utility_for_diversity():
    members = [member(1, 0, [1, 0], q=95), member(2, 1, [1, .01], q=94),
               member(3, 2, [.98, .2], q=20)]
    phases = [ps.Phase(0, (0, 1, 2), (), True, ('TIME_MISSING',))]
    candidate = ps.select_phase_keepers(members, phases, diversity_similarity=.9999,
                                       diversity_policy='quality_banded', diversity_quality_slack=.08)
    assert candidate['keepers'] == [0, 1]


def test_pose_protection_is_opt_in_bounded_and_review_only():
    members = [member(1, 0, [1, 0], q=90), member(2, 1, [1, .001], q=85)]
    for m, pose in zip(members, [record(), record(body(move=.18))]):
        meta = json.loads(m['quality_meta'])
        meta['pose_evidence'] = pose
        m['quality_meta'] = json.dumps(meta)
    phases = ps.segment_phases(members)
    assert ps.select_phase_keepers(members, phases)['keepers'] == [0]
    candidate = ps.select_phase_keepers(members, phases, pose_policy='consensus')
    assert candidate['keepers'] == [0, 1]
    assert candidate['group_keeper_budget'] == 2
    assert candidate['phases'][0]['keepers'] == [0, 1]
    assert candidate['mandatory_review']
    assert candidate['authority'] == 'shadow_review_only'
    assert not candidate['physical_group_split']
    assert not candidate['pose_coverage']['semantic_phase_authority']


def test_missing_pose_data_preserves_baseline_keepers():
    members = [member(1, 0, [1, 0]), member(2, 1, [1, .001])]
    phases = ps.segment_phases(members)
    baseline = ps.select_phase_keepers(members, phases)
    candidate = ps.select_phase_keepers(members, phases, pose_policy='consensus')
    assert candidate['keepers'] == baseline['keepers']
    assert candidate['group_keeper_budget'] == baseline['group_keeper_budget']


@pytest.mark.parametrize('key,value', [
    ('keeper_diversity_policy', 'unknown'), ('keeper_diversity_policy', []),
    ('keeper_diversity_quality_slack', True), ('keeper_diversity_quality_slack', float('nan')),
    ('keeper_pose_policy', 'unknown'), ('keeper_pose_policy', []),
    ('keeper_pose_displacement_threshold', 0), ('keeper_pose_displacement_threshold', float('inf')),
])
def test_coverage_config_rejects_invalid_values(tmp_path, key, value):
    import yaml
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump({'cluster': {key: value}}))
    with pytest.raises(ValueError):
        load_config(path).validate()
