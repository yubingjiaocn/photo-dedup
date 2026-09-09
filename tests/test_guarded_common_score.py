import pytest

from src import phase_selection as PS
from src.config import load_config
from tests.test_keeper_score_policy import member


def choose(rows, margin=.06):
    return PS.select_phase_keepers(rows, [PS.Phase(0, tuple(range(len(rows))), (), False)],
                                   score_policy='guarded_common', score_change_margin=margin)


def test_weak_primary_gain_preserves_old_ranking():
    rows = [member(1,80,.1),member(2,75)]
    result = choose(rows)
    assert result['keepers'] == [1]
    assert result['utility_scores'] == PS.utility_scores(rows)
    assert result['scoring_context']['fallback'] == 'INSUFFICIENT_PRIMARY_GAIN'
    assert result['scoring_context']['effective_policy'] == 'per_member'


def test_decisive_primary_gain_uses_shared_evidence():
    rows = [member(1,90,.1),member(2,75)]
    result = choose(rows)
    assert result['keepers'] == [0]
    assert result['utility_scores'] == PS.utility_scores(rows,score_policy='common_evidence')
    assert result['scoring_context']['primary_gain'] > .06
    assert result['scoring_context']['effective_policy'] == 'common_evidence'
    assert result['review_required'] is True
    assert 'KEEPER_SCORE_COMPARABILITY_CHANGE' in result['reason_codes']


def test_unchanged_primary_does_not_recalibrate_secondary_diversity():
    rows = [member(1,95,.5),member(2,75),member(3,60,.1)]
    result = choose(rows)
    assert result['utility_scores'] == PS.utility_scores(rows)
    assert result['scoring_context']['fallback'] == 'NO_PRIMARY_RANK_CHANGE'


def test_margin_is_a_generic_explicit_parameter_not_an_id_rule():
    rows = [member(9001,80,.1),member(3,75)]
    assert choose(rows,.02)['keepers'] == [0]
    assert choose(rows,.06)['keepers'] == [1]
    assert choose(rows,.99)['keepers'] == [1]


@pytest.mark.parametrize('margin',[-.1,1.1,float('nan'),float('inf'),None,True,[],'.06'])
def test_invalid_margin_fails_closed(margin):
    with pytest.raises(ValueError,match='score_change_margin'):
        choose([member(1)],margin)


def test_guarded_scores_exposed_by_public_helper_match_selection():
    rows = [member(1,90,.1),member(2,75)]
    assert choose(rows)['utility_scores'] == PS.utility_scores(rows,score_policy='guarded_common')


def test_guarded_policy_does_not_expand_budget_or_authority():
    result = choose([member(1,90,.1),member(2,75)])
    assert result['group_keeper_budget'] == 1
    assert result['authority'] == 'shadow_review_only'
    assert result['phase_coverage_diagnostics'][0]['auto_remove_authority'] == 'BYTE_IDENTICAL_ONLY'


def test_guarded_config_validates_gain_margin(tmp_path):
    path = tmp_path / 'guard.yaml'
    path.write_text('cluster:\n  keeper_score_policy: guarded_common\n  keeper_score_change_margin: 0.06\n')
    load_config(path).validate()
    for invalid in ['true', 'null', '[]', '.nan', '.inf', '-0.1', '1.1', 'text']:
        path.write_text(f'cluster:\n  keeper_score_policy: guarded_common\n  keeper_score_change_margin: {invalid}\n')
        with pytest.raises(ValueError, match='keeper_score_change_margin'):
            load_config(path).validate()
