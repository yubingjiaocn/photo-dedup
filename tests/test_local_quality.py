import json
import sqlite3

import numpy as np
import pytest

from src import db, local_quality as lq, phase_selection as ps
from src.config import load_config
from tests.test_phase_selection import member


def observed(fid, musiq, clip, *, kind='subject', cls=16, axis=0):
    m = member(fid, fid, [1, 0], q=80 if fid == 1 else 79)
    v = np.zeros(768, dtype='<f2')
    v[axis] = 1
    m['local_quality_embedding'] = v.tobytes()
    meta = json.loads(m['quality_meta'])
    meta['local_quality'] = {'schema_version': 1, 'method': 'native_coco_crop_v1',
        'regions': [{'kind': kind, 'class_id': cls, 'confidence': .99,
                     'box_normalized': [.1, .1, .9, .9], 'embedding_offset': 0,
                     'native_size': [500, 500], 'musiq': musiq, 'clipiqa': clip}]}
    m['quality_meta'] = json.dumps(meta)
    return m


def phase(*indices):
    return ps.Phase(0, tuple(indices), (), False, ())


def test_joint_native_quality_dominance_is_bounded():
    ms = [observed(1, 60, .4), observed(2, 70, .7)]
    scores, evidence = lq.adjust_scores(ms, [phase(0, 1)], {0: .8, 1: .79})
    assert scores == pytest.approx({0: .72, 1: .79})
    assert evidence['changed_members'] == [0]
    assert not evidence['eye_state_authority']
    assert not evidence['amodal_completeness_authority']


@pytest.mark.parametrize('other', [observed(2, 70, .7, cls=0),
    observed(2, 70, .7, axis=1), observed(2, 70, .39), observed(2, 59, .7)])
def test_disagreement_or_unknown_association_abstains(other):
    scores, evidence = lq.adjust_scores([observed(1, 60, .4), other], [phase(0, 1)], {0: .8, 1: .79})
    assert scores == {0: .8, 1: .79}
    assert not evidence['changed_members']


@pytest.mark.parametrize('meta,blob', [('{}', None), ('null', b'abc'),
    ('{"local_quality":[]}', b'abc'), ('{"local_quality":null}', b'abc'), ('invalid', b'abc')])
def test_malformed_observations_fail_closed(meta, blob):
    assert lq.observations({'quality_meta': meta, 'local_quality_embedding': blob}) == {}


def test_faces_require_matched_enclosing_person():
    ms = [observed(1, 60, .4, kind='face', cls=0), observed(2, 70, .7, kind='face', cls=0)]
    scores, _ = lq.adjust_scores(ms, [phase(0, 1)], {0: .8, 1: .79})
    assert scores == {0: .8, 1: .79}


def test_matched_native_views_cross_shadow_boundaries_without_changing_phases():
    ms = [observed(1, 60, .4), observed(2, 70, .7)]
    phases = [phase(0), phase(1)]
    scores, _ = lq.adjust_scores(ms, phases, {0: .8, 1: .79})
    assert scores == pytest.approx({0: .72, 1: .79})
    assert [p.members for p in phases] == [(0,), (1,)]


def test_distinct_native_views_across_shadow_phases_abstain():
    ms = [observed(1, 60, .4), observed(2, 70, .7, axis=1)]
    scores, _ = lq.adjust_scores(ms, [phase(0), phase(1)], {0: .8, 1: .79})
    assert scores == {0: .8, 1: .79}


def test_selector_is_opt_in_same_budget_and_review_only():
    ms = [observed(1, 60, .4), observed(2, 70, .7)]
    phases = [phase(0, 1)]
    base = ps.select_phase_keepers(ms, phases)
    assert base['keepers'] == [0]
    candidate = ps.select_phase_keepers(ms, phases, local_quality_policy='dominance')
    assert candidate['keepers'] == [1]
    assert candidate['group_keeper_budget'] == base['group_keeper_budget']
    assert candidate['mandatory_review'] and candidate['review_required']
    assert 'LOCAL_QUALITY_DOMINANCE' in candidate['reason_codes']
    assert not candidate['physical_group_split']
    assert candidate['authority'] == 'shadow_review_only'


@pytest.mark.parametrize('key,value', [('keeper_local_quality_policy', 'bad'),
    ('keeper_local_quality_similarity', float('nan')), ('keeper_local_quality_penalty', True),
    ('keeper_local_quality_penalty', .26)])
def test_invalid_config_fails_before_runtime(tmp_path, key, value):
    import yaml
    p = tmp_path / 'config.yaml'
    p.write_text(yaml.safe_dump({'cluster': {key: value}}))
    with pytest.raises(ValueError):
        load_config(p).validate()


def test_additive_blob_migration_and_feature_refresh_invalidation(tmp_path):
    p = tmp_path / 'old.sqlite'
    c = sqlite3.connect(p)
    legacy = '\n'.join(line for line in db.SCHEMA.splitlines() if 'local_quality_embedding BLOB' not in line)
    c.executescript(legacy)
    c.execute("INSERT INTO files(id,path) VALUES(1,'example.jpg')")
    c.execute('INSERT INTO features(file_id,quality_score) VALUES(1,70)')
    c.commit()
    c.close()
    c = db.open_db(p)
    assert c.execute('SELECT quality_score,local_quality_embedding FROM features').fetchone()[:] == (70, None)
    c.execute("UPDATE features SET local_quality_embedding=x'0102'")
    db.batch_insert_features(c, [{'file_id': 1, 'phash': None, 'dinov2_embedding': None,
        'quality_score': 71, 'quality_meta': '{}', 'face_count': 0, 'faces_json': '[]', 'status': 'done'}])
    assert c.execute('SELECT local_quality_embedding FROM features').fetchone()[0] is None
    c.close()
