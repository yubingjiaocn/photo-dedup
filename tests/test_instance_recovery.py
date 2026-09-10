import copy
import json

import numpy as np
import pytest

from src.instance_recovery import (PRODUCER, appearance_unique, confirm_target,
                                   review_context, track_box, novel_detection,
                                   pose_pair_support, stable_tracks, sift_box)
from src import phase_selection as ps
from src.config import Config, _DEFAULTS
from tests.test_region_set_quality import frame, phases


def packet():
    return {'schema_version':1, 'producer':PRODUCER, 'review_only':True,
            'keeper_authority':False, 'group_member_ids':[1,2],
            'stable_observed_set':False,
            'proposals':[{'status':'confirmed_review_proposal', 'source_file_id':1,
                          'target_file_id':2, 'class_id':0, 'box_normalized':[.1,.1,.5,.9],
                          'target_confirmation':True, 'appearance_unique':True,
                          'motion_consistent':True, 'novelty_checked':True}]}


def members(p=None):
    result = [frame(1),frame(2)]
    for m in result:
        meta = json.loads(m['quality_meta'])
        meta['instance_recovery'] = packet() if p is None else p
        m['quality_meta'] = json.dumps(meta)
    return result


def test_default_disabled_and_optin_never_changes_keeper_order_scores_or_budget():
    ms = members()
    baseline = ps.select_phase_keepers(ms,phases(0,1))
    review = ps.select_phase_keepers(ms,phases(0,1),instance_recovery_policy='review_only')
    assert 'instance_recovery_context' not in baseline
    for key in ['keepers','utility_scores','group_keeper_budget','phases']:
        assert baseline[key] == review[key]
    assert review['mandatory_review'] and review['review_required']
    assert len(review['instance_recovery_context']['proposed_groups']) == 1
    assert not review['instance_recovery_context']['keeper_authority']


@pytest.mark.parametrize('change', [
    {'producer':'flow_box'}, {'schema_version':2}, {'review_only':False},
    {'keeper_authority':True}, {'group_member_ids':[1,3]}, {'proposals':{}},
])
def test_invalid_packets_abstain(change):
    p = packet()
    p.update(change)
    assert not review_context(members(p))['proposed_groups']


@pytest.mark.parametrize('change', [
    {'target_confirmation':False}, {'appearance_unique':False}, {'motion_consistent':False},
    {'novelty_checked':False}, {'source_file_id':True},
    {'target_file_id':9}, {'source_file_id':2}, {'class_id':99},
    {'box_normalized':[0,0,float('nan'),1]}, {'box_normalized':[.5,.5,.1,.1]},
])
def test_unconfirmed_and_malformed_proposals_abstain(change):
    p = packet()
    p['proposals'][0].update(change)
    assert not review_context(members(p))['proposed_groups']


def test_partial_or_cross_group_records_cannot_leak():
    ms = members()
    ms[1]['quality_meta'] = '{}'
    assert not review_context(ms)['proposed_groups']
    ms = members()
    ms[1]['id'] = 3
    assert not review_context(ms)['proposed_groups']


def test_no_subject_global_baseline_even_if_evaluation_label_is_injected():
    ms = [frame(1),frame(2)]
    for m in ms:
        m['quality_meta'] = '{}'
        m['evaluation_regime'] = 'multi'
    a = ps.select_phase_keepers(ms,phases(0,1))
    b = ps.select_phase_keepers(ms,phases(0,1),instance_recovery_policy='review_only')
    assert a['keepers'] == b['keepers']
    assert not b['instance_recovery_context']['proposed_groups']


def test_flat_pixels_and_wrong_geometry_fail_closed():
    im = np.zeros((400,400),np.uint8)
    assert track_box(im,im,[50,50,300,350])['reason'] == 'LOW_TEXTURE'
    assert track_box(im,im[:200],[50,50,300,350])['reason'] == 'FRAME_GEOMETRY_CHANGED'


def test_native_translation_produces_search_window_not_identity():
    import cv2
    rng = np.random.default_rng(42)
    im = cv2.GaussianBlur(rng.integers(0,255,(500,500),dtype=np.uint8),(5,5),0)
    shifted = cv2.warpAffine(im,np.array([[1,0,7],[0,1,5]],dtype=np.float32),(500,500))
    result = track_box(im,shifted,[100,100,350,400])
    assert result['eligible'] and not result['identity_authority']
    assert np.allclose(result['box'],[107,105,357,405],atol=2)


def test_target_absent_or_ambiguous_fails_even_with_perfect_flow():
    box = [0,0,200,300]
    assert not confirm_target(box,0,[])['eligible']
    d = {'box':box,'class_id':0,'confidence':.9}
    assert confirm_target(box,0,[d])['eligible']
    assert not confirm_target(box,0,[d,d])['eligible']
    assert not confirm_target(box,16,[d])['eligible']


def test_identical_costumes_cannot_use_position_to_resolve_identity():
    objects = [{'class_id':0},{'class_id':0}]
    vectors = np.array([[1.,0.],[1.,0.]])
    assert not appearance_unique(0,0,objects,objects,vectors,vectors)['eligible']
    vectors = np.eye(2)
    assert appearance_unique(0,0,objects,objects,vectors,vectors)['eligible']
    assert not appearance_unique(0,1,objects,objects,vectors,vectors)['eligible']


def test_policy_validation():
    with pytest.raises(ValueError,match='instance_recovery'):
        ps.select_phase_keepers([],[],instance_recovery_policy='on')
    settings = copy.deepcopy(_DEFAULTS)
    settings['cluster']['keeper_instance_recovery_policy'] = 'on'
    with pytest.raises(ValueError,match='instance_recovery'):
        Config(settings).validate()


def test_partial_cross_class_duplicate_is_not_new_instance():
    # Different classes and IoU below .5 cannot hide a contained/overlapping body.
    dog = {'box':[.24,.16,.91,.83], 'class_id':16}
    partial_cat = {'box':[.035,.14,.697,.86], 'class_id':15}
    assert not novel_detection(partial_cat,[dog])
    assert novel_detection({'box':[.92,.1,.99,.7],'class_id':0},[dog])


def test_cached_pose_needs_native_torso_texture_and_independent_anchors():
    from tests.test_pose_evidence import body
    image=np.random.default_rng(11).integers(0,255,(600,600),dtype=np.uint8)
    obj={'box':[120,60,480,570],'class_id':0,'pose':body()}
    result=pose_pair_support(image,image,obj,obj)
    assert result['eligible'] and not result['identity_authority']
    flat=np.full_like(image,100)
    assert pose_pair_support(flat,flat,obj,obj)['reason']=='POSE_TORSO_LOW_TEXTURE'
    weak=copy.deepcopy(obj)
    weak['pose']['box_confidence']=.5
    assert not pose_pair_support(image,image,obj,weak)['eligible']
    missing=copy.deepcopy(obj)
    del missing['pose']
    assert not pose_pair_support(image,image,obj,missing)['eligible']


def test_cycle_stable_subset_is_not_complete_set_or_largest_actor():
    catalogs=[[{},{}],[{},{}],[{},{}]]
    edges={(s,t,1):1 for s in range(3) for t in range(3) if s!=t}
    assert stable_tracks(catalogs,edges)==[[1,1,1]]
    assert len(stable_tracks(catalogs,edges))<len(catalogs[0])
    edges[1,2,1]=0
    assert stable_tracks(catalogs,edges)==[]


def test_sift_low_texture_is_not_identity():
    image=np.zeros((400,400),np.uint8)
    assert not sift_box(image,image,[50,50,300,350])['eligible']
