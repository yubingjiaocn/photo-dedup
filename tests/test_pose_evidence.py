import copy
import json

import pytest

from src.pose_evidence import compare_poses, protect_pose_variants


def body(dx=0.0, move=0.0):
    points = [[0.5 + dx, 0.2, 0.99] for _ in range(17)]
    for i, xy in {5: (0.4, 0.35), 6: (0.6, 0.35), 7: (0.3, 0.45),
                  8: (0.7, 0.45), 9: (0.25, 0.55), 10: (0.75, 0.55),
                  11: (0.45, 0.65), 12: (0.55, 0.65), 13: (0.43, 0.78),
                  14: (0.57, 0.78), 15: (0.42, 0.9), 16: (0.58, 0.9)}.items():
        points[i] = [xy[0] + dx + (move if i in {7, 8, 9, 10} else 0), xy[1], 0.99]
    return {'box': [0.2 + dx, 0.1, 0.8 + dx, 0.95],
            'box_confidence': 0.99, 'keypoints': points}


def record(person=None):
    return {'producer': 'local-coco17-model', 'status': 'ok',
            'shape': [1000, 1000], 'people': [person or body()]}


def test_same_pose_has_zero_displacement():
    result = compare_poses(record(), record())
    assert result['eligible']
    assert result['displacement'] == pytest.approx(0)
    assert not result['semantic_phase_authority']


def test_translation_is_not_body_action():
    result = compare_poses(record(), record(body(dx=0.04)))
    assert result['eligible']
    assert result['displacement'] == pytest.approx(0, abs=1e-12)


def test_visible_limb_displacement_is_measured_in_torso_lengths():
    result = compare_poses(record(), record(body(move=0.12)))
    assert result['eligible']
    assert result['displacement'] == pytest.approx(0.4)


@pytest.mark.parametrize('missing', [None, {}, {'producer': 'local-coco17-model', 'status': 'unknown'}])
def test_missing_data_abstains(missing):
    assert not compare_poses(missing, record())['eligible']


def test_unmatched_or_ambiguous_people_abstain():
    left = record()
    right = record()
    right['people'].append(copy.deepcopy(right['people'][0]))
    assert not compare_poses(left, right)['eligible']


def test_missing_core_keypoint_abstains():
    left = record()
    left['people'][0]['keypoints'][11][2] = 0.1
    assert not compare_poses(left, record())['eligible']


def test_one_joint_outlier_is_not_action_evidence():
    right = record()
    right['people'][0]['keypoints'][9][0] = 0.95
    assert compare_poses(record(), right)['displacement'] == pytest.approx(0)


def test_producer_and_shape_validation():
    right = record()
    right['producer'] = 'other-model'
    assert not compare_poses(record(), right)['eligible']
    right = record()
    right['shape'] = [float('nan'), 1000]
    assert not compare_poses(record(), right)['eligible']


def member(pose):
    return {'quality_meta': json.dumps({'pose_evidence': pose})}


def test_bounded_pose_variant_preserves_existing_keeper_and_adds_one():
    members = [member(record()), member(record(body(move=0.18)))]
    kept, evidence = protect_pose_variants(members, [0], [0.8, 0.7], displacement_threshold=0.4)
    assert kept == [0, 1]
    assert evidence['added_keeper'] == 1
    assert len(evidence['comparisons']) == 1
    assert not evidence['semantic_phase_authority']


def test_variant_must_contrast_with_every_existing_keeper():
    members = [member(record()), member(record(body(move=0.18))), member(record(body(move=0.18)))]
    kept, evidence = protect_pose_variants(members, [0, 1], [0.9, 0.8, 0.7], displacement_threshold=0.4)
    assert kept == [0, 1]
    assert evidence['added_keeper'] is None


def test_unknown_is_not_new_pose_and_large_groups_abstain():
    assert protect_pose_variants([member(record()), {}], [0], [0.8, 0.9], displacement_threshold=0.4)[0] == [0]
    large = [member(record())] * 6 + [member(record(body(move=0.18)))]
    assert protect_pose_variants(large, [0], [0.8] * 7, displacement_threshold=0.4)[0] == [0]


@pytest.mark.parametrize('threshold', [0, -1, True, float('nan'), float('inf')])
def test_invalid_pose_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        protect_pose_variants([member(record())], [0], [0.8], displacement_threshold=threshold)
