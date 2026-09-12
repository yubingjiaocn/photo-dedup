import numpy as np
from PIL import Image

from src.instance_correspondence import (
    mask_for,
    view_agreement,
    masked_crop,
    foreground_keypoints,
    foreground_motion,
    match_appearance,
    track_components,
)


def obj(cls=0, dx=0):
    return {
        "class_id": cls,
        "box": [100 + dx, 80, 450 + dx, 500],
        "polygon": [[100 + dx, 80], [450 + dx, 80], [450 + dx, 500], [100 + dx, 500]],
        "crop_edge": False,
        "view_confirmed": True,
    }


def test_independent_views_can_change_detector_class_without_claiming_identity():
    result = view_agreement(obj(0), obj(77), (600, 600))
    assert result["eligible"]
    assert result["whole_body_completeness"] == "unknown"


def test_crop_edge_or_head_only_view_cannot_confirm_full_extent():
    a = obj()
    b = obj()
    b["crop_edge"] = True
    assert not view_agreement(a, b, (600, 600))["eligible"]
    b = obj()
    b["box"][3] = 200
    b["polygon"] = [[100, 80], [450, 80], [450, 200], [100, 200]]
    assert not view_agreement(a, b, (600, 600))["eligible"]


def test_background_is_neutral_not_native_identity_evidence():
    image = Image.fromarray(np.full((600, 600, 3), 250, dtype=np.uint8))
    o = obj()
    o["polygon"] = [[200, 150], [400, 150], [400, 400], [200, 400]]
    pixels = np.asarray(masked_crop(image, o))
    assert np.all(pixels[0, 0] == 127)
    assert np.all(pixels[100, 150] == 250)
    assert mask_for((20, 20), [[float("nan"), 2], [2, 3], [3, 4]]).sum() == 0


def test_class_switch_uses_appearance_not_class_or_position():
    a = [obj(0), obj(0, 30)]
    b = [obj(77), obj(0, 30)]
    accepted, refused = match_appearance(a, b, np.eye(2), np.eye(2))
    assert len(accepted) == 2 and not refused
    assert (
        accepted[0]["source_detector_class"] == 0
        and accepted[0]["target_detector_class"] == 77
    )


def test_unconfirmed_peer_still_blocks_manufactured_uniqueness():
    a = [obj(), obj(77)]
    b = [obj(77), obj()]
    a[1]["view_confirmed"] = False
    b[1]["view_confirmed"] = False
    values = np.array([[1.0, 0.0], [1.0, 0.0]])
    accepted, refused = match_appearance(a, b, values, values)
    assert not accepted and len(refused) == 2


def test_unconfirmed_target_abstains_even_with_unique_embedding():
    a = [obj()]
    b = [obj(77)]
    b[0]["view_confirmed"] = False
    accepted, refused = match_appearance(a, b, np.ones((1, 1)), np.ones((1, 1)))
    assert not accepted and refused[0]["reason"] == "INDEPENDENT_VIEW_UNCONFIRMED"


def test_native_foreground_translation_needs_reciprocal_distributed_points():
    import cv2

    rng = np.random.default_rng(12)
    source = cv2.GaussianBlur(
        rng.integers(0, 255, (600, 600), dtype=np.uint8), (3, 3), 0
    )
    target = cv2.warpAffine(
        source, np.array([[1, 0, 10], [0, 1, 0]], np.float32), (600, 600)
    )
    a, b = obj(), obj(77, 10)
    left, right = foreground_keypoints(source, a), foreground_keypoints(target, b)
    result = foreground_motion(left, right, a, b)
    assert result["eligible"] and not result["identity_authority"]
    assert result["inliers"] >= 12
    assert all(
        100 <= x <= 450 and 80 <= y <= 500 for x, y in result["source_inliers_native"]
    )


def test_flat_foreground_does_not_inherit_background_texture():
    image = np.random.default_rng(33).integers(0, 255, (600, 600), dtype=np.uint8)
    image[75:505, 95:455] = 100
    points = foreground_keypoints(image, obj())
    assert not foreground_motion(points, points, obj(), obj())["eligible"]


def test_missing_observations_stay_missing():
    edges = [((0, 0), (1, 0)), ((1, 0), (0, 0))]
    tracks = track_components([1, 1, 0, 1], edges)
    assert len(tracks) == 1 and tracks[0]["slots"] == [0, 0, None, None]
    assert tracks[0]["missing_frames"] == [2, 3]
    assert not tracks[0]["all_frames_observed"]


def test_transitivity_alone_does_not_create_complete_identity_track():
    edges = [((0, 0), (1, 0)), ((1, 0), (0, 0)), ((1, 0), (2, 0)), ((2, 0), (1, 0))]
    assert not track_components([1, 1, 1], edges)
    edges.extend([((0, 0), (2, 0)), ((2, 0), (0, 0))])
    assert track_components([1, 1, 1], edges)[0]["all_frames_observed"]


def test_two_instances_in_one_frame_cannot_share_a_track():
    edges = [((0, 0), (1, 0)), ((1, 0), (0, 0)), ((0, 1), (1, 0)), ((1, 0), (0, 1))]
    assert not track_components([2, 1], edges)
