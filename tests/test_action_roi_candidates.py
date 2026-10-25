"""Proposal gates are review-only, independent of person/face detections."""
import copy

import numpy as np
import pytest

from src import action_roi_candidates as roi
from src import vlm_visual_budget


def _candidate(box=(.35, .3, .6, .8)):
    return {"box_normalized": list(box), "area_fraction": .10,
            "centroid_normalized": [.475, .55], "changed_fraction": .8,
            "mask": np.ones((40, 40), bool), "lab_mean": np.array([120., 140., 160.]),
            "hist": np.ones((16, 16), np.float32) / 256}


def _pipeline(monkeypatch, components):
    monkeypatch.setattr(roi, "_registration", lambda _: (np.array([[1., 0., 0.], [0., 1., 0.]]), {}))
    values = iter(components)
    monkeypatch.setattr(roi, "_components", lambda *_: (next(values), []))
    return roi.propose_pair([np.zeros((400, 500, 3), np.uint8)] * 2)


def test_candidate_requires_independent_unique_components_and_has_no_authority(monkeypatch):
    result = _pipeline(monkeypatch, [[_candidate()] for _ in range(4)])
    assert result["status"] == "candidate"
    assert result["review_only"] is True
    assert result["semantic_identity_authority"] is False
    assert result["keeper_authority"] is False
    assert result["whole_body_completeness"] == "unknown"
    assert result["bystander_face_crops"] == 0
    assert len(result["masks"]) == 2
    assert result["consistency"]["reciprocal_unique"] is True
    # Candidate packets cannot authorize the production primary-subject planner.
    with pytest.raises(vlm_visual_budget.VisualBudgetError, match="PRIMARY_SUBJECT_UNLOCALIZED"):
        vlm_visual_budget.plan([{"file_id": 1, "quality_meta": {"action_roi_candidate": result}}], regime="action")


@pytest.mark.parametrize("components", [[], [_candidate(), _candidate()]])
def test_missing_or_ambiguous_candidate_abstains(monkeypatch, components):
    assert _pipeline(monkeypatch, [components])["reason"] == "NO_UNIQUE_CENTRAL_MOVING_COMPONENT"


def test_segmentation_extent_disagreement_abstains(monkeypatch):
    result = _pipeline(monkeypatch, [[_candidate()], [_candidate((.1, .1, .4, .4))]])
    assert result["reason"] == "SEGMENTATION_EXTENT_UNSTABLE"


@pytest.mark.parametrize("field,value", [("lab_mean", np.array([20., 20., 20.])),
                                         ("area_fraction", .02),
                                         ("centroid_normalized", [.8, .8])])
def test_cross_frame_disagreement_abstains(monkeypatch, field, value):
    other = copy.deepcopy(_candidate())
    other[field] = value
    result = _pipeline(monkeypatch, [[_candidate()], [_candidate()], [other], [other]])
    assert result["reason"] == "CROSS_FRAME_REGION_INCONSISTENT"


def test_low_texture_geometry_and_small_native_image_abstain():
    blank = np.zeros((400, 500, 3), np.uint8)
    assert roi.propose_pair([blank, blank])["reason"] == "BACKGROUND_LOW_TEXTURE"
    assert roi.propose_pair([blank, blank[:300]])["reason"] == "FRAME_GEOMETRY_CHANGED"
    assert roi.propose_pair([blank[:200], blank[:200]])["reason"] == "NATIVE_IMAGE_TOO_SMALL"
    assert roi.propose_pair([blank])["reason"] == "TWO_RGB_FRAMES_REQUIRED"


def test_static_components_do_not_claim_action_region():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    result = roi.propose_pair([image, image])
    assert result["status"] == "abstain"
    assert result["reason"] == "NO_UNIQUE_CENTRAL_MOVING_COMPONENT"


def test_boundary_and_offcenter_components_cannot_spend_crop_budget():
    image = np.zeros((300, 400, 3), np.uint8)
    image[60:240, :90] = [240, 80, 30]
    image[60:240, 320:] = [30, 240, 80]
    peer = np.zeros_like(image)
    eligible, audit = roi._components(image, peer, np.array([[1., 0., 0.], [0., 1., 0.]]), 8)
    assert not eligible
    changed = [record for record in audit if record["changed_fraction"] >= .5]
    assert changed
    assert all("BOUNDARY_TOUCHING" in r["rejections"] and "OFF_CENTER" in r["rejections"] for r in changed)
