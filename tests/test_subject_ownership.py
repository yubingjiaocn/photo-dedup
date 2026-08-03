"""Fixture-style tests for the pure, decision-isolated subject ownership gate."""

from __future__ import annotations

from dataclasses import asdict, replace
import json

import pytest

from src.routing_schema import ReasonCode, State
from src.subject_ownership import (
    Box,
    CropContext,
    EyeObservation,
    GateApplicability,
    GateThresholds,
    MappedMPFace,
    OwnershipEvidence,
    SubjectBox,
    YuNetFace,
    evaluate_subject_ownership,
)


def _face(face_id="f1", box=Box(300, 180, 160, 190), iod=55):
    return YuNetFace(face_id, box, iod)


def _mp(face_id="f1", context=CropContext.CROP_1_3, box=Box(305, 185, 155, 180),
        observation=EyeObservation.OPEN, iod=54, **kwargs):
    return MappedMPFace(face_id, context, box, iod, observation, **kwargs)


def _clear_person(**changes):
    base = OwnershipEvidence(
        routing_state=State.KNOWN,
        subject_tags=frozenset({"REAL_PERSON_SUBJECT"}),
        image_width=1200,
        image_height=900,
        yunet_faces=(_face(),),
        subject_boxes=(SubjectBox("person-1", Box(220, 80, 420, 720)),),
        mp_faces=(_mp(), _mp(context=CropContext.CROP_1_6, box=Box(302, 182, 158, 184))),
        printed_or_screen_risk=False,
        nonhuman_face_risk=False,
    )
    return replace(base, **changes)


def _assert(result, applicability, reason=None):
    assert result.applicability == applicability
    assert result.accepted_subject_face_ids == (() if applicability != GateApplicability.APPLICABLE else ("f1",))
    assert result.reasons == (() if reason is None else (reason,))


def test_clear_real_person_subject_is_applicable():
    result = evaluate_subject_ownership(_clear_person())
    _assert(result, GateApplicability.APPLICABLE)


def test_cosplay_masked_person_is_eligible_when_other_gates_pass():
    evidence = _clear_person(subject_tags=frozenset({"COSTUME_MASKED_PERSON"}))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.APPLICABLE)


def test_winnie_foreground_with_background_tourist_is_not_applicable():
    evidence = _clear_person(
        subject_tags=frozenset({"REAL_PERSON_SUBJECT", "NONHUMAN_CHARACTER_DOLL_STATUE"}),
        subject_boxes=(SubjectBox("winnie", Box(100, 50, 650, 800)),),
        yunet_faces=(_face(box=Box(1000, 120, 130, 160)),),
        mp_faces=(),
    )
    _assert(
        evaluate_subject_ownership(evidence),
        GateApplicability.NOT_APPLICABLE,
        ReasonCode.NONHUMAN_FACE_RISK,
    )


@pytest.mark.parametrize(
    ("tags", "reason"),
    [
        ({"DOCUMENT_SCREENSHOT"}, ReasonCode.PRINTED_OR_SCREEN_FACE_RISK),
        ({"NONHUMAN_CHARACTER_DOLL_STATUE"}, ReasonCode.NONHUMAN_FACE_RISK),
    ],
)
def test_explicit_anime_poster_and_doll_are_not_applicable(tags, reason):
    evidence = _clear_person(subject_tags=frozenset(tags))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.NOT_APPLICABLE, reason)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"printed_or_screen_risk": True}, ReasonCode.PRINTED_OR_SCREEN_FACE_RISK),
        ({"nonhuman_face_risk": True}, ReasonCode.NONHUMAN_FACE_RISK),
    ],
)
def test_ambiguous_printed_screen_mascot_or_material_risk_abstains(changes, reason):
    evidence = replace(_clear_person(), **changes)
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, reason)


def test_missing_material_risk_evidence_abstains():
    evidence = replace(_clear_person(), printed_or_screen_risk=None)
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.FEATURE_MISSING)


def test_unknown_subject_tag_is_rejected():
    evidence = replace(
        _clear_person(),
        subject_tags=frozenset({"REAL_PERSON_SUBJECT", "ALIEN"}),
    )
    with pytest.raises(ValueError, match="subject_tags"):
        evaluate_subject_ownership(evidence)


@pytest.mark.parametrize(
    "tags",
    [
        frozenset({"FOOD_STILL_LIFE"}),
        frozenset({"NO_DOMINANT_SUBJECT"}),  # fireworks fixture: no person subject
    ],
)
def test_food_and_fireworks_are_not_applicable_not_bad_photos(tags):
    evidence = OwnershipEvidence(State.KNOWN, tags, 1200, 900)
    result = evaluate_subject_ownership(evidence)
    _assert(result, GateApplicability.NOT_APPLICABLE, ReasonCode.DETECTOR_NOT_APPLICABLE)
    rendered = json.dumps(asdict(result))
    assert "review" not in rendered.lower() and "delete" not in rendered.lower()


def test_unknown_routing_fails_closed_before_detector_evidence():
    evidence = replace(_clear_person(), routing_state=State.UNKNOWN)
    _assert(
        evaluate_subject_ownership(evidence),
        GateApplicability.UNKNOWN,
        ReasonCode.SUBJECT_ASSIGNMENT_UNKNOWN,
    )


def test_missing_subject_box_never_guesses_from_largest_face():
    evidence = replace(_clear_person(), subject_boxes=None)
    _assert(
        evaluate_subject_ownership(evidence),
        GateApplicability.UNKNOWN,
        ReasonCode.SUBJECT_ASSIGNMENT_UNKNOWN,
    )


def test_face_outside_subject_box_is_background_only():
    evidence = replace(
        _clear_person(),
        subject_boxes=(SubjectBox("person-1", Box(0, 0, 200, 700)),),
    )
    _assert(
        evaluate_subject_ownership(evidence),
        GateApplicability.NOT_APPLICABLE,
        ReasonCode.BACKGROUND_FACE_ONLY,
    )


def test_owned_subject_face_plus_extra_background_face_abstains():
    evidence = _clear_person()
    evidence = replace(
        evidence,
        yunet_faces=evidence.yunet_faces + (
            _face("background", Box(950, 100, 120, 150), 38),
        ),
    )
    result = evaluate_subject_ownership(evidence)
    assert result.applicability == GateApplicability.UNKNOWN
    assert result.reasons == (ReasonCode.BACKGROUND_FACE_ONLY,)
    assert result.accepted_subject_face_ids == ()


def test_multiple_subject_assignment_ambiguity_abstains():
    evidence = replace(
        _clear_person(),
        subject_boxes=(
            SubjectBox("person-1", Box(200, 50, 500, 750)),
            SubjectBox("person-2", Box(250, 100, 500, 750)),
        ),
    )
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.MULTIPLE_SUBJECTS)


def test_yunet_mp_count_mismatch_abstains():
    evidence = replace(_clear_person(), mp_faces=(_mp(),))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.FACE_COUNT_MISMATCH)


def test_duplicate_mapped_detection_is_count_mismatch_not_adopted():
    evidence = _clear_person()
    duplicate = _mp(context=CropContext.CROP_1_3, box=Box(310, 190, 150, 175))
    evidence = replace(evidence, mp_faces=evidence.mp_faces + (duplicate,))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.FACE_COUNT_MISMATCH)


def test_mapped_box_inconsistent_with_yunet_abstains():
    evidence = _clear_person()
    bad = _mp(box=Box(800, 600, 155, 180))
    evidence = replace(evidence, mp_faces=(bad, evidence.mp_faces[1]))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.FACE_BOX_MISMATCH)


def test_crop_context_observation_conflict_abstains():
    evidence = _clear_person()
    closed = _mp(context=CropContext.CROP_1_6, observation=EyeObservation.CLOSED)
    evidence = replace(evidence, mp_faces=(evidence.mp_faces[0], closed))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.CROP_CONTEXT_CONFLICT)


def test_crop_context_geometry_conflict_abstains():
    evidence = _clear_person()
    left = _mp(box=Box(250, 185, 155, 180))
    right = _mp(context=CropContext.CROP_1_6, box=Box(350, 185, 155, 180))
    evidence = replace(evidence, mp_faces=(left, right))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.CROP_CONTEXT_CONFLICT)


@pytest.mark.parametrize(
    "face",
    [
        _face(box=Box(300, 180, 80, 190), iod=55),
        _face(box=Box(300, 180, 160, 190), iod=20),
    ],
)
def test_small_original_face_or_iod_cannot_be_rescued_by_crop(face):
    evidence = replace(_clear_person(), yunet_faces=(face,))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.SMALL_FACE)


def test_missing_original_iod_cannot_be_replaced_by_crop_iod():
    evidence = replace(_clear_person(), yunet_faces=(_face(iod=None),))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.FEATURE_MISSING)


@pytest.mark.parametrize("kwargs", [{"profile": True}, {"occluded": True}])
def test_profile_or_occluded_face_abstains(kwargs):
    evidence = _clear_person()
    affected = _mp(context=CropContext.CROP_1_6, **kwargs)
    evidence = replace(evidence, mp_faces=(evidence.mp_faces[0], affected))
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, ReasonCode.PROFILE_OR_OCCLUDED)


@pytest.mark.parametrize(
    "affected",
    [
        _mp(context=CropContext.CROP_1_6, eyes_visible=False),
        _mp(context=CropContext.CROP_1_6, observation=EyeObservation.UNKNOWN),
    ],
)
def test_eye_visibility_or_unknown_observation_abstains(affected):
    evidence = _clear_person()
    evidence = replace(evidence, mp_faces=(evidence.mp_faces[0], affected))
    expected = ReasonCode.CROP_CONTEXT_CONFLICT if affected.observation == EyeObservation.UNKNOWN else ReasonCode.EYE_STATE_UNKNOWN
    _assert(evaluate_subject_ownership(evidence), GateApplicability.UNKNOWN, expected)


def test_reaching_configured_max_faces_abstains():
    evidence = _clear_person()
    result = evaluate_subject_ownership(evidence, GateThresholds(max_faces=1))
    _assert(result, GateApplicability.UNKNOWN, ReasonCode.MAX_FACES_REACHED)


def test_result_contract_has_only_applicability_reasons_and_face_ids():
    result = evaluate_subject_ownership(_clear_person())
    payload = asdict(result)
    assert set(payload) == {"applicability", "reasons", "accepted_subject_face_ids"}
    rendered = json.dumps(payload).lower()
    assert not any(word in rendered for word in ("decision", "action", "quality_score", "review", "delete"))
