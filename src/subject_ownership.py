"""Pure, fail-closed subject ownership gate for face-detector evidence.

The caller supplies already-structured routing and detector evidence.  This
module never opens an image and never invokes a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .routing_schema import ReasonCode, State, SUBJECT_TAGS


class GateApplicability(StrEnum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


class CropContext(StrEnum):
    CROP_1_3 = "1.3x"
    CROP_1_6 = "1.6x"


class EyeObservation(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    MAYBE = "MAYBE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values) or self.width <= 0 or self.height <= 0:
            raise ValueError("box coordinates must be finite and dimensions positive")


@dataclass(frozen=True)
class SubjectBox:
    subject_id: str
    bbox: Box


@dataclass(frozen=True)
class YuNetFace:
    face_id: str
    bbox: Box
    iod_px: float | None = None


@dataclass(frozen=True)
class MappedMPFace:
    """One MediaPipe result already mapped into original-image coordinates."""

    yunet_face_id: str
    crop_context: CropContext
    bbox: Box
    iod_px: float | None
    observation: EyeObservation
    profile: bool = False
    occluded: bool = False
    eyes_visible: bool = True


@dataclass(frozen=True)
class OwnershipEvidence:
    routing_state: State
    subject_tags: frozenset[str]
    image_width: int
    image_height: int
    yunet_faces: tuple[YuNetFace, ...] = ()
    subject_boxes: tuple[SubjectBox, ...] | None = None
    mp_faces: tuple[MappedMPFace, ...] = ()
    printed_or_screen_risk: bool | None = None
    nonhuman_face_risk: bool | None = None


@dataclass(frozen=True)
class GateThresholds:
    min_face_width_px: float = 96.0
    min_iod_px: float = 24.0
    min_subject_overlap: float = 0.5
    min_detector_iou: float = 0.25
    max_center_distance_ratio: float = 0.35
    max_faces: int = 10


@dataclass(frozen=True)
class OwnershipResult:
    applicability: GateApplicability
    reasons: tuple[ReasonCode, ...]
    accepted_subject_face_ids: tuple[str, ...] = ()


_PERSON_TAGS = frozenset({"REAL_PERSON_SUBJECT", "COSTUME_MASKED_PERSON"})
_NONHUMAN_TAG = "NONHUMAN_CHARACTER_DOLL_STATUE"
_DOCUMENT_TAG = "DOCUMENT_SCREENSHOT"


def evaluate_subject_ownership(
    evidence: OwnershipEvidence,
    thresholds: GateThresholds | None = None,
) -> OwnershipResult:
    """Evaluate all seven safety gates without performing inference."""
    t = thresholds or GateThresholds()
    _validate_evidence(evidence, t)
    tags = evidence.subject_tags

    # Semantic eligibility and explicit negative classes.
    if not tags.issubset(SUBJECT_TAGS):
        raise ValueError("subject_tags must use the validated routing schema")
    if evidence.routing_state != State.KNOWN:
        return _unknown(ReasonCode.SUBJECT_ASSIGNMENT_UNKNOWN)
    if _DOCUMENT_TAG in tags:
        return _not_applicable(ReasonCode.PRINTED_OR_SCREEN_FACE_RISK)
    if _NONHUMAN_TAG in tags:
        return _not_applicable(ReasonCode.NONHUMAN_FACE_RISK)
    if not tags.intersection(_PERSON_TAGS):
        return _not_applicable(ReasonCode.DETECTOR_NOT_APPLICABLE)
    if evidence.printed_or_screen_risk is None or evidence.nonhuman_face_risk is None:
        return _unknown(ReasonCode.FEATURE_MISSING)
    if evidence.printed_or_screen_risk:
        return _unknown(ReasonCode.PRINTED_OR_SCREEN_FACE_RISK)
    if evidence.nonhuman_face_risk:
        return _unknown(ReasonCode.NONHUMAN_FACE_RISK)
    if len(evidence.yunet_faces) >= t.max_faces:
        return _unknown(ReasonCode.MAX_FACES_REACHED)

    # Ownership must be established from explicit subject regions.
    if not evidence.subject_boxes:
        return _unknown(ReasonCode.SUBJECT_ASSIGNMENT_UNKNOWN)
    assignments: dict[str, str] = {}
    ambiguous = False
    for face in evidence.yunet_faces:
        owners = [subject.subject_id for subject in evidence.subject_boxes if _belongs(face.bbox, subject.bbox, t)]
        if len(owners) == 1:
            assignments[face.face_id] = owners[0]
        elif len(owners) > 1:
            ambiguous = True
    if ambiguous:
        return _unknown(ReasonCode.MULTIPLE_SUBJECTS)
    if not assignments:
        reason = ReasonCode.BACKGROUND_FACE_ONLY if evidence.yunet_faces else ReasonCode.SUBJECT_NOT_OWNED
        return _not_applicable(reason)
    # Never silently discard extra detector faces outside the explicit subject
    # regions.  A valid foreground assignment does not make background people
    # safe: crop/landmark results could otherwise be attributed to the subject.
    if len(assignments) != len(evidence.yunet_faces):
        return _unknown(ReasonCode.BACKGROUND_FACE_ONLY)
    if len(set(assignments.values())) != len(evidence.subject_boxes):
        return _unknown(ReasonCode.MULTIPLE_SUBJECTS)
    accepted = tuple(sorted(assignments))
    faces = {face.face_id: face for face in evidence.yunet_faces if face.face_id in assignments}

    # Original-image size evidence cannot be replaced by crop magnification.
    for face in faces.values():
        if face.bbox.width < t.min_face_width_px:
            return _unknown(ReasonCode.SMALL_FACE)
        if face.iod_px is None:
            return _unknown(ReasonCode.FEATURE_MISSING)
        if face.iod_px < t.min_iod_px:
            return _unknown(ReasonCode.SMALL_FACE)

    # Exactly one mapped result per accepted face and crop context is required.
    expected = {(face_id, context) for face_id in accepted for context in CropContext}
    grouped: dict[tuple[str, CropContext], list[MappedMPFace]] = {}
    for detected in evidence.mp_faces:
        grouped.setdefault((detected.yunet_face_id, detected.crop_context), []).append(detected)
    if set(grouped) != expected or any(len(items) != 1 for items in grouped.values()):
        return _unknown(ReasonCode.FACE_COUNT_MISMATCH)

    for face_id in accepted:
        yunet = faces[face_id]
        crop_faces = [grouped[(face_id, context)][0] for context in CropContext]
        if any(not _boxes_consistent(yunet.bbox, item.bbox, t) for item in crop_faces):
            return _unknown(ReasonCode.FACE_BOX_MISMATCH)
        if not _boxes_consistent(crop_faces[0].bbox, crop_faces[1].bbox, t):
            return _unknown(ReasonCode.CROP_CONTEXT_CONFLICT)
        if crop_faces[0].observation != crop_faces[1].observation:
            return _unknown(ReasonCode.CROP_CONTEXT_CONFLICT)
        if any(item.iod_px is None for item in crop_faces):
            return _unknown(ReasonCode.FEATURE_MISSING)
        if any(item.iod_px < t.min_iod_px for item in crop_faces if item.iod_px is not None):
            return _unknown(ReasonCode.SMALL_FACE)
        if any(item.profile or item.occluded for item in crop_faces):
            return _unknown(ReasonCode.PROFILE_OR_OCCLUDED)
        if any(not item.eyes_visible for item in crop_faces):
            return _unknown(ReasonCode.EYE_STATE_UNKNOWN)
        if crop_faces[0].observation in {EyeObservation.MAYBE, EyeObservation.UNKNOWN}:
            return _unknown(ReasonCode.EYE_STATE_UNKNOWN)

    return OwnershipResult(GateApplicability.APPLICABLE, (), accepted)


def _validate_evidence(evidence: OwnershipEvidence, t: GateThresholds) -> None:
    if evidence.image_width <= 0 or evidence.image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if t.max_faces <= 0 or not 0 <= t.min_subject_overlap <= 1 or not 0 <= t.min_detector_iou <= 1:
        raise ValueError("invalid gate thresholds")
    face_ids = [face.face_id for face in evidence.yunet_faces]
    subject_ids = [subject.subject_id for subject in evidence.subject_boxes or ()]
    if len(face_ids) != len(set(face_ids)) or len(subject_ids) != len(set(subject_ids)):
        raise ValueError("face and subject ids must be unique")


def _belongs(face: Box, subject: Box, t: GateThresholds) -> bool:
    center_x, center_y = face.x + face.width / 2, face.y + face.height / 2
    center_inside = subject.x <= center_x <= subject.x + subject.width and subject.y <= center_y <= subject.y + subject.height
    return center_inside and _intersection_area(face, subject) / (face.width * face.height) >= t.min_subject_overlap


def _boxes_consistent(left: Box, right: Box, t: GateThresholds) -> bool:
    if _iou(left, right) >= t.min_detector_iou:
        return True
    left_center = (left.x + left.width / 2, left.y + left.height / 2)
    right_center = (right.x + right.width / 2, right.y + right.height / 2)
    distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
    return distance <= t.max_center_distance_ratio * max(left.width, left.height, right.width, right.height)


def _intersection_area(left: Box, right: Box) -> float:
    width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    height = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    return width * height


def _iou(left: Box, right: Box) -> float:
    intersection = _intersection_area(left, right)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def _unknown(reason: ReasonCode) -> OwnershipResult:
    return OwnershipResult(GateApplicability.UNKNOWN, (reason,))


def _not_applicable(reason: ReasonCode) -> OwnershipResult:
    return OwnershipResult(GateApplicability.NOT_APPLICABLE, (reason,))
