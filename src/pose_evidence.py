"""Conservative COCO-17 pose comparison; observations, not semantic labels.

No image/model loading, identity naming, external phase labels or deletion
policy lives here. Missing or ambiguous body association is an abstention.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

CORE = frozenset({5, 6, 11, 12})
BODY = frozenset(range(5, 17))
LIMBS = BODY - CORE


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _candidates(record):
    result = []
    people = record.get('people', [])
    if not isinstance(people, list):
        return result
    for index, person in enumerate(people):
        if not isinstance(person, Mapping):
            continue
        box, points = person.get('box'), person.get('keypoints')
        confidence = person.get('box_confidence')
        if not _finite(confidence) or not 0.5 <= confidence <= 1:
            continue
        if (not isinstance(box, list) or len(box) != 4
                or not all(_finite(x) and 0 <= x <= 1 for x in box)
                or box[2] <= box[0] or box[3] <= box[1]
                or not isinstance(points, list) or len(points) != 17):
            continue
        usable = {i for i in BODY if isinstance(points[i], list) and len(points[i]) == 3
                  and all(_finite(x) for x in points[i])
                  and 0.6 <= points[i][2] <= 1
                  and all(0 <= x <= 1 for x in points[i][:2])}
        width, height = box[2] - box[0], box[3] - box[1]
        result.append({'index': index, 'points': points, 'usable': usable,
                       'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                       'width': width, 'height': height, 'area': width * height})
    return result


def _normalize(person, shape):
    width, height = shape
    points = {i: (person['points'][i][0] * width, person['points'][i][1] * height)
              for i in person['usable']}
    center = [sum(points[i][k] for i in CORE) / 4 for k in (0, 1)]
    shoulder = [(points[5][k] + points[6][k]) / 2 for k in (0, 1)]
    hip = [(points[11][k] + points[12][k]) / 2 for k in (0, 1)]
    axis = [hip[k] - shoulder[k] for k in (0, 1)]
    scale = math.hypot(*axis)
    diagonal = math.hypot(width, height)
    if scale < max(0.01 * diagonal, 0.05 * math.hypot(person['width'] * width, person['height'] * height)):
        return None
    vertical = [v / scale for v in axis]
    horizontal = [vertical[1], -vertical[0]]
    normalized = {i: [sum((points[i][k] - center[k]) * basis[k] for k in (0, 1)) / scale
                      for basis in (horizontal, vertical)] for i in person['usable']}
    return normalized, scale / diagonal


def compare_poses(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None,
                  *, participant_policy: str = 'consensus') -> dict[str, Any]:
    """Measure matched limb displacement in torso lengths, or return unknown.

    Crowds require either two independently matched moving bodies or a genuinely
    large, uniquely dominant body. A background detector changing alone is not
    sufficient. Displacement does not claim the user's preferred phase grain.
    """
    if participant_policy not in {'consensus', 'visible_participant'}:
        raise ValueError('unknown pose participant policy')
    unknown = {'eligible': False, 'displacement': None, 'matched_people': 0}
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {**unknown, 'reason': 'POSE_EVIDENCE_MISSING'}
    if left.get('producer') != right.get('producer') or not left.get('producer'):
        return {**unknown, 'reason': 'POSE_PRODUCER_MISMATCH'}
    for record in (left, right):
        shape = record.get('shape')
        if (record.get('status') != 'ok' or not isinstance(shape, list) or len(shape) != 2
                or not all(_finite(x) and x > 0 for x in shape)):
            return {**unknown, 'reason': 'POSE_EVIDENCE_INVALID'}
    aa, bb = _candidates(left), _candidates(right)
    if not aa or not bb:
        return {**unknown, 'reason': 'NO_USABLE_BODY'}
    distances = [[_distance(a['center'], b['center']) for b in bb] for a in aa]
    matches = []
    for i, a in enumerate(aa):
        order = sorted(range(len(bb)), key=lambda j: distances[i][j])
        j = order[0]
        b, distance = bb[j], distances[i][j]
        reverse = sorted(range(len(aa)), key=lambda k: distances[k][j])
        if reverse[0] != i or distance > 0.1:
            continue
        if ((len(order) > 1 and distances[i][order[1]] < max(distance + 0.02, 1.5 * distance))
                or (len(reverse) > 1 and distances[reverse[1]][j] < max(distance + 0.02, 1.5 * distance))):
            continue
        if not all(0.67 <= b[key] / a[key] <= 1.5 for key in ['width', 'height']):
            continue
        shared = a['usable'] & b['usable']
        if len(shared) < 8 or not CORE <= shared:
            continue
        na, nb = _normalize(a, left['shape']), _normalize(b, right['shape'])
        if na is None or nb is None or not 0.67 <= nb[1] / na[1] <= 1.5:
            continue
        displacement = sorted((_distance(na[0][k], nb[0][k]) for k in shared & LIMBS), reverse=True)[2]
        forearm_motions = []
        for elbow, wrist in ((7, 9), (8, 10)):
            if not {elbow, wrist} <= shared:
                continue
            if min(a['points'][elbow][2], a['points'][wrist][2],
                   b['points'][elbow][2], b['points'][wrist][2]) < 0.8:
                continue
            va = [na[0][wrist][k] - na[0][elbow][k] for k in (0, 1)]
            vb = [nb[0][wrist][k] - nb[0][elbow][k] for k in (0, 1)]
            la, lb = math.hypot(*va), math.hypot(*vb)
            if not (0.15 <= la <= 1.2 and 0.15 <= lb <= 1.2 and 0.5 <= lb / la <= 1.5):
                continue
            cosine = sum(x * y for x, y in zip(va, vb)) / (la * lb)
            if cosine <= math.cos(math.pi / 4):
                forearm_motions.append(_distance(na[0][wrist], nb[0][wrist]))

        def dominant(person, people):
            return person['area'] >= 0.15 and all(
                person is other or person['area'] >= 2 * other['area'] for other in people)

        matches.append({'displacement': displacement,
                        'forearm_displacement': max(forearm_motions, default=0.0),
                        'dominant': dominant(a, aa) and dominant(b, bb),
                        'well_resolved': min(a['area'], b['area']) >= 0.06
                        and min(a['height'], b['height']) >= 0.30,
                        'shared_joints': len(shared)})
    if not matches:
        return {**unknown, 'reason': 'BODY_ASSOCIATION_ABSTAIN'}
    scores = sorted((m['displacement'] for m in matches), reverse=True)
    if max(len(aa), len(bb)) < 2:
        options = scores
    else:
        options = [m['displacement'] for m in matches if m['dominant']]
        if len(scores) >= 2:
            options.append(scores[1])
    if participant_policy == 'visible_participant':
        # One reliably matched participant can change a joint interaction.
        # A large forearm rotation plus wrist displacement is localized action
        # evidence; it does not require three unrelated joints to move.
        options.extend(max(m['displacement'], m['forearm_displacement'])
                       for m in matches if m['well_resolved'])
    if not options:
        return {**unknown, 'matched_people': len(matches), 'reason': 'CROWD_SUPPORT_ABSTAIN'}
    return {'eligible': True, 'displacement': max(options), 'matched_people': len(matches),
            'reason': 'MATCHED_LIMB_DISPLACEMENT', 'units': 'torso_lengths',
            'semantic_phase_authority': False}


def protect_pose_variants(
    members: Sequence[Mapping[str, Any]], keepers: Sequence[int], scores: Sequence[float] | Mapping[int, float],
    *, displacement_threshold: float, participant_policy: str = 'consensus',
) -> tuple[list[int], dict[str, Any]]:
    """Propose at most one clearly contrasting pose in a small nonempty group.

    Existing keepers are never removed or reranked. A new representative must
    have eligible, above-threshold comparisons with EVERY existing keeper;
    unknown association is not evidence of a new state. This bounded opt-in
    recall/retention tradeoff cannot authorize deletion or label a human phase.
    """
    if not _finite(displacement_threshold) or displacement_threshold <= 0:
        raise ValueError('pose displacement threshold must be positive and finite')
    try:
        values = [scores[i] for i in range(len(members))]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('one finite utility score per member is required') from exc
    if len(scores) != len(members) or any(not _finite(s) for s in values):
        raise ValueError('one finite utility score per member is required')
    if (not keepers or len(set(keepers)) != len(keepers)
            or any(type(k) is not int or not 0 <= k < len(members) for k in keepers)):
        raise ValueError('keepers must be nonempty, unique, valid member indices')
    result = list(keepers)
    evidence = {'added_keeper': None, 'comparisons': [], 'policy': 'bounded_pose_variant',
                'displacement_threshold': displacement_threshold, 'max_extra_keepers': 1,
                'participant_policy': participant_policy,
                'semantic_phase_authority': False}
    if not 2 <= len(members) <= 6 or len(keepers) == len(members):
        return result, evidence

    def record(member):
        try:
            meta = json.loads(member.get('quality_meta') or '{}')
            return meta.get('pose_evidence') if isinstance(meta, dict) else None
        except (TypeError, ValueError):
            return None

    records = [record(m) for m in members]
    candidates = []
    comparisons = {}
    for index in range(len(members)):
        if index in keepers:
            continue
        paired = [{'keeper': k, **compare_poses(records[index], records[k], participant_policy=participant_policy)} for k in keepers]
        comparisons[index] = paired
        if all(p['eligible'] and p['displacement'] >= displacement_threshold for p in paired):
            candidates.append(index)
    if candidates:
        chosen = max(candidates, key=lambda i: (values[i], -i))
        result.append(chosen)
        evidence.update(added_keeper=chosen, comparisons=comparisons[chosen])
    return result, evidence
