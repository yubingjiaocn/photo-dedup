"""Optional local perceptual-quality dominance; no eye/completeness semantics."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DIM = 768
STRIDE = DIM * 2


def _finite(value: Any, low: float, high: float) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and low <= value <= high)


def observations(member: Mapping[str, Any]) -> dict[tuple[str, int], dict]:
    """Malformed, ambiguous or unavailable native evidence abstains."""
    try:
        meta = json.loads(member.get('quality_meta') or '{}')['local_quality']
        blob = member.get('local_quality_embedding')
        if (meta.get('schema_version') != 1 or meta.get('method') != 'native_coco_crop_v1'
                or not isinstance(blob, (bytes, bytearray, memoryview))
                or not 0 < len(blob) <= 16 * STRIDE):
            return {}
        regions = meta['regions']
        if not isinstance(regions, list) or len(regions) > 16:
            return {}
        result = {}
        for r in regions:
            key = (r['kind'], r['class_id'])
            box, offset = r['box_normalized'], r['embedding_offset']
            if (key in result or key[0] not in {'subject', 'face'} or type(key[1]) is not int
                    or len(box) != 4 or not all(_finite(x, 0, 1) for x in box)
                    or not box[0] < box[2] or not box[1] < box[3]
                    or type(offset) is not int or offset < 0 or offset % STRIDE
                    or offset + STRIDE > len(blob)
                    or not _finite(r['confidence'], 0.6, 1)
                    or not _finite(r['musiq'], 0, 100) or not _finite(r['clipiqa'], 0, 1)
                    or len(r['native_size']) != 2
                    or not all(_finite(x, 96, 100000) for x in r['native_size'])):
                return {}
            vector = np.frombuffer(blob, dtype='<f2', count=DIM, offset=offset).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if not np.all(np.isfinite(vector)) or not math.isfinite(norm) or norm <= 0:
                return {}
            result[key] = {**r, 'vector': vector / norm}
        return result
    except (KeyError, ValueError, TypeError, OverflowError, AttributeError, BufferError):
        return {}


def _iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def adjust_scores(members: Sequence[Mapping[str, Any]], phases, scores: Mapping[int, float],
                  *, similarity: float = 0.9, penalty: float = 0.08):
    """Bounded penalty for jointly worse native local IQA in a matched view.

    Association uses same detected class, aligned frame box and local DINO.
    Both MUSIQ and CLIP-IQA must improve; no missing-component renormalization.
    Face comparisons also require a matched enclosing person crop. This never
    changes group/phase budgets, deletes frames, or establishes semantic truth.
    """
    if not _finite(similarity, 0, 1) or not _finite(penalty, 0, 0.25):
        raise ValueError('local quality similarity/penalty out of range')
    result = dict(scores)
    obs = [observations(m) for m in members]
    comparisons = []
    # Logical phase boundaries are weak detector/time hypotheses, not semantic
    # truth. Matched native views remain comparable across such boundaries;
    # the supplied phase structure and its budgets are never modified here.
    population = range(len(members))
    for worse in population:
        for better in population:
            if worse == better:
                continue
            for key in obs[worse].keys() & obs[better].keys():
                a, b = obs[worse][key], obs[better][key]
                if _iou(a['box_normalized'], b['box_normalized']) < 0.4:
                    continue
                appearance = float(np.dot(a['vector'], b['vector']))
                if key[0] == 'face':
                    parent = ('subject', key[1])
                    if parent not in obs[worse] or parent not in obs[better]:
                        continue
                    pa, pb = obs[worse][parent], obs[better][parent]
                    appearance = float(np.dot(pa['vector'], pb['vector']))
                    if _iou(pa['box_normalized'], pb['box_normalized']) < 0.4:
                        continue
                mg, cg = b['musiq']-a['musiq'], b['clipiqa']-a['clipiqa']
                if appearance >= similarity and mg >= 1.0 and cg >= 0.05:
                    result[worse] = max(0.0, float(scores[worse])-penalty)
                    comparisons.append({'worse': worse, 'better': better, 'region': key[0],
                                        'class_id': key[1], 'appearance_similarity': appearance,
                                        'musiq_gain': mg, 'clipiqa_gain': cg})
    return result, {'policy': 'native_local_dominance', 'similarity': similarity,
                    'max_utility_penalty': penalty, 'comparisons': comparisons,
                    'changed_members': sorted(i for i in result if result[i] != scores[i]),
                    'eye_state_authority': False, 'amodal_completeness_authority': False}
