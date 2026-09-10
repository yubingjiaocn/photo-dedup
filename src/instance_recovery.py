"""Local, review-only cross-frame observation proposals. Never keeper authority.

A transported box is a search window, not a detection or an identity. Target
confirmation and appearance-unique correspondence are separate requirements.
"""
from __future__ import annotations

import json
import math
import numpy as np

from .local_quality import _iou

PRODUCER = 'native_instance_recovery_v0'


def _failure(reason, **evidence):
    return {'eligible': False, 'reason': reason, **evidence}


def track_box(source, target, box):
    """Native grayscale LK forward/back + robust affine, without pixel upsampling."""
    import cv2
    if source.shape != target.shape or source.ndim != 2:
        return _failure('FRAME_GEOMETRY_CHANGED')
    h, w = source.shape
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    if min(bw, bh) < 96:
        return _failure('NATIVE_REGION_TOO_SMALL')
    mask = np.zeros_like(source)
    # Avoid boundaries where background motion can dominate an instance.
    mask[max(0, int(y1+bh*.12)):min(h, int(y2-bh*.12)),
         max(0, int(x1+bw*.12)):min(w, int(x2-bw*.12))] = 255
    p = cv2.goodFeaturesToTrack(source, maxCorners=160, qualityLevel=.02,
                                minDistance=8, mask=mask, blockSize=7)
    if p is None or len(p) < 12:
        return _failure('LOW_TEXTURE', points=0 if p is None else len(p))
    # A coarse image smaller than the LK support window is not a valid level.
    levels = max(0, min(5, int(math.log2(min(source.shape)/64))))
    kwargs = dict(winSize=(31,31), maxLevel=levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
    q, status, error = cv2.calcOpticalFlowPyrLK(source, target, p, None, **kwargs)
    if q is None:
        return _failure('FLOW_FAILED')
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(target, source, q, None, **kwargs)
    if back is None:
        return _failure('REVERSE_FLOW_FAILED')
    fb = np.linalg.norm(p.reshape(-1,2)-back.reshape(-1,2), axis=1)
    good = (status.ravel()==1) & (reverse.ravel()==1) & (fb <= 2.0) & (error.ravel() <= 25)
    a, b = p.reshape(-1,2)[good], q.reshape(-1,2)[good]
    info = {'points': len(p), 'fb_consistent': len(a), 'fb_threshold_native_px': 2.0}
    if len(a) < 12 or len(a)/len(p) < .5:
        return _failure('FORWARD_BACK_INCONSISTENT', **info)
    matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                               ransacReprojThreshold=3, maxIters=2000)
    if matrix is None or inliers is None:
        return _failure('MOTION_UNRESOLVED', **info)
    ok = inliers.ravel().astype(bool)
    count = int(ok.sum())
    coverage = float(np.prod(np.ptp(a[ok], axis=0))/(bw*bh)) if count else 0.
    scale = float(math.hypot(matrix[0,0], matrix[1,0]))
    info.update(inliers=count, inlier_ratio=count/len(a), spatial_coverage=coverage, scale=scale)
    if count < 12 or count/len(a) < .65 or coverage < .15 or not .8 <= scale <= 1.25:
        return _failure('DRIFT_OR_LOCAL_PATCH_ONLY', **info)
    corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float32)
    transformed = corners @ matrix[:,:2].T + matrix[:,2]
    lo, hi = transformed.min(axis=0), transformed.max(axis=0)
    if lo.min() < 0 or hi[0] > w or hi[1] > h:
        return _failure('TARGET_TRUNCATED', **info)
    return {'eligible': True, 'reason': 'SEARCH_WINDOW_ONLY', 'box': [*lo.tolist(), *hi.tolist()],
            'identity_authority': False, **info}


def sift_box(source, target, box):
    """Contrast-normalized native local keypoints for larger inter-frame changes.

    Reciprocal ratio-test matches and spatially distributed RANSAC support are
    motion evidence only; a separate detector and instance appearance must agree.
    """
    import cv2
    if source.shape != target.shape or source.ndim != 2:
        return _failure('FRAME_GEOMETRY_CHANGED')
    h,w = source.shape
    x1,y1,x2,y2 = box
    bw,bh = x2-x1,y2-y1
    if min(bw,bh) < 96:
        return _failure('NATIVE_REGION_TOO_SMALL')
    roi = [max(0,int(x1-bw*.75)),max(0,int(y1-bh*.5)),
           min(w,int(x2+bw*.75)),min(h,int(y2+bh*.5))]
    source_roi = [max(0,int(x1)),max(0,int(y1)),min(w,int(x2)),min(h,int(y2))]
    a = source[source_roi[1]:source_roi[3],source_roi[0]:source_roi[2]]
    b = target[roi[1]:roi[3],roi[0]:roi[2]]
    mask = np.zeros_like(a)
    ih,iw = a.shape
    mask[int(ih*.1):int(ih*.9),int(iw*.1):int(iw*.9)] = 255
    sift = cv2.SIFT_create(nfeatures=1800)
    ka, da = sift.detectAndCompute(a,mask)
    kb, db = sift.detectAndCompute(b,None)
    if da is None or db is None or min(len(da),len(db)) < 12:
        return _failure('SIFT_LOW_TEXTURE')
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(da,db,k=2)
    reverse = matcher.knnMatch(db,da,k=2)
    back = {m.queryIdx:m.trainIdx for pair in reverse if len(pair)==2
            for m,n in [pair] if m.distance < .75*n.distance}
    matches = [m for pair in forward if len(pair)==2 for m,n in [pair]
               if m.distance < .75*n.distance and back.get(m.trainIdx)==m.queryIdx]
    if len(matches)<12:
        return _failure('SIFT_NO_RECIPROCAL_SUPPORT', reciprocal_matches=len(matches))
    p = np.float32([ka[m.queryIdx].pt for m in matches]) + source_roi[:2]
    q = np.float32([kb[m.trainIdx].pt for m in matches]) + roi[:2]
    matrix, inliers = cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000)
    if matrix is None or inliers is None:
        return _failure('SIFT_MOTION_UNRESOLVED')
    ok = inliers.ravel().astype(bool)
    count = int(ok.sum())
    coverage = float(np.prod(np.ptp(p[ok],axis=0))/(bw*bh)) if count else 0.
    scale = float(math.hypot(matrix[0,0],matrix[1,0]))
    info = {'reciprocal_matches':len(matches),'inliers':count,'inlier_ratio':count/len(matches),
            'spatial_coverage':coverage,'scale':scale,'method':'native_sift_reciprocal_ransac'}
    if count < 12 or count/len(matches)<.65 or coverage<.15 or not .8<=scale<=1.25:
        return _failure('SIFT_DRIFT_OR_LOCAL_PATCH_ONLY',**info)
    corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    transformed = corners @ matrix[:,:2].T + matrix[:,2]
    lo,hi = transformed.min(axis=0),transformed.max(axis=0)
    if lo.min()<0 or hi[0]>w or hi[1]>h:
        return _failure('TARGET_TRUNCATED',**info)
    return {'eligible':True,'reason':'SEARCH_WINDOW_ONLY','box':[*lo.tolist(),*hi.tolist()],
            'identity_authority':False,**info}


def confirm_target(predicted_box, class_id, detections):
    """Reciprocal association is still required after this independent detection."""
    choices = sorted([(_iou(predicted_box, d['box']), i) for i,d in enumerate(detections)
                      if d['class_id'] == class_id and d['confidence'] >= .6], reverse=True)
    if not choices or choices[0][0] < .5:
        return _failure('TARGET_NOT_CONFIRMED')
    if len(choices)>1 and choices[0][0]-choices[1][0]<.15:
        return _failure('TARGET_DETECTION_AMBIGUOUS')
    overlap, index = choices[0]
    if min(detections[index]['box'][2]-detections[index]['box'][0],
           detections[index]['box'][3]-detections[index]['box'][1]) < 96:
        return _failure('TARGET_NATIVE_REGION_TOO_SMALL')
    return {'eligible': True, 'target_index': index, 'iou': overlap,
            'reason': 'INDEPENDENT_TARGET_DETECTION'}


def novel_detection(candidate, catalog):
    """Overlapping cross-class/partial boxes are not recovered new instances."""
    box = candidate['box']
    area = (box[2]-box[0])*(box[3]-box[1])
    if area <= 0:
        return False
    for old in catalog:
        b = old['box']
        other = (b[2]-b[0])*(b[3]-b[1])
        intersection = max(0,min(box[2],b[2])-max(box[0],b[0])) * max(0,min(box[3],b[3])-max(box[1],b[1]))
        if _iou(box,b) >= .5 or (other>0 and intersection/min(area,other)>=.6):
            return False
    return True


def pose_pair_support(source, target, a, b):
    """Independent cached native COCO17 detections, not transported boxes.

    Only rigid torso anchors gate geometry; limb action is not required to stay
    fixed. Identity is separately tested against every same-class appearance.
    """
    import cv2
    if source.shape != target.shape or a['class_id'] != 0 or b['class_id'] != 0:
        return _failure('POSE_PAIR_UNSUPPORTED')
    pa,pb = a.get('pose'),b.get('pose')
    if pa is None or pb is None:
        return _failure('INDEPENDENT_POSE_NOT_CONFIRMED')
    if min(pa['box_confidence'],pb['box_confidence']) < .6:
        return _failure('INDEPENDENT_POSE_LOW_CONFIDENCE')
    if any(sum(p[2]>=.5 for p in rec['keypoints'])<8 for rec in [pa,pb]):
        return _failure('INSUFFICIENT_POSE_ANCHORS')
    torso = [i for i in [5,6,11,12] if pa['keypoints'][i][2]>=.5 and pb['keypoints'][i][2]>=.5]
    if len(torso)<3 or _iou(a['box'],b['box'])<.5:
        return _failure('POSE_GEOMETRY_UNCONFIRMED')
    h,w = source.shape
    centers=[]
    textures=[]
    for image,obj,rec in [(source,a,pa),(target,b,pb)]:
        x1,y1,x2,y2=obj['box']
        if min(x2-x1,y2-y1)<96 or x1<=1 or y1<=1 or x2>=w-1 or y2>=h-1:
            return _failure('POSE_NATIVE_REGION_UNASSESSABLE')
        anchors=np.array([[rec['keypoints'][i][0]*w,rec['keypoints'][i][1]*h] for i in torso])
        centers.append(anchors.mean(axis=0))
        # Restrict texture support to the detected torso triangle/quad. A screen
        # behind the actor must not satisfy the native texture requirement.
        mask=np.zeros_like(image)
        hull=cv2.convexHull(anchors.astype(np.int32))
        cv2.fillConvexPoly(mask,hull,255)
        points=cv2.goodFeaturesToTrack(image,80,.02,5,mask=mask,blockSize=5)
        count=0 if points is None else len(points)
        textures.append(count)
        if count<12:
            return _failure('POSE_TORSO_LOW_TEXTURE',texture_points=textures)
    shift=float(np.linalg.norm((centers[1]-centers[0])/np.array([w,h])))
    if shift>.15:
        return _failure('POSE_CENTER_DRIFT',normalized_shift=shift)
    return {'eligible':True,'reason':'INDEPENDENT_POSE_PAIR_GEOMETRY',
            'method':'cached_native_pose_anchor', 'identity_authority':False,
            'torso_anchors':torso,'texture_points':textures,'normalized_shift':shift,
            'box':b['box']}


def appearance_unique(source_index, target_index, source_objects, target_objects,
                      source_vectors, target_vectors):
    """No position shortcut: unique reciprocal local appearance across all peers."""
    similarity = source_vectors @ target_vectors.T
    cls = source_objects[source_index]['class_id']
    allowed = np.array([[a['class_id'] == b['class_id'] for b in target_objects]
                        for a in source_objects])
    similarity = np.where(allowed, similarity, -2.)
    value = float(similarity[source_index,target_index])
    alternatives_row = np.delete(similarity[source_index], target_index)
    alternatives_col = np.delete(similarity[:,target_index], source_index)
    margin = min(value-float(alternatives_row.max()) if alternatives_row.size else 2.,
                 value-float(alternatives_col.max()) if alternatives_col.size else 2.)
    good = (cls == target_objects[target_index]['class_id'] and np.isfinite(value)
            and value >= .9 and margin >= .03)
    return {'eligible': bool(good), 'similarity': value, 'reciprocal_margin': margin,
            'reason': 'APPEARANCE_UNIQUE' if good else 'IDENTITY_UNCONFIRMED'}


def stable_tracks(catalogs, edges):
    """All reciprocal/cycle-consistent tracks, without claiming a complete set.

    No largest-instance priority or dropping awkward detections to claim complete
    coverage. Unmatched observations remain explicitly outside the stable subset.
    """
    tracks = []
    if len(catalogs)<2:
        return tracks
    for anchor in range(len(catalogs[0])):
        track = [anchor] + [edges.get((0,t,anchor)) for t in range(1,len(catalogs))]
        if any(type(i) is not int or not 0<=i<len(catalogs[t]) for t,i in enumerate(track)):
            continue
        if all(edges.get((s,t,track[s]))==track[t] for s in range(len(track))
               for t in range(len(track)) if s!=t):
            tracks.append(track)
    # An ambiguous many-to-one component is not a stable pair of identities.
    return [track for track in tracks if all(sum(other[t]==track[t] for other in tracks)==1
                                              for t in range(len(track)))]


def _legacy_review_context(members):
    """Read only producer proposals bound to the exact group; fail closed.

    This consumer never edits member evidence, keeper selection, scores or order.
    The full proposal packet is deliberately not interpreted as feature truth.
    """
    ids = sorted(m.get('id') for m in members if type(m.get('id')) is int)
    base = {'policy': 'review_only', 'producer': PRODUCER, 'keeper_authority': False,
            'semantic_subject_set_authority': False, 'proposed_groups': [], 'refusals': []}
    if len(ids) != len(members) or len(set(ids)) != len(ids):
        base['refusals'] = ['GROUP_BINDING_UNAVAILABLE']
        return base
    records = []
    for member in members:
        try:
            r = json.loads(member.get('quality_meta') or '{}').get('instance_recovery')
        except (ValueError, TypeError, AttributeError):
            r = None
        if r is not None:
            records.append(r)
    if not records:
        base['refusals'] = ['NO_RECOVERY_PROPOSALS']
        return base
    if len(records) != len(members):
        base['refusals'] = ['INCOMPLETE_GROUP_PACKET']
        return base
    first = records[0]
    if (not isinstance(first, dict) or any(r != first for r in records)
            or first.get('producer') != PRODUCER or first.get('schema_version') != 1
            or first.get('group_member_ids') != ids or first.get('review_only') is not True
            or first.get('keeper_authority') is not False
            or type(first.get('proposals')) is not list):
        base['refusals'] = ['INVALID_GROUP_PACKET']
        return base
    proposals = first['proposals']
    valid = []
    for p in proposals:
        if not isinstance(p, dict):
            continue
        box = p.get('box_normalized')
        if (p.get('source_file_id') in ids and p.get('target_file_id') in ids
                and p.get('source_file_id') != p.get('target_file_id')
                and p.get('status') == 'confirmed_review_proposal'
                and p.get('target_confirmation') is True and p.get('appearance_unique') is True
                and p.get('motion_consistent') is True and p.get('novelty_checked') is True
                and type(p.get('source_file_id')) is int and type(p.get('target_file_id')) is int
                and type(p.get('class_id')) is int and p.get('class_id') in (0,15,16)
                and isinstance(box,list) and len(box)==4
                and all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1 for v in box)
                and box[0]<box[2] and box[1]<box[3]):
            valid.append(p)
    if valid:
        base['proposed_groups'] = [{'group_member_ids': ids, 'proposals': valid,
                                   'stable_observed_set': first.get('stable_observed_set') is True,
                                   'review_required': True}]
    else:
        base['refusals'] = ['NO_CONFIRMED_UNIQUE_PROPOSALS']
    return base


def review_context(members):
    """Optional independent producers share review flags, never keeper authority."""
    base = _legacy_review_context(members)
    from .dense_recovery_review import context
    dense = context(members)
    if dense is not None:
        base['dense_visible_region_context'] = dense
        if dense['proposals']:
            base['legacy_refusals'] = base['refusals']
            base['refusals'] = []
            base['proposed_groups'].append(dense)
    return base
