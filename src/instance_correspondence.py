"""Experimental foreground-only, class-agnostic observation correspondence.

No keeper or semantic subject authority. COCO labels describe detector output,
not actor identity. Missing observations remain missing; no imputed identities.
"""
from __future__ import annotations

import numpy as np
from .local_quality import _iou


def mask_for(shape, polygon):
    import cv2
    mask=np.zeros(shape[:2],np.uint8)
    points=np.asarray(polygon,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
        return mask
    cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],255)
    return mask


def view_agreement(a,b,shape):
    """Independent detector views on the same native frame, not box propagation."""
    if a.get('crop_edge') or b.get('crop_edge'):
        return {'eligible':False,'reason':'CROP_EDGE_OR_PARTIAL_VIEW'}
    if a.get('class_id') not in (0,77) or b.get('class_id') not in (0,77):
        return {'eligible':False,'reason':'UNSUPPORTED_OBSERVATION_CLASS'}
    overlap=_iou(a['box'],b['box'])
    ma=mask_for(shape,a['polygon'])>0
    mb=mask_for(shape,b['polygon'])>0
    union=np.count_nonzero(ma|mb)
    miou=float(np.count_nonzero(ma&mb)/union) if union else 0.
    area_ratio=min(np.count_nonzero(ma),np.count_nonzero(mb))/max(1,max(np.count_nonzero(ma),np.count_nonzero(mb)))
    eligible=overlap>=.65 and miou>=.65 and area_ratio>=.75
    return {'eligible':bool(eligible),'reason':'INDEPENDENT_VIEW_CONFIRMED' if eligible else 'VIEW_MASK_OR_EXTENT_UNSTABLE',
            'box_iou':overlap,'mask_iou':miou,'mask_area_ratio':float(area_ratio),
            'whole_body_completeness':'unknown'}


def masked_crop(image,observation):
    """Neutralize pixels outside the native segmentation before DINO preprocessing."""
    from PIL import Image
    box=tuple(round(x) for x in observation['box'])
    pixels=np.asarray(image.crop(box)).copy()
    polygon=np.asarray(observation['polygon'])-np.array(box[:2])
    mask=mask_for(pixels.shape,polygon)
    pixels[mask==0]=127
    return Image.fromarray(pixels)


def foreground_keypoints(gray,observation):
    import cv2
    x1,y1,x2,y2=(round(x) for x in observation['box'])
    crop=gray[y1:y2,x1:x2]
    polygon=np.asarray(observation['polygon'])-np.array([x1,y1])
    mask=mask_for(crop.shape,polygon)
    # Erode only a small native border; mask-edge artifacts cannot supply points.
    mask=cv2.erode(mask,np.ones((5,5),np.uint8))
    kp,desc=cv2.SIFT_create(nfeatures=1800).detectAndCompute(crop,mask)
    for point in kp:
        point.pt=(point.pt[0]+x1,point.pt[1]+y1)
    return kp,desc


def foreground_motion(left,right,a,b):
    """Reciprocal native foreground descriptors and distributed robust support."""
    import cv2
    ka,da=left;kb,db=right
    if da is None or db is None or min(len(da),len(db))<12:
        return {'eligible':False,'reason':'FOREGROUND_LOW_TEXTURE'}
    matcher=cv2.BFMatcher()
    forward=matcher.knnMatch(da,db,k=2)
    reverse=matcher.knnMatch(db,da,k=2)
    back={m.queryIdx:m.trainIdx for pair in reverse if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance}
    matches=[m for pair in forward if len(pair)==2 for m,n in [pair]
             if m.distance<.75*n.distance and back.get(m.trainIdx)==m.queryIdx]
    if len(matches)<12:
        return {'eligible':False,'reason':'FOREGROUND_RECIPROCAL_SUPPORT_LOW','matches':len(matches)}
    p=np.float32([ka[m.queryIdx].pt for m in matches]);q=np.float32([kb[m.trainIdx].pt for m in matches])
    matrix,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000)
    if matrix is None or inliers is None:
        return {'eligible':False,'reason':'FOREGROUND_MOTION_UNRESOLVED'}
    valid=inliers.ravel().astype(bool)
    n=int(valid.sum())
    areas=[(x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]) for x in [a,b]]
    coverage=[float(np.prod(np.ptp(points[valid],axis=0))/max(area,1)) if n else 0.
              for points,area in zip([p,q],areas)]
    scale=float(np.hypot(matrix[0,0],matrix[1,0]))
    eligible=n>=12 and n/len(matches)>=.65 and min(coverage)>=.15 and .8<=scale<=1.25
    return {'eligible':bool(eligible),'reason':'NATIVE_FOREGROUND_SUPPORT' if eligible else 'FOREGROUND_DRIFT_OR_LOCAL_PATCH',
            'matches':len(matches),'inliers':n,'inlier_ratio':n/len(matches),'coverage':coverage,'scale':scale,
            'source_inliers_native':p[valid][:64].tolist(),'target_inliers_native':q[valid][:64].tolist(),
            'identity_authority':False}


def match_appearance(a,b,va,vb):
    """Class-agnostic reciprocal uniqueness, including every potential peer."""
    if not len(a) or not len(b):
        return [],[]
    similarity=va@vb.T
    pairs=[];refusals=[]
    for i,x in enumerate(a):
        j=int(np.argmax(similarity[i]))
        value=float(similarity[i,j])
        row=np.delete(similarity[i],j);col=np.delete(similarity[:,j],i)
        margin=min(value-float(row.max()) if row.size else 2.,value-float(col.max()) if col.size else 2.)
        info={'source_index':i,'target_index':j,'similarity':value,'margin':margin,
              'source_detector_class':x['class_id'],'target_detector_class':b[j]['class_id']}
        if value<.9:
            refusals.append({**info,'reason':'FOREGROUND_APPEARANCE_BELOW_FLOOR'})
        elif margin<.03 or int(np.argmax(similarity[:,j]))!=i:
            refusals.append({**info,'reason':'FOREGROUND_APPEARANCE_NOT_UNIQUE'})
        elif _iou(x['box'],b[j]['box'])<.5:
            refusals.append({**info,'reason':'OBSERVATION_GEOMETRY_UNCONFIRMED'})
        elif not x.get('view_confirmed') or not b[j].get('view_confirmed'):
            refusals.append({**info,'reason':'INDEPENDENT_VIEW_UNCONFIRMED'})
        else:
            pairs.append(info)
    return pairs,refusals


def track_components(frame_counts,edges):
    """Missing slots stay None. A track needs every present pair, never transitivity alone."""
    nodes={(f,i) for f,count in enumerate(frame_counts) for i in range(count)}
    adjacent={n:set() for n in nodes}
    edges=set(edges)
    for u,v in edges:
        if u in nodes and v in nodes and (v,u) in edges and u[0]!=v[0]:
            adjacent[u].add(v)
    seen=set();tracks=[]
    for root in sorted(nodes):
        if root in seen:
            continue
        component=set();todo=[root]
        while todo:
            node=todo.pop()
            if node in component:
                continue
            component.add(node);todo.extend(adjacent[node]-component)
        seen|=component
        if len(component)<2 or len({x[0] for x in component})!=len(component):
            continue
        if not all((u,v) in edges for u in component for v in component if u!=v):
            continue
        slots=[None]*len(frame_counts)
        for f,i in component:
            slots[f]=i
        tracks.append({'slots':slots,'all_frames_observed':all(i is not None for i in slots),
                       'missing_frames':[f for f,i in enumerate(slots) if i is None],
                       'semantic_identity':'unknown'})
    return tracks
