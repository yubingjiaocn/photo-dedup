"""Read bounded, local visible-region proposals; never keeper/identity authority."""
import json
import math

PRODUCER='native_dense_visible_region_v1'


def number(value,low,high):
    return type(value) in (int,float) and math.isfinite(value) and low<=value<=high


def box_valid(box):
    return (isinstance(box,list) and len(box)==4 and all(number(v,0,1) for v in box)
            and box[0]<box[2] and box[1]<box[3])


def context(members):
    records=[]
    for member in members:
        try:
            records.append(json.loads(member.get('quality_meta') or '{}').get('dense_instance_recovery'))
        except (ValueError,TypeError,AttributeError):
            records.append(None)
    if all(r is None for r in records):
        return None
    result={'producer':PRODUCER,'proposals':[],'refusals':[],
            'identity_scope':'visible_region_only','keeper_authority':False,
            'stable_observed_set':False,'stable_observed_set_status':'not_established',
            'review_required':True}
    ids=sorted(m['id'] for m in members if type(m.get('id')) is int)
    first=records[0] if records else None
    if (len(ids)!=len(members) or len(set(ids))!=len(ids) or not isinstance(first,dict)
            or any(r!=first for r in records) or first.get('producer')!=PRODUCER
            or first.get('schema_version')!=1 or first.get('group_member_ids')!=ids
            or first.get('review_only') is not True or first.get('keeper_authority') is not False
            or not isinstance(first.get('proposals'),list) or len(first['proposals'])>64):
        result['refusals']=['INVALID_DENSE_GROUP_PACKET']
        return result
    result['group_member_ids']=ids
    for p in first['proposals']:
        if not isinstance(p,dict):
            continue
        counts=[p.get(k) for k in ['matches','inliers','source_patches','target_patches']]
        valid=(all(type(v) is int for v in counts)
               and 12<=counts[1]<=counts[0]<=min(counts[2:]) and max(counts[2:])<=512)
        if not valid:
            continue
        if (type(p.get('source_file_id')) is not int or type(p.get('target_file_id')) is not int
                or p['source_file_id'] not in ids or p['target_file_id'] not in ids
                or p['source_file_id']==p['target_file_id']):
            continue
        if (p.get('identity_scope')!='visible_region_only' or p.get('whole_body_completeness')!='unknown'
                or p.get('semantic_identity_authority') is not False
                or p.get('source_detector_confirmed') is not True or p.get('target_detector_confirmed') is not True
                or p.get('object_unique') is not True or p.get('novelty_checked') is not True
                or p.get('native_no_resize') is not True or p.get('native_patch_size')!=14):
            continue
        if any(type(p.get(k)) is not int or p[k] not in (0,77) for k in ['source_detector_class','target_detector_class']):
            continue
        if not all(box_valid(p.get(k)) for k in ['source_box_normalized','target_box_normalized']):
            continue
        coverage=p.get('coverage')
        if (not isinstance(coverage,list) or len(coverage)!=2 or not all(number(v,.15,1) for v in coverage)
                or not number(p.get('source_confidence'),.6,1) or not number(p.get('target_confidence'),.6,1)
                or not number(p.get('median_similarity'),.9,1) or not number(p.get('scale'),.8,1.25)
                or counts[1]/counts[0]<.65 or counts[1]/min(counts[2:])<.08):
            continue
        result['proposals'].append(p)
    if not result['proposals']:
        result['refusals']=['NO_VALID_DENSE_VISIBLE_REGION_PROPOSALS']
    return result
