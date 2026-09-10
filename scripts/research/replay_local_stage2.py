"""Actual Stage2/optional Stage3; evaluation references opened only afterwards."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import yaml

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src import db, stage2_cluster, stage3_report  # noqa: E402

ROOT=Path('/home/ubuntu/photo-dedup-eval/astra-local-quality-20260910')


def replay(source, config_path, mapping_path, reference_path, reference_variant, out, report, verify_only=False,
           variants_path=ROOT/'frozen-variants.json', variant='native_local_quality'):
    if out.exists() and not verify_only:
        raise ValueError('Existing replay directory; inspect it, or use --verify-only')
    out.mkdir(parents=True,exist_ok=True)
    destination=out/'inventory.sqlite'
    if not verify_only:
        a=sqlite3.connect(f'file:{source}?mode=ro&immutable=1',uri=True)
        b=sqlite3.connect(destination)
        a.backup(b)
        a.close()
        b.close()
    options=json.loads(variants_path.read_text())[variant]
    config=yaml.safe_load(config_path.read_text())
    keys={'local_quality_policy':'keeper_local_quality_policy',
          'local_quality_similarity':'keeper_local_quality_similarity',
          'local_quality_penalty':'keeper_local_quality_penalty',
          'pose_policy':'keeper_pose_policy', 'pose_displacement_threshold':'keeper_pose_displacement_threshold'}
    config['cluster'].update({keys[k]:v for k,v in options.items()})
    config['paths'].update(db=str(destination),output_dir=str(out),trash=str(out/'unused-trash'))
    config['execute']['dry_run']=True
    cfg=out/'config.yaml'
    if verify_only:
        existing=yaml.safe_load(cfg.read_text())
        assert existing==config
        stats,rendered=None,None
        if report:
            assert (out/'review.html').is_file()
    else:
        cfg.write_text(yaml.safe_dump(config))
        stats=stage2_cluster.run(str(cfg))
        rendered=stage3_report.run(str(cfg)) if report else None
    mapping=json.loads(mapping_path.read_text())
    reference={g['group_alias']:g for g in json.loads(reference_path.read_text())['rows']}
    c=sqlite3.connect(f'file:{destination}?mode=ro',uri=True)
    c.row_factory=sqlite3.Row
    if stats is None:
        stats=json.loads(db.get_meta(c,'stage2_stats','{}'))
    actual={}
    for g in c.execute('SELECT id,group_type,decision_state,decision_json FROM groups'):
        members=list(c.execute('SELECT file_id,is_keep FROM group_members WHERE group_id=?',(g['id'],)))
        actual[frozenset(x['file_id'] for x in members)]=(g,members)
    assert sum(g['group_type']!='sha_exact' for g,_ in actual.values())==len(mapping)
    evidence_groups=0
    pose_groups=0
    for g in mapping:
        aliases={f['id']:f['alias'] for f in g['frames']}
        saved,members=actual[frozenset(aliases)]
        assert saved['group_type']==g['group_type']
        assert saved['decision_state']=='REVIEW_REQUIRED'
        selected={aliases[m['file_id']] for m in members if m['is_keep']}
        assert selected==set(reference[g['group_alias']]['variants'][reference_variant]['keepers'])
        decision=json.loads(saved['decision_json'])
        context=decision['phase_selection']['local_quality_context']
        assert not context['eye_state_authority'] and not context['amodal_completeness_authority']
        evidence_groups+=bool(context['changed_members'])
        pose=decision['phase_selection'].get('pose_coverage') or {}
        pose_groups+=pose.get('added_keeper') is not None
    unsafe=c.execute("SELECT COUNT(*) FROM group_members gm JOIN groups g ON gm.group_id=g.id JOIN features f ON gm.file_id=f.file_id JOIN features k ON g.keep_file_id=k.file_id WHERE gm.decision='AUTO_REMOVE' AND (g.group_type!='sha_exact' OR f.content_sha256 IS NULL OR f.content_sha256!=k.content_sha256)").fetchone()[0]
    assert unsafe==0
    c.close()
    result={'verified_groups':len(mapping),'actual_stage2':True,'keeper_sets_match':True,
            'group_membership_unchanged':True,'visual_groups_review_required':True,
            'local_evidence_persisted_groups':evidence_groups,'pose_addition_groups':pose_groups,
            'unsafe_auto_remove':unsafe,
            'stage4_executed':False,'runtime_label_input':False,'config':str(cfg),
            'stats':stats,'report':rendered}
    (out/'replay-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['db','config','mapping','reference','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reference-variant',required=True)
    p.add_argument('--report',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--variants',type=Path,default=ROOT/'frozen-variants.json')
    p.add_argument('--variant',default='native_local_quality')
    a=p.parse_args()
    replay(a.db,a.config,a.mapping,a.reference,a.reference_variant,a.output,a.report,a.verify_only,a.variants,a.variant)
