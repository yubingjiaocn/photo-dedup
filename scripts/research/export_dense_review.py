"""Export valid visible-region review packets and self-contained local proof."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys

from PIL import Image,ImageDraw,ImageOps

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src.dense_recovery_review import PRODUCER,context


def picture(image):
    stream=io.BytesIO();image.save(stream,format='JPEG',quality=91)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Export exists; inspect rather than overwrite')
    a.output.mkdir(parents=True)
    queue=[];blocks=[]
    for path in sorted(a.source.glob('*/result.json')):
        r=json.loads(path.read_text());proposals=[]
        if not r['new_observation_pair_proposals']:
            continue
        frames=r['group']['frames']
        for pair in r['new_observation_pair_proposals']:
            s,t=pair['source_frame'],pair['target_frame'];i,j=pair['source_index'],pair['target_index']
            left,right=r['objects'][s][i],r['objects'][t][j]
            images=[]
            for f in [s,t]:
                with Image.open(frames[f]['path']) as raw:
                    images.append(ImageOps.exif_transpose(raw).convert('RGB'))
            proposal={'identity_scope':'visible_region_only','whole_body_completeness':'unknown',
                      'semantic_identity_authority':False,'source_file_id':frames[s]['id'],'target_file_id':frames[t]['id'],
                      'source_frame':frames[s]['alias'],'target_frame':frames[t]['alias'],
                      'source_detector_class':left['class_id'],'target_detector_class':right['class_id'],
                      'source_confidence':left['confidence'],'target_confidence':right['confidence'],
                      'source_detector_confirmed':True,'target_detector_confirmed':True,'object_unique':True,
                      'novelty_checked':bool(pair['source_new_vs_v0'] or pair['target_new_vs_v0']),
                      'native_no_resize':True,'native_patch_size':14,
                      'source_crop_edge':left['crop_edge'],'target_crop_edge':right['crop_edge'],
                      'source_new_vs_v0_catalog':pair['source_new_vs_v0'],'target_new_vs_v0_catalog':pair['target_new_vs_v0'],
                      **{k:pair[k] for k in ['matches','inliers','source_patches','target_patches','coverage','scale','median_similarity']}}
            figures=[]
            for side,obj,image,frame_index in zip(['source','target'],[left,right],images,[s,t]):
                w,h=image.size
                proposal[side+'_box_normalized']=[v/(w if k%2==0 else h) for k,v in enumerate(obj['box'])]
                box=tuple(round(v) for v in obj['box']);crop=image.crop(box)
                scale=min(1,600/max(crop.size));crop.thumbnail((600,600));draw=ImageDraw.Draw(crop)
                for x,y in pair[side+'_inlier_centers']:
                    x=(x-box[0])*scale;y=(y-box[1])*scale
                    draw.ellipse((x-2,y-2,x+2,y+2),outline='#00ff80',width=2)
                figures.append(f'<figure><img alt="{frames[frame_index]["alias"]} 原生区域与匹配点" src="{picture(crop)}"><figcaption>{frames[frame_index]["alias"]} · class {obj["class_id"]} · 范围完整度 unknown</figcaption></figure>')
            proposals.append(proposal)
            blocks.append(f'<section><h2>{r["day"]} {r["group_alias"]}：{frames[s]["alias"]} ↔ {frames[t]["alias"]}</h2><div class="pair">'+''.join(figures)+f'</div><p>局部描述子互匹配 {pair["matches"]} 个，几何一致 {pair["inliers"]} 个；median cosine {pair["median_similarity"]:.3f}。绿色点是保留的原生前景 patch 中心，展示最多64个，非人工身份标签。</p><p><b>仅可见区域对应提案：</b>裁切边界/遮挡仍存在，不能认为完整身体、完整演员集合或服装内人类身份得到确认；源组其他帧仍可缺测。不能据此自动补保或删图。</p></section>')
        packet={'schema_version':1,'producer':PRODUCER,'group_member_ids':sorted(f['id'] for f in frames),
                'review_only':True,'keeper_authority':False,'proposals':proposals}
        fake=[{'id':f['id'],'quality_meta':json.dumps({'dense_instance_recovery':packet})} for f in frames]
        validated=context(fake)
        assert len(validated['proposals'])==len(proposals) and not validated['refusals']
        out=a.output/path.parent.name;out.mkdir()
        (out/'result.json').write_text(json.dumps({'day':r['day'],'group_alias':r['group_alias'],'dense_packet':packet},indent=2))
        queue.append({'day':r['day'],'group_alias':r['group_alias'],'packet':packet,'human_review_status':'pending'})
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';img-src data:;style-src 'unsafe-inline'"><title>可见角色区域对应 · review-only</title><style>body{font:16px/1.6 system-ui;background:#111923;color:#e2eaf2;max-width:1100px;margin:30px auto;padding:20px}section{background:#1e2c3c;border-radius:12px;padding:24px;margin:24px 0}.pair{display:flex;flex-wrap:wrap;gap:24px;justify-content:center}figure{margin:0;max-width:45%}img{max-width:100%;height:auto}b{color:#ffd280}figcaption{color:#b8c5d4}</style><h1>原生局部描述子：可见角色区域对应</h1><p>默认关闭、review-only。新提案只提供人工检查信息，keeper集合/顺序/评分/预算均不改变。检测类别77不等于真实“毛绒熊”或人类身份。</p>'''
    document+=''.join(blocks)+'<p>自包含本地文件；无网络依赖、脚本或上传。模型临时视觉评价不是人工真值；完整主体集合仍未确认。</p></html>'
    (a.output/'review.html').write_text(document)
    (a.output/'proposed-groups.json').write_text(json.dumps({'review_only':True,'groups':queue},indent=2))
    print('groups',len(queue),'pairs',sum(len(q['packet']['proposals']) for q in queue),'review',a.output/'review.html')


if __name__=='__main__':
    main()
