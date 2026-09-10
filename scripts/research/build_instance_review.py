"""Self-contained local review carrier; no app/UI changes or external assets."""
import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageOps


def embedded(data):
    return 'data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    root=a.root
    if (root/'review.html').exists():
        raise ValueError('Review carrier exists; inspect before replacing')
    summary=json.loads((root/'candidate/summary.json').read_text())
    replay=json.loads((root/'stage2-3/summary.json').read_text())
    results=[json.loads(p.read_text()) for p in sorted((root/'candidate').glob('*/result.json'))]
    queue=[]
    blocks=[]
    for r in results:
        day,alias=r['day'],r['group_alias']
        key=f'{day}-{alias}'
        panels=[]
        for f,panel in zip(r['group']['frames'],r['panels']):
            url=embedded((root/'candidate'/key/panel).read_bytes())
            panels.append(f'<figure><img alt="{key} {f["alias"]} 全景与候选框" src="{url}"><figcaption>{f["alias"]}</figcaption></figure>')
        if r['proposal_count']:
            packet=r['packet']
            queue.append({'day':day,'group_alias':alias,'group_member_ids':packet['group_member_ids'],
                          'proposals':packet['proposals'],'stable_observed_set':r['stable_observed_set'],
                          'stable_tracks':r['stable_tracks'],'keeper_changed':False,'human_review_status':'pending'})
            crops=[]
            for proposal in packet['proposals']:
                f=next(f for f in r['group']['frames'] if f['id']==proposal['target_file_id'])
                with Image.open(f['path']) as raw:
                    image=ImageOps.exif_transpose(raw).convert('RGB')
                    w,h=image.size
                    box=[v*(w if k%2==0 else h) for k,v in enumerate(proposal['box_normalized'])]
                    image=image.crop(tuple(round(v) for v in box))
                    image.thumbnail((480,600))
                    buffer=io.BytesIO()
                    image.save(buffer,format='JPEG',quality=93)
                crops.append(f'<figure><img class="crop" alt="{f["alias"]} 新观测 native crop" src="{embedded(buffer.getvalue())}"><figcaption>{f["alias"]} 新观测 #{proposal["target_index"]} · 原生像素裁切（仅缩小）</figcaption></figure>')
            blocks.append(f'<section class="proposed"><h2>待审阅 {key}</h2><p><b>{r["proposal_count"]} 条帧级观测提案；不是 {r["proposal_count"]} 个独立主体。</b> 不改变原keeper，不自动补保，不授权删除。</p><div class="grid crops">'+''.join(crops)+'</div><div class="grid">'+''.join(panels)+'</div><p>绿色框：通过独立姿态检测、native躯干纹理、对应几何及互唯一外观检查的提案。它不是身份真值。此组仍有未确认角色，<b>完整主体集合未恢复</b>。</p><p>人工核对：是否同一演员？是否原检测确实漏掉而非重复框？候选是否主要包含该演员？确认之前仅保留为审阅条目。</p></section>')
        else:
            blocks.append(f'<details><summary>{key} · 无可采纳提案 · 原始框数 {r["original_counts"]} → 候选框数 {r["candidate_counts"]}</summary><p>原始/候选框数不是主体人数；黄色框不构成已确认实例。拒绝原因：{html.escape(json.dumps(r["refusals"],ensure_ascii=False))}</p><div class="grid">'+''.join(panels)+'</div></details>')
    metrics=replay['metrics']
    rows=[]
    for label,stratum in [('单主体','single'),('多主体','multi'),('无明确主体','no_subject'),('合计','overall')]:
        m=metrics[stratum]['review_only']
        rows.append(f'<tr><td>{label}</td><td>{m["groups"]}</td><td>{m["keepers"]}</td><td>{m["high_confidence_phase_misses"]}</td><td>{m["quality_wrong"]}</td><td>{m["proposed_groups"]}</td></tr>')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"><title>Photo-dedup 实例恢复 v0 · 本地审阅</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#10151d;color:#e5eaf0;max-width:1400px;margin:32px auto;padding:0 22px}h1,h2{line-height:1.3}.muted{color:#aebdcb}.summary,section,details{padding:22px;background:#1b2430;border-radius:12px;margin:22px 0}.proposed{border:2px solid #3c8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}figure{margin:0}img{max-width:100%;height:auto;border-radius:6px}.crop{max-height:500px;width:auto}.crops{display:flex;justify-content:center;gap:36px;margin-bottom:24px}figcaption{color:#b5c4d2}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #455263;padding:10px;text-align:left}b{color:#ffd37a}summary{cursor:pointer}.tag{padding:5px 12px;background:#293b4d;border-radius:20px;display:inline-block;margin:4px}</style><h1>跨帧实例恢复 v0</h1><p class="muted">默认关闭 · review-only · 自包含本地文件 · 不调用网络或上传照片</p>'''
    document+=f'<div class="summary"><span class="tag">窄切片 8 组 / 19 帧</span><span class="tag">提案 {summary["proposed_groups"]} 组 / {summary["proposals"]} 帧级观测</span><span class="tag">keeper变更 0</span><span class="tag">完整集合恢复 0</span><p>实际改善：G020中同一舞台女演员在两帧获得漏检观测提案。原有keeper及其顺序全部不动。模型临时目视检查为2/2正确，但这只是<b>一个演员的一次双向对应</b>，不是总体准确率，也不是人工真值。</p><p>LK方法无提案；SIFT方法的唯一提案实际是同一犬的错误类别重复框，已拒绝。当前候选改用已有独立pose观测，并增加跨类别重叠/局部框拒绝。主体集合依然不完整，不能宣称多主体覆盖已经解决。</p></div>'
    document+='<h2>实际 Stage2 / Stage3 回放</h2><p>188组/498张，off与review_only均实际运行。keeper集合、顺序、首选图、评分与预算完全相同；unsafe AUTO_REMOVE=0；未运行Stage4。</p><table><thead><tr><th>分层</th><th>组数</th><th>keepers（两配置相同）</th><th>高置信漏保（未改善）</th><th>质量错误（未改善）</th><th>提案组</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table><p class="muted">no_subject仍走原全图baseline。文档/屏幕没有样本，不声明支持。以上是已有模型临时开发标签，不是新留出。</p>'
    blocks.sort(key=lambda b:0 if b.startswith('<section') else 1)
    document+=''.join(blocks)+'<footer><p>人工审阅队列：proposed-groups.json · keeper变更队列：changed-groups.json（空）。算法提案不能直接用于删除；AUTO_REMOVE边界仍仅BYTE_IDENTICAL。</p></footer></html>'
    (root/'review.html').write_text(document)
    (root/'proposed-groups.json').write_text(json.dumps({'schema_version':1,'review_only':True,'groups':queue},ensure_ascii=False,indent=2))
    (root/'changed-groups.json').write_text(json.dumps({'keeper_changed_groups':[],'observation_proposed_groups':[{'day':q['day'],'group_alias':q['group_alias']} for q in queue]},indent=2))
    print(root/'review.html',len(document.encode()),'bytes; embedded images',document.count('data:image/jpeg;base64,'))


if __name__=='__main__':
    main()
