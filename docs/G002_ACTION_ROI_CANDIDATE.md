# G002 动作区域 review-only 候选

日期：2026-09-30

## 当前结论

原始 G002 两帧已产生可看的跨帧动作区域候选，覆盖中央橙色前景及手脚，未将旁观者脸用于 crop。候选只表示同一大前景区域的外观/几何一致性，**不建立吉祥物身份、动作阶段或全身完整性真值**。

Willy已确认单主体动作comparison strip的ROI方向正确。JPEG quality=90 重编码的稳定性控制会弃权，因此可靠性仍未通过，不接生产视觉规划器，不扩展到22组。独立多主体窄切片见 `docs/MULTI_SUBJECT_REVIEW_CANDIDATE.md`；其真实主体覆盖门禁尚未通过，新的 decision / multi_select_reason / irreplaceable schema仍延期。

- 对比条：`artifacts/fable-roi-narrow/G002-comparison-strip.jpg`
- 每格原生 crop：1158×1896；实际展示图像：512×838，短边512，未上采样。
- 全图定位叠框：`artifacts/fable-roi-narrow/G002-candidate-overlay.jpg`
- 来源、SHA、全部候选/拒绝理由、参数、跨帧指标：`artifacts/fable-roi-narrow/G002-candidate-report.json`
- 两帧分割 mask：`G002-F01-mask.png`、`G002-F02-mask.png`（同目录）；报告包含原始 mask pixels SHA。
- F01=file_id 99，F02=file_id 107；不使用旧 panel alias join。

## 已有能力的边界

| 模块 | 可复用能力 | 不足以直接定位本组主体的原因 |
|---|---|---|
| `instance_recovery.py` | LK前后向、SIFT/RANSAC搜索窗、独立目标确认 | 需要既有 source box；target confirmation 依赖检测，不能凭空发现漏检的非person主体 |
| `instance_correspondence.py` | mask前景SIFT、masked DINO appearance、reciprocal uniqueness | 输入仍需独立 detector segmentation；view gate 限 class 0/77；无现成 class-agnostic proposal generator |
| `dense_recovery_review.py` | visible-region packet 校验、面积/对应约束 | reader 要求两帧 detector confirmed、class 0/77 和已有密集patch证据；G002 quality_meta未持久化此包 |
| `local_region_set` | 审计检测框、局部质量 | 冻结框只覆盖左右人物；不能给中央动作区域授权 |
| 本地 OpenCV | SIFT、Lab k-means、连通域、warp | 当前安装无 `cv2.saliency`；不下载模型/权重；显著性本身也不能确定动作主体 |

## 有界实现

`src/action_roi_candidates.py` 仅接收两帧已定向 RGB，不接收 person/face detector 结果或旧 keeper labels：

1. 在最长边1024的分析图上做 SIFT ratio match 和 robust partial affine，要求至少40 inliers、inlier ratio≥0.40、分布面积≥0.30、边缘背景支持≥20、scale∈[0.9,1.1]、median error≤2px。未通过即弃权。
2. 每帧分别做 Lab k-means（K=8和12，各3次初始化，固定seed），不指定橙色或吉祥物类别。闭运算后提取连通域。
3. 候选 mask 面积必须为全图2.5–30%；中心x∈[0.25,0.75]、y∈[0.20,0.80]；边界至少留1%；注册后的RGB变化比例≥50%。每个粒度必须恰好有一个候选；缺失或歧义即弃权。
4. 两粒度box IoU≥0.75。跨帧独立候选需box IoU≥0.40、mask面积比≥0.60、中心距离≤0.15、Lab均值距离≤25、色度histogram Bhattacharyya distance≤0.35。颜色/几何一致性不升级为身份。
5. 取两帧候选union加10%余量，使用同一个normalized window裁切原图。越界或原生/显示短边<256即停止。无face crop兜底。

报告把每个拒绝的分量也保留下来，可检查静止大字、边缘人物、地面为何不占用候选预算。旁观者仍可能出现在完整全图上下文，但本次动作comparison strip无旁观者脸。

## 现场结果和停止点

原始两帧：box IoU **0.6572**、mask面积比 **0.9359**、中心距离 **0.06793**、Lab距离 **20.073**、histogram距离 **0.12993**。原图重复运行和反转帧序产生完全相同的共享窗口；半分辨率输入窗口IoU **0.99656**。同帧对照正确弃权。

JPEG90对照以 `NO_UNIQUE_CENTRAL_MOVING_COMPONENT` 弃权：file_id 107的K=8分量延伸至右/下边界，触发 `BOUNDARY_TOUCHING`，该粒度合格候选数为0。完整分量证据在 `stability-jpeg90-refusal.json`。这说明低层颜色分量仍受编码扰动影响；弃权没有回退到旁观者，但不能把原图成功推广成通用稳定定位。`stability-report.json`记录所有5项控制，不隐藏失败项。

下一最小路线：由人工在这两张原图确认当前共享窗口（或每帧画一个区域），存为 `file_id + SHA` 绑定的 review annotation，再测试独立颜色/边缘分割是否能在不放宽边界/歧义约束下复用该窗。无需先解决吉祥物身份，也无需引入外部VLM。保持单组，不以更多阈值调参掩盖JPEG敏感性。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/probe_g002_action_roi.py
OPENBLAS_NUM_THREADS=1 python3 artifacts/fable-roi-narrow/check_stability.py
```

SQLite始终`mode=ro`，source SHA逐帧核验，脚本输出必须在repo内。输出不写`vlm-review/groups`、不改变decision tables、不能被生产primary-subject authority gate接受。外部图像模型/Google Photos调用为0。
