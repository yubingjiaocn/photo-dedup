# 跨帧实例恢复提案 v0

本地、默认关闭、review-only。提案只补充人工审阅信息，不替换原始 region catalog，不改 keeper 集合、顺序、首选图、评分或预算，不授予删除权限。

**完整日期最终验收见 [FINAL_ACCEPTANCE.md](FINAL_ACCEPTANCE.md)。** 37组/143张的三臂回放保持63keepers，未证实净收益；multi样本不足，模型标签不是人标，浏览器与Windows真机门禁未过。以下8组/19帧和188组/498张均为此前已暴露development证据，不与新日期混算。

## 运行时开关

```yaml
cluster:
  keeper_instance_recovery_policy: off  # 可选 review_only
```

`review_only` 读取每个成员 `quality_meta.instance_recovery` 中绑定到完整 group file IDs 的一致 packet，将有效提案保存在 `decision_json.phase_selection.instance_recovery_context`，并要求 review。没有 packet、成员不齐、关联/目标未确认或格式不合法时 abstain。`AUTO_REMOVE` 仍仅允许 `BYTE_IDENTICAL`；本功能不执行 Stage4。

当前 v0 **不自动补保照片**。即使某条实例对应可靠，也不能由此推导动作阶段、完整主体覆盖或照片质量。

## 本地提案生产

`run_instance_recovery.py` 使用显式 slice 中的既有 feature DB 和只读原图。支持：

- `flow`：native 灰度 LK 前后向一致性及 RANSAC；传递框只是搜索区域，必须由目标帧独立检测确认。
- `redetect_sift`：现有本地 COCO segmentation detector 的局部重检测，结合 native SIFT reciprocal ratio matches、RANSAC 和局部 DINO 外观。裁切边界框及跨类别重复/局部框不作为新实例。
- `pose_anchor`：复用已有独立 COCO17 pose cache。检测置信度至少 0.6、至少 8 个可靠关节；两帧至少 3 个可靠肩/髋锚点，native 躯干内各至少 12 个纹理角点；几何重叠与中心位移必须合理。它不是 optical flow，不依靠肢体保持不动。

三种方法均在所有同类候选之间要求 DINO cosine ≥ 0.90、双向唯一 margin ≥ 0.03；不能靠位置或最大框补足身份歧义。新框还必须通过跨类别重叠/包含排除，不能把同一实例的框变体记成恢复新实例。低纹理、裁切、漂移、无法独立确认的目标全部 abstain。

`stable_tracks` 返回所有通过双向与跨帧 cycle consistency 的轨迹子集，没有最大主体优先级。`stable_observed_set=true` 要求子集覆盖**每帧全部 catalog observations**。少数稳定轨迹不等于完整主体集合；完整 detector-defined set 也不是人工意图主体真值。

## 可复现的本轮入口

在仓库根目录执行；所有输出必须使用新的目录，已完成组不盲重跑。离线模型缓存路径是已有本地缓存，不是下载目标。

```bash
HF_HOME=/home/ubuntu/hf-cache-photo-dedup \
.venv/bin/python scripts/research/run_instance_recovery.py \
  --slice /home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/slice.json \
  --output /home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/NEW-proposals \
  --method pose_anchor \
  --model /home/ubuntu/photo-dedup-eval/astra-usable-algorithm-20260909/person-instance-probe/models/yolo26s-seg.pt
```

脚本设置 Hugging Face offline 和关闭 YOLO sync；不安装或下载模型。slice 包含 `date/group_alias/source_db/config/frames`；每帧有 `alias/id/path`。`source_db` 应含已验证 `local_region_set` 与 native `pose_evidence`。无 pose cache 时该机制 abstain，不伪造关节或偷偷推理。

本轮权威 candidate 已落盘，直接复用，不必运行上面的新推理：

```bash
.venv/bin/python scripts/research/replay_instance_recovery.py \
  --proposals /home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/candidate \
  --output /home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/NEW-stage2-3
```

该研究回放入口复制已有五个 development DB，仅在副本附加 packet，实际运行 off/review_only 两种 Stage2/3，并验证 keeper 顺序、评分、预算、物理分组与 BYTE_IDENTICAL 删除边界。标签和 single/multi/no_subject 侧表只在运行后评价，不进入生产提案或 Stage2。

## 审阅载体与边界

本轮入口：`/home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/review.html`，为自包含本地 HTML，21 张内嵌缩略图/裁切，无脚本或外部资源。相邻 `proposed-groups.json` 是人工审阅队列；`changed-groups.json` 单独列出 keeper 变更，当前为空。

8 组已暴露窄切片得到 1 个提案组：同一舞台演员的 2 帧观测。临时模型目视检查为 2/2，但只有 1 个物理轨迹，不能当作总体 precision。完整集合恢复仍是 0/8；暂不扩展提案生产至全部 development。188 组全量运行只是隔离性/回归验证，不是全量恢复实验。

no_subject 保持原全图 baseline；不使用评测 regime 标签推断运行时主体。文档、票据和屏幕没有样本，未验证。新完整事件 holdout 需在参数冻结、标签封存后才揭盲；本轮不是新留出净改善证据。

## 原生 dense 可见区域补充

同一 `review_only` 开关可以消费另一个严格绑定到 group 的 `quality_meta.dense_instance_recovery` packet。它只追加 `dense_visible_region_context` 与人工审阅条目；没有该 packet 时，legacy 输出完全相同。默认依旧关闭，不更改 keeper 或评分。

生产入口为 `scripts/research/run_native_dense_slice.py`，使用已安装 YOLO 分割与 DINOv2-base：显式 native crop 不 resize/center-crop；14px patch 的分割前景占比至少75%、native纹理足够，最多均匀采样512个。DINO先处理未中和背景的矩形crop，再筛选token，因此描述子仍可含背景及上下文信息，不能视作已消除背景混淆。局部描述子互最近、cosine至少0.90且双向patch margin至少0.01；RANSAC支持至少12点、inlier ratio至少0.65、双方空间覆盖至少0.15、scale在0.8–1.25，inlier数量占较小patch样本数至少0.08。物体竞争只在已通过上述门槛的eligible pairs中比较（至少1.5倍竞争分数，且IoU至少0.5），不等于所有潜在对象中的外观唯一性。

这里允许 detector class0/77 跨类别参与候选，但**原始检测类别不是语义身份**。审阅范围明确为 `visible_region_only`；裁切边界、遮挡和身体完整度仍是 unknown，不把局部区域当完整身体、服装内人类身份或全组主体集合。该机制没有自动补保或删除权限。新数据必须使用完整隔离的root，并核验slice里的source_db、config、原图路径和输入身份；不要复制旧dense-pilot，不能仅凭G018/F01、observation index或result文件存在判定缓存可复用。现有研究producer不是任意数据根之间可安全共享缓存的通用服务。

本轮8组/19帧只新增G018 F01↔F02的一条服装角色区域对应、2个帧级观测。它与既有pose-v0的G020不同，合计2个proposed groups，keeper变更仍0。局部对不能宣称覆盖G018全部4帧。`dense-review/review.html` 是单独的自包含匹配点载体，原v0审阅页保留。

导出与新接入回放：

```bash
.venv/bin/python scripts/research/export_dense_review.py \
  --source /home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/dense-slice \
  --output /home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/NEW-dense-review

.venv/bin/python scripts/research/replay_instance_recovery.py \
  --proposals /home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/candidate \
  --dense-proposals /home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/dense-review \
  --days 2026-07-25 \
  --output /home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/NEW-dense-stage2-3
```

全部188组与冻结f79cc0d的keeper顺序、评分、预算、phase输出逐组相同；本补充的真实Stage2/3新增回放范围是受影响的Jul25 52组，两配置unsafe AUTO_REMOVE均为0。原188组Stage2/3证据仍归属v0，不重复宣称为本补充的新回放。
