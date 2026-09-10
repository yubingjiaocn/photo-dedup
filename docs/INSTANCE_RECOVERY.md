# 跨帧实例恢复提案 v0

本地、默认关闭、review-only。提案只补充人工审阅信息，不替换原始 region catalog，不改 keeper 集合、顺序、首选图、评分或预算，不授予删除权限。

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
