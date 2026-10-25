# 多主体 / 舞台 review-only 窄切片

日期：2026-09-30

## 当前结论

三个真实 development 组已建立主体集合证据、跨帧视觉对应候选、完整覆盖账本及全图 contact sheet。**没有一组通过完整主体 ROI / correspondence 门禁**，因此按停止条件保持 attention-only，不实现新的 decision / multi_select_reason / irreplaceable 推荐 schema 或 UI，也不扩展22组。

舞台样例直接复现“检测到的人不是表演主体”：两帧各3个框全部位于前景观众，舞台表演者未被框住。以框数一致、最大脸或最高质量框代替 ensemble coverage 会得出错误的完整覆盖结论。本轮观众不占主体 strip 预算。

## 三组与实际结果

来源均为只读 `astra-regime-validation-20260910/development`，每帧读取原图并验证 inventory content SHA。内部统一使用既有 `vlm_identity.identity(file_id, content_sha256)`；展示 file_id，不依赖封存 alias，不读旧偏好标签。

| 组 | file_id | 场景 / 观察 | 检测数 | 对应候选 | 停止原因 |
|---|---|---|---|---|---|
| 2026-04-04 G014 | 248、284 | 人与犬互动，多个人体被画面边缘裁切 | 4 / 5 | 4对 | 数量变化、未匹配框、完整性与body extent未知 |
| 2026-04-05 G007 | 35、131 | 三人可见、镜面/遮挡环境 | 3 / 3 | 3对 | 主体框未覆盖所有可见肢体；完整集合与反射关系未经独立确认 |
| 2026-07-25 G004 | 470、481 | 远景舞台、前景观众 | 3 / 3 | 3对 | 框仅覆盖观众；演员漏检；关键主体 ROI为0 |

视觉观察由本轮 Astra 对最多6张旧集派生 artifact进行检查，属于候选说明，不是 Willy 人标或身份真值。前三张检索预览是单人物/布景例，未纳入三组窄切片，也未当作多主体成功样本。

## 实现边界

`src/subject_set_evidence.py`：

- L0保存输入区域/姿态/人脸和SHA绑定的候选scope；L1逐帧列出主体数量、检测置信、body/face可见性、眼睛、face/local quality、pose、边缘裁切/遮挡、formation/interaction/phase与对应冲突。眼睛没有可靠主体级join时明确unknown，不能把组级眼睛状态广播到每个成员。
- 跨帧 region_id 只在本帧有效。对应使用同class、IoU≥0.50、双向唯一且margin≥0.15的几何候选；不会把候选升级为人物身份，也不能处理遮挡交换、镜像、相机运动或相同服装身份确认。漏框与歧义保留unknown/conflict。
- 检测数相等不等于完整集合；视觉模型自写的scope不能授权verified completeness。实际三组没有human_verified或independent_subject_set_validation覆盖授权。
- stage/group判据优先顺序为ensemble completeness→所有关键脸眼/遮挡→同步动作/队形→互动→舞台光→不同动作阶段。L1能保留多个成员与阶段的不可替代候选位置；本轮无可靠语义证据，候选为空，不虚构独有动作、阶段或推荐。
- L2仅从当前L1投影，不包含boxes/keypoints/embedding或旧选择权威。该小模块是隔离的离线证据切片，**尚未接入生产compiler / model invocation / ready sidecar**。
- ROI候选以跨帧body变化及局部质量差异排序，最多2个，可显式选择3个；不按脸大小排序。所有主体都在coverage ledger列明proposed/shown、旁观者降权、预算未展示、face未展示及全图上下文。
- 真正完整的body/face strips需要通过范围、对应与原生短边≥256门禁。实际阻塞组仅输出proposed windows供诊断，`shown=false`；**本轮subject strips为0**，不把看似合理但截断人体的框发布成完整覆盖。face strip也未实现可靠主体级join与原生尺寸保证。
- 缓存键覆盖版本及全部L0输入，SHA、scope或detector变化使其失效；生产group planner仍拒绝未授权候选。

## 可看产物

`artifacts/multi-subject-narrow/`：

1. `2026-04-04-G014-contact-sheet.jpg`
2. `2026-04-05-G007-contact-sheet.jpg`
3. `2026-07-25-G004-contact-sheet.jpg`

每格完整图像实际短边512，无上采样，满足≥400。三份同名前缀 `*-report.json` 包含source identities、L0/L1/L2、对应、候选窗口、覆盖账本、尺寸与decision table前后摘要。`report.json`为总报告，`inspection/`仅为检索及视觉诊断过程，不是模型输入或成功证据。

## 验证与复现

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/probe_multi_subject.py
PYTHONPATH=/home/ubuntu/.cache/uv/archive-v0/arVVMhmvO8QpKbeh/lib/python3.13/site-packages \
  python3 -m pytest -q tests/test_subject_set_evidence.py tests/test_action_roi_candidates.py \
  tests/test_vlm_review.py tests/test_vlm_evidence_layers.py
```

Focused 47 passed：多主体对应、frame-local编号重排、漏检/歧义、完整性未知、旁观者降权、全覆盖账本、2–3区域预算、原生尺寸、L1→L2、stale cache、生产planner拒绝、三组decision tables不变；旧multi-select/low-confidence attention测试保持通过。新strict recommendation schema及其UI测试按真实样例前置门禁延期，不能将旧测试说成新schema验收。

完整Python、ruff、frontend与diff结果见 `artifacts/multi-subject-narrow/run-log.md`。所有写入在repo，SQLite `mode=ro`，未调用外部/付费图像模型，无Google Photos、服务/config、发布、commit或自动删除。

## 下一最小步骤

由人工在三组中任选一组，按file_id+SHA确认完整主体集合、关键成员/旁观者、body/face extent与反射/遮挡；或提供能独立覆盖这些成员的检测结果。随后只在该组复验对应与每成员body/face条，不先开22组。G007可作为多人范围修正候选；G004须解决远景演员漏检，不能用观众框补齐。

G002已获Willy对动作comparison strip方向确认，其既有产物与算法未改；JPEG90稳定性弃权仍成立。多主体模块没有绕过这一限制。最终验收交由parent独立执行。
