# VLM 疑难组选片实验审计轨迹

> 本文只记录实验弯路、反例、停止原因与收敛过程。当前正确设计见 `../CLOUD_JUDGE_BATCH.md` 与 `../VLM_REVIEW_WORKFLOW.md`。本文不作为生产配置或准入说明。

记录日期：2026-09-30  
范围：冻结开发/holdout 集，6 个唯一日期，2,065 张照片、194 个相似组、510 张组内照片。所有实验均为 review-only；未访问 Google Photos，未改 keeper、排序、group decision 或删除状态。

## 最终收敛到的当前方向

当前主链是：

1. CV 完成分组并生成 L0 原始审计数据、L1 派生判据、L2 精简判据；
2. 按可信 EXIF 生成完整组图、时序页或确定性辅助视图；
3. 每个疑难组调用一次云端 VLM，完整组图始终在输入中；
4. VLM 只输出 `select | attention_only | abstain` 的人工复核建议，允许多选，最多三个区分点；
5. low-confidence 观察可以共存，但不能单独支撑推荐或不可替代结论。

本地 VLM、固定网格、光流和本地 proposal ensemble 只保留为实验或辅助证据，不是必经决策层。

---

## 1. 最初把 VLM 定义成“只补 CV 空白”

### 当时假设

CV 先做主要判断，VLM 只描述互动、表情和叙事价值等 CV 不会的内容。

### 为什么不成立

用户明确要求疑难组由 VLM 同时看图片、局部区域和全部 CV 判据。CV 的单项分数没有足够权威先裁掉照片；若先让 CV 定赢家，再让 VLM 解释，容易变成对旧 keeper 的事后合理化。

### 留下的规则

CV 在疑难组中是测量来源，不是最终裁决器。旧 keeper、utility score 和 pair margin 不进入 VLM prompt，也不作偏好真值。

---

## 2. 把“所有判据都用上”实现成原始数据倾倒

### 当时实现

将持久化 CV 树、region catalog、接口说明、向量和运行字段大面积直接放入 prompt。

### 实测后果

首轮 22 个疑难组约消耗 1,058,615 input tokens。大量内容是 embedding、原始数组、框坐标、telemetry 和重复接口常量，既昂贵也没有形成“哪张更好、差多少、在哪个区域”的可读证据。

首轮表面得到 7/22 medium-grounded、15/22 abstain；后续发现 identity 和视觉输入均有硬错误，这 7 组结果全部作废，不能视为模型质量信号。

### 修正

引入三层：

- L0：完整 path-redacted audit raw，永不进 prompt；
- L1：pairwise、rank、delta、冲突、适用性和 authority；
- L2：只保留组内有区分力的精简证据。

22 组重编译后，含 prompt 的文本 token 估算中位约 4,480、最大约 9,962。coverage ledger 要求每个判据族注明 exact / derived / summarized / non-evidence，避免通过“压缩”静默丢字段。

证据：`../FABLE_CRITERIA_REVIEW.md`、`../FABLE_REBUILD_STATUS.md`、`../../artifacts/fable-rebuild/report-22.json`。

---

## 3. compiler guard 误杀自己的 pHash 接口字段

### 问题

半成品 compiler 把 `measurement_interfaces.phash` 的接口描述误判为原始二进制，导致自身输出无法通过 raw-binary guard。

### 原因

guard 只按字段名匹配，没有区分“接口描述”与“实际测量 payload”。

### 修正

仅允许精确路径、精确类型的 interface descriptor；真实 bytes、base64、原始大数组和几何 flood 仍 fail closed。新增回归覆盖接口描述、二进制、几何数组和上下文上限。

---

## 4. F01/F02 alias 在 panel 与 sidecar 中反序

### 问题

G002 封存 panel 为 F01=file 107、F02=file 99；旧 sidecar 却为 F01=file 99、F02=file 107。此前按 alias 对照“腾空/近地面”和旧标签的评测发生串图。

### 修正

内部和评测 join 统一使用 `file_id + content_sha256`。alias 只在最终展示层生成，并在 sidecar 显式保存 `alias_map`。直接 alias join 一律拒绝。旧 22 组身份 join 已做双射和源 SHA 校验。

### 教训

展示顺序不是 identity；任何 panel、缓存和评测标签都不能靠 F01/F02 关联。

---

## 5. ROI 机械追随最大脸，抓到旁观者

### 问题

G002 动作主体是中央吉祥物，额外 face crops 却来自左右旁观者。视觉预算被错误主体占用。

### 尝试

先用 person/face detector 定主体；失败后又尝试背景配准 + Lab 多粒度颜色组件寻找中央运动区域。

### 结果

颜色路径能在原图中生成正确的吉祥物动作 comparison strip，Willy 确认裁切方向正确；但 JPEG90 重编码会让一个颜色分量触边并安全弃权。它可作为已确认单组正例，不能推广成通用主体定位器。

证据：`../G002_ACTION_ROI_CANDIDATE.md`、`../../artifacts/fable-roi-narrow/G002-comparison-strip.jpg`。

### 留下的规则

- action 看主体全身、手脚、接触点和道具；
- portrait 看脸眼；group 看主体集合；stage 保留完整舞台；
- 旁观者脸不能作为 fallback；
- ROI 不可靠时保留完整图并 attention/abstain，不能硬裁。

---

## 6. 试图用 person detector 完成多主体/舞台覆盖

### 实测反例

- 2026-04-04 G014：检测数 4→5，存在漏框、身体裁切和未匹配主体；
- 2026-04-05 G007：三个人框可几何对应，但身体范围、镜面和遮挡关系不完整；
- 2026-07-25 G004：两帧检测到的三个 person 都是前景观众，舞台演员完全漏检。

### 结论

检测数相等不代表 ensemble complete；最大脸通常更可能是观众。多主体集合、互动、队形和舞台主体不能由 person detector 授权。

证据：`../MULTI_SUBJECT_REVIEW_CANDIDATE.md`、`../../artifacts/multi-subject-narrow/`。

---

## 7. 两段式云 VLM：理论合理，实际过度复杂

### 方案

Pass A 云 VLM 找 subject/stage/interaction map；本地 CV 对 ROI 补测；Pass B 云 VLM 深度判断。

### fake E2E

严格 schema、multi-select、irreplaceable、attention-only、hash/cache 和零写入均可在 fake response 上通过，但 fake 只证明接口可运行，不证明真实模型会按约输出。

### 真实 Sonnet Pass A

- G004 正确识别中央舞台，把前景观众标为 bystander，并主动报告 resolution uncertainty；
- G007 找到前景人物和右侧人物/互动区，没有退化成最大脸。

### 被门禁错误拒绝的地方

1. 无关 bystander 小框面积 0.0042，略低于全局 0.005 下限，导致整组失败；
2. `supporting` 人物被禁止进入主对比 track；
3. coverage uncertain 被当成禁止 enrichment，而不是“允许继续但最终最高 attention-only”。

这些属于契约设计过严，不是模型没有找到主体。随后规则改为：key、supporting 和 stage_area 可进入比较；bystander/unknown 不支撑推荐；coverage uncertainty 可以继续，但限制 decision ceiling。

### 为什么最终仍不采用两次云调用

每组需要重复发送完整图，状态、缓存、失败恢复和成本都翻倍。Willy 将架构收敛为“一次云 judge”，不再用云端 Pass A。

证据：`../TWO_PASS_MULTI_SUBJECT_REVIEW.md`、`../TWO_PASS_REAL_PILOT.md`、`../../artifacts/two-pass-real-pilot/`。

---

## 8. 本地 Qwen3-VL-8B 作为精确 ROI locator

### 模型与性能

`Qwen/Qwen3-VL-8B-Instruct-FP8`，revision `9cdc6310a8cb770ce18efaf4e9935334512aee45`。权重约 10.606 GB；L40S 峰值约 10.27 GiB；四组平均约 58.05 秒/组。

### 结果

严格首轮只通过 2/4：

- G007 输出 Markdown JSON 围栏；
- G014 使用 schema 外的 `role=other`；
- G002 虽找到橙色吉祥物，但紧框分别裁掉脚和跳起后的头，还把巨大 EXPO 背景字设成 key；
- G004 舞台/观众方向有用。

所有组都倾向自报 candidate complete，未充分暴露镜像、遮挡和裁切不确定性。重复运行显示错误也可以很稳定。

### 结论

本地 8B 的框只能作不可信 attention hint，不能独占裁剪、定义 coverage 或决定哪些完整照片不送云端。

证据：`../../artifacts/path-experiments/local-vlm/REPORT.md`。

---

## 9. 将本地 8B 降级成极简场景路由器

### 方案

只输出 action / stage / group / interaction / ordinary / unknown 和视觉 packet 类型，不输出 boxes。

### 结果

JSON 8/8，重复一致 4/4，warm 平均约 1.67 秒/组。G002、G004、G014 合理；G007 被高置信判为 ordinary/full_only，且 uncertainty 为空。它结构上完全合法，却会漏掉多人/镜像复杂性。

### 结论

3/4 的小样本表现不足以成为必经路由层。错误的高置信路由比显式 schema failure 更危险。最终建议是不依赖本地 VLM：完整照片按可信时间排序，超限分页，直接进入单次云 judge。

证据：`../../artifacts/path-experiments/local-vlm/scene-router-v2/REPORT.md`。

---

## 10. 三条并行辅助路径

### 固定多尺度网格

G004 的中心窗口明显改善舞台结构可读性；G007 的 high-change 窗口却只突出裙子、包和腿，裁掉脸。四象限能保证几何覆盖，但带来约 7 图/组、12–19M pixels、区域重复和跨 tile 断体。

结论：保留为无 detector 的舞台/大场景辅助视图，不作为语义主体定位器。

证据：`../../artifacts/path-experiments/multiscale-grid/REPORT.md`。

### 时序、差分和光流

可信原图亚秒 EXIF 很有价值：

- G002 正确顺序为 file 107→99，间隔 1.536 秒，近地面与腾空阶段不同；
- G014 为 284→248，间隔 0.342 秒，包含舔鼻、张嘴/眯眼等互动微阶段；
- 数据库整数 EXIF 在样本中偏移 +8 小时；存储和实际 mtime 都不能作拍摄顺序。

光流/差分会同时强调主体、反光、文字和旁观者；它只能提示“哪里变化”，不能命名动作阶段或自动判定不可替代。

结论：可信 EXIF storyboard 是主链输入；差分/flow 默认留在调试层。

证据：`../../artifacts/path-experiments/temporal-phase/REPORT.md`。

### 本地 class-agnostic proposal ensemble

Lab、边缘、显著性、配准残差、flow 和 DINO patch 共生成 1,170 条组件记录；稳定性和多源共识后留下 180 条 attention-only，最终 0 条稳定三族 ROI candidate。

它能把 G004 注意力从观众移向舞台，也能在 G002 找到橙色角色，但无法恢复小演员、完整多人集合或镜像身份。

结论：可作本地 attention 辅助，不替代 VLM 语义判断。

证据：`../../artifacts/path-experiments/hybrid-proposals/REPORT.md`。

---

## 11. 输出门禁多次把“弱观察”误当“整组失败”

### 240 字硬限

真实 G004/G007 Pass B 返回可用 attention-only，三条观察长度约 193–249 字；旧 validator 因超过 240 字直接拒绝整组。该限制并非用户需求，用户要求的是最多三个区分点。

修正：安全硬限改为 500 Unicode chars；UI 超长折叠但保留原文，不静默截断事实。

### bystander 引用一律禁止

旧规则禁止任何 bystander evidence。实际 attention-only 可以合法指出“旁观者遮挡、干扰或构图拥挤”。

修正：bystander/unknown 不能支撑 select、recommended、multi-select reason 或 irreplaceable；但可以在 attention-only 中绑定具体 frame/region 描述遮挡和干扰。

### select 中出现 low evidence 即整组失败

单次云 judge canary 前三组中：G057（11张）正常 validated；G014 和 G007 的推荐都有 medium/high 支撑，但另有次要 discriminator 含 low evidence，旧 validator 将两组全部拒绝并早停。

修正中的 contract v3：

- 推荐、multi-select reason 和 irreplaceable 必须由 medium/high 支撑；
- low 可以作为弱观察共存；
- low 不得成为推荐的唯一或必要证据；
- 原始响应保持不变，通过版本化重校验，不靠改写模型答案“制造成功”。

本段初稿时 contract v3 与 194 组续跑仍在进行；随后按“只有原片安全是全局硬门禁”的最新优先级完成全量。最终 194/194 组各消费一次云 judge 槽位，187 组 validated、7 组保留 raw 后转人工，decision/keeper/delete 写入为 0。该完成状态只证明批处理和结构覆盖，不等于偏好准确率。

证据：`../CLOUD_JUDGE_BATCH.md`、`../../artifacts/cloud-judge-batch/authorized-scene-run-v2/`、`../../artifacts/cloud-judge-batch/authorized-scene-run-v3/`。

---

## 12. VLM 比例估算曾使用了错误口径

### 早期估算

用 `group_trusted + keeper_budget + utility_margin` 的临时阈值推算 194 组中约 51 组可由 CV 直接处理，得到约 143 个疑难组、每 100 张约 7 次云调用。

### 为什么不能作为当前权威口径

该阈值没有通过人标校准，而且 utility/pair margin 正是后来决定不能作为疑难组选片权威的旧选择信号。按当前保守正式 route，只有 `sha_exact + AUTO_READY + 全成员同 SHA` 才直接 CV-only；冻结六日期中没有这类组，因此 194/194 都属于 hard route。

### 当前正确数字

- 2,065 张总照片；
- 510 张进入 194 个相似组，占 24.7%；
- 最多 194 次单次云 judge，相当于每 100 张约 9.4 次调用；
- 不是 2,065 次，也不是已验证的 143 次；
- 后续要降低比例，必须用人标验证的新 CV 规则，而不是重新启用旧 utility margin。

---

## 13. 当前不可再重复的错误

1. 不把所有存储字段直接塞进 prompt；
2. 不用 alias 作 identity 或评测 join；
3. 不让最大脸/person detector 定义动作或舞台主体；
4. 不用 local VLM box 裁掉完整组图；
5. 不把 detector 数量相等解释成多主体完整；
6. 不把 motion magnitude 当语义阶段或审美质量；
7. 不让任意一个 low observation 拖垮有 medium/high 支撑的推荐；
8. 不因格式长度、无关小框或 supporting 角色而整组 fail；
9. 不拿 fake E2E、模型自报 confidence 或旧 keeper 充当偏好准确率；
10. 不在新生日照片上边接入边调参；先完成冻结集 review-only 验收。

## 14. 保留的成功机制

- `file_id + content_sha256` 全链身份；
- L0→L1→L2 分层与 coverage ledger；
- 完整组图始终保留；
- 原图亚秒 EXIF 时序；
- 最多三个区分点、多选、attention-only、abstain；
- 所有模型输出原文、request/response hash、usage、cost 与失败原因落盘；
- 每组最多一次云 judge；无隐式 retry、升级或 fallback；
- review-only，零 keeper/delete/decision 写入。
