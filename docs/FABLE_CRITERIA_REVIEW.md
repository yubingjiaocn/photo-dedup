# Fable 5.1 独立评审：疑难组 VLM 判据与验收

日期：2026-09-30

## 结论

当前产品方向成立：简单组由确定性 CV 解释，疑难/高相似组由 VLM 综合图像与 CV 判据，所有结果仅用于人工 review。但现有 7/22 结果应作废：当前 compiler 不能稳定生成合法上下文，旧集 alias join 有错，视觉 ROI 也未围绕真正区分点分配。停止继续调 prompt，先修数据层、join 和视觉输入，再做小规模有界复测；全部 gate 通过前不接 Google Photos。

## 三层数据模型

| 层 | 内容 | 是否进 prompt |
|---|---|---|
| L0 audit raw | 原始 boxes/landmarks、逐区域数组、embedding、telemetry、接口版本；按 file_id + content SHA 索引 | 永不 |
| L1 derived evidence | pairwise 距离、逐图标量/rank/delta、region 摘要、冲突、applicability/authority；带 compiler 版本 | 部分 |
| L2 prompt-visible | 每组约 1.5–2K 文本 token，只含组内有区分力的判据和明确方向/置信度 | 唯一入模数据层 |

硬规则：L2 只能由 L1 生成；schema/raw-binary guard 失败即停止；统一以 file_id + content SHA 为主键。alias 只在 L2 按 panel 顺序生成并显式写 `alias -> file_id`，评测禁止用 alias join。

## 各判据的最小表示

- SHA/pHash：完全相同集合，或小组 Hamming 上三角/大组最近邻对。
- DINO：pairwise cosine，保留两位；大组只给 min/max 对。
- MUSIQ/CLIP-IQA/总质量：逐图 score、rank、相对 best 的 delta；组内极差很小时只记“无区分”。
- 人脸：数量、主体脸占画面比例、中心/边缘、主体脸清晰度 rank；不传原始坐标和 landmarks。
- 眼睛：open / closed / uncertain(detector conflict)。
- 曝光：相对组中位数的偏差、clip 百分比；不传冗余底层 histogram。
- local region：cross-file 配准后只保留方差最大的 top-3 shared regions；使用语义标签或相对位置，不传完整 catalog。
- pose/action/phase：每图短标签和 phase 序号；明确 apex/landing 等不可替代状态。
- completeness/occlusion：主体裁切部位、遮挡来源。
- utility/pair margin：不进 prompt，避免用旧选择结果锚定 VLM。
- scene/regime：header 一行，用于决定本组适用判据。
- applicability/authority：header 中列适用/不适用和不确定判据。

组内一致的判据不展开，只列入“已检查但无区分”的摘要。所有 criterion family 必须在 coverage ledger 中标记为 exact / derived / summarized / non-evidence，不能静默消失。

## Region 与视觉预算

- 对 shared regions 做组内配准和差异排序，只传 top-3 区分区域。
- crop 服务于区分点，不服务于“最大脸”：action 优先主体全身和手/脚/道具；portrait 优先脸/眼；group 优先各主体脸；landscape 优先高信息细节。
- 旁观者脸降权；当前 G002 额外 face crops 实际来自旁观者，属于视觉预算错配。
- 每组一张 contact sheet；最多两个跨文件同一区域对比条，每格至少 256px。关键动作部位在 sheet 中过小则 fail closed。

## 多阶段与低置信

建议 schema 包含：`alias_map`、`decision(select|attention_only|abstain)`、可多选 `recommended`、`multi_select_reason`、最多3条 `distinctions`、以及 `irreplaceable[]`。不可替代文件必须进入 recommended，且不能同时标为 worse。

低置信但证据绑定的观察只能进入 `attention_only`：不返回推荐，只告诉人工看哪1–2个位置。attention-only 是降低扫图成本的有效结果，不应当作模型失败或选择依据。

## 旧集验收

使用 <=30 组分层旧集，按 scene 和组大小覆盖。真值只来自 Willy 的小样本盲标：允许多选、标不可替代文件、标“若只看1–2个点应看哪里”。旧 keeper 和 provisional labels 仅作参考。

指标：
- 推荐必须落入人工可接受集合，目标 >=85%；推荐中含人工明确应弃图不得超过5%。
- top-1 区分点与人工“该看哪里”重合 >=60%。
- attention-only + abstain <=40%。
- 文本 token 中位数 <=8K、最大 <=15K（理想 L2 约1.5–2K）。
- 相比只看 contact sheet，人工复核时间下降 >=30%。

## Google Photos 前 gate

1. 22组 L2 编译全部通过 schema/raw-binary guard。
2. alias/file_id/SHA 三处 join 自检100%。
3. token 中位数<=8K且最大<=15K。
4. 动作组 ROI 正确，旁观者脸不占 crop 预算。
5. 有界验收的推荐、区分点、attention 指标达标；attention-only 有帮助率>=70%。
6. 全链路无自动删除，输出只写 review sidecar。

六项全绿后才进入 Google Photos <=100 组的 review-only 试点。

## 优先级

### P0
- 作废当前7/22；修 compiler guard；修 file_id/SHA join；完成 L0/L1/L2 分层和 coverage ledger。

### P1
- scene/差异驱动 ROI；多阶段 schema；attention-only；Willy 对 <=30组做盲标。

### P2
- 按评测调 scene 判据；比较 Sonnet 和更小模型；满足 gate 后做 Google Photos 小批试点。

明确不做：继续把全量存储字段塞进 prompt；把旧 keeper/provisional labels当真值；G1/G2未修前继续跑批；自动删除。
