# 两段式真实模型小测：G004 / G007

## 当前结果

两组均完成 **原始真实Pass A复用 → 本地CV enrichment → 真实Sonnet Pass B**。Pass B的原始结果都是attention_only，未给推荐；严格输出校验未通过，因此没有可发布的validated B判断。当前已按Willy的成本要求暂停云扩测，保留全部输入与响应供本地VLM对照。

| 组 | A结构复验 / 主strips | Coverage / ceiling | B真实返回 | 严格验收 |
|---|---|---|---|---|
| 2026-07-25 G004，470/481 | T1 key_subject/stage_area；T2/T3 bystander排除 | uncertain/resolution → attention_only | 3点舞台结构、取景与遮挡观察 | 两条text为242/249字符，超过240上限 |
| 2026-04-05 G007，35/131 | T1 key_subject + T2 supporting；T3 bystander排除 | uncertain/occlusion → attention_only | 3点前景姿态、右侧人物与遮挡观察 | 第三条243字符；还引用不允许的bystander R3/R7 |

G007的bystander引用是第二项独立问题：即使缩短该句，跨字段grounding检查仍会拒绝。原始响应未修改、截短或重新解释为通过。两组sidecar均保持attention_only，recommended/irreplaceable为空；`pass-B-validated.json`记录not_produced和失败原因。不能称完整“验收成功”的A/CV/B链。

## 可观察结果与边界

**G004**：完整舞台页和舞台strip进入B，模型观察到中央蓝色屏幕、两侧色块、桁架及紫色穹顶布局相近，屏幕/灯光和取景有小变化；前景遮挡使舞台地面及细小人物不可判断。该输出最多支持舞台结构、灯光和画面变化attention，不证明逐演员、脸或眼睛覆盖。自由文本中“更宽取景”“更高sharpness”仍是模型观察/引用，不是人工偏好真值。

**G007**：B看到了前景主体及右侧有意义supporting人物的两条strips，并观察姿态、脚步与画面位置的差别。主体间遮挡未解决，不能select，也不能从矩形ROI推断镜像身份或完整多人覆盖。第三点的bystander evidence不符合主证据契约，因此不交给推荐消费方。

没有Willy人标，不计算偏好准确率，不使用旧keeper/provisional label作真值。两组结果没有进入生产review API、keeper或删除路径。

## 验证契约

`src/two_pass_review.py`版本为`two-pass-review-v2`：

- Pass A仍严格验证file_id+SHA、有限坐标、边界、full-page grounding、区域重复及角色冲突。
- 合法bystander/unknown小框不消耗主ROI预算，在coverage ledger显式保留；对应track无论低置信或ambiguous都不能进入主strips。
- 主track只允许一致role和kind的key_subject/supporting区域，包含stage_area；主track歧义、低置信、关键主体缺失/未比较仍拒绝。
- uncertain/resolution/occlusion可进入本地CV/B；sidecar与请求明确`decision_ceiling=attention_only`。只有覆盖candidate_complete、无未解决限制、中/高置信且所有关键区域被比较时，才允许select。模型越过ceiling返回select会被拒绝，输出空推荐的attention-only sidecar。
- B最多3点，每条text≤240字符，证据只能引用key_subject/supporting区域；attention/abstain不能包含推荐、多选理由或不可替代项。

本地CV保留完整L0/L1/L2、ROI像素测量、已有face/eye/pose/local-quality的空间join及明确unknown。不会把检测区域质量冒充新ROI重新测出的IQA，也不把全图眼睛状态广播到舞台人物。

## A复用与账本

版本化入口：`scripts/revalidate_two_pass_real.py`。`archive/v1/`保留原实现、schema与完整首次产物；`revalidate-v2/`保存当前A结构复验、CV、主strips、B请求和响应。

复用前逐项检查：原implementation manifest及归档文件hash、原schema与请求一致、原请求payload/wire/provider/raw响应hash、wire与页图字节一致、原图SHA及定向尺寸、frame identity、原数据库文件hash与decision表摘要。两组均通过全部本地验证及预算后，才各调用一次B。A没有重调用。

Converse/us-east-1，模型请求ID固定`global.anthropic.claude-sonnet-5-5`，SDK `total_max_attempts=1`。总计4次调用（2A+2B），没有retry/fallback/升级，没有G002/G014扩测。provider均end_turn、RetryAttempts=0，cache读写0；provider未返回独立resolved-model字段，不声称额外served-model证明。

## 实际usage与估算成本

| 组/Pass | Input | Output | Total | Latency | 假设费率估值 |
|---|---:|---:|---:|---:|---:|
| G004/A | 10,090 | 1,782 | 11,872 | 10.923s | $0.057000 |
| G007/A | 10,060 | 1,689 | 11,749 | 10.171s | $0.055515 |
| G004/B | 21,312 | 411 | 21,723 | 5.591s | $0.070101 |
| G007/B | 26,742 | 439 | 27,181 | 5.535s | $0.086811 |
| 合计 | **68,204** | **4,321** | **72,525** | | **$0.269427** |

B合计48,054 input / 850 output，估值$0.156912。美元按显式假设 **$3/M input、$15/M output** 计算，Sonnet5.5 Bedrock具体tariff未独立核实；这是规划估算，不是AWS实际账单。失败的B也全额纳入账本。

G004/B实际3图、6,647,552 pixels、920,226图像bytes、25,920 wire文本bytes；G007/B为4图、8,121,600 pixels、1,985,844图像bytes、32,888文本bytes。均在7图/20M pixels/16MB/36K文本byte上界内。

## 本地交付

总览：`artifacts/two-pass-real-pilot/index.html`，直接打开，不启动服务。

- 权威总报告：根目录`report.json`、`call-ledger.json`、`artifact-verification.json`。
- 当前链：`revalidate-v2/2026-07-25-G004/`与`revalidate-v2/2026-04-05-G007/`。
- 每组：`revalidation.json`、`pass-A-validated.json`、`L0/L1/L2.json`、完整页、`strip-T*.jpg`、`pass-B-request.json`、wire request、provider response、`pass-B-raw.txt`、raw/validated状态、sidecar。
- 原始A及首次拒绝状态保留在组根目录及`archive/v1/`；当前入口明确指向revalidate-v2，避免把历史拒绝诊断strips当作B输入。
- 完整私人影像与wire base64只用于本地私有审查；授权Bedrock请求是唯一外部影像传输。

## 验证与停止状态

本轮focused **101 passed**；Python broad **1125 passed / 179.53s**（Python 3.11环境）；前端 **48 passed**、typecheck/build通过；scoped Ruff与diff-check通过。日志见`gates/*v2*`。未宣称全仓库Ruff干净，parent仍需独立验收。旧评测树仅只读，冻结SQLite文件、decision表和旧真实JSON保持不变。没有Google Photos、服务/config、发布、commit或自动删除。

云端已暂停。现有原始B反映出明确的短文本和evidence角色遵循问题，适合作为本地VLM的固定输入对照；不得将未经校验的自由文本直接展示成选片推荐。
