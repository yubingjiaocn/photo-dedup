# 三张以上组：阶段覆盖候选 v4

当前合同：`large-group-v4-candidate-2`。使用 Sonnet 5.5 对开发集进行单次 judge 实测；现有单次 judge v3 与二张组行为保持不变，候选不接入 live runner 或 Review UI。开发期不设美元预算上限，首要容量目标是未来进入 VLM 的 unique source photos。独立 holdout 不参与本轮。

## 数据边界与开发集结果

只使用 `artifacts/large-group-tuning/development-labels.json` 的 39 组标签、`split.json` 的分组名/分层方法，以及这 39 组在 `authorized-scene-run-v3/<group>/` 下的原有请求、图像、判断、CV 和 usage。21 组 holdout 只知道名字，未读其标签、图像或结果，留给 parent 独立评估。完整人标分析与 live review state 均不作为输入。

开发集为按 member_count 分层、确定性 SHA256 顺序的约 2/3 分割。基线集合比较：15/39 完全一致、11 组只漏选、3 组只多选、10 组换选/混合；漏 30 张，多 14 张。这里的“漏/多”指与 assisted review 人选的集合差异，不直接等价于视觉阶段真值。

可复现完整 39 组明细：`candidate/development-baseline.json`，含具体 file_id+SHA、请求/判断 hash、完整页数和时序信息；视觉分类：`candidate/error-taxonomy.json`。每组完整画面拼版为 `candidate/<group>.jpg`，只辅助离线审阅，不替代后续调用中的完整高分辨率页。

### 错误分类与可核验组例

| 类别 | 例子及证据 | 解释边界 |
|---|---|---|
| 漏阶段/表情 | 04-04 G041：380 闭嘴仰望、309 张嘴；96 与309更可替代。G047：311 靠近、369 接触、11 收手后的分离；v3称最后阶段“interaction lost”而省略。G035：329侧视、42回看、44低头回看。07-25 G018：412/104/415/298连续转身、手势和角色相对位置变化，人选四张 | 图像可见变化；不能只按“最好表情”压到一张，也不能认为动作结束天然无价值 |
| 过度多选 | 04-04 G029：267与437的犬群互动布局近同，v3自己称“near-equivalent”，又以时间差保留两张。G018：548牵引绳挡脸、轻微抬脚，被当成额外阶段；G044：93顶部手指与382干净画面，被当成独特互动 | 时间、背景手指和微小构图变化并非充分多选理由；需要“删去会损失什么可读状态” |
| 代表帧选错/偏好不明 | 04-04 G016：人选392，AI523；v3声称后图狗头被右边裁掉，完整拼版不支持这个绝对说法。G058：人选313，AI221；都是推车内犬的相似状态，不能仅凭人选推断不同阶段。07-31 G034：人选162，AI84+303，包含挥手表情代表选择和额外手势取舍 | 一部分是视觉描述错误，一部分是审美偏好；不能靠多选统一修复 |
| 大组比较负荷 | 04-04 G057：11张完整页，AI保留3、人选8；中间闭嘴/张嘴侧面、近远正面、转身等被压缩为三种姿态。G056：8张、AI与人选均为233+292，是规模对照 | 39组全部有完整页且已有validated结果，没有证据证明硬预算截断或传输失败。G057是过度合并/比较负荷的候选机制，不把大小当已证根因 |
| 视觉/时序/CV不足 | 07-25 G043：远处小舞台，v3自己承认个体coverage无法确认；G016远景灯光/队形也不能夸大为细微表情。G057有同秒帧，不能从文件ID推断细分先后。G051：40与216非常近似，却一起入选而漏侧脸372/335，理由与选集不完全一致 | CV全局分数只是同阶段代表的辅助，不能定义阶段；低可见性不升级为保守推荐；存储时间不等于可靠动作因果顺序 |

39组分类逐条区分“集合差异事实”“视觉解释假设”“未证实因果”。15组exact也保留为对照，不因exact就宣称理由每字正确。

## 合同与实现

`src/large_group_review.py` 为纯 request adapter 与 validator，无 provider、DB 或文件写接口。`contract_for(2)` 固定 `v3`，请求深拷贝后字节等价、响应委托现有v3验证；只有3–16张进入新合同。16张以上拒绝完整组预算，不抽样、不静默裁帧。

单次 JSON 中增加：

1. `phase_census`：每帧恰好一条 file_id+SHA、可见状态、可读性、证据、cluster；是短观察记录，不要求输出推理过程。
2. `clusters`：完整互斥分区、每簇代表、替代关系、每张省略图相对代表的比较。近似关系不能传递合并，只在真正可替代的簇内选最佳。
3. `recommended`：select 时必须等于每个可读簇的代表集合。独特阶段数与最多三个用户区分点解耦；上限16而非8，多选理由可绑定全部16帧。

`uncertain_distinct` 仅表示“可看清，但是否可由别帧替代不确定”，保守保留一个代表；自身视觉证据仍须 medium/high。`unreadable/uncertain` 可读性不能靠这个标签升级，整组走 attention/abstain。near-duplicate簇仍仅留最佳，不用“宁可多留”把所有帧拆成独特阶段。

无语义主体定位的 deterministic region 只支撑画面观察；`irreplaceable` 仍为空，不声称个体身份或完整主体覆盖。完整组图、file_id+SHA、region绑定、decision ceiling和每张推荐的medium/high证据仍验证。

全部用户 prose 要求简体中文。v4单独使用中文兼容的privacy instruction和文本检查，保留联系信息/身份猜测词法提示；不修改v3英文输出或既有privacy函数。自动门禁可排除纯英文，不是完整繁简识别器；后续人工canary需检查自然简体中文。

## 离线复现

```bash
PYTHONPATH=. /tmp/photo-dedup-py311-test/bin/python -m scripts.prepare_large_group_candidate
/tmp/photo-dedup-py311-test/bin/python -m pytest -q tests/test_large_group_review.py
/tmp/photo-dedup-py311-test/bin/python -m pytest -q tests
/tmp/photo-dedup-test-venv/bin/ruff check src tests scripts
git diff --check
```

环境路径为本次测试机已有venv；其他机器使用安装requirements的Python。准备脚本只写 `artifacts/large-group-tuning/candidate/`，不调用模型。运行时代码和schema不读取标签、不含group ID规则；脚本中的canary组名仅是离线抽样名单。标签只进入baseline统计文件，模型请求由v3请求和图像身份构建。

## Development 实测与曝光容量

Canary顺序：04-04 G041（闭嘴/张嘴）、G035（回看表情）、G029（过度多选负对照）、G057（11张）、07-25 G018（舞台全保留）、G030（6张代表替换）、04-04 G056（8张exact对照）、07-31 G028（互动阶段替代）。当前权威请求、raw、validated与usage分别在 `candidate/development-run-a-r2/<group>/` 和 `candidate/development-run-b-r2/<group>/`；最终对比和锁定hash以 `candidate/development-comparison.json` 为准。

单次judge、零自动重试；模型 `global.anthropic.claude-sonnet-5-5`，输出上限8192 tokens / 48KB。开发期费用无上限，不以美元决定输入取舍。`scripts/run_large_group_development.py` 使用独立 Converse transport、中文 instruction、原始响应落盘与逐组持久 reservation。不得调用固定4096 tokens且追加英文指令的旧 `SonnetJudge`。

候选2由 `src/large_group_projection.py` 定义。模型只输出逐帧清单和cluster代表这一份选择依据；应用确定性生成 `recommended` 与多选理由的evidence，避免同一响应中的两份选择数组相互矛盾。原始模型schema与带 `derived_fields` 的validated sidecar明确区分，不把客户端生成字段冒充模型原文。唯一一次有假设的合同修正及首轮失败原文保存在 `candidate/revision-2-rationale.json` 与 `development-run-a/`。Converse原生reasoning metadata只保存在provider receipt，不进入选片JSON；保存响应的本地重新验证不触发付费重试。

A路线完整曝光组内全部成员。B路线先用本地64×64 RGB平均差、局部tile差、dHash、图像梯度代表分数和不超过1秒的存储时间作micro-cluster；只直接比较每张候选与代表，禁止相似度传递合并。预筛策略在读取人选标签前生成全部39组计划并落盘，标签只用于后续曝光召回验收。相同输入组复用A结果而非重复调用制造随机差异；改变输入的组只发送完整代表页，所有可能混入被排除原图的storyboard/contact sheet/crop均移除。contact sheet包含N张原图就计N张曝光，绝不按一个image part记一张。

固定本地预筛结果：strict为149→149，无省图；balanced为149→148，human-selected exposure recall 100%；aggressive为149→142，但人选曝光召回仅97.37%，因此拒绝且不发起B模型调用。balanced仅改变04-04 G058的5→4张输入，其余38组不变。这个规模下的保守省图仅0.67%，不能宣传为显著容量突破。B若任一组漏选增加或人选曝光召回不足100%，即拒绝，即使图片数量更少。

运行入口：`python -m scripts.run_large_group_development canary`；八组结构/中文验证完成后用 `development` 只调用剩余31组；`python -m scripts.evaluate_photo_exposure` 为纯本地预筛；`python -m scripts.run_exposure_canary` 只调用具有省图且召回达标的B组；`python -m scripts.report_large_group_development` 输出development对比、Pareto容量与锁定hash。不得用这些脚本读取或运行holdout。

停线：原片/DB/v3/decision/keeper/delete写入、holdout访问或身份错误立即停止；任何模型输出失败保留raw且零自动retry；本轮最多一轮有明确假设的模型合同修正。完成39组development对比并锁定候选后停止，不继续调参。完整请求、响应、usage、逐组曝光与版本见candidate下的development-run目录；执行口径以 `candidate/execution-policy.json` 为准。

## 39组真实云端 A/B 结果与候选状态

权威实测结果：`candidate/development-comparison.json`；处置：`candidate/disposition.json`。以下是39组development assisted-label集合比较，不是holdout或盲测准确率。

| 路线 | exact | precision | recall | F1 | 漏选 / 多留 | 成员→VLM曝光原图 |
|---|---:|---:|---:|---:|---:|---:|
| 现有v3基线 | 15/39（38.5%） | 76.7% | 60.5% | 67.6% | 30 / 14 | 149→149 |
| A：完整组v4候选2 | 10/39（25.6%） | 57.1% | 84.2% | 68.1% | 12 / 48 | 149→149 |
| B：balanced本地预筛+v4候选2 | 10/39（25.6%） | 57.5% | 85.5% | 68.8% | 11 / 48 | 149→148 |

A、B的VLM组数均为39；高分辨率候选和image parts分别为149、148；总像素分别为468,713,472、465,567,744；人选曝光召回均100%。B仅G058实际更换视觉输入并新增一次调用，其余38组输入一致，复用A结果。G058的A选221/364，B选221/364/313，人选313，因此B未增加漏选，但只省一张输入。该单次差异不能单独证明预筛导致质量提升。

**候选2锁定为实验结果，不替换当前默认v3。** 阶段漏选虽减少，但多留大幅上升，exact下降；15个原exact组中11个退化，18/39组被全保留。G035、G041、G047、07-25 G018/G040等阶段覆盖得到改善；G019/G028/G032/G036等近重复组却出现过度拆分。B的balanced通过必要曝光/漏选门槛，0.67%的省图不构成显著规模优势；aggressive因漏掉2/76张人选，在模型调用前已拒绝。

本轮judge合计44次调用（首版4次、候选2的39次、B的1次），累计173次原图提交；8组canary已包含在39组中，未重复跑完39组。请求模型为Sonnet 5.5，共1,181,229 input及76,787 output tokens。唯一一次合同修正用于去掉重复shortlist字段；一次Converse envelope本地重新验证未新增模型调用。失败原文保留，自动重试为0。673个授权development v3文件hash未变，原片/DB未由实验runner打开，decision/keeper/delete写入为0，holdout未参与。

这里的B真实调用结果与下方production funnel的teacher replay是不同评估口径；不可把replay表当作B的新模型调用结果。

## Production funnel：VLM teacher，本地 CV student/router

候选锁定后的生产架构不再是“所有相似组完整送 VLM”。`src/photo_exposure.py` 的 `photo-exposure-router-v1` 是无标签、无 provider、无 DB、无写接口的本地路由层：从调用方提供的解码图、可信/unknown 时间和 `file_id+SHA` 身份构造 descriptor；每张候选只与当前代表直接比较，避免由相似度传递误合并。输出每组 `local_resolved`、`vlm_required`、micro-cluster、代表身份及完整 VLM exposure allowlist。生产链只能用 allowlist 构造视觉资产；contact sheet 含 N 张仍算 N 张 exposure。

当前可计算 surrogate 包括 64×64 RGB 差、局部 tile P95、dHash、颜色直方图、edge layout、时间间隔和局部梯度质量差。已有 Sonnet 5.5 development teacher 的 112 个阶段簇提供风格参照：同簇 pair 的 RGB MAE 中位数 0.079、dHash 中位数10、时间中位数1秒；异簇分别为0.106、17、2秒。分布明显重叠，证明这些 CV 特征适合做保守近重复压缩和 hard-router，不足以单独判定表情、动作阶段或用户偏好。脸部/表情/姿态 detector 后续可作为附加 veto 或 hard signal；状态 unknown 时不得把照片裁掉。

`recall_first / balanced / efficient` 三档均先做本地 micro-cluster；只有极同质、单簇且质量领先明确的三张以上组才允许 `local_resolved`，否则将保守代表送 VLM。二张组始终标记 `v3_unchanged`，router 不产出替代选择。39组development本来就是筛过的疑难大组，因此三档均为0个local-resolved；这不是全图库local route为0的生产预测。实测 Pareto：

| profile | dev member→VLM source | VLM groups | human-selected decision coverage / VLM exposure recall | v4 teacher replay exact / P / R / F1 | 漏 / 多 |
|---|---:|---:|---:|---:|---:|
| recall_first | 149→149 | 39 | 100% / 100% | 25.6% / 57.1% / 84.2% / 68.1% | 12 / 48 |
| balanced | 149→148 | 39 | 100% / 100% | 25.6% / 57.1% / 84.2% / 68.1% | 12 / 48 |
| efficient | 149→142 | 39 | 97.37% / 97.37% | 23.1% / 56.9% / 81.6% / 67.0% | 14 / 47 |

因此 balanced 是当前 development 的保守省图点，efficient 因漏两张人选而拒绝；任何生产 profile 也必须允许一键回退 recall_first。按 PRD 已记录冻结图库“大组照片占输入24.7%”作为显式外推假设，development 路由对应每1000输入约64.65个VLM大组，unique source photos约247.0 / 245.34 / 235.40；它不是生产实测，且未包含二张组容量。不得用 overview 包含全组却把曝光记成代表数。

可复现入口：

```bash
PYTHONPATH=. /tmp/photo-dedup-py311-test/bin/python -m scripts.run_production_funnel
```

权威输出在 `artifacts/large-group-tuning/production-funnel/`：`router-plans.json` 逐组列出 local/VLM路由与代表SHA，`frontier.json` 给出 teacher surrogate、开发集质量和每1000输入外推，`source-integrity.json` 证明本轮读取的development v3请求/结果/图片前后hash一致及六类写入为0。脚本只读取39组development；holdout仅从split读取21个名字/计数，不读取其图片、标签、旧AI结果，也不调用模型。以上全部是development拟合与容量估算，最终是否提升须由parent独立holdout决定。

## 二张组本地 resolver 候选（development-only）

当前候选 `pair-exposure-resolver-v3 / A`：raw DINO cosine distance≤0.045、RGB MAE≤0.09、dHash≤8、去重后的可用质量family票数≥6、winner margin≥4。本地只允许 `local_select_one`，其他全部 `vlm_required`，禁止本地keep-both。原始DINO仅从只读库存中按88组development的176个date-scoped file IDs查询，精确按little-endian float16×768解码，校验bytes/dim/finite/nonzero norm后计算cosine。

质量票按detail、CLIP-IQA、MUSIQ、exposure、face quality、face composition、eye-open七个来源family合并，每family至多一票；同源矛盾弃权，quality_score别名不计票。这是来源去重，不声称统计独立。88组中14组通过近似门，但可用family最多5个，因此A本地解决 **0/88**；88组/176张继续v3，15个双选组0 leak。本地精度无样本、不可估计；没有为追预期组数降低门槛。固定六日期压力测试与strict/A/wide三点frontier均0本地。

整库route-rate投影为510/2065张，即246.97张unique source photos/千输入，当前节省0。候选未接live，43个pair holdout未打开。权威结果、错误上界、DB/原片hash、逐组与日期折证据、复现方式见 [PAIR_ROUTER_TUNING.md](PAIR_ROUTER_TUNING.md) 和 `artifacts/pair-router-tuning/candidate/current.json`。
