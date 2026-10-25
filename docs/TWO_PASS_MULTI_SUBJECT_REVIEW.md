# 两段式多主体 / 舞台 review-only

## 交付与状态

离线链路已实现于 `src/two_pass_review.py`，入口为 `scripts/probe_two_pass_review.py`。离线入口只使用明确命名的 `fake:` 模型和确定性 FakeModel；真实模型入口单独使用显式授权、allowlist 限制的 Bedrock adapter，见 `TWO_PASS_REAL_PILOT.md`。两条入口均无重试、升级、生产接入或 decision table 写接口。离线`select`仅是fixture契约输出；真实pilot结果与调用状态以`TWO_PASS_REAL_PILOT.md`为准。

`artifacts/two-pass-multi-subject/` 保存 G004 舞台、G007 多人组的完整页、ROI comparison strips、请求、L0/L1/L2、sidecar、调用账本和测试日志。原 `artifacts/multi-subject-narrow/` 三组真实结果保持 **attention_only**，其 JSON hash 在新总报告中固定核验，未改为通过。

## 为什么由 VLM 先做 subject map

舞台的关键区域可能是远处演员、道具、队形与舞台光，最大脸和最高置信 person 框往往属于前景观众。多人互动还存在身体交叠、镜像和局部遮挡。CV/person-detector-first 会在主体进入比较链之前漏掉它们。局部质量检测再精确，也无法补回没有进入 ROI 的主体。

Pass A 从完整高分辨率组图识别匿名关键主体与互动/舞台区域；CV只在这些经过结构验证的区域上测量；Pass B同时看到全图、局部对照、subject map及CV派生证据。模型提出的完整覆盖仍是候选声明，不能升级为人标或独立验证的 completeness。

## Pass A：subject map

每帧一张完整 group page，不裁脸、不省略帧。原生短边至少1024，显示短边目标1536，缩小但不上采样。请求仅包含最简场景提示、frame file_id+SHA、完整页与输出 schema；不传 detector 框、旧 keeper、utility、pair margin 或旧 decision authority。

Schema：`schemas/two-pass-subject-map-v1.schema.json`；运行时 `validate_map` 同时检查跨字段约束：

- `scene/regime/confidence`；每帧独立 coverage state、uncertainty 和 confidence。
- frame-bound normalized `[x1,y1,x2,y2]`，严格有限数、顺序和边界，主ROI面积0.5%–85%；合法bystander/unknown小框只进入coverage ledger，不因面积阻止整组。每帧最多4区域，全组16区域。
- `role=key_subject|supporting|bystander|unknown`；`kind=subject|interaction|stage_area`。
- 区域 grounding 必须指向当前完整页的 file_id+SHA，`visible=true`。不允许别名join、额外身份字段或实名字段。
- 同类区域IoU>0.8拒绝；关键区域与旁观者IoU>0.3视为role conflict。该阈值是重复/冲突防护，不是语义主体检测器。
- 最多3条匿名跨帧 correspondence 候选；一帧在一条track最多一个区域，一个区域最多属于一条track，所有关键区域都必须被比较，不能因预算静默丢主体。
- 主strips允许同role、同kind的key_subject/supporting tracks，包括stage_area；bystander/unknown tracks保留账本但不入strips。主track歧义、角色冲突、缺关键区域或未比较仍拒绝。
- uncertain/resolution/occlusion和低覆盖置信可进入CV/B，sidecar与B请求的`decision_ceiling=attention_only`禁止推荐；只有无覆盖限制且关键区域全被比较时才允许select。missing_key、reflection与unresolved_roles仍在B前停止。

结构验证只证明引用与框合法，不能识别一个自信但错误的主体图。尤其“漏掉主体但自称完整”、把观众错误标成 key_subject、错误镜像对应，仍需要真实模型/人工验证，当前没有宣称已解决。

## 本地 enrichment 与视觉预算

复用 `subject_set_evidence.CRITERIA/iou` 的集合判据和几何操作，`vlm_visual_budget.subject_map_plan` 增加隔离的 VLM-map planner；原生产 detector planner与三组旧证据保持原样。

- 原有CV通过现有 `vlm_review.build_layers` 的完整 L0→L1→L2 compiler。所有持久化判据族保留在L1/coverage ledger；L2保留可用的区分性标量、pairwise及明确的未测/无区分状态，不能偷换成只看top质量分。全图L2的展示标签另附 `frame_bindings` 到file_id+SHA，不能按数组位置猜测身份。
- 每个Pass A区域本地计算亮度均值、暗部/亮部占比、梯度能量；它们是像素测量，不是审美评分。
- 将现有local quality按空间重叠映射；保留来源region_id/kind，并明确是旧检测区域的测量，不是新ROI重新跑出的MUSIQ/CLIP-IQA。
- face与eye只从其独立bbox做空间映射，不将全图 `ALL_OPEN` 广播到主体。pose有唯一IoU join时记录可见关键点数；多重/无join保持unknown。新ROI的模型IQA、语义阶段、完整性、遮挡等无测量时保持unknown。
- 各track增加32×32 RGB平均绝对差，明确为未配准外观差，不是身份或语义相似性。无本地模型下载/推理。
- 每条track输出一张逐帧comparison strip，每格是真实ROI短边≥256，显示目标384，不上采样。任一关键ROI不够大即停止，不拼padding冒充像素尺寸，也不降权丢弃小主体。
- 旁观者保留在完整页与coverage ledger，不进入主ROI strips。area内部仍可能自然包含观众/路人像素；矩形区域不是分割mask，不能宣称像素级排除旁观者。

G004 fake fixture使用舞台整体区域，不冒充逐个演员/眼睛已解析；G007使用前景和右侧互动区域，不将镜像关系当成人物身份真值。手写fake框的完整候选声明仅用于测试后续链路。

## Pass B：deep judge

输入含所有完整页、所有关键region comparison strips、scene map与全部ROI / full-frame派生L2判据。原始二进制向量、文件路径和旧选片权威不入模。

Schema：`schemas/two-pass-deep-judge-v1.schema.json`；`validate_judgment`追加证据和置信检查：

- `decision=select|attention_only|abstain`；低置信或attention_only ceiling不能select，attention/abstain不携带推荐或不可替代项。
- `recommended`允许多帧；每一项绑定本帧 file_id+SHA和key_subject/supporting ROI证据。select至少一个中/高置信区分点，最多3个。
- 多选必须提供 `multi_select_reason`，证据覆盖全部被推荐帧。
- `irreplaceable`按frame+subject ROI及可选phase组织；phase限定interaction/gesture/formation/lighting/appearance，不虚构时间顺序。每项必须属于推荐帧，并引用同帧同主体的证据。
- 中高置信只支持review-only语义建议，不能写keeper或触发删除；区域引用也不能证明一句视觉断言一定正确。

## 硬成本上限

| 项目 | Pass A | Pass B |
|---|---:|---:|
| 调用次数 | 1 | 1 |
| 图片数 | 4完整页 | 4完整页+3 strips=7 |
| 输入文本token保守上界 | 6,000 | 36,000 |
| 请求最大输出tokens | 4,096 | 4,096 |
| 响应JSON最大bytes | 12,000 | 12,000 |

全组最多4帧、16区域、3条track。每pass图片合计≤16MB且≤20M pixels。超过任一限制立即attention_only，不能拆出无限调用、隐式缩图、丢关键主体或换更贵模型。

文本上界按完整序列化请求（含prompt、schema、image manifest）的UTF-8字节数计，作为byte-level tokenizer的保守上界；不是provider账单token。本次正例A为3,777–3,779，B为20,976–30,022；按既有byte/3启发式仅约1.3K及7–10K，成本门禁使用更保守的上界而非该估计。视觉token由provider/model决定，当前以图片数量/像素/bytes约束，没有假装知道价格。真实adapter记录provider实际usage及显式假设费率；它不将文本字节上界冒充账单token或实付金额。两pass完整页重复输入必然增加成本；本实现用硬停控制，不宣称成本优于单pass。

## Sidecar、隐私与缓存

每pass显式模型ID。账本记录request hash、输入文本上界、图数、响应bytes和external calls=0；billed tokens为null而非伪造为0。sidecar包含版本、两模型、完整输入/CV、fake响应、prompt/schema/实现hash；原图每次重验SHA，资产每次重验SHA，过期缓存不复用。读取缓存时本次调用数=0，与原生成账本区分。

匿名R/T只表示当前组视觉候选，不建立跨组人物身份。拒绝未声明字段，prompt禁止实名、身份推断/人脸识别。schema不能防止模型在自由理由文本内泄露姓名，未来真实模型输出仍需隐私检查。派生图片与完整页同样含人脸和现场细节，按原图私有数据保存；产物保存在repo私有目录；真实pilot的授权外部调用和隐私边界见`TWO_PASS_REAL_PILOT.md`。

## 确定性证据与失败边界

每组5种fake场景：正例、声明漏检、旁观者污染、歧义对应、low-confidence。共10例：正例走完A/CV/B并测试多选/不可替代；漏检、污染、歧义均只执行A并attention_only；low只attention。**fake证明接口与门禁，不证明真实VLM主体召回、镜像理解或偏好正确率。**

运行：

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/probe_two_pass_review.py
PYTHONPATH=/home/ubuntu/.cache/uv/archive-v0/arVVMhmvO8QpKbeh/lib/python3.13/site-packages \
  OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests/test_two_pass_review.py
```

优先查看：

- `artifacts/two-pass-multi-subject/2026-07-25-G004-positive/strip-T1.jpg`
- `artifacts/two-pass-multi-subject/2026-04-05-G007-positive/strip-T1.jpg`、`strip-T2.jpg`
- 同目录的 `pass-A-request.json` / `pass-B-request.json`、`L1.json`、`sidecar.json`
- 总 `report.json` 与 `run-log.md`。

旧评测树仅mode=ro SQL与原图读取；前后decision-table摘要一致。未访问Google Photos、调用外部模型、改服务/config、发布、commit或自动删除。离线架构与测试完成后停止，真实效果和生产准入留给parent独立验收及后续明确授权。
