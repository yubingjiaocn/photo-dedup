# 单次云端 judge 批处理

入口：`scripts/run_cloud_judge_batch.py`。数据流为 **冻结 CV → 本地 8B 粗粒度 scene router → 确定性完整视觉 packet → 一次 Sonnet judge**。8B 输出只决定补充视图的类型，不定义主体覆盖，不决定哪些完整照片可以进入云端。

当前冻结批次 **194/194 组已执行完毕**，使用 12 组分层 canary 记录与剩余 182 组顺序执行；并发始终为 1。当前续跑规则把单组 schema、语义、措辞和成本问题记为 warning 或 failed，不因这些质量问题全局早停。只有原片/冻结 DB 意外改写、输入身份错绑、路径逃逸、重复调用失控，或连续 10 组完全没有有效响应才硬停。429 和不确定 I/O 消费槽位，不自动重试。

权威结果：`artifacts/cloud-judge-batch/authorized-scene-run-v3/summary.md` 与 `summary.json`；逐组证据在 `call-ledger.json`、`report.json` 和各组目录。`execution-policy.json` 记录当前执行策略，canary 阈值保留为诊断数据。

## 当前执行结果

| 指标 | 结果 |
|---|---:|
| 冻结组 / 完整组图页 | 194 / 510 |
| 新增 Sonnet 5.5 调用 | 191 |
| 继承 v2 已消费调用 | 3 |
| 总调用 / provider receipts | 194 / 194 |
| Canary 校验 | 11/12 validated |
| 全量校验 | 187 validated、7 failed_no_retry |
| Validated decision | 185 select、2 abstain、0 attention_only |
| 有效多选组 | 54 |
| 保留 low 弱观察的有效组 | 105 |
| 新增 tokens | 3,421,088 input + 104,420 output = 3,525,508 |
| 总 tokens | 3,514,159 input + 106,396 output = 3,620,555 |
| 新增 / 总估算 USD | $11.829564 / $12.138417 |
| 最高单组估算 USD | $0.191721 |
| SDK 重试 / 决策表写入 | 0 / 0 |

成本使用 input $3/M、output $15/M 的明确假设，**不是核实过的官方费率或 AWS 账单**。全部 194 组都有 provider receipt，未知成本组为 0。按日期、组大小、输入 scene route、有效 decision 和 validation 分层的完整结果见 `summary.md`。失败组使用 `unvalidated` 单独统计，不伪称为 attention_only。

G014、G057、G007 通过 hash、wire、冻结原片重渲染、region、时序和 CV L2 核对后纯本地导入；新增调用为 0，raw 未改。v2 原始目录、三次调用账单信息与原失败记录保留原样。

| 单组失败类型 | 组数 | 组 |
|---|---:|---|
| Provider 含非 text 内容块 | 3 | 04-04 G034、04-05 G016、07-31 G018 |
| 推荐 evidence 引用了其他 frame | 2 | 06-23 G003、07-25 G013 |
| 多选理由包含额外非推荐 frame | 2 | 04-04 G042、07-25 G044 |

这些组均保留 `failed_no_retry`，没有补调或修补 raw，也没有阻断后续组。多选失败的两组已经覆盖全部推荐，但还引用了一个未推荐 frame，触发当前严格集合绑定规则；原始证据可直接复查。含非 text 内容块的完整 receipt 保留，最终 text 另存于 `rejected-text/`。

`deterministic-audit.json` 核对全部已消费组的产物 hash、完整组页、原始 raw 与 validated 一致性、v2 证据未变、冻结 inventory 未变及 usage/cost 对账。没有 keeper/delete/decision 写入，没有 Google Photos、服务/config、发布或 commit 操作。

`validated` 只表示格式、置信度与证据绑定通过，不等于描述正确或符合 Willy 偏好。三组 Astra 视觉抽查发现：G014 的睁眼/舌头差异可见；G007 的背景鞋子裁切描述有误；G004 的舞台上下位置描述有误。详见 `astra-visual-check.json`。没有 Willy 人标，不报告偏好准确率；最终由 parent 独立验收。

## 视觉 packet

每组的所有照片始终各有一张完整高分辨率页。scene router 的低置信、失败、缺少组条目都按 unknown 构造完整页，不因缺少 local subject map 取消云端审阅。

| scene | 必须包含 | 补充视图 |
|---|---|---|
| action / interaction | 全组完整页 + 按时间排列的完整 storyboard | 有独立人工/parent 确认记录时增加动作 strip；否则保留完整图 |
| stage | 全组完整高分辨率页 | 每帧固定四象限网格 + 两个中心尺度 |
| group | 全组完整页 | 每帧固定四象限网格；可增加独立确认的 subject strips |
| ordinary / unknown | 全组完整页 | 不依赖 8B boxes |

- 8B box 仅能作为 `untrusted_local_attention_hints` 文本元数据，不进入裁切、CV 空间映射、主体覆盖或推荐证据。其引用没有可支持选择的 region ID。
- 错误、越界或 hash 不符的可选 hints 被显式记录在 `visual_warnings` 后忽略；完整图审阅继续。没有模型补框、repair 或重调。
- 补充网格是确定性几何视图，不宣称其中存在某个主体。所有视图都有 frame 的 file_id+SHA 和 box 绑定。
- 完整页按 EXIF 时间再按 file_id 排列；无 EXIF 时按 file_id，时间字段为 null。不沿用旧 keeper 排序，不推断动作阶段。
- 8B scene 猜错时，云端仍看到整组完整图，并被要求自行识别场景与有效区别。视图类型不是场景真值。

单组硬预算：最多 16 帧、20 张视觉资产、32MB JPEG、48M pixels；当前冻结组最多 11 帧。完整页原生短边至少 1024，目标 1536；固定网格格子原生短边至少 256，均不放大。超预算时明确 attention，不抽样、漏帧或冒充整组。补充 mosaic 每轴不超过 8000 pixels。

CV 使用既有 derived compiler，绑定所有 frame，去除 selection authority 和原始向量。批处理 L2 按测量类别轮转、类别内按归一化差异选择最多 24 个区分性 scalar，保留全部 frame 和有界 pairwise 信息，总计最多 30KB；projection 显式记录原始/选中/省略数量，完整测量保留在 L0/L1，不改写测量值。完整请求含 schema/metadata/隐私约束的文本上限 48KB，输出最多 4096 tokens / 12000 bytes。完整图、时序和未知值不会为缩短文本而被悄悄删减。

## 云端输出与显示

批处理 schema：`schemas/cloud-scene-judge-v1.schema.json`。云端每组只返回一次独立 judgment：

- 最多 **3 个区分点**，每条文本不超过 **500 Unicode code points**。
- `region` 指向 packet 中的完整 frame 或确定性几何视图，不指向 8B 的 subject box。推荐需要 medium/high 的云端视觉证据，并逐 frame 绑定；允许有证据的多选，最多 8 帧。
- Contract v3 将 **decision support** 与 **weak attention note** 分开。select 的 global confidence 必须 medium/high；每个 recommended frame 至少有一个 medium/high evidence binding。推荐可以附带低置信观察，但低置信不能成为唯一或必要支撑。
- discriminator 允许 medium/high/low 混合，也允许全 low 的明确弱观察；不因一个 low 丢弃整组。multi-select reason 必须覆盖所有推荐且仅使用 medium/high；irreplaceable 若适用则逐项 medium/high 并绑定推荐 frame。非 select 不允许 recommended 或 multi-select reason，attention_only/abstain 的限制保持不变。
- 几何视图不证明个体身份或关键主体完整性；当前 scene-judge schema 的 `irreplaceable` 必须为空。不得把某个网格当成已验证关键主体来宣称个体不可替代。
- `evidence-bindings.json` 显式列出每个区分点的 region、frame、box、role；role 是 `frame_view`，只描述视觉输入类型，不是语义主体角色。

raw text、解析后的 raw JSON、provider receipt、validated JSON 都保留。校验不会改写或截短事实。前端 `ReviewText` 超过 240 字时折叠显示 180 字预览，完整文本保留在可展开段落中；已接入现有独立 VLM 建议与低置信注意点区域；低置信项按 confidence 显示“弱观察（不支持推荐）”，不丢弃同组的有效建议。批处理 sidecar 暂不自动接入生产 review API。

隐私词法检查允许“holding a phone”这类可见物体描述，仍拒绝 phone number / phone:、号码、邮箱、姓名/身份推断等触发项。历史 contract v3 冻结批次保留原始英文输出；后续调用要求所有面向用户的理由使用简体中文，JSON key、枚举、frame/region ID 仍保持 schema 定义的英文。词法检查不是通用隐私/事实分类器。

## 已保存的真实 attention-only 基线

`artifacts/cloud-judge-batch/real-baseline/`：

| 组 | 三条文本长度 | 纯重校验结果 |
|---|---|---|
| 2026-07-25 G004 | 242 / 249 / 193 | validated attention_only |
| 2026-04-05 G007 | 209 / 221 / 243 | validated attention_only |

先逐项核对原账本绑定的 A/B request、wire request、provider response、raw text SHA，再重新执行 schema 与 frame/region 门禁。A coverage 与原请求 file_id+SHA 一致，B scene_map 与 A 原文一致。原 `artifacts/two-pass-real-pilot/` 不改写；新目录保留原始 `.txt`、解析后 raw JSON、validated JSON 和 provenance。validated B 与原始解析对象完全相等；没有新增模型调用。

这两份保存结果使用 `two-pass-deep-judge-v1.schema.json` 和原 subject map。其角色规则是：key_subject/supporting 可支撑 select；bystander/unknown 仅能在 attention_only 中支持遮挡、重叠、干扰、构图或取景观察，必须能解析到具体 frame。它们不能支撑推荐、多选理由或 irreplaceable。G007 的 R3/R7 frame binding 单列在 provenance。当前 scene-router 批处理的视觉输入与这份历史真实基线分开保存，不把历史 cloud A 伪称为新 local 8B 结果。

## 冻结范围

权威快照：`artifacts/cloud-judge-batch/frozen.json`。

| 日期 | files | groups | inventory |
|---|---:|---:|---|
| 2026-04-04 | 556 | 62 | development |
| 2026-04-05 | 160 | 20 | development |
| 2026-06-23 | 113 | 6 | local-holdout |
| 2026-07-25 | 559 | 52 | development |
| 2026-07-26 | 77 | 14 | development |
| 2026-07-31 | 600 | 40 | development |
| **合计** | **2065** | **194** | **6 个唯一日期** |

不重复加入 pose-holdout 或 stage2 variants。快照记录 inventory SHA、成员 file_id+SHA；入口与收尾均以 `mode=ro` 核对 DB 字节 hash、计数与组成员。

## Scene-router manifest

Schema：`schemas/local8b-scene-router-v1.schema.json`。入口必须先读取 manifest；文件不存在就停止，不自行运行 local/cloud 模型。例：

```json
{
  "version": "local8b-scene-router-v1",
  "snapshot_sha256": "<frozen.json 中的 snapshot_sha256>",
  "groups": [
    {
      "group": "2026-07-25-G004",
      "status": "routed",
      "scene": "stage",
      "origin": "local8b",
      "model": "Qwen/Qwen3-VL-8B-Instruct@<实际revision>"
    },
    {
      "group": "2026-04-05-G007",
      "status": "routed",
      "scene": "group",
      "origin": "local8b",
      "model": "Qwen/Qwen3-VL-8B-Instruct@<实际revision>",
      "attention_hints": {"path": "G007/hints.json", "sha256": "<文件字节SHA256>"}
    }
  ]
}
```

允许 scene：`action|stage|group|interaction|ordinary|unknown`。status：`routed|failed|pending`。failed/pending 或缺条目的组均使用 unknown 完整页 packet。可选 hints JSON 是列表，每项严格为 `{frame:{file_id,content_sha256}, box:[x,y,x2,y2], confidence:low|medium|high}`；没有角色、身份、推荐或 coverage 权限。相对路径不得逃逸 manifest 所在目录。

### 独立确认的补充 strips

通过独立 `--confirmed-visuals` 文件输入，不能由 scene-router 条目声明确认。例：

```json
{
  "snapshot_sha256": "<冻结集SHA>",
  "groups": {
    "2026-04-04-G003": {
      "confirmed_by": "parent_visual_review",
      "tracks": [{"regions": [
        {"frame": {"file_id": 1, "content_sha256": "<原图SHA>"}, "box": [0.2,0.2,0.8,0.8]}
      ]}]
    }
  }
}
```

`confirmed_by` 只允许 `human` 或 `parent_visual_review`，由受信任操作方提供，不能用本地模型自称确认。runner 核对 frame、边界、原生像素后从原图重渲染，仍保留全部完整页。不可用的确认条会生成显式 warning 并回到完整图与该 scene 的确定性视图。该文件是本地操作方的审核记录，不是密码学签名。

## 单次调用、恢复、计费

唯一模型：`global.anthropic.claude-sonnet-5-5`，Bedrock Converse / us-east-1。每组最多一个云端 judge dispatch，无云端 Pass A、repair、重试、升级或 fallback；SDK `total_max_attempts=1`。授权入口并发固定 1，授权上限 2；本轮不升并发。通用底层 runner 的离线测试允许到 4；每个真实调用使用独立 SDK Session。

普通命令只 prepare；`--execute` 才开启 transport。dispatch 前原子写入并 fsync reservation，进程锁保护同一批目录。超时、schema 拒绝或不确定 I/O 都消费此组槽位。恢复时 reserved/failed_no_retry 不再调用，成功结果复核全部产物 hash 后 cache hit。始终使用同一权威输出目录，不能换目录绕过已消费的槽位。contract v3 continuation 仅导入原来三个已消费槽位及其账单信息，不清零、不重发。

`contract.json` 固定 frozen SHA、模型、费率、关键实现/schema、独立确认文件 hash；已预约组再固定自身 scene-router 条目。未消费组允许后续补充路由，已消费组不可更改输入后重跑。实现或契约变化会拒绝直接恢复，不提供 force-replay。

每组保存 request/response/assets/raw/validated SHA、usage、延时、状态及路由。provider receipt 在内容校验前落盘，格式失败也保留已知 usage/cost；超时或 usage 缺失则成本 null，不当作免费。`external_calls_total` 是保守 dispatch reservation 数，异常时请求可能尚未到 provider；`provider_receipts` 是已有回执数。`external_calls_this_run` 排除 cache hit。

默认 $3/M input、$15/M output 是明确标记的估算假设，**不是核实过的 Sonnet 5.5 费率或 AWS 账单**。可传入确认费率及来源。provider 返回 cache tokens 时，不静默套用无缓存费率。

## 命令

本机已有完整依赖 Python：`/tmp/photo-dedup-py311-test/bin/python`。以下在 repo 根执行。

```bash
# 只读冻结清单；不同的已有快照拒绝覆盖
/tmp/photo-dedup-py311-test/bin/python scripts/run_cloud_judge_batch.py --freeze

# 历史真实 A/B 纯重校验，0模型调用
/tmp/photo-dedup-py311-test/bin/python scripts/revalidate_saved_judge.py

# 六类 scene 的明确离线 fixture，不能连接网络
/tmp/photo-dedup-py311-test/bin/python scripts/run_cloud_judge_batch.py \
  --scene-router artifacts/cloud-judge-batch/scene-router-fixture/scene-router.json \
  --fixture-responses artifacts/cloud-judge-batch/scene-router-fixture/responses.json \
  --output artifacts/cloud-judge-batch/scene-router-fixture/release-run

# 真实 local scene-router 交付后，只 prepare、不调用
/tmp/photo-dedup-py311-test/bin/python scripts/run_cloud_judge_batch.py \
  --scene-router <scene-router.json> \
  --output artifacts/cloud-judge-batch/authorized-scene-run
```

首次建立 v3 continuation（纯本地重校验，完成后不要重复运行）：

```bash
/tmp/photo-dedup-py311-test/bin/python scripts/continue_cloud_judge_v3.py
```

当前授权执行/恢复入口（先记录 canary，再顺序执行冻结剩余组；已消费组不重调）：

```bash
/tmp/photo-dedup-py311-test/bin/python scripts/run_authorized_cloud_judge.py \
  --scene-router artifacts/cloud-judge-batch/authorized-input/scene-router.json \
  --output artifacts/cloud-judge-batch/authorized-scene-run-v3 \
  --execute
```

批次结束后可重复运行纯本地验收与分层汇总，不发模型请求：

```bash
/tmp/photo-dedup-py311-test/bin/python scripts/audit_cloud_judge_v3.py
```

canary 每个日期两组：优先已有 local 粗场景组（否则最小组），再选该日期其他组中最大的一个；含两个 11 张大组，不挑低成本组冒充覆盖。10/12、单组 $0.20 等阈值保留为 canary 诊断指标。单组质量、成本及传输问题留档并继续；原始输入/DB、身份、路径、调用唯一性安全错误或连续 10 组无有效响应会停止并写 `stopped.json`。每个已消费槽位均不自动重试。恢复始终使用同一目录和输入；stopped 记录不会自行解除。

当前 `authorized-input/scene-router.json` 从已有 Qwen3-VL-8B 的四份原始响应提取并 hash/身份核验粗场景：action、stage、group、interaction；190 个没有 local 场景结果的组明确 pending，走 unknown 完整页。没有新增 local 推理，也不沿用其主体框。映射、外层 code fence 去除和来源哈希见 `conversion-provenance.json`。如以后需要人工确认的 strips，可通过普通 prepare 入口的独立 `--confirmed-visuals` 参数准备；当前授权 canary 使用完整页与确定性补充视图。

## 离线验收

当前代码门禁：focused **151 passed**、broad **1187 passed**、frontend **51 passed**；typecheck、build、scoped Ruff 与 diff-check 均通过。日志和最终确定性验收见 v3 目录的 `verification.json`。

权威 fixture：`artifacts/cloud-judge-batch/scene-router-fixture/release-run/`。六类 scene 各一个 fixture，8B hints 故意含极小框，证明它们不裁掉完整图。fixture 是手写路由/输出，不代表真实 8B 或 Sonnet 准确率。仅六个有 fixture response 的组执行离线完整链，其余 188 组明确标记 `fixture_case_not_selected`；真实执行模式不使用这个 fixture 排除规则。

统计 schema：`schemas/cloud-judge-batch-v1.schema.json`，含每组 route/state、scene_routes、usage/cost、未知成本和零 decision 写入统计。结果见 `artifacts/cloud-judge-batch/verification.json`，确定性复核见 `deterministic-verification.json`。

门禁命令（前端 build 完成后再跑 Python broad，避免 dist 瞬时清空）：

```bash
(cd frontend && npm test && npm run typecheck && npm run build)
/tmp/photo-dedup-py311-test/bin/python -m pytest tests/test_cloud_judge_batch.py tests/test_two_pass_review.py tests/test_two_pass_bedrock.py tests/test_vlm_evidence_layers.py tests/test_vlm_review.py -q
/tmp/photo-dedup-py311-test/bin/python -m pytest tests -q
/tmp/photo-dedup-test-venv/bin/ruff check src/cloud_judge_batch.py src/two_pass_review.py scripts/run_cloud_judge_batch.py scripts/revalidate_saved_judge.py scripts/run_authorized_cloud_judge.py scripts/continue_cloud_judge_v3.py scripts/audit_cloud_judge_v3.py tests/test_cloud_judge_batch.py
git diff --check
```

`pytest tests` 避免把 artifacts 内封存的旧测试副本重复收集。真实偏好正确率和 8B scene 准确率均未测量。批处理结构化通过率不能代替人工偏好或语义正确率。
