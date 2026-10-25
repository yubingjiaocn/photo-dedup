# 可学习 Review MVP

每次 cloud review 保存都记录一次不可变事件；现有 review-state 仍是 UI 当前状态。学习闭环是 **本地导出文件 → 手工云训练 → 导回数据 artifact → shadow 比较**。不建账号或 SaaS，不自动上传、重训或调用外部模型；learner 永远没有 route、execute、keeper、删图或原片写权限。

## 数据边界

- 默认训练包只有已审阅非 holdout 组的 2048 review JPEG、事件、完整 L0/L1、共享只读 SQLite reader 导出的全部 features 列、AI raw/validated/provider receipt 与派生 pair 标签。没有原图、数据库文件、凭据、主机名、用户名或绝对路径。
- `file_id` 只在 date 对应库存内唯一；持久身份是 `date + file_id + content_sha256`。DINO 是 768 维 little-endian float16，严格校验 1536 bytes、finite、nonzero norm，再归一化。训练特征不使用 VLM L2 的稀疏投影。
- 43 pair + 21 large holdout 继续封存。一次性 194 组冻结 label snapshot 回填允许把这 64 组标签写为 `holdout=true` 的 sealed 事件，但不读取其图像、AI、L0/L1 或 features。live 手工保存也使用相同 sealed schema，封存事件的这些字段为 unavailable/null。默认 CloudReviewData/CLI 在读取任何 batch 图片前拒绝含 holdout 的 manifest；agent/自动化必须提供 development-only manifest。Willy 如需手工 review 全部组，显式加 `--manual-holdout-review`；保存仍 sealed，不能进入 training/shadow。导出和独立 validator 都重新查可信 split 名单及 frozen file/SHA，不能通过修改 event.holdout 绕过；相同源 SHA 也排除。
- 当前 130 组训练可用、336 张 review 图。6 日期、约 288 对 gold pairwise 只够 baseline；不能认证 live 折叠。第二阶段积累千级 pair、至少 15 日期后再训练 safe-to-collapse / ranker，并另建 time-based fresh holdout；现有 64 组不回流训练。

## 事件与恢复

`schemas/review-event-log-v1.schema.json` + `src/review_events.py` 定义 event ID、UTC 时间、session、单调 sequence、组身份、AI 建议/理由/evidence refs/prompt hash/model/contract/receipt、Willy 最终集合/added/removed/replaced/ignored reasons/feedback/action、L0/L1 引用及 review/original SHA 映射。

每个事件是 `output/review-events-v1/000000000001-<id>.json`，进程锁下分配连续序号，fsync 后原子落盘；已存在事件不改写/删除。同 idempotency key 重放只返回既有事件，内容冲突拒绝。UI 为 save intent 产生 request_id；同一未变 intent 的 HTTP 失败重试复用 ID，选择/action/feedback/ignored reasons 变化产生新 ID，保存成功清除。

保存先在锁内刷新 UI state，原子写入 state 与 pending_event outbox，再追加 journal，最后清除 outbox。事件写失败不撤销已保存选择；错误明确返回，pending_event 在 CloudReviewData 启动/load、下一次保存以及 CLI export 前恢复。export 扫描同一 output 下 review-state 的全部可见 JSON outbox，恢复失败直接报错，不能静默漏事件。恢复沿用 save lock → journal lock 顺序；已追加未清 outbox 的事件幂等重放，页面刷新不会创建新动作。

## 文件交换命令

仓库根目录运行（Python 环境须安装 requirements.txt 中的运行依赖）：

交付的 development-only UI manifest 是 `output/learning-mvp/development-manifest.json`（130组）。默认 review 示例：

```sh
python -m src.serve_review --output output/development-review \
  --cloud-batch artifacts/cloud-judge-batch/authorized-scene-run-v3 \
  --cloud-manifest output/learning-mvp/development-manifest.json
```

Willy 手工全组 review 时将 manifest 换为 `artifacts/cloud-judge-batch/frozen.json` 并显式加 `--manual-holdout-review`。agent/自动化不使用该开关。


```sh
python scripts/learning_loop.py backfill \
  --snapshot artifacts/cloud-review-label-analysis/willy-labels-2026-09-30T1704Z.json \
  --journal output/learning-mvp/review-events-v1

python scripts/learning_loop.py export \
  --journal output/learning-mvp/review-events-v1 \
  --destination output/my-training-bundle

python scripts/learning_loop.py validate --bundle output/my-training-bundle
```

可用 `--start N --end N` 选择事件范围；每组采用范围内最后状态，pending/later 不进入训练。目录必须新建，不覆盖旧包。exporter 与原片补包先写 sibling `.staging-*` 目录，完成校验后在 publish lock 下原子 rename；普通异常删除 staging，进程崩溃遗留 staging 不视为交付，确认无活跃 writer 后可删除。manifest 给出版本、provenance、development split、全部文件 SHA、照片数、unique source photos 和 bytes。bundle SHA 是 canonical manifest 的 SHA256。接收方用 `schemas/learning-registry-trust.json` 固定可信 frozen JSON canonical SHA256 与两份 split 文件 SHA256；三者写入 provenance 并逐项核验，registry 不从待验证包里取。CLI 默认不能替换 registry；其他数据集须显式提供新的 `--registry-trust` / `--split-registry` / `--frozen`，新 fingerprint 无法验证旧 bundle。

状态 receipt 中可能带本地导入路径，因此仅保留该文件的相对 ref + source SHA；完整 provider-response/raw/validated 复制进包。`external_refs` 仅允许 `<group>/state.json`，不能把照片、特征或 L0/L1 改成外部引用。L0/L1 保留原始内容；数据库 features 独立保留完整列，包括二进制 base64。包内默认照片可能含人物与场景，手工上云前检查隐私、传输和存储成本。

## 原片按需补包与 source catalog

原片库数 TB，不走本地全量上行。`source-catalog-v1` 保存 provider(local/google_photos)、稳定 asset ID、可选 Google media item ID、content SHA、尺寸/mtime/bytes、review asset SHA。不保存下载 URL、token 或本地路径；短期 URL 不是稳定身份。

trainer 可输出 `original-request-v1` 的 `original-request.json`，绑定 bundle SHA，最多 32 个 hard examples，逐条写 date/file_id/SHA/provider/asset_id/reason。可以作为 learner artifact 的唯一可选附件。import 当场校验附件 bundle SHA 和非 holdout catalog allowlist，不等补包时再拒绝。

```sh
# 首先显示照片数/bytes，不复制
python scripts/learning_loop.py export-original-supplement \
  --request output/original-request.json --bundle output/my-training-bundle \
  --destination output/original-supplement
# 用户检查数量/成本后显式确认；只复制请求清单
python scripts/learning_loop.py export-original-supplement \
  --request output/original-request.json --bundle output/my-training-bundle \
  --destination output/original-supplement --confirm
```

没有 all-originals/include-originals 快捷入口。补包只能请求原训练包已有的非 holdout 身份，从本地只读库存解析源路径，copy 前后和目标均验证 SHA，不转码、不修改原图。独立 supplement manifest 记录请求、照片数、bytes 和 checksums。默认 dry-run。

未来 cloud fetch interface 消费同一 request + catalog，按稳定 provider ID 取少量原图，并核对 SHA；本轮 `google_photos` 明确拒绝，未实现 OAuth/API，也未访问 Google Photos。M2 必须现场确认 2026 年可用 API、Picker 权限和用户授权范围，不能假设可静默拉全库。凭据留在 fetch 执行环境，不能进入 bundle/artifact。

## learner 导入、回滚与 shadow

`learner-artifact-v1` 包含 dataset bundle SHA、trainer code/model/version/fixture、feature schema、metrics/gates、versioned preference_profile、非 holdout exemplar metadata、checksums。profile 仅接受固定摄影 criterion + bounded weight，不执行自由文本规则；拒绝代码、pickle、任意二进制与权限字段。

```sh
# 明确标注未云训练的端到端 fixture
python scripts/learning_loop.py fixture-artifact --bundle output/my-training-bundle --destination output/fixture-learner
python scripts/learning_loop.py import --bundle output/my-training-bundle --artifact output/fixture-learner --output output/learning-runtime
python scripts/learning_loop.py shadow --bundle output/my-training-bundle --output output/learning-runtime --group 2026-04-04-G002
python scripts/learning_loop.py rollback --output output/learning-runtime
python scripts/learning_loop.py rollback --output output/learning-runtime --none
```

版本在 `output/learners/versions/<sha>/`；`active-shadow.json` 原子更新，permissions 固定为空，只能启用 shadow。rollback 是 active/previous 两个槽位的单步 toggle；连续调用在两版之间往返，不是历史栈。`--none` 切到 none，不删除历史版本。fixture 不是云训练结果，也没有真实训练指标。

shadow 用完整 DB DINO 的组均值归一化 embedding，cosine + group 稳定排序，最多 K=3；排除 query 自身、相同 source SHA 与 holdout。历史示例只包含 AI 建议 → Willy 最终集合与差异，不生成个人属性推断。UTF-8 bytes 默认上限 8192，token 上限 4096，采用每 byte 至多一 token 的保守上界；超预算整条 exemplar 不加入，不截坏 JSON。输出独立 `output/shadow/<group>.json`，含 baseline prompt hash/原建议和 shadow context；`shadow_recommendation=null`、`model_called=false`。默认不注入 live v3。

## 验证与产物

- `tests/test_learning_loop.py`：schema、差异、并发/idempotency、save outbox 恢复、只读/WAL/close/DINO、bundle 篡改/holdout/split/identity、artifact 权限/兼容、rollback、确定性 shadow、原片单条补包。
- `scripts/qa_learning_loop.py`：真实 Chromium，仅 development 组，保存追加事件、刷新不重复、全部保留、零 original requests。
- `scripts/verify_learning_loop.py`：130 组完整包校验、fixture 导入/两版 rollback/none、确定性检索、单图原片补包与源 SHA。
- 当前权威包为 `output/learning-mvp/bundle-v3`，`output/learning-mvp/current.json` 指向它；bundle-v2 保留为按旧 validator 验证过的历史交付，不覆盖，也不冒充新 provenance 验证结果。
- `artifacts/learning-loop-mvp/holdout-incidents.json` 记录 G001 `bytes_copied_not_decoded`。64 组继续硬排除训练；未来评估必须同时报告完整 64 组与 exclude-G001 的 63 组敏感性结果。
- 本次证据在 `artifacts/learning-loop-mvp/`；完整事件与训练包在 `output/learning-mvp/`。独立验收由 parent 完成。
