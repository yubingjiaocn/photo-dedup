# 二张组本地路由候选 v3

当前离线候选为 `pair-exposure-resolver-v3 / A`，未接入 live。88 组 development 重放中 **0 组本地选择，88 组 / 176 张全部回到 v3**；15 个人工保留两张组无本地漏保。当前候选没有容量收益，不能把零次本地决策写成 100% 本地准确率。43 组 holdout 保持封存，下一步由 parent 独立审查后决定是否值得开启。

## 固定政策

本地只有 `local_select_one` 一个路径，`local_keep_both` 硬禁。以下条件同时满足才可本地选一：

- 原始 DINO cosine distance ≤ 0.045；
- 完整页 64×64 RGB MAE ≤ 0.09；
- local dHash Hamming distance ≤ 8；
- 可用独立来源 family votes ≥ 6；
- 第一张/第二张的 family 票数差绝对值 ≥ 4。

near-pixel 与 DINO 是相似性门，不参与偏好投票。保存请求中的 round 后 DINO 距离不进入判定。每个 family 最多一票，内部有反向信号则弃权；缺失、相同值不贡献票。未知 profile、错误输入、非有限值、缺失或异常 DINO 均回 `vlm_required`。身份损坏时同时标记 `input_valid=false`，调用方必须恢复原始 pair，不能把空 exposure list 解释为本地成功。有效身份的正常弃权保留两张完整曝光。

| 独立来源 family | 合并的输入 |
|---|---|
| detail | 保存 sharpness 与完整页 gradient |
| CLIP-IQA | global 与 local-region CLIP-IQA |
| MUSIQ | global 与 local-region MUSIQ |
| exposure | entropy、mass_usable、clip_hi、clip_lo |
| face_quality | face quality |
| face_composition | face edge margin、near-edge count |
| eye_open | eye-open score |

这里的“独立”是测量来源去重，**不宣称统计独立或因果独立**。`quality_score` 是原始 MUSIQ 的别名，完全排除；同模型的 global/local 不重复计票。未观测到的面部或眼睛信号不能被解释为质量合格。指标之间仍可能相关，不能将票数当作概率。

## 只读输入与 DINO 合同

`run_pair_router_development.py` 只使用 development 标签和 88 组已有 v3 请求/完整页。split 中 holdout 仅检查名字、计数、不相交；不打开其标签、图片或旧 AI。原始 DINO 来自六份冻结 inventory，对每个日期仅执行带 `WHERE file_id IN (...)` 的明确列查询，共 **176 个 date-scoped file identities**。file_id 跨日期不是全局身份。

SQLite 以 `mode=ro&immutable=1` 打开并设置 `query_only=ON`；非空 WAL 拒绝使用，已有空 WAL/SHM 纳入前后 hash。查询只取 development 的 file_id、content SHA、DINO BLOB 与原片路径；原片只做 SHA 校验，不修改。原片、DB、sidecar、development v3 输入/旧结果文件均前后 hash 核验。旧结果只核验字节，基线选集由 development-labels 提供。

BLOB 必须是 bytes、恰好 1536 bytes，按 **little-endian float16 × 768** 解码到 float64；维度与全部数值有限性严格校验，L2 norm 必须有限且 > 1e-12，再归一化计算 cosine。库存保存的是未归一化向量，不能要求原始 norm=1；本轮176向量全部通过合同。raw bytes没有 endian 标识，因此 decoder遵循固定 `<f2`，不会猜测或自动修正其他编码。

本轮 RGB/dHash/gradient 来自已保存的完整页缩放图，因此这是冻结输入上的离线候选。未来若换成直接原片描述符，必须重新验证预处理等价性；不能将此 runner 当作已接入生产。

## Development、日期压力测试与 frontier

88 组为 73 个人工单选、15 个双选。Profile A 有14组通过 DINO+RGB+dHash 近似门，但没有一组具有6个可投票 family。全部88组的有效票数分布为：0票1组、1票4组、2票18组、3票38组、4票21组、5票6组。eye-open 在本集无有效票；face quality仅9组有票。因而不能靠重复质量指标凑到本地放行条件。

预先声明的三点离散 frontier：A；strict（DINO≤0.03、MAE≤0.06）；wide_diagnostic（DINO≤0.055、dHash≤12）。三点均0本地、0 keep-both leak；没有继续降低票数或追求某个预期组数。

六个日期 leave-one-date-out 采用完全固定 A，训练侧仅报告门禁，测试侧不拟合、不选择阈值；各折均0本地。它是设计后的按日期压力检查，不能冒充真正独立或嵌套交叉验证。

混合重放等于旧 v3：exact **69/88（78.41%）**，照片级 precision **89.11%**、recall **87.38%**、F1 **88.24%**，漏13/多11。88的分母包括旧AI失败/弃权组，不能与83个select组分母混用。

零事件单侧95% exact binomial上界使用 `1−0.05^(1/n)`。本地决策 n=0，错误率与上界为 **null / 无法估计**；keep-both测试 n=15、leak=0，其漏保率上界仍为 **18.10%**（且有开发集选择偏差与IID假设）。零泄漏是必要门，不是精度证据。

## 容量

冻结图库共2,065张；1,555张未分组不进VLM，131个pair共262张，其他63组共248张全部送入。A的development route rate投影为 **194个VLM组、510张unique source photos**，即每千输入 **93.95组 / 246.97张**，节省 **0张**。这是基于既有整库计数的投影，未跑holdout。

本开发集 VLM 输入为88组、176张完整页/高分辨率parts；准确像素数见 `frontier.json`。任何contact sheet里包含的每张源图都计入unique曝光，不能按sheet数打折。未生成新视觉packet，也没有外部VLM调用。

## 权威产物与复现

`artifacts/pair-router-tuning/candidate/current.json` 指向本轮唯一权威 `lock.json`；锁文件覆盖实现、测试、文档、政策、逐组计划、frontier、日期折、源/DB完整性与本轮测试日志。归档内容只作追溯，说明见 `candidate/superseded-sol-v2/README.md`。

```bash
PYTHONPATH=. /tmp/photo-dedup-py311-test/bin/python -m scripts.run_pair_router_development
/tmp/photo-dedup-py311-test/bin/python -m pytest -q tests/test_pair_exposure_resolver.py tests/test_photo_exposure.py tests/test_large_group_review.py
/tmp/photo-dedup-py311-test/bin/python -m pytest -q tests
/tmp/photo-dedup-test-venv/bin/ruff check src/pair_exposure_resolver.py scripts/run_pair_router_development.py tests/test_pair_exposure_resolver.py
git diff --check
```

重放会更新candidate评估产物；变更后须重新跑gate并重新锁hash，不能沿用旧锁宣称验证完成。代码、模型/用户授权和live准入彼此分离。实现模型为 `amazon-bedrock-mantle/global.openai.gpt-6-astra`，harness为OpenClaw native subagent直接文件工具+本地Python，无ACP、无外部VLM。独立代码审查由parent安排。
