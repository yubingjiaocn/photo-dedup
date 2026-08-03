# Photo Dedup P0 决策策略：置信度、Maybe 与“每组至少留一张”

> 状态：设计建议（不改代码）
> 目标：让 6 万级照片批处理在**默认不永久删除**的前提下，能自动处理明显重复项，把不确定项送入可解释的 `MAYBE` 队列，并以机器可审计的规则保证每组至少保留一张。
> 结论先行：P0 不应把一个总分直接解释成“删除概率”。应采用 **硬安全门（hard gates）→ 组内比较 → 弃权/Maybe（reject option）→ 保底约束 → 可恢复执行** 的决策链；阈值用人工标注的本地样本校准，而不是凭模型原始分数拍脑袋。

---

## 1. 现有 Stage 2 / Stage 3 的事实与风险

### 1.1 已有能力

现有实现有很好的低侵入改造基础：

- Stage 1 缓存 pHash、DINOv2 embedding、MUSIQ、CLIP-IQA、全图 Laplacian sharpness、YuNet 人脸框与 detector score；昂贵图像读取和可重跑的决策已分离。
- Stage 2 按 `exact_dup → burst → similar_scene` 运行，DSU 合并后用
  `0.6*MUSIQ + 0.3*face_quality + 0.1*resolution` 选唯一 keeper。
- Stage 2 已有“人脸中心移动超过阈值则不合并”的打卡照保护。
- Stage 3 只生成 review 和删除 manifest；真正执行在下一阶段。Motion Photo 的 still/sidecar 也已绑定处理。
- 默认执行配置为 `mode: move`、`dry_run: true`，方向正确。

### 1.2 当前 P0 缺口

1. **二元输出过早。** 每个组只存在一个 `is_keep=1`，其余全部进入删除清单；没有 `MAYBE/UNKNOWN`。
2. **总分没有置信语义。** `keep_score=0.73` 只是排序效用，不是“73% 正确”，不同相册/场景也不可直接比较。
3. **没有组内 margin。** 第一名只比第二名高 0.001，仍会自动淘汰第二名。
4. **没有绝对缺陷门。** “A 比 B 稍差”与“A 明确闭眼/糊/严重曝光失败”没有分开；当前也尚未计算闭眼和曝光。
5. **模型分歧未利用。** MUSIQ、CLIP-IQA、sharpness、face quality 可能给出不同排序，现逻辑会用加权和掩盖冲突。
6. **缺失值被当作低分。** 无 MUSIQ/无人脸/人脸框异常会变成 0，可能把“未知”误判成“差”。
7. **DSU chaining 风险。** A≈B、B≈C 即可把 A/B/C 合成组，即使 A 与 C 不像；只记录最终组件，不记录边强度/组紧致度，无法解释组纯度。
8. **多人脸保护不够。** 只比较“最高置信 dominant face”的中心；合影中少了一个人、有人闭眼或 detector 漏检都可能被总分盖掉。
9. **Stage 3 manifest 不区分风险层。** 抽查页面存在，但不是对 `MAYBE` 全审、对自动项分层抽样的正式 gate。
10. **永久删除仍可配置。** 对未经人工确认的自动决策，P0 必须从策略层禁止直接 hard-delete，而不只依赖用户记得保留默认值。

---

## 2. 设计原则：这是“带弃权的排序”，不是普通分类

学术上的 selective classification / reject option 提供了正确抽象：模型只在足够有把握时自动决策，其余样本弃权给人工，以自动覆盖率换错误率。Geifman & El-Yaniv 明确把它描述为在目标风险水平下通过拒绝样本来控制 coverage；这里可映射为：

- **coverage**：自动判为可移出的候选占全部候选的比例；
- **selective risk**：自动候选中，人审认为“不该移出”的比例；
- **reject / abstain**：`MAYBE`；
- **UNKNOWN**：特征缺失、超出适用域或规则冲突；执行上也归入 `MAYBE`，但原因必须区分。

关键原则：

1. **宁可降低 coverage，不以不可逆风险换吞吐。** 大规模吞吐来自缓存、批处理和只审不确定项，不来自取消安全门。
2. **分开“组是否可信”和“组内谁更好”。** 聚错组时，组内排序再准也无意义。
3. **分开“相对更差”和“绝对不可取”。** 仅相对落后通常只能进入 Maybe；同时满足明确缺陷、强替代品、纯组，才可自动移入隔离区。
4. **置信度必须经本地标注校准。** Guo 等指出现代神经网络原始 confidence 往往未校准；temperature scaling 虽简单有效，但本项目首先应校准最终规则/分数，而不是把 YuNet/MUSIQ 原始值冒充概率。
5. **安全不变量由代码结构保证，不靠阈值。** “每组至少一张”“自动项不能永久删除”“特征缺失不得自动淘汰”必须是不可绕过的 invariant。

---

## 3. 建议状态机

### 3.1 成员级状态

```text
KEEP          明确保留；每组至少 1 张
AUTO_REMOVE   高置信、可自动移入 quarantine/trash；绝不直接永久删除
MAYBE         需要人工比较；不进入自动 manifest
UNKNOWN       特征缺失/不适用/异常；UI 可单独筛选，执行语义等同 MAYBE
USER_KEEP     人工覆盖，优先级最高且跨重跑保留
USER_REMOVE   人工覆盖；仍先进入可恢复区，除非用户另行显式永久清空
```

建议不要使用成员级 `DELETE` 命名；它会混淆“模型建议”“已移到回收站”“永久删除”三个不同事实。

### 3.2 组级状态

```text
AUTO_READY       组可信且至少存在一个 AUTO_REMOVE，允许生成隔离 manifest
REVIEW_REQUIRED  有 MAYBE / UNKNOWN / 多人脸风险 / 分歧 / 低 margin
KEEP_ALL         组不纯或所有候选都不够差；本轮不清理
INVALID          数据不完整或 invariant 校验失败；停止输出该组
```

### 3.3 执行级状态（与模型决策分离）

```text
PROPOSED → QUARANTINED/TRASHED → USER_CONFIRMED_PURGE
                  ↘ RESTORED
```

P0 默认只允许 `AUTO_REMOVE → QUARANTINED/TRASHED`。永久清空必须是**独立命令、独立确认、带批次 ID 与保留期**的人工动作；建议 30 天保留期。Google Photos 也只进 Trash，不做自动清空。

---

## 4. 最小侵入数据模型

不建议 P0 新建复杂模型服务。保留现有 `features / groups / group_members`，仅做 additive migration；旧数据库可重跑 Stage 2/3。

### 4.1 `features.quality_meta`：继续作为特征扩展点

现已有 JSON，P0 可追加，不破坏 schema：

```json
{
  "schema_version": 2,
  "musiq": 63.1,
  "clipiqa": 0.57,
  "sharpness": 418.2,
  "face_quality": 0.81,
  "exposure": {
    "shadow_clip_ratio": 0.004,
    "highlight_clip_ratio": 0.012,
    "median_luma": 0.47,
    "status": "ok"
  },
  "defects": {
    "blur": {"score": 0.08, "state": "pass", "available": true},
    "underexposed": {"score": 0.02, "state": "pass", "available": true},
    "overexposed": {"score": 0.11, "state": "pass", "available": true},
    "closed_eyes": {"score": null, "state": "unknown", "available": false}
  }
}
```

- `score` 只要求方向统一（0 好、1 坏），**不宣称是概率**。
- 每项必须有 `available/state`；缺失是 `unknown`，不是 0 分。
- P0 曝光可用 luminance histogram；模糊应至少同时保留全图 sharpness 与 face crop sharpness。闭眼检测若没有经过验证的轻量模型，先输出 `unknown`，不要用 YuNet landmarks 猜眼睛开合。

### 4.2 `groups` 添加 4 列

```sql
ALTER TABLE groups ADD COLUMN decision_state TEXT;
ALTER TABLE groups ADD COLUMN confidence REAL;
ALTER TABLE groups ADD COLUMN policy_version TEXT;
ALTER TABLE groups ADD COLUMN decision_json TEXT;
```

`decision_json` 保存可重算、可解释的组级证据：

```json
{
  "group_purity": 0.97,
  "min_similarity_to_medoid": 0.95,
  "weak_edge_count": 0,
  "keeper_file_id": 123,
  "keeper_margin": 0.14,
  "rank_agreement": 0.83,
  "face_risk": "none",
  "abstain_reasons": [],
  "threshold_profile": "default-v1"
}
```

### 4.3 `group_members` 添加 4 列

```sql
ALTER TABLE group_members ADD COLUMN decision TEXT;
ALTER TABLE group_members ADD COLUMN confidence REAL;
ALTER TABLE group_members ADD COLUMN evidence_json TEXT;
ALTER TABLE group_members ADD COLUMN user_override TEXT;
```

- 保留现有 `is_keep` 兼容旧代码，但它只作为派生字段：`KEEP/USER_KEEP → 1`，其余为 0；Stage 3 新逻辑必须读 `decision`，不能再用 `is_keep=0` 等同删除。
- `reason` 保留简短人类文本；详细证据进 JSON。
- `user_override` 不应被 Stage 2 `clear_groups()` 擦掉。真正落地时，最好将人工覆盖单独放进以 `(file_id, scope/policy)` 为键的 `review_decisions` 表；若 P0 尚无回写 UI，可先留列但在重跑前迁出/恢复。

### 4.4 为什么这是最小侵入

- 不改变 Stage 1 主表和昂贵 embedding 缓存。
- Stage 2 仍选 keeper，只是多算纯度、margin、分歧并允许 abstain。
- Stage 3 仍生成 HTML/manifest，但 manifest 只收 `AUTO_REMOVE`；`MAYBE/UNKNOWN` 单独展示。
- JSON 允许先快速迭代解释字段；等 schema 稳定后再正规化。

---

## 5. 决策逻辑：硬门优先，置信度最后

### 5.1 第零步：建立“受保护集合”

以下成员直接进入 protected set，不允许 `AUTO_REMOVE`：

- `USER_KEEP`；
- 路径位于 reference/protected album（建议借鉴 Czkawka 的 reference paths：参与比较但自动动作不可修改/移动/删除）；
- 特征读取失败、文件损坏状态未确认、关键字段缺失；
- 唯一 RAW/编辑成品/特殊格式（若未来纳入）；
- Motion Photo 的 still 与视频必须作为一个 asset 原子决策。

### 5.2 组纯度门（先回答“它们真是同一组吗？”）

对每组选择 embedding medoid `m`（与其他成员平均距离最小），计算：

```text
sim_medoid(i) = cosine(embedding_i, embedding_m)
group_purity  = min_i sim_medoid(i)
spread        = max_i distance(i, m)
```

并记录形成组件的边类型与强度。建议：

- `exact_dup` 需要区分 **字节/内容哈希完全相同** 与 `pHash≤2`。前者是最高安全层；后者仍是感知近似，不叫“exact”。
- burst 自动层要求所有成员对 medoid 达到校准后的阈值；只靠 DSU 可达性不够。
- 若最弱成员仅通过 chaining 加入，先把它切成子组或标 `MAYBE(group_chain)`。
- `similar_scene` 默认不产生 `AUTO_REMOVE`；仅用于 review grouping，直到有实证校准。

> P0 简单实现无需新聚类算法：DSU 完成后做一次 medoid compactness 校验；不达标的成员转 `MAYBE`，或按强边重新分量。这样可堵住最危险的 chaining。

### 5.3 组内排序与 margin

对每个指标先做**本地校准后的单调归一化**，避免把 MUSIQ 0–100、Laplacian 长尾值和 CLIP-IQA 生硬相加。无标注冷启动时，可在同相册/同设备分层用稳健分位数：

```text
z_k(i) = clip((x_k(i) - median_k) / IQR_k, -3, 3)
```

保留现有 `keep_score` 作为 `utility_score`，但新增：

```text
keeper = argmax utility_score(i)
keeper_margin = utility(keeper) - utility(second_best)
pair_margin(i) = utility(keeper) - utility(i)
```

**只用绝对 margin 不够**：不同组分布不同。建议同时记录 percentile/归一化 margin。冷启动候选阈值（仅用于 shadow mode，不可直接上线）可从：

- `keeper_margin ≥ 0.08`；
- 候选 `pair_margin ≥ 0.10`；

开始观察，再由标注数据选择。这里的 0.08/0.10 是现有 0–1 加权效用空间里的工程起点，不是论文阈值，也不是概率。

组内大小策略：

- `n=2`：margin 最脆弱；除内容哈希一致外，要求更严格的绝对缺陷/一致性证据。
- `n≥3`：若一个候选在所有指标上稳定落后，证据更强；但组越大 chaining 风险也越高，必须先过纯度门。

### 5.4 绝对缺陷门

`AUTO_REMOVE` 不能只因为“排名末位”。候选至少满足以下之一：

**A. 真正副本通道（duplicate lane）**

- 内容哈希相同；或经校准证明非常安全的近副本规则；
- keeper 不低于候选的分辨率/位深/文件完整性；
- 元数据保留策略明确（若候选有唯一 metadata，转 Maybe）；
- 不涉及受保护路径。

**B. 缺陷替代通道（quality lane）**

- 候选至少一个**高精度绝对缺陷**达到 reject threshold；
- keeper 对同一缺陷达到 pass threshold；
- pair margin 足够；
- 没有其他指标强烈支持候选；
- 人脸安全门通过。

每个缺陷采用三段阈值而不是一刀切：

```text
score <= T_pass        PASS
T_pass < score < T_bad MAYBE / gray zone
score >= T_bad         DEFECT
```

建议 P0 先落地可解释、便宜的：

- **严重模糊**：全图/tiles sharpness + face crop sharpness；有脸时不得只看背景。
- **严重欠曝/过曝**：luminance histogram（clip ratio、median、动态范围）；高反差/剪影落 gray zone。
- **闭眼**：仅在经过人工验证的 eye-openness 检测可用时启用；大笑、侧脸、墨镜、低分辨率眼部一律 unknown。
- **人脸模糊**：face crop sharpness 明显差且 keeper 对应脸清晰。

“构图不好、表情不好、姿态不好”主观性强，P0 不作为自动移出硬缺陷。

### 5.5 模型/指标分歧门

无需训练 deep ensemble，也可借鉴 ensemble uncertainty 的核心：多个独立视角不一致时降低置信。对每组计算以下 rankings：

1. technical：MUSIQ；
2. aesthetic：CLIP-IQA（仅 tie-break，不作为硬缺陷）；
3. sharpness：全图/subject；
4. face：face quality（仅在人脸检测可靠时）；
5. fidelity：resolution、文件大小/原始性。

定义简单可解释的 agreement：

```text
top_vote = 支持最终 keeper 的可用 ranking 数 / 可用 ranking 总数
rank_agreement = 平均两两 Spearman/Kendall 一致性（可选）
```

P0 可先只实现 `top_vote`：

- 可用指标少于 2 个 → `UNKNOWN(insufficient_evidence)`；
- `top_vote < 2/3` → `MAYBE(model_disagreement)`；
- 任一强冲突（例如 MUSIQ 选 A、face sharpness 明显选 B）→ 人脸组 `MAYBE`；
- CLIP-IQA 单独反对不应否决技术性强缺陷，但要降低 confidence 并进入抽检高优先级。

Deep Ensembles 论文表明模型集合是简单、可扩展的 uncertainty 手段，并能在分布外样本上表达更高不确定性；本项目不必复制深模型训练，只采用“分歧即不确定证据”的稳健工程思想。

### 5.6 UNKNOWN / Maybe 的明确触发条件

任何一条成立即不得自动移出：

- embedding、MUSIQ 或文件尺寸等关键特征缺失/NaN/越界；
- 图像解码与 metadata 异常；
- 组纯度不足、存在 weak/chained member；
- keeper margin 或 pair margin 位于灰区；
- 指标 top-vote 不足；
- 绝对缺陷位于灰区，或缺陷模型不适用；
- 多人脸安全门失败；
- keeper 自己也有同类严重缺陷（“一组都糊”）；
- 候选有某项独特优势（唯一更高分辨率、唯一清晰人脸、唯一不同人物/姿态）；
- `similar_scene` 组；
- 数据分层落在校准样本覆盖外（新设备、截图、扫描件、夜景等）。

原因码必须枚举化，例如：

```text
LOW_MARGIN, GROUP_CHAIN, GROUP_IMPURE, MODEL_DISAGREEMENT,
FEATURE_MISSING, ALL_DEFECTIVE, MULTIFACE_RISK, UNIQUE_FACE,
FACE_COUNT_MISMATCH, CLOSED_EYE_UNKNOWN, SUBJECTIVE_ONLY,
OUT_OF_CALIBRATION_DOMAIN, PROTECTED_PATH
```

### 5.7 每组至少留一张：硬 invariant

不要把“至少留一张”仅实现为 `argmax`；需在**输出 manifest 前再次验证**：

```python
protected = USER_KEEP + KEEP + MAYBE + UNKNOWN
if len(protected) == 0:
    promote argmax(utility_score) to KEEP with reason="group_safety_floor"

assert count(AUTO_REMOVE) <= member_count - 1
assert keep_file_id not in auto_remove_manifest
assert motion_asset_not_split()
```

更保守的 P0：

- 每组至少 1 个 `KEEP`，不是仅“未删除”；
- 如果 keeper 有严重缺陷，也照样保留，并把组设为 `REVIEW_REQUIRED/ALL_DEFECTIVE`；不能为了满足质量阈值把整组清空；
- 人工批量操作也不能越过 invariant，除非用户在组外明确执行“删除整个事件/整组”，并收到单独警告。

---

## 6. 多人脸规则（P0 必须保守）

### 6.1 为什么 dominant face 不足

当前 face-shift 只取最高 detector score 的脸。多人合影中 detector 排名会换人；“最大脸位置相近”不能证明所有人物相同，更不能证明每个人眼睛都睁开。

### 6.2 无人脸识别模型时的 P0 规则

1. 任一成员 `face_count ≥ 2`，默认组级 `REVIEW_REQUIRED`。
2. 以下情况直接 `MAYBE`：
   - face count 不同；
   - 某张出现其他成员没有的可靠 face box；
   - 按归一化中心排序后无法一一匹配；
   - 任一脸过小、遮挡、低 detector score，导致 eye/focus unknown；
   - 不同脸的 sharpness/eye 状态结论冲突。
3. 多人脸组只有在以下情况可 `AUTO_REMOVE`：
   - 内容哈希完全相同；或
   - 经过标注校准的近副本，所有可靠 face boxes 可匹配、face count 相同、keeper 对每张脸都不差，且候选有明确绝对缺陷。
4. “只要有一人闭眼就删”不可直接上线：若其他成员都没有完整覆盖同一组人物，必须保留；大笑/侧脸/墨镜均为 unknown。

### 6.3 单人脸组

- 两图都检测到 1 张可靠脸且位置/尺度匹配，才比较 face sharpness/eye state。
- 一图有脸一图无脸，不能自动认作缺陷；可能是人物进出画面，转 Maybe。
- face center 大幅移动继续沿用现有 split；建议再加 bbox IoU/尺度变化，因为中心相近但人物大小或身份可能不同。

---

## 7. 置信度的构造与校准

### 7.1 不要直接手工相乘成“概率”

初期可以有一个排序用的 `raw_confidence`，但 UI 应显示 `HIGH/MEDIUM/LOW + reasons`，不要显示未经校准的“97%”。建议原始分由以下可解释项组成：

```text
raw_confidence = min(
    purity_confidence,
    margin_confidence,
    defect_confidence,
    agreement_confidence,
    face_safety_confidence,
    feature_completeness
)
```

使用 `min` 而不是平均，可避免一个超强分数补偿致命短板。任一 hard gate 为 0 即 abstain。

### 7.2 校准数据

从真实库分层抽取组，由人标注：

- 组是否纯（成员确实可互相替代）；
- 推荐 keeper；
- 每个候选：`safe_to_remove / maybe / must_keep`；
- 原因（不同人物/表情、模糊、曝光、闭眼、唯一 metadata 等）。

初始建议至少 **500 组或 1,500 个 pair decisions**，覆盖：

- exact/content duplicates、burst、similar_scene；
- 无脸/单脸/多人脸；
- 白天/夜景/室内/逆光；
- 2 张小组与 ≥5 张大组；
- 不同设备、截图/扫描件；
- 阈值附近样本要过采样。

同一 group 不能拆到 train/calibration/test 两侧，防止泄漏。若只做规则阈值，无需训练集，可用 calibration + 独立 holdout。

### 7.3 校准方法

P0 推荐按复杂度递进：

1. **分桶可靠性表**：按 raw confidence decile、group type、face bucket 统计实际错误率；数据少时最可解释。
2. **Isotonic regression**：样本足够且只假设单调时，将 raw score 映射为经验正确率。
3. **Logistic/temperature scaling**：若最终有稳定 logit，可做轻量后处理；不改变排序。
4. **风险控制阈值**：在 calibration set 上选择最大 coverage 的阈值，使“误移出率”的保守上置信界低于目标。

Conformal Risk Control 将 conformal prediction 扩展到控制单调 loss 的期望，可作为后续严格风险控制方案；P0 不必一次实现完整 conformal machinery，但应保留 calibration split、policy version 和 risk/coverage 报表，避免未来推倒重来。

### 7.4 建议风险目标

照片误删代价高度不对称。建议上线门槛：

- 自动项 holdout **错误率点估计 ≤ 0.5%**；
- 同时要求 95% 单侧上置信界 ≤ 1%；
- 多人脸和 `similar_scene` 分层单独达标，否则该层 coverage=0（全部 Maybe）；
- 任何抽检发现“唯一人物/唯一时刻被自动移出”，立即暂停相应 profile。

注意：小样本“0 错误”不等于安全。比如抽检 100 个零错误，仍不足以证明错误率 <1%；需报告二项分布上界，而不只报 accuracy。

### 7.5 漂移与版本

每次输出记录：

```text
policy_version, feature_schema_version, model/version,
threshold_profile, calibration_dataset_id, run_id, created_at
```

新相机、模型权重、缩放方式或阈值变化都要重新看 calibration。没有匹配 profile 时自动降级 `UNKNOWN(OUT_OF_CALIBRATION_DOMAIN)`。

---

## 8. 抽检设计：不只“随便看几十组”

Stage 3 应输出三个队列：

1. `review_maybe.html/json`：**100% 人审** `MAYBE/UNKNOWN`；按风险原因排序。
2. `auto_remove_sample.html/json`：对自动项做正式抽检。
3. `auto_remove_manifest`：只有抽检 gate 通过后才生成或解锁执行。

### 8.1 分层抽样

每次 run 至少包含：

- 每个 `group_type × face_bucket(0/1/2+) × confidence_band` 的样本；
- 全部罕见/高风险层（多人脸若尚允许自动、极大组、组纯度刚过线、margin 刚过线）；
- 自动项随机样本；
- 最大可回收空间样本（大文件会放大损失）；
- 新设备/新日期范围样本。

冷启动建议：`max(200, 自动候选的 2%)`，每层至少 20；若自动候选少则全审。稳定运行后可降到 `max(100, 0.5%)`，但风险边界层继续高采样。

### 8.2 抽检 gate

- 发现任何 catastrophic error（唯一人物/不同事件/唯一高质量版本被自动移出）→ 停止该 run，相关规则全转 Maybe。
- 普通错误超过目标上界 → 提高阈值、缩小 coverage，重跑 Stage 2/3。
- 记录人审 override，进入下一轮 calibration；不能只改阈值不保存证据。

### 8.3 UI 可解释性

每个候选显示：

```text
AUTO_REMOVE (HIGH, calibrated band 99–100%)
因为：同组纯度 0.972；比 keeper 低 0.143；严重人脸模糊；
      MUSIQ/face sharpness/resolution 3/3 支持 keeper；无 unique face。
保护：keeper #123；原文件将进入 30 天 quarantine，不会永久删除。
```

Maybe 显示具体阻塞：

```text
MAYBE — LOW_MARGIN + MODEL_DISAGREEMENT
MUSIQ 偏好 #123；face sharpness 偏好 #124；差值 0.021 < 阈值 0.080。
```

---

## 9. 推荐 P0 判定伪代码

```python
def decide_group(group, policy):
    # 0. user/reference protection and data validity
    protected = protected_members(group)
    if invalid_group_data(group):
        return keep_best_and_mark_others_unknown("FEATURE_MISSING")

    # 1. grouping confidence before quality ranking
    purity = compactness_to_medoid(group)
    if group.type == "similar_scene" or purity < policy.purity_min:
        return keep_best_and_mark_others_maybe("GROUP_IMPURE_OR_UNCALIBRATED")

    keeper, utility, rankings = rank_members(group)
    mark(keeper, "KEEP")

    for x in group.members - {keeper}:
        if x in protected:
            mark(x, "KEEP", "PROTECTED")
            continue

        evidence = compare(keeper, x)

        if evidence.missing_critical:
            mark(x, "UNKNOWN", "FEATURE_MISSING")
        elif evidence.multiface_unsafe:
            mark(x, "MAYBE", evidence.face_reason)
        elif evidence.group_chain_or_outlier:
            mark(x, "MAYBE", "GROUP_CHAIN")
        elif evidence.model_agreement < policy.agreement_min:
            mark(x, "MAYBE", "MODEL_DISAGREEMENT")
        elif evidence.pair_margin < policy.margin_min:
            mark(x, "MAYBE", "LOW_MARGIN")
        elif is_content_identical(x, keeper) and fidelity_not_better(x, keeper):
            mark(x, "AUTO_REMOVE", "CONTENT_DUPLICATE")
        elif (evidence.absolute_bad_x
              and evidence.keeper_passes_same_defect
              and evidence.no_unique_advantage_x):
            mark(x, "AUTO_REMOVE", evidence.defect_reason)
        else:
            mark(x, "MAYBE", "RELATIVE_ONLY_OR_GRAY_ZONE")

    # 2. non-bypassable safety floor
    if count_non_remove(group) == 0:
        mark(argmax(utility), "KEEP", "GROUP_SAFETY_FLOOR")
    assert count(group, "AUTO_REMOVE") <= len(group.members) - 1

    # 3. group status
    group.state = "REVIEW_REQUIRED" if has_maybe_or_unknown(group) else "AUTO_READY"
    group.confidence = calibrated_confidence(min_gate_evidence(group))
    return group
```

**重要次序：**先组纯度，再 keeper，再逐候选判断；不是先给所有非 keeper 打 delete，再事后补救。

---

## 10. Stage 2 / Stage 3 的最小改造边界

### Stage 2

- 保留现有三层候选生成和 `select_keep()`。
- 持久化 pair edge 的类型/强度，或至少在组件形成后重算 medoid compactness。
- 输出成员 `decision/confidence/evidence_json` 与组级 policy metadata。
- `clear_groups()` 重跑时不得丢人工 override。
- 在 commit 前运行 invariants。

### Stage 3

- `collect_deletions()` 改为只收 `decision='AUTO_REMOVE'` 或显式 `USER_REMOVE`；绝不能继续用 `is_keep=0`。
- `MAYBE/UNKNOWN` 从所有本地、云端删除 manifest 排除。
- review 页面按状态和原因分区，显示 margin、纯度、分歧、缺陷、人脸风险、policy version。
- manifest 带 `run_id/policy_version/keeper_file_id/decision_reason`，执行前校验 keeper 仍存在且源文件未变化。
- 自动项默认目标为 quarantine/OS trash；禁止自动 hard delete。

### 执行阶段

- 执行前对路径、size、mtime（最好内容 hash）做 optimistic concurrency check；源文件变化则跳过并报 `STALE_DECISION`。
- 以 asset 为事务单位处理 Motion Photo；任一半失败则回滚/停止。
- 写完整操作日志与恢复清单。

---

## 11. 冷启动阈值策略（推荐，不是假装已有真值）

在没有本地标注前：

| 层 | 自动行为 | 冷启动策略 |
|---|---|---|
| 内容哈希完全一致 | 可 `AUTO_REMOVE` 到 quarantine | 保留更高 fidelity / protected copy；仍至少留一张 |
| pHash≤2 + 高 DINO 相似 | 先 Maybe 或极小 coverage shadow | 必须验证 metadata/fidelity/纯度 |
| burst | shadow mode | 仅记录如果启用会移哪些，不真正执行 |
| similar_scene | 全 Maybe | 默认继续关闭自动移出 |
| 单人脸 | 高门槛 | face count/位置/尺度匹配；冲突即 Maybe |
| 多人脸 | 全 Maybe（内容哈希一致除外） | 有足够标注后再开放 |
| 闭眼 | unknown | 没有经验证模型不启用 |

上线顺序：

1. **Phase 0：shadow**——生成决策，不移动文件，完成标注与 risk-coverage 曲线。
2. **Phase 1：只开放内容哈希重复**——验证执行/恢复链。
3. **Phase 2：开放无脸 burst 的高置信绝对缺陷通道**。
4. **Phase 3：开放单人脸；多人脸继续 Maybe**。
5. 任何层未独立达到风险目标，不因总体平均很好而开放。

---

## 12. 测试与验收清单

必须新增的策略级测试场景：

- 两张同分：1 KEEP + 1 MAYBE，不自动移出。
- 一组三张全糊：仍 1 KEEP，其余 Maybe，原因 `ALL_DEFECTIVE`。
- A≈B、B≈C、A≉C：C 不得因 DSU chaining 自动移出。
- 一张缺 MUSIQ/embedding：UNKNOWN，不是低分自动淘汰。
- MUSIQ 与 face sharpness 选不同 keeper：Maybe。
- 两人合影中一张少检测到一个人：Maybe。
- 某张闭眼但包含唯一人物：KEEP/Maybe，不自动移出。
- 单人打卡照脸位置明显移动：拆组或全部保留。
- exact/content dup 三张：恰好至少 1 KEEP，其余只进 quarantine manifest。
- protected/reference path 的副本永不被自动移动。
- Motion Photo still/video 决策一致。
- Stage 2 重跑后人工 override 保留。
- Stage 3 输出不包含 Maybe/Unknown。
- keeper 文件在 Stage 2 后被移动/修改：执行拒绝 stale decision。
- 任意随机生成组满足 property：`AUTO_REMOVE <= n-1`。

验收指标：

- risk-coverage 曲线（总体 + 每个 group/face/profile 分层）；
- Maybe/Unknown 比例及 top reason；
- 自动项人工错误率与单侧置信上界；
- keeper override rate；
- grouping impurity rate；
- quarantine restore rate；
- 每次 run 的 invariant violations 必须为 0。

---

## 13. 参考来源与可借鉴点

1. **Geifman, Y.; El-Yaniv, R. (2017), _Selective Classification for Deep Neural Networks_.** 提出 reject option，以 coverage 换取目标风险；本设计的 `MAYBE` 即 abstention。
   https://arxiv.org/abs/1705.08500
2. **Guo, C. et al. (2017), _On Calibration of Modern Neural Networks_.** 说明现代网络 confidence 常未校准，并验证 temperature scaling 作为简单后处理；支持“不把原始模型分数当概率”。
   https://arxiv.org/abs/1706.04599
3. **Lakshminarayanan, B.; Pritzel, A.; Blundell, C. (2017), _Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles_.** ensemble 可提供高质量 uncertainty，并对分布外样本表达更高不确定；本设计采用低成本的多指标排名分歧作为工程近似。
   https://arxiv.org/abs/1612.01474
4. **Angelopoulos, A. N. et al., _Conformal Risk Control_.** 将 conformal prediction 扩展到单调 loss 的期望风险控制；可作为后续从经验阈值升级到有限样本风险控制的理论路线。
   https://arxiv.org/abs/2208.02814
5. **digiKam 9.2 Similarity View 官方文档。** 使用 reference image、可配置 reference 选择法、相似度范围、组级/成员级相似度展示；说明成熟工具把“选择参考图”和“删除副本”显式分开。
   https://docs.digikam.org/en/left_sidebar/similarity_view.html
6. **Czkawka 官方仓库说明。** Reference paths 可参与比较，但不能被自动修改、移动或删除；这是本设计 protected/reference set 的直接工程参照。
   https://github.com/qarmin/czkawka/blob/master/instructions/Instruction.md
7. **OpenPhotoCull（开源）README。** 采用 tile-grid Laplacian + EXIF intent、subject-vs-background focus、luminance histogram、时间+pHash 分组；支持 duplicate side-by-side、K/D/U 状态、可调阈值、OS trash。可借鉴其“轻量缺陷 + 人审 + 可恢复删除”，但其性能/准确性陈述未经本项目独立验证。
   https://github.com/zwoodard/OpenPhotoCull
8. **Blurry（开源）README。** 强调相似组对比、sharpness/contrast 相对比较、face highlight 和人工选择；支持 P0 优先增强 review/解释，而非立刻上复杂深模型。
   https://github.com/genotrance/blurry

---

## 最终建议

P0 的核心不是再加一个“置信度字段”，而是把现有 `keeper vs all-delete` 改成**受约束的选择性决策系统**：

1. 先确认组纯度，堵住 DSU chaining；
2. 永远选出至少一个 KEEP；
3. 只有“强替代关系 + 足够 margin + 绝对缺陷/真正副本 + 多指标一致 + 人脸安全”才 `AUTO_REMOVE`；
4. 缺失、分歧、多人脸、灰区和主观判断全部 `MAYBE/UNKNOWN`；
5. 用真实库标注做分层 calibration 和 risk-coverage 选择；
6. Stage 3 只把 `AUTO_REMOVE` 放入可恢复隔离清单，永久删除永远需要后续人工确认。

这条路线不依赖更深模型，改动集中在 Stage 2 决策与 Stage 3 输出，能最大程度复用当前缓存和批处理架构，同时把最危险的误删路径变成结构上不可发生。
