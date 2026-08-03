# Scene Gating Implementation Spec（shadow-first）

> 状态：设计规格；不授权生产删除。适配当前 `src/config.py`、`src/stage1_features.py`、`src/decision.py`。
> 核心不变量：scene/subject 只路由 detector；quality 只在可信近重复组内比较；**P0 仅 `BYTE_IDENTICAL` 可 `AUTO_REMOVE`（且只进 quarantine）**。
>
> **产品边界与规模目标：** 本项目服务于 Willy 与琳合计接近 1 TB、数万张且仍会增长的个人照片库，重点是让机械盘上的大图库先完成一轮可恢复、低人工的整理。本项目不以复刻像素蛋糕等商业摄影 AI 的综合审美、精修或成片质量为目标；它定位为本地离线、可解释、低误伤的第一轮整理工具：可靠地分组，识别少数明确技术问题，减少人工浏览量；遇到主观审美、特殊场景、模型冲突或证据不足时主动弃权。成功标准是“在不误伤独特照片的前提下，显著压缩需要人工逐张查看的集合”，不是追求商业黑盒的全自动覆盖率。

## 1. 数据契约：四类信息不得混写

建议先落在 `quality_meta["routing"]`（additive JSON，`schema_version=1`），稳定后再正规化：

```json
{
  "schema_version": 1,
  "model": {"name":"siglip-base-patch16-224","revision":"...","prompt_bank_hash":"..."},
  "scene_context": {
    "tags":[{"code":"NIGHT_LOW_LIGHT","score":0.0,"support":[],"conflict":[]}],
    "state":"KNOWN|UNKNOWN", "reasons":[]
  },
  "subject_protection": {
    "tags":[{"code":"REAL_PERSON_SUBJECT","score":0.0,"support":[]}],
    "state":"KNOWN|UNKNOWN", "reasons":[]
  },
  "applicability": {
    "eye_quality":"APPLICABLE|NOT_APPLICABLE|UNKNOWN",
    "face_quality":"APPLICABLE|NOT_APPLICABLE|UNKNOWN",
    "global_sharpness":"APPLICABLE|WEAK_ONLY|UNKNOWN",
    "exposure":"APPLICABLE|SCENE_SPECIFIC|UNKNOWN",
    "reasons":[]
  },
  "quality_evidence": {
    "scope":"WITHIN_TRUSTED_GROUP_ONLY",
    "signals":[], "missing":[], "conflicts":[], "reasons":[]
  }
}
```

### 1.1 固定 coarse tags（共 11 个，multi-label）

**Scene/context（5）**

- `INDOOR`
- `OUTDOOR`
- `NIGHT_LOW_LIGHT`
- `FIREWORKS`
- `LANDSCAPE_CITYSCAPE`

**Subject/protection（6）**

- `REAL_PERSON_SUBJECT`
- `COSTUME_MASKED_PERSON`
- `NONHUMAN_CHARACTER_DOLL_STATUE`
- `FOOD_STILL_LIFE`
- `DOCUMENT_SCREENSHOT`
- `NO_DOMINANT_SUBJECT`

一张图可同时为 `OUTDOOR + NIGHT_LOW_LIGHT + FIREWORKS + REAL_PERSON_SUBJECT`。`UNKNOWN` 是两个 namespace 的 `state`，**不是第 12 个内容标签**；低绝对分、互斥簇 margin 小、高 entropy、prompt 不一致、模型/metadata 冲突或推理失败均进入 UNKNOWN。不得强制 top-1。

### 1.2 枚举 reason codes

- 输入/metadata：`MOTION_PHOTO_BOUND_ASSET`、`DECODE_FAILED`、`DECODE_PARTIAL`、`MODEL_UNAVAILABLE`、`FEATURE_MISSING`
- router：`LOW_ABSOLUTE_SCORE`、`LOW_EXCLUSIVE_MARGIN`、`HIGH_EXCLUSIVE_ENTROPY`、`PROMPT_INCONSISTENT`、`ROUTER_CONFLICT`、`OUT_OF_CALIBRATION_DOMAIN`
- subject：`SUBJECT_NOT_OWNED`、`BACKGROUND_FACE_ONLY`、`NONHUMAN_FACE_RISK`、`PRINTED_OR_SCREEN_FACE_RISK`、`MULTIPLE_SUBJECTS`、`SUBJECT_ASSIGNMENT_UNKNOWN`
- detector：`DETECTOR_NOT_APPLICABLE`、`FACE_COUNT_MISMATCH`、`CROP_CONTEXT_CONFLICT`、`SMALL_FACE`、`PROFILE_OR_OCCLUDED`、`EYE_STATE_UNKNOWN`
- group/quality：沿用 `BYTE_IDENTICAL`、`GROUP_IMPURE`、`LOW_MARGIN`、`MODEL_DISAGREEMENT`、`RELATIVE_ONLY`、`SUBJECTIVE_ONLY`，补 `CROSS_ROUTING_GROUP`、`SCENE_CALIBRATION_MISSING`。

`score` 一律称 raw/routing score；本地 calibration 前不得命名 `probability` 或显示百分比置信。

## 2. 单次解码路由状态机

```text
METADATA_GATE
  ├─ Motion Photo/sidecar/受保护路径/格式异常 → 绑定 asset；保护或 UNKNOWN
  └─ still candidate
       → DECODE_ONCE（EXIF 转正；失败即 UNKNOWN）
       → RGB_FANOUT
          ├─ 现有 DINO embedding：近似分组/组纯度，不作语义 tag
          ├─ SigLIP-B/16 224：coarse multi-label router（先 shadow）
          └─ cheap technical evidence：exposure/YuNet 等
       → SUBJECT_PROTECTION
       → APPLICABILITY_MAP
       → RUN_APPLICABLE_DETECTORS（不适用必须显式记录）
       → TRUSTED_GROUP_CHECK（byte hash / DINO / pHash；scene 不建删除组）
       → WITHIN_GROUP_QUALITY_COMPARISON
       → ABSTAIN_OR_MANIFEST
```

顺序要求：

1. metadata 硬信号先于视觉；Motion Photo 必须原子绑定，单边不可动作。
2. 原文件只读、decode 一次；缩放/crop 在内存产生，缓存模型版本与 prompt hash。
3. SigLIP 只产生 shadow routing record；不改变当前 decision。
4. 先确定受保护主体，再决定 eyes/face/global sharpness 等是否适用。
5. quality 仅在可信同组、相同/兼容 routing context 内比较；跨路由组 `MAYBE:CROSS_ROUTING_GROUP`。
6. 任意 UNKNOWN、冲突、缺失、不适用误用均 abstain；manifest 仅接收 policy 明确授权的 `AUTO_REMOVE`。

## 3. SigLIP-B/16 最小 prompt bank

新增 SigLIP，而不是拿现有 DINO embedding 硬做 prompt 分类：DINO 已稳定服务视觉相似/组纯度，但没有 text tower，无法提供可审计的 open-vocabulary label/prompt 对照；为 DINO 另训 classifier 需要标注且把 grouping 与 routing 耦合。**取舍：保留 DINO 的 grouping 职责，新增一个串行、可关闭、可缓存的 SigLIP shadow router；不替换 DINO。**

### 3.1 Prompt 结构

每 label 最小：

- 3 个 positive paraphrases：`a photo whose main subject is ...`、`a photograph of ...`、`the prominent subject is ...`
- 2 个 hard negatives：针对最危险相邻类；例如真人 vs 海报/人偶，costume person vs mascot/statue，fireworks vs city lights。
- 1 个 generic null prompt：`an ordinary photo not described by the candidate labels`

固定英文 prompt、模板版本和顺序；每个 label 保存各 prompt raw similarity/sigmoid score、mean/median、range/std、hard-negative gap。prompt 不得按单张图动态生成。

### 3.2 shadow calibration（不假装概率）

先采本库分层 hard set，逐 label 做：

- **absolute score**：仅 sweep 候选 gate；原始 sigmoid 仍叫 `score`。
- **margin**：只在预先声明的互斥簇内算，如 `REAL_PERSON_SUBJECT` vs `NONHUMAN_CHARACTER_DOLL_STATUE`；multi-label 全集不做 top-1 margin。
- **entropy**：仅对互斥簇的 scores 归一化后计算 normalized entropy；不可对全部 sigmoid tags 强套 softmax。
- **prompt consistency**：range/std、正向多数票、leave-one-prompt-out 是否翻转；不一致即 UNKNOWN。

shadow 报表至少含 per-label precision/recall、保护类 false-negative、UNKNOWN coverage、score/margin/entropy 分布与 reliability diagram。样本足够后可按 label 做 temperature scaling/isotonic；只有独立 holdout 校准后字段才能改称 `calibrated_probability`。保护标签目标优先高 recall，coverage 可牺牲。

## 4. 真人主体人脸旁路

调用条件必须是：`REAL_PERSON_SUBJECT` 或 `COSTUME_MASKED_PERSON` 的**主体候选**，且无非人/印刷冲突。执行 `YuNet → 1.3×/1.6× crop → MediaPipe`；结果仍只是 eyes/face evidence。

七个不可绕过的安全门：

1. **主体归属门**：face bbox 必须与主体彩域/显著主体关联；背景脸标 `BACKGROUND_FACE_ONLY`。
2. **语义负例门**：海报、屏幕、动漫、雕像、人偶、mascot 或材质不明，统一 abstain；不得套真人闭眼规则。
3. **框一致性门**：YuNet 与映射回原图的 MP bbox 中心/IoU 合理；1.3×/1.6×结论冲突为 `CROP_CONTEXT_CONFLICT`。
4. **原图像素门**：用原图 face width/IOD 判断；crop 放大不能绕过 `SMALL_FACE`。
5. **计数/去重门**：同脸跨 crop 去重；YuNet/MP count 不一致、达到 max_faces、多人无法匹配均 abstain。
6. **姿态/可见性门**：侧脸、遮挡、墨镜、低光、强反光、眼部不可判均 `PROFILE_OR_OCCLUDED/EYE_STATE_UNKNOWN`。
7. **决策隔离与验证门**：旁路只能写 evidence；`NO_FACE/MAYBE/UNKNOWN`、背景脸和非人主体永不触发删除；真人/重妆/海报/人偶/背景人群分层验证通过前保持 shadow。

现有诊断仅 10 张、救回 3/4 个全图 MP 漏检案例，同时已出现动漫假脸与维尼背景真人；因此它证明“可增加召回”，没有证明“可授权淘汰”。

## 5. Quality 与 MUSIQ 边界

- 本 spec **不新增或提升 MUSIQ 为生产依赖**。当前 `stage1_features.py` 已可经 `pyiqa` 计算 MUSIQ，但 scene gating 的正确性、router 和 `AUTO_REMOVE` 都不得依赖它。
- MUSIQ 仅列为后续 shadow quality signal：其 `pyiqa` 分发/商业许可边界及 checkpoint 条款仍需复核，且跨 scene/设备的绝对分尚未本地校准。
- 先复用已有 exposure、sharpness、face quality、eyes evidence；所有 score 都只做同一可信组内相对比较。烟花禁用“全局 Laplacian 越高越好”的解释；MUSIQ/CLIP-IQA 单独不能构成 defect。

## 6. P0/P1 决策权限与升级门槛

### 当前 P0

`src/decision.py` 的权限保持不变：仅 `content_sha256` 相同且 fidelity/保护条件允许时，reason=`BYTE_IDENTICAL` 可 `AUTO_REMOVE`；动作只到可恢复 quarantine/trash，组内至少 1 个明确 KEEP。pHash≤2、DINO 高相似、burst、similar_scene、scene/subject/eyes/quality 单独或组合均最多 MAYBE/UNKNOWN。

### 视觉近重复何时才可升级

必须同时满足后才另立 policy version、小流量开放：

1. grouping 有直接边证据、medoid compactness，无 DSU chaining；候选与 keeper 可互相替代且 metadata/fidelity 无唯一价值；
2. routing 非 UNKNOWN，主体归属一致，无 Motion Photo 拆分、保护路径、多人/非人/印刷风险；
3. 候选有经独立真实集验证的**绝对技术缺陷**，keeper 同项 pass；不仅是相对排名或审美偏好；
4. 至少 2 个独立 quality signals 一致，pair margin 超过分 scene 校准门槛，缺失/冲突即 abstain；
5. 独立 holdout 上自动候选错误率点估计 ≤0.5%，95% 单侧上界 ≤1%；高风险 scene/face bucket 分层单独达标；
6. shadow 全库与正式分层抽检通过，任何“唯一人物/唯一时刻”误入自动项即暂停该 profile；
7. manifest/invariant/恢复链验证：`AUTO_REMOVE≤n-1`、keeper 存在、源文件未变、只 quarantine、可恢复、全量 audit trail。

未全部满足时，视觉近重复 coverage 固定为 0；不得因总体平均指标良好而开放失败分层。

## 7. 后续小 task（均 5–10 分钟）

1. **Schema 常量与 fixture（先）**
   产物：`src/routing_schema.py`、`tests/test_routing_schema.py`。
   验收：四 namespace 分离；11 tags/reason codes 枚举校验；UNKNOWN 不被序列化为内容 tag。
2. **Router 状态机 shadow 骨架（先）**
   产物：`src/scene_router.py`、`tests/test_scene_router.py`；config 默认 `enabled:false, mode:shadow`。
   验收：metadata→decode result→routing→applicability；任何异常 fail closed；decision 输出与未启用前逐字等价。
3. **一次 decode fan-out 接线（先）**
   产物：`src/stage1_features.py` 的最小接线 diff、decode-count test。
   验收：每文件一次 `_open_image_and_sha`；DINO 与 shadow router 共享 RGB；缓存 model/prompt hash。
4. **SigLIP prompt bank + recorder**
   产物：`research/siglip_prompt_bank_v1.yaml`、`src/siglip_router.py`、golden serialization test。
   验收：每 label 3 positive/2 hard-negative；只输出 raw scores/一致性，不输出 probability，不改变 decision。
5. **主体归属与人脸旁路 gate（后于 router）**
   产物：`src/subject_ownership.py`、`tests/test_subject_ownership.py`。
   验收：维尼+背景人、海报/动漫、人偶 fixture 全 abstain；七门任一失败均不运行/不采用 eyes defect。
6. **Shadow calibration 报表（后于主体）**
   产物：`scripts/report_scene_shadow.py`、`output/scene-shadow-summary.json`。
   验收：按 label/危险簇输出 recall、FN、UNKNOWN coverage、score/margin/entropy/prompt stability；无生产阈值写回。
7. **组内 quality applicability（最后）**
   产物：`src/quality_routing.py`、策略测试。
   验收：fireworks/global sharpness、nonhuman/eyes、cross-routing group 均正确禁用或 abstain；`AUTO_REMOVE` 集合仍严格等于 byte-identical lane。

## 8. 研究依据（关键 URL）

### 学术（至少三处）

- Places365：scene-centric 封闭类别适合 baseline，不足以表达主体保护：<https://places2.csail.mit.edu/PAMI_places.pdf>
- CLIP：自然语言 zero-shot/open-vocabulary 能力及任务/prompt 依赖：<https://arxiv.org/abs/2103.00020>
- SigLIP：pairwise sigmoid objective 更适合非互斥候选：<https://arxiv.org/abs/2303.15343>
- Calibration：现代网络 confidence 常过度自信：<https://arxiv.org/abs/1706.04599>
- Selective classification：以 abstention 换风险控制：<https://arxiv.org/abs/1705.08500>
- MUSIQ：native-resolution multi-scale IQA，但不是主体意图/组内淘汰真值：<https://arxiv.org/abs/2108.05997>

### 产品（至少三处，均只支持“组织/复核”，不反推删除模型）

- Apple Photos Duplicates：候选集合由用户 Merge：<https://support.apple.com/guide/photos/remove-duplicates-pht5a3157c1d/mac>
- Apple Photos Burst：burst 是可查看、可挑选单元：<https://support.apple.com/guide/photos/view-photo-bursts-pht8745d2677/mac>
- Google Photos stacks：可选整理视图，不改变存储量：<https://support.google.com/photos/answer/14169846?hl=en&co=GENIE.Platform%3DAndroid>
- Adobe Lightroom masking：subject/people/object 识别用于非破坏性编辑：<https://helpx.adobe.com/lightroom-classic/help/masking.html>
- Narrative face/focus assessments：评分是可调 warning lights，保留人工纠错：<https://help.narrative.so/en/articles/7337369-face-and-focus-assessments>
