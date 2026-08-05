# 个人图库自动挑片的 Scene / Subject Gating：限定范围学术调研

## Executive summary

1. 不应把 Places365 的 365 类做成单一 flat 决策；本项目最合适的是**小型层次化 multi-label routing taxonomy + UNKNOWN abstain**。
2. 首选组合：**SigLIP-B/16（scene + subject coarse routing）+ MUSIQ（quality）**；二者共用一次解码后的 RGB，但保持独立输出与决策权限。
3. 若 SigLIP 在面具重妆、coser、人偶之间混淆，再把 **RAM/RAM++ 作为 shadow tagger**，不要一开始同时部署三套 recognition。
4. Places365-ResNet50 适合作为低成本 scene baseline / shadow validator，但其封闭场景类不能表达 coser、人偶、烟花等主体，也不能控制删除。
5. Scene 回答“拍摄环境/事件是什么”，subject 回答“要保护和评估什么”，quality 回答“在同 subject、同近重复组内哪帧更好”；三者禁止合成一个不可解释总分。
6. Scene/subject 只选择质量规则和保护策略；**scene tag 永不直接触发删除**。
7. 推荐 coarse labels：people-real、costume/masked、character/statue/doll、food/still-life、fireworks、night/low-light、landscape/cityscape、document/screenshot、other；可多标签并带 indoor/outdoor 与 single/group 属性。
8. SigLIP 的 sigmoid 输出不是可直接信任的校准概率；阈值须用本地人工标注集按 label 校准。
9. UNKNOWN 条件应同时看 absolute confidence、top-1/top-2 margin、候选集 entropy、prompt ensemble 一致性与跨模型冲突。
10. IQA（MUSIQ/MANIQA/TOPIQ）能作组内排序证据，但跨 scene 的绝对分不可比较；烟花和重妆尤其需要 scene-specific quality features。
11. 自动删除必须另有近重复关系、组内保留数、质量显著劣势和硬保护条件共同授权；recognition/tagging/aesthetic score 单独最多 shadow。
12. RTX 5070 Ti 16GB 足够离线推理；HDD 成本应以“一次 decode、内存 fan-out、缓存 embedding/score”为核心，而不是反复读取原图。

---

## 1. 问题拆分：flat、hierarchical multi-label、open-vocabulary

| 方案 | 优点 | 本项目风险 | 结论 |
|---|---|---|---|
| **Flat closed-set**（如 Places365 的 365-way softmax） | 快、输出互斥且容易算 entropy/margin；Places 在 scene recognition 上成熟 | 真实照片天然多义：`night + fireworks + people`；365 个 leaf 难校准且与“采用哪套质量规则”不一一对应；coser/人偶是 subject，不是 scene | 只适合 baseline 或映射到少量 coarse parent，不直接使用 365 leaf 决策 |
| **Hierarchical multi-label** | 同时表达 parent/child，例如 `people → costume/masked`、`night → fireworks`；可在细类不确定时退回父类 | 标签相关性、父子一致性和每类阈值需显式实现；训练式方法需要本地标注 | **推荐作为系统 taxonomy**；不一定要训练专用分类头，可用 VLM prompts 实现 |
| **Open-vocabulary**（CLIP/SigLIP/RAM open-set） | 可快速加入 `cosplayer in elaborate makeup`、`theme-park character` 等个人图库标签；零训练 | prompt 敏感、web 数据偏差、分数未必校准；“任何标签”不等于能识别个人定义的边界 | **推荐作 coarse router，必须固定 prompts + 本地校准 + abstain** |

学术依据：Places 提供 scene-centric 表征，但类别是封闭的环境语义 [Zhou et al., 2017](https://places2.csail.mit.edu/PAMI_places.pdf)；CLIP 证明自然语言可作 open-vocabulary classifier，同时论文也明确零样本性能依赖任务与文本 [Radford et al., 2021](https://arxiv.org/abs/2103.00020)；SigLIP 把全局 softmax 对比损失改成 pairwise sigmoid，更自然地支持非互斥候选 [Zhai et al., 2023](https://arxiv.org/abs/2303.15343)。多标签研究表明 label correlation / semantic-specific representation 有价值，而不是把所有标签当互不相关的二分类 [Chen et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.html)；但这些 benchmark 多为 COCO/VOC，不能直接证明在个人 cos/夜景图库中校准良好。

**实践折中：** 系统只暴露 8–10 个 coarse routing labels；底层 SigLIP 可用 3–6 个 prompt paraphrases/label，先聚合为 label score，再做层次一致性约束。细粒度自由 tags 只进搜索索引，不进入安全决策。

## 2. 模型候选与最多 2–3 个组合

### 推荐 A（默认）：SigLIP-B/16 + MUSIQ

| 项 | SigLIP-B/16 | MUSIQ |
|---|---|---|
| 职责 | open-vocabulary scene + subject routing | 单图 technical/perceptual quality，**仅组内比较** |
| 论文 | SigLIP，2023 arXiv [论文](https://arxiv.org/abs/2303.15343) | MUSIQ，ICCV 2021 [论文](https://arxiv.org/abs/2108.05997) / [官方代码](https://github.com/google-research/google-research/tree/master/musiq) |
| 许可 | Google HF checkpoint 标为 **Apache-2.0** [model card](https://huggingface.co/google/siglip-base-patch16-224) | Google Research 代码通常 Apache-2.0；本项目拟经 `pyiqa` 使用时，**pyiqa 整体是 PolyForm Noncommercial 1.0**，个人离线可用但商业化须重审 [仓库](https://github.com/chaofengc/IQA-PyTorch)；checkpoint/数据各自条款也需单独核对 |
| 规模/显存（推理估算） | base image+text 双塔约 200M 量级；FP16 权重约 0.4GB，连同 activation 通常约 1–3GB | 参数/实现随 checkpoint；16GB 单图或小 batch 充裕，通常数 GB 内 |
| Windows | `transformers` / PyTorch CUDA 路径成熟；建议固定版本并先测 CUDA 12.x wheel | `pyiqa` 为 PyTorch；Windows 通常可行，但需 smoke test checkpoint 下载、timm/torchvision 版本 |
| decode | 224 RGB；**无需额外 HDD 全图 decode**，复用一次 decode 后的缩略 RGB | multi-scale / patch IQA 应从同一次全图 decode 产生较大输入；不可只拿 224 thumbnail 替代。若 pipeline 本来已全图 decode，则无额外读取；否则这是唯一需要全图像素的分支 |

**为什么默认它：** 一个通用 embedding 同时处理小型 scene 与 subject prompt bank，避免 Places + RAM 双 recognition；MUSIQ 与语义路由解耦，符合现有项目判断。局限：SigLIP 分数并非本图库上的校准概率，面具重妆、真人 coser、迪士尼人偶很可能受 prompt 和训练数据共现偏差影响；MUSIQ 在 KonIQ/SPAQ 等自然失真分布上训练/评估，不知道“烟花形态完整”“coser 眼睛是否闭合”等任务偏好。

### 推荐 B（精度诊断/可选）：SigLIP-B/16 + RAM（shadow）+ MUSIQ

- RAM（2023 arXiv，CVPRW 2024）以大规模自动标签和 data engine 学习 common-category tagging，[论文](https://arxiv.org/abs/2306.03514)、[项目页](https://recognize-anything.github.io/)、[官方 GitHub](https://github.com/xinyu1205/recognize-anything)。RAM++ 支持 custom/open-set category embedding。
- **许可：** 官方 repo 是 Apache-2.0；仍需保留 NOTICE，并核对下载 checkpoint 的附加说明。
- **规模/显存：** RAM/RAM++ 使用 Swin-Large（backbone 本身约 197M 参数）外加 recognition/text modules，整体是数亿参数量级；checkpoint/推理明显重于 SigLIP。保守规划 FP16 单图约 4–8GB，16GB 可行，但必须实测峰值；不要与 IQA 大 batch 并发常驻。
- **Windows：** PyTorch 可跑，但研究 repo 比 `transformers` 更易碰到编译/依赖和路径问题；建议 WSL2 或独立 venv。输入通常 384 RGB，可复用同一次 decode。
- **角色限制：** RAM 输出 6400+ tags 的强项是召回和诊断，不是删除授权。只用它检查 SigLIP 漏掉 `mask/costume/doll/fireworks/food` 的案例，或作为“冲突则 UNKNOWN”的第二意见。其自动构造标签也会带噪，长尾 open-set 不等于可靠校准。

### 备选 C（最轻 baseline）：Places365-ResNet50 + MUSIQ

- Places 数据/模型奠定 scene-centric recognition [Places project](http://places.csail.mit.edu/)；官方实现 [CSAILVision/places365](https://github.com/CSAILVision/places365)，**代码 MIT**。
- ResNet-50 约 25.6M 参数、FP32 权重约 100MB；推理显存通常 <1GB，Windows PyTorch 易跑；224 crop 可复用 decode。
- 只将 365 logits 聚合为 `indoor/outdoor/urban/natural/event-like/other` 等 coarse parent，并以 entropy/margin abstain。它不能可靠区分真人、面具、人偶，也不应把 `amusement_park` 当作“迪士尼人偶”。Places365 的论文/数据发布早于题设重点窗口，但它是 2018–2026 scene work 的基础 baseline，故保留。
- 若默认 A 已满足，不建议再部署 C；它最适合做一次离线 benchmark，验证“专用 scene 模型是否真比 SigLIP prompts 稳”。

## 3. Scene、subject、quality 必须解耦

### 3.1 三个独立记录

1. **Scene/context（可多标签）**：indoor/outdoor、night、theme-park/event、landscape/cityscape、fireworks context。回答“规则环境是什么”。
2. **Subject/protection（可多标签）**：real person、costume/masked person、character/statue/doll、food/object、no dominant subject。回答“主体是什么、哪里应测清晰/闭眼、是否属于高风险人物内容”。
3. **Quality（连续分 + 分项）**：technical IQA、主体清晰度、曝光/高光、眼睛、构图/美学、burst-relative score。回答“同语义、同近重复组内，哪张更适合保留”。

禁止 `final_score = scene_confidence + subject_confidence + MUSIQ`。正确关系是：scene/subject **选择 feature 与 threshold**，quality **在已有 duplicate cluster 内排序**。例如：

- `fireworks`：停用“全局 Laplacian 越大越好”的解释，启用高光保留、烟雾/拖影、轨迹完整度、burst 峰值；MUSIQ 只作弱证据。
- `real_person | costume/masked`：主体区域清晰、脸/眼检测仅在可信时启用；face detector 找不到面具脸不能等价于“无人”。
- `character/statue/doll`：禁止真人闭眼规则，保留主体显著性与局部清晰。
- `food/still-life`：以显著主体 ROI 清晰、曝光和构图为主。
- `landscape/cityscape`：全局/多区域清晰与高光阴影；不强行寻找单一主体。

显著性模型如 U²-Net（Pattern Recognition 2020，[官方代码](https://github.com/xuebinqin/U-2-Net)）或 DIS/IS-Net（ECCV 2022，[论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780036.pdf)、[Apache-2.0 code](https://github.com/xuebinqin/DIS)）可提供 class-agnostic ROI。但其训练目标是 foreground mask / dichotomous segmentation，不是“摄影者意图主体”；多人、烟花、景观、反射和复杂服装会失败，故 ROI 只能调节质量测量区域，不能证明照片可删。

## 4. 推荐 coarse routing taxonomy

采用**非互斥** labels；每张可同时命中多个：

### Level 0：安全/输入状态
- `motion_photo`（来自容器/metadata，不由视觉模型猜）
- `decode_partial_or_error`
- `unknown_visual`

### Level 1：主体路由
- `people_real`
- `people_costume_or_masked`（包括重妆、面具；不要求 face detector 成功）
- `character_statue_doll`（主题乐园人偶、雕像、玩具）
- `food_or_still_life`
- `no_dominant_subject_or_scene`

### Level 1：环境/事件路由
- `fireworks`
- `night_or_low_light`
- `landscape_or_cityscape`
- `document_or_screenshot`（若实际图库存在；这类通常应用不同重复/质量策略）
- 辅助属性：`indoor`, `outdoor`, `single_subject`, `group_or_crowd`

父子规则示例：`people_costume_or_masked ⇒ people-present parent`，但**不强迫** `people_real` 与 `character_statue_doll` 互斥；两者高分说明模型不确定，应 UNKNOWN/双路由保护，而不是硬选 top-1。

不建议把 `Disney`、具体角色、具体菜名、具体地点放进 routing taxonomy：这些是检索 tags，不对应稳定 quality policy，并有商标/长尾/域偏差。

## 5. Confidence、entropy、margin 与 abstention

### 5.1 SigLIP / multi-label（主方案）

对每个 coarse label 预先定义 K 个正向 prompts 与 2–4 个 hard-negative prompts。例如 `people_costume_or_masked` 的正向覆盖 cosplay、elaborate makeup、mask；负向覆盖 statue/doll/theme-park mascot without visible human。分数先在 prompt 内聚合，再作本地 calibration。

必须用从图库抽样的人工集（建议按上述 label 分层，含 hard cases）估计每类阈值，不能照搬 sigmoid 0.5。建议输出：

- `p_l`：经 temperature scaling 或 isotonic regression 的每 label 概率；校准集不够时叫 `score`，不要伪称 probability。
- **absolute gate**：所有保护/路由 label 都低于各自 `T_low,l` → UNKNOWN。
- **margin gate**：若两个**互斥语义簇**（如 people-real vs character-doll）的 top score 差 `< M_l`，UNKNOWN 或同时路由到更保守策略。
- **entropy gate**：只在预定义互斥 coarse set 内归一化后算 `H=-Σq log q / log N`；若 `H>H_max` 则 UNKNOWN。不要对本来可共存的全部 sigmoid labels 直接套 softmax entropy。
- **prompt stability**：同 label 的 prompts 方差过大，或 leave-one-prompt-out 后结论翻转 → UNKNOWN。
- **model conflict**（启用 RAM shadow 时）：SigLIP 与 RAM 在保护性 parent 上冲突 → UNKNOWN；一致只增加路由可信度，不增加删除权限。
- **OOD proxy**：最大相似度低、与本地已知 embedding prototype 距离大、或增强前后预测不稳定 → UNKNOWN。

具体数值（如 `T_low=0.65, margin=0.10, H=0.65`）只能作为初始 sweep，**不能在没有本地 ROC / reliability diagram 的情况下写死**。阈值选择目标应是保护类高 recall：对 `people/costume/character` 优先压低 false negative，即使多进 UNKNOWN。

### 5.2 Places365 flat softmax（若做 baseline）

将 365 logits 先按 parent 用 `logsumexp` 聚合，再校准 parent softmax；同时观察：

- parent `p_max < T_scene`；
- normalized entropy 过高；
- top-1/top-2 margin 过小；
- 365 leaf top-1 很高但映射 parent 与 SigLIP subject 冲突。

任一触发只产生 `scene_unknown`。神经网络 softmax 常过度自信，因此温度校准仍是必要条件；经典 calibration 证据见 [Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599)。选择性预测/coverage-risk 思路可参考 Deep Gambler [Liu et al., NeurIPS 2019](https://arxiv.org/abs/1907.00208)：本项目应优化“在 coverage 降低时错误风险是否单调下降”，而不是追求 100% 分类覆盖。

## 6. IQA / aesthetics：能说明什么，不能说明什么

| 工作 | 年份/场合 | 对本项目的价值 | 关键局限 |
|---|---|---|---|
| KonIQ-10k [paper/data](https://database.mmsp-kn.de/koniq-10k-database.html) | TIP 2020 | 真实世界自然失真与 MOS，常用 NR-IQA benchmark | web 图像分布；MOS 不是个人“这组留哪张”偏好 |
| SPAQ [CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.html) | CVPR 2020 | 11k smartphone photos，并有亮度/色彩等属性，域较接近 | 手机型号/摄影分布有限；绝对 MOS 不等于 burst preference |
| MUSIQ [paper](https://arxiv.org/abs/2108.05997) | ICCV 2021 | native-resolution multi-scale patch Transformer；成熟 checkpoint | 多尺度仍是全图感知；无法理解闭眼、烟花时刻或摄影意图 |
| MANIQA [paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.html) | CVPRW 2022 | patch weighting 与多维 attention，NTIRE NR-IQA 强基线 | challenge/失真数据优势不保证个人图库组内 culling；实现较复杂 |
| TOPIQ [paper](https://arxiv.org/abs/2308.03060) | arXiv 2023 / TIP 2024 | 从高层语义到局部失真，跨数据集表现强；`pyiqa` 可试 | 语义引导不等于 subject/scene safety；仍输出单一 MOS-like score |
| Burst selection [paper](https://arxiv.org/abs/1803.07212) | arXiv 2018 | 直接学习同一 burst 内 pairwise ranking，契合“细微帧差” | crowd preference 与本用户偏好不同；真正 burst 假设，不适合姿势/主体大变的 cluster |
| BuIQA [paper](https://arxiv.org/abs/2511.07958) | AAAI 2026 | 明确区分 subjective/objective burst task；Photo Triage 上做相对选择 | 现无可直接采用的成熟权重；主观 pairwise accuracy 约 0.706，且对齐/差分假设近 burst；详见本项目 `BUIQA_ASSESSMENT.md` |

**选择：** 当前继续 MUSIQ，不因 benchmark 新就换 MANIQA/TOPIQ。若要比较，做同一人工 pairwise culling 集上的 blind A/B，metric 是 pairwise accuracy、错误删风险和各 scene coverage，而不是引用跨数据集 SROCC。Aesthetic/IQA 都只能在**同一 cluster + 同 routing context**内当排序证据，不能跨人物、烟花、食物比较绝对分，更不能单独删图。

## 7. 证据权限分级

### 可作为自动决策证据（但仍非单独删除授权）

- 文件身份、Motion Photo 配对、decode 完整性、EXIF/时间等确定性 metadata。
- 已验证的近重复/同组关系；这是 culling 的前提。
- 在同组、同 subject routing 下的 technical signals：严重失焦、严重曝光失败、主体 ROI 清晰度、经本地验证的 MUSIQ **相对差值**。
- 多个独立 quality signal 一致、差距超过本地验证 margin；且组内至少保留 N 张、无 UNKNOWN/保护冲突。

### 只能 shadow / 选择质量策略

- SigLIP、Places365、RAM 的 scene/subject/tag 输出。
- U²-Net/DIS 显著性 mask（只定位测量 ROI）。
- MUSIQ/MANIQA/TOPIQ 的跨 scene absolute score；CLIP-IQA / generic aesthetics。
- burst selection / BuIQA（直至在本地真实 clusters 上有充分验证）。
- 任意 entropy、margin、模型一致性：它们决定是否 abstain，不证明内容“无价值”。

### 不能参与自动删除（当前）

- 单个 scene tag 或“这不是人像”的结论。
- `Disney/cosplay/doll/food/fireworks` 等语义本身。
- face detector `NO_FACE`（面具、重妆、背影、人偶均会破坏该推理）。
- 单一 aesthetic/IQA 排名、caption、自由文本 tags、低 open-vocabulary similarity。
- UNKNOWN、模型冲突、decode 异常、Motion Photo 未安全配对的任何文件。

建议自动删除规则形态：

`eligible = near_duplicate_confirmed AND cluster_keep_floor_satisfied AND no_hard_protection AND routing_not_unknown AND quality_failure_strong AND independent_evidence_count>=2`

其中 scene/subject 永远不出现在 `quality_failure_strong` 内，只用于选择如何计算它。

## 8. HDD 单次读图与 Windows 落地

1. 先读取 metadata/embedded thumbnail；只有进入视觉阶段才 decode。
2. 对每个文件**一次全图 decode**，立即在内存生成：SigLIP 224、RAM 384（如启用）、MUSIQ 所需 multi-scale/patch 输入、显著性输入；不要每个模型重新开 JPEG/HEIC。
3. 缓存内容哈希、模型版本、prompt-bank hash、embedding、raw/calibrated scores、abstention reason。模型或 prompts 变化时只失效对应派生项。
4. GPU 模型顺序执行或按显存分组，优先 SigLIP→释放 text tower/不必要 tensors→MUSIQ；RAM shadow 另批运行。16GB 不是容量瓶颈，451GB HDD I/O 才是。
5. Windows 原生优先 `torch + transformers + pyiqa`；RAM 研究 repo 若依赖冲突则放 WSL2。首次落地必须 smoke test：CUDA wheel、FP16 numerical stability、HEIC/超大 JPEG、中文/长路径、断点续跑。

## 9. 最小验证设计（在启用任何 gating 前）

- 从 6.6 万张分层抽取 hard set：真人普通妆、重妆/面具 coser、迪士尼人偶、真人与人偶同框、食物、静物、纯烟花、烟花+人、夜景、风景、Motion Photo representative frame。
- 每个 coarse label 至少收集足够 positive/negative，特别提高保护类 hard negatives；报告 per-label precision/recall、AUROC、ECE/reliability、UNKNOWN coverage、selective risk。
- culling 另做 pairwise/cluster-level 标注：哪张必须留、可删、无法决定；禁止用 scene annotation 代替 culling annotation。
- 上线顺序：`shadow log → 人工 review UI → 只切换 quality policy → 在严格 duplicate + 强质量失败下小范围自动动作`。任何阶段发现保护类 false negative，回退到 UNKNOWN。

## 10. 总结性推荐

**部署目标应是“小 taxonomy、高 abstention、语义只路由、质量才排序”。** 默认只上 SigLIP-B/16 + MUSIQ；RAM 是处理 cos/人偶难例的 shadow 诊断器，Places365 是轻量 academic baseline，不建议三者常驻。显著性只用于 ROI。所有 recognition 与 aesthetics 输出都不能直接触发删除；真正自动删除仍由近重复确认、组内保留底线、强质量差和多证据一致共同控制。
