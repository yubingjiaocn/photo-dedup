# BuIQA 落地可行性评估

> 调研对象：*Burst Image Quality Assessment: A New Benchmark and Unified Framework for Multiple Downstream Tasks*（arXiv 2511.07958，**AAAI 2026**，Liang et al., 北航 BUAA）
> 目标：判断是否值得用 BuIQA 替换现有 MUSIQ 作为照片去重的主 IQA。
> 调研日期：2026-07-23。仅调研，未跑代码。

---

## 结论：**不推**（作为主 IQA 替换 MUSIQ）。最多列为**低优先级未来实验**，默认继续用 MUSIQ + CLIP-IQA。

**5 条硬事实证据：**

1. **在我们的场景（主观挑最佳帧）上，BuIQA 只比第二名强 2.6 个百分点。** 论文 Table 2（BI-SQA 主观任务，pairwise accuracy）：Photo Triage 上 Ours=0.706 vs 次优 ELTA=0.680（+0.026）；SPAQ 上 Ours=0.582 vs 次优 SPAQ-model=0.566（+0.016）。这是"挑best"任务的天花板——即便最好的方法也只有 70.6% 的对子跟人类偏好一致。**BuIQA 的碾压级优势全在 objective 任务（超分/去噪挑帧），不是我们的场景。**

2. **论文根本没跟 MUSIQ / CLIP-IQA 对比。** 对比的 7 个 baseline 是 PAU / SPAQ / PAUQA / ELTA / ESFD（IQA）+ FasterVQA / KVQ（VQA）。所以"比 MUSIQ 强多少"这个问题**没有直接答案**，只能推断：它比同代 5 个 IQA 方法在主观任务上仅微弱领先。

3. **没有开源预训练权重。** 官方 repo（`github.com/Hiuyee124/AAAI26-BuIQA`）只有训练/评测代码 + 标注 json，`No releases published`、Packages 为空、HuggingFace 无权重。要用就得**自己下 Photo Triage + SPAQ 数据集从头训**。不是 `pip install` 能跑的东西。

4. **架构上是"连拍序列"专用，跟我们的聚类场景有本质错配。** 模型输入是整个 burst 序列 `B ∈ ℝ^{T×H×W×3}`，核心机制是把每帧对齐到参考帧后**做特征差分**（`P = Fea³ - Fea³_ref`）来捕捉"帧间细微差异"。这假设各帧是近乎同一场景的亚像素抖动。我们 DINOv2 聚类里"同一人换姿势、主体都变了"的打卡照根本不是 burst，参考帧差分会失效或行为不可预测。MUSIQ 是单帧独立打分，对任何分组都稳。

5. **它的杀手锏（知识蒸馏）只对 objective 任务有效。** BuIQA 靠从下游模型（去噪/超分网络）蒸馏 per-frame 贡献度拿到 SRCC 0.81 vs baseline 0.42、下游 +0.33dB PSNR。这套机制在主观挑帧上**明确不启用**（论文原文："for subjective tasks, it learns the subtle difference between frames... without the need of distillation"）。也就是说主观分支退化成一个普通的 burst-aware IQA，优势所剩无几。

**一句话：** BuIQA 是给"burst 超分/去噪流水线选喂哪几帧"设计的神器，不是给"消费级相册挑最好看那张"设计的。我们的场景落在它最弱的那一半，还没有现成权重。ROI 不成立。

---

## 1. 论文本体

| 项目 | 内容 |
|------|------|
| **会议** | AAAI 2026（vol 40, no 9, pp 6880–6888），已录用发表 |
| **机构** | 北航（BUAA），Mai Xu / Lai Jiang 组 |
| **任务定义** | BuIQA：评估 burst 序列中**每一帧**的 task-driven 质量分，用于自适应挑帧 |
| **输入** | 整个 burst 序列 `B ∈ ℝ^{T×H×W×3}`（T 帧全喂进去，不是单帧）；合成数据用到 14 帧/序列 |
| **输出** | 序列里所有帧的质量分 `Ŝ = {Ŝ₁,...,Ŝₙ}`（相对排序，不是绝对分） |

### 模型架构（两段式）

- **Prompt Generation Network（TPG 模块）**：conv+ReLU → 2 个 residual block → feature alignment（每帧对齐参考帧）→ **差分**得到 prompt `P`（`T×H×W×3`）。objective 任务额外做**异构知识蒸馏**（从下游去噪/超分模型学 per-frame 贡献），subjective 任务不蒸馏。
- **Quality Assessment Network**：**冻结的 Swin Transformer backbone** + 可学习 adapter encoder（prompt 注入）+ multi-scale attention（4 尺度 softmax 加权融合）+ MLP 头出分。
- **backbone 可选**：README `--model` 支持 `swin` / `res`(ResNet) / `vgg`；论文正文用 **Swin Transformer**。baseline 用 VGG-16。
- **参数量**：论文没给显式数字。但 backbone 冻结、只训 adapter+attention+MLP+TPG，**可训练参数很小**；Swin-T/B 本体 28M/88M 量级。**推理显存需求极低**，不是瓶颈。
- **损失**：蒸馏 KL loss（`L_Dist`）+ margin/ranking loss（`L_Mrg`，带 grouping-rank 抗序列长度扰动），`L = α·L_Dist + β·L_Mrg`，α=1, β=10。

### 训练数据（BuIQA 数据集，自建，7,346 序列 / 45,827 图 / 191,572 标注）

分两个子集：

- **BI-OQA（客观，2,237 序列 / 30,543 图 / 176,288 标注）**：源自 BurstSR + HDR+ + 合成 RAW burst（Zurich）。GT 分数 = 跑 8 个下游模型（去噪 HDR21/INN/DBD/BPN，超分 EBSR/BSRT/DBSR/BIP）测每帧对 PSNR 的贡献。**RAW / 计算摄影域，跟我们无关。**
- **BI-SQA（主观，5,109 序列 / 15,284 图）**——**这才是跟我们相关的**：
  - **Photo Triage**（Princeton，Chang 2016）：4,175 序列 / 11,314 图。消费级"同一场景连续拍多张、人工标哪张该留"。**这就是打卡照/连拍挑best 的原型问题。**
  - **SPAQ**（智能手机照片质量库）：把视觉相似样本**分组**成 934 个伪 burst / 3,970 图。

> **Domain 判断：** BI-SQA 确实是消费级/手机域，且 SPAQ 的"把相似图分组当序列"跟我们 DINOv2 聚类的做法**几乎一模一样**——这是它唯一站得住的加分项。但注意：Photo Triage/SPAQ 里的组仍是"同场景相似帧"，论文**没有在强姿势变化/主体变化的数据上测过**，对齐+差分机制在那种数据上是否成立没证据。

### 关键 metric 与 baseline 对比

**主观任务（Table 2，pairwise accuracy，我们的场景）** — 数值越高越好，0.5=随机：

| 方法 | Photo Triage | SPAQ |
|------|:---:|:---:|
| **Ours (BuIQA)** | **0.706** | **0.582** |
| ELTA | 0.680 | 0.510 |
| ESFD | 0.665 | 0.538 |
| PAUQA | 0.639 | 0.501 |
| PAU | 0.632 | 0.524 |
| Baseline (VGG-16) | 0.631 | 0.501 |
| SPAQ | 0.584 | 0.566 |
| FasterVQA | 0.587 | 0.503 |
| KVQ | 0.533 | 0.472 |

→ BuIQA 领先次优仅 **+0.026 / +0.016**。**MUSIQ、CLIP-IQA 都不在表里。**

**客观任务（Table 1，SRCC，非我们场景）**：BuIQA 碾压。例：HDR21 去噪 R0 下 Ours=0.810 vs 次优 0.421；EBSR 超分 SRCC 至少 +0.343。下游实用性：按 BuIQA 挑帧喂去噪/超分，**+0.33 dB PSNR**。

**SOTA 场景归属：** BuIQA 的真本事在 **objective 挑帧**（超分/去噪）。**subjective 挑帧**（=我们的场景）它只是"略好于同代 IQA"。我们的场景恰好吻合它最弱的一半。

---

## 2. 代码 / 权重可用性

| 项目 | 状态 |
|------|------|
| **官方 GitHub** | ✅ `github.com/Hiuyee124/AAAI26-BuIQA`（Python 100%，含 `main.py`/`trainer.py`/`model`/`data`/`utils`/`ann`）。⚠️ 仅 2 star，无 issue 活跃度 |
| **预训练权重** | ❌ **无。** `No releases published`、Packages 空、HF 搜不到。**必须自己训。** |
| **pip 包** | ❌ 无。研究 repo，clone 后 `python main.py` 跑 |
| **依赖** | ✅ 轻量（见第 5 节）：torch≥2.0 / torchvision / numpy / scipy / sklearn / opencv / PIL / tqdm / tensorboard。无异国依赖 |
| **数据集** | ❌ 需自行下载 Zurich/BurstSR/HDR+/SPAQ/Photo-Triage（repo 只给标注 json `ann/<Dataset>/result_<model>.json`） |
| **能否独立跑** | 训练/评测 pipeline 齐全，但**跑主观分支要下 Photo Triage + SPAQ 从头训**。跑通到有可用模型 ≈ **数据下载 + 训练 pipeline 调通几天**的工作量，不是即插即用 |

**自实现工作量估计（若不用官方 repo）**：论文 + repo 描述已足够精确到能复现（架构、loss、超参 α=1/β=10/lr=1e-3/Adam/4:1 split 都给了）。但没必要——直接用官方 repo 训更快。真正成本是**数据准备 + 训练 + 在我们自己数据上验证**，乐观 3-5 天，悲观一周+（含 debug 与效果达不到预期的返工）。

---

## 3. 场景匹配度

| 维度 | 判断 |
|------|------|
| Domain（消费级/手机连拍打卡照） | 🟡 **部分匹配。** BI-SQA=Photo Triage(消费连拍) + SPAQ(手机照分组)，域对。SPAQ 的"相似图分组成序列"跟我们 DINOv2 聚类思路一致 |
| "同一人换姿势、主体变了"的打卡照 | 🔴 **高风险失效。** 模型靠对齐参考帧+特征差分捕捉"帧间细微差异"，假设近同场景。姿势/主体大变时 prompt 差分无意义，论文未在此类数据验证 |
| 需要 burst 序列结构 | 🟡 输出是**序列内相对排序**，得把 cluster 当"序列"整体喂。对"组内挑best"逻辑上兼容，但对齐模块仍假设 burst-like |
| 主观挑帧效果绝对值 | 🔴 天花板才 0.706 pairwise acc（Photo Triage），且仅 +2.6pp |
| **单帧 IQA（MUSIQ）兜底** | 🟢 **更稳。** 单帧独立打分，对任意分组/姿势变化免疫；pyiqa 一行调用、有权重、社区成熟。作为主 IQA 的鲁棒性明显优于"需自训 + burst 假设"的 BuIQA |

---

## 4. 三档落地推荐 → **落在"不推"**

- **强推（换主 IQA）**：❌ 不成立。要求代码好+权重开源+效果显著+依赖轻。**权重缺失、主观效果仅微弱领先、未跟 MUSIQ 直接对比**——三项硬伤。
- **可选（opt-in，默认 MUSIQ）**：🟡 **勉强够格但低优先级。** 依赖确实轻、域部分匹配。若未来想做"objective 挑帧"（比如给某个超分流程选帧）可考虑。但对当前"相册去重挑best"收益不明。
- **不推（用 MUSIQ + CLIP-IQA 兜底）**：✅ **当前推荐。** 现有 pyiqa 方案（MUSIQ 技术质量 + CLIP-IQA 美学 tie-break）即插即用、鲁棒、零训练成本，覆盖我们的主观挑best 场景已足够。BuIQA 的增量价值（主观 +2.6pp、且需自训 + burst 假设）不值当替换。

**建议动作：** 继续 MUSIQ 为主、CLIP-IQA 做美学补充。把 BuIQA 标记为"若日后引入 objective 挑帧需求再评估"的 backlog 项，不投入。

---

## 5. 技术依赖清单（若真要落地，供参考）

对齐目标机 **RTX 5070 Super 16GB / PyTorch CUDA 12.x**：

| 项 | 需求 | 显存/大小 | 16GB 是否够 |
|----|------|:---:|:---:|
| 框架 | `torch>=2.0` + `torchvision>=0.15`（CUDA 12.1 轮子） | — | ✅ |
| 其他库 | numpy/scipy/scikit-learn/opencv-python/Pillow/tqdm/tensorboard/matplotlib/pandas/dill | 轻量 | ✅ |
| backbone | Swin Transformer（冻结）+ adapter | Swin-T ~28M / Swin-B ~88M 参数 | ✅ 推理/训练均绰绰有余 |
| 训练数据 | Photo Triage + SPAQ（主观）；BurstSR/HDR+/Zurich（客观，我们用不上） | 数据集数十 GB | 磁盘非显存问题 |
| 预训练权重 | **无——必须自训** | — | 训练 batch 下 16GB 足够（模型小） |

**结论：** 显存/依赖完全不是障碍，16GB 富余。真正的成本是**数据准备 + 自训 + 在我们数据上验证**的工程时间，以及"训完发现主观场景只比 MUSIQ 好一丢丢"的返工风险。

---

## 附：引用与链接

- 论文（arXiv HTML）：https://arxiv.org/html/2511.07958v1 ｜ abs：https://arxiv.org/abs/2511.07958
- AAAI 2026 官方：https://ojs.aaai.org/index.php/AAAI/article/view/37621
- 官方代码：https://github.com/Hiuyee124/AAAI26-BuIQA （无权重、无 release）
- BibTeX：`liang2026burst`，AAAI vol 40(9): 6880–6888, 2026
