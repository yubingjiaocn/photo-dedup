# P0 — 过曝 / 欠曝 / 曝光不可恢复检测

> **调研目标**：为 photo-dedup 增加"安全废片标签"能力中的曝光维度。零付费、本地离线、开源可商用/可自用。
> **环境**：Windows 11 + RTX 5070 Ti 16GB + Python 3.11/3.12，约 66,000 张（主要手机 JPG，含 Motion Photo）。
> **明确不做**：自动调色、曝光矫正、HDR 融合。只出**标签**。
> **调研日期**：2026-08-03。本文中标 `【实测】` 的数字是我在本机跑出来的，脚本与原始输出见文末附录 A。

---

## 0. 一句话结论

**不要用任何 IQA 模型做曝光判废。用纯 CV 的"信息丢失度量"三层规则 + YuNet 人脸 ROI 兜底，全部跑在 OpenCV/numpy 上，零权重、零 license 风险、43 ms/图。**

推荐组合（详见 §5）：

| 层 | 干什么 | 依赖 | 成本 |
|----|--------|------|------|
| **L1 硬 clipping 统计** | 全通道 clip 面积 + 最大连通块 | numpy | ~5 ms |
| **L2 信息存活度** | usable tone mass / anchor mass / 非 clip 区熵 | numpy + cv2 | ~15 ms |
| **L3 主体豁免** | YuNet 人脸 ROI 内曝光（复用已有 stage1 人脸） | cv2 (已装) | ~7 ms（已在跑） |
| **判定** | 三档 reject / maybe / ok，reject 需**多条件同时成立** | — | 0 |

**关键设计决定：曝光判废必须"多条件 AND"，不能单看 clipping 比例。** 因为最容易误杀的三类照片（剪影、夜景、雪景/逆光）单看 clipping 比例跟真废片长得一模一样。区分它们的是"画面里还有没有一块曝光正常的区域（anchor）"和"非 clip 区还有多少结构（熵）"。这两个是本次调研最有价值的产出。

---

## 1. 先把问题拆开：这不是一个指标能解决的

任务要求区分四件事，它们的判据完全不同：

| 现象 | 本质 | 该用什么判 | 该不该判废 |
|------|------|-----------|-----------|
| **A. 简单 histogram clipping** | 有像素撞到 0 或 255 | clip 面积比 | ❌ **不该**。几乎所有正常照片都有一点点 |
| **B. 局部高光/阴影** | 小面积高光溢出（金属反光、太阳、灯泡）或小面积死黑 | 最大**连通块**面积，不是总面积 | ❌ **不该**。这是物理必然，不是拍坏 |
| **C. 审美曝光** | 剪影、低调人像、夜景、high-key 雪景 | 「主体是否正常曝光」+「非 clip 区是否还有结构」 | ❌ **绝对不该**。误杀这类是最严重的错误 |
| **D. 真实信息丢失** | 整幅画面主体区落在不可恢复带，无任何正常曝光锚点 | clip 面积 **AND** anchor 缺失 **AND** 熵塌陷（多条件） | ✅ **该**（且要留 maybe 缓冲） |

**A/B 是干扰项，C 是雷区，只有 D 是目标。** 现有商业工具（像素蛋糕、Aftershoot）之所以给"弱/标准/强"三档而不是一个开关，就是因为 C 和 D 的边界是**主观的、随题材变化的**。我们的做法应该是：把阈值调到"只抓最没争议的 D"，剩下全进 maybe。

### 1.1 RAW / JPEG 差异（对我们的库：**基本不用管，但有一个真陷阱**）

| 维度 | RAW | 相机出的 JPEG（我们的情况） |
|------|-----|---------------------------|
| 位深 | 12–14 bit 线性 | **8 bit，已做 gamma + tone curve** |
| clipping 定义 | 传感器 saturation，`> 1.0 after white level normalization`（darktable 原文）| 编码值 255 |
| 高光可恢复性 | 单/双通道 clip 常可从未 clip 通道重建 | **255 就是 255，没有任何余量** |
| 阴影可恢复性 | 有噪声底之上的真实数据 | **已被 tone curve 压死 + JPEG 量化** |
| 检测阈值参照 | libraw/rawspeed 里的 sensor saturation 值 | 固定 250/255 与 5/255 |

**结论 1：我们的库是相机 JPEG，"可恢复性"这个问题在很大程度上已经被相机替我们回答了 —— 撞到 255 就是不可恢复。** 这让判据大大简化，也让判废更安全（不会出现"其实 RAW 里还有数据"的冤案）。

**结论 2（真陷阱）：手机 HDR / 夜景模式会主动把画面拉到 mid-tone，导致"看起来曝光很好但实际单帧信息很差"，以及反过来"直方图看着很凶但实际是相机故意的"。** 具体表现：
- 小米/华为夜景模式合成后画面 `Ymean` 会被抬到 0.25–0.35，clip 都很低 → 我们的规则会判 ok。**正确**，不该动它。
- 手机 HDR 逆光人像：天空会被压到 0.85–0.95 不 clip，人脸被提亮 → 判 ok。**正确**。
- **但如果是老手机/关了 HDR 的逆光直出**，天空整片 255、人脸整片死黑 → 这是真 D，规则要能抓到。**这就是为什么必须有人脸 ROI 那一层**（§5 L3）。

**结论 3：`.MP` / `.MP4` Motion Photo 伙伴不单独判曝光**，跟随 jpg 的判定（沿用项目现有 `motion_partner_id` 生死绑定原则）。

**结论 4：EXIF 不可靠，不要当主判据。** 【实测】10 张测试图里 6 张（Wikimedia 转存）**完全没有曝光相关 EXIF**；4 张有 `ExposureTime`/`FNumber` 的，`ExposureBiasValue` 全是 0.0（因为是包围曝光靠快门实现的）。手机照片的 `BrightnessValue` 字段 SPAQ 论文也明确说"部分厂商不提供，需要用曝光方程反推"。
→ **EXIF 只能做 side-channel 加分项**（例如 `Flash` 触发过 → 对过曝更宽容；`ISO` 极高 → 对暗部噪声更宽容），不能做判据。

---

## 2. 候选方案盘点：license / 权重 / 性能 / 适用性

### 2.1 纯 CV / 传统方法（**全部推荐**）

| 方法 | 出处 | License | 权重 | 我的判断 |
|------|------|---------|------|---------|
| **直方图 clipping 计数** | 通用；RawTherapee `Clip %`（缩略图固定 **0.2%**）、darktable raw-overexposed 模块（默认阈值 1.0，社区反馈需调到 0.99 才标全区域） | 无（自己写 3 行） | 无 | ✅ **必用**，L1 基座 |
| **连通域分析** | `cv2.connectedComponentsWithStats` | Apache 2.0（opencv-python，**已装 5.0.0**） | 无 | ✅ **必用**，区分 B 和 D 的关键 |
| **Mertens well-exposedness** `E = Π_c exp(−(I_c−0.5)²/(2σ²))`, σ=0.2 | Mertens et al. 2007；IPOL 2018 复现文（Hessel）明确 σ=0.2，并实测 σ∈[0.1,0.2] 最优 | 论文公式，无 license 约束；`cv2.createMergeMertens` 现成 | 无 | 🟡 **参考不采纳为主判据**，见 §4.2 |
| **skimage `is_low_contrast`** | scikit-image，BSD-3 | 无 | 🟡 可选，逻辑就是 `p99−p1 < 0.05`，自己写更可控 |
| **YuNet 人脸 ROI 亮度** | opencv_zoo（**项目已在用**） | 模型 MIT（opencv_zoo） | 232 KB ONNX（已下载验证） | ✅ **必用**，L3 主体豁免 |

### 2.2 IQA 模型（**全部不推荐做曝光判废**）

| 方案 | License | 权重 | 为什么不推 |
|------|---------|------|-----------|
| **CLIP-IQA `brightness` prompt**（"Bright photo." vs "Dark photo."） | prompt 本身无约束；`openai/clip-vit-base-patch16` 权重可用 | 600 MB | 🔴 **实测证伪，见 §4.1。它测的是语义"明亮感"，不是光度学曝光。夕阳剪影拿 0.998（最"亮"），比 +4EV 全白的 0.900 还高。** |
| **pyiqa (`clipiqa` / `musiq` / `topiq`)** | ⚠️ **PolyForm Noncommercial 1.0.0**（我在本机 `pyiqa-0.1.16.dist-info/licenses/LICENSE` 里确认了原文） | 各几百 MB | 🔴 输出是**单一综合质量分**，把曝光、锐度、噪声、构图揉在一起，**不可解释**，无法给出"因为过曝所以废"的理由。用来判废等于黑箱杀片。（另：非商业 license 对 Willy 个人用途 OK，但如果这套东西以后要开源/给别人用就是雷） |
| **SPAQ MT-A**（CVPR'20，唯一直接输出 **brightness attribute score** 的模型） | 🔴 **代码/权重 license 至今未澄清** —— repo issue #37 "License Clarification for Code and Trained Models" 无回复；issue #29 另问 Dataset License；数据集侧称 MIT | ResNet-50 级，权重在 Google Drive / Baidu（issue #9 报下载失效） | 🟡 **学术上最对口**（brightness attribute 与人类 MOS 的 SRCC = 0.784 人工 / 0.704 模型），但：① license 悬空；② 输出是 0–100 的**审美 brightness 分**，不是"信息是否丢失"，正是我们要避免的 C/D 混淆；③ 权重下载不稳。**列为参考文献，不落地。** |
| **Q-Align / OneAlign**（ICML'24） | 代码 Apache，模型走 HF `q-future/one-align`；也能 `pyiqa create_metric('qalign')`（则受 PolyForm 约束） | **mPLUG-Owl2 级 LMM，~7B**，16GB 显存跑 fp16 勉强 | 🔴 66,000 张跑 LMM 不现实（哪怕 0.5 s/图也是 9 小时纯 GPU，且 5070 Ti 16GB 要 4bit 量化）。输出仍是综合分，同上不可解释 |
| **Afifi CVPR'21 Exposure Correction** | 数据集随 MIT-Adobe FiveK 原 license（**FiveK 是研究用途，非自由商用**）；代码需 **MATLAB 2019b + Deep Learning Toolbox** | — | 🔴 **要 MATLAB，直接淘汰。** 且它是**矫正**模型不是**检测**模型。数据集的 EV 分级（−1.5/+1.5 等相对 EV 渲染）可以借来做验证集思路，但不落地 |
| **BuIQA (AAAI'26)** | — | **无权重，需自训** | 🔴 已在 `research/BUIQA_ASSESSMENT.md` 结论"不推"。此处一致：不碰 |

### 2.3 同类开源项目（借鉴，不直接依赖）

GitHub `photo-culling` topic 下 2026 年爆发了一批本地 AI 挑片工具，值得注意的是它们的**分档语言**几乎统一：

- **`ncoevoet/facet`**（182 ★，最成熟）：本地 AI 打分 + gallery，用 **topiq** 做质量，有 **genre profiles**（sports/wedding/concert/wildlife 预设捆绑 strictness + keeper budget），auto-cull 带 **dry-run preview + keeper budget**。
- **`joneilcaoile/photocull-ai`**：明确 **Keep / Maybe / Reject** 三档 + 可调权重阈值。
- **`a45674567/fixxer`**：burst 分组 + sharpness/exposure/blink 打分 + XMP 导出。
- **`host452b/ShutterSift`**：Keep/Review/Reject。

**借鉴到的三点**（跟我们 P0 目标完全吻合）：
1. **三档而非二档**是行业共识（Aftershoot 极端筛选也另设 Maybe）；
2. **genre / scene profile** 是解决"C 类审美曝光"误杀的正统解法（夜景/演出场景整套放宽阈值）；
3. **keeper budget + dry-run** 是安全阀（项目已有 `dry_run: true`，方向对）。

不直接依赖它们的代码：曝光这块逻辑很短（<150 行），自己写可控性远好于引入依赖。

---

## 3. 【实测】数据一：真实照片 + 合成 EV 阶梯

测试集（10 张真图，CC 授权，Wikimedia Commons）：
- `StLouisArchMultExpEV{+4.09,+1.51,−1.82,−4.72}` —— 同场景**已标注相对 EV** 的包围曝光四连，这是天然的 ground truth
- `aes_snow`（high-key 雪景）、`aes_night_city`（夜景）、`aes_sunset_silhouette`（夕阳剪影）、`aes_silhouette2`（逆光剪影）、`aes_concert` / `aes_stage_dark`（暗调舞台，含人脸）
- 再对 3 张真图做 **±1/2/3 EV 合成偏移**（sRGB→线性→乘 2^EV→回 sRGB，模拟相机曝光误差），共 28 个样本

### 3.1 关键特征表（512px 长边解码）

| file | clip_hi | clip_lo | blob_hi | blob_lo | mass_usable | **anchor** | **entropy_keep** | Ymean |
|------|--------:|--------:|--------:|--------:|------------:|-----------:|-----------------:|------:|
| `EV+4.09`（严重过曝） | 0.1408 | 0.0000 | 0.1075 | 0.0000 | 0.8379 | 0.7097 | 5.4776 | 0.5602 |
| `EV+1.51` | 0.0517 | 0.0000 | 0.0246 | 0.0000 | 0.9410 | 0.1985 | 4.6864 | 0.2487 |
| `EV−1.82` | 0.0022 | **0.6724** | 0.0002 | **0.4612** | 0.2689 | 0.0773 | 4.5973 | 0.0675 |
| `EV−4.72`（严重欠曝） | 0.0000 | **0.8900** | 0.0000 | **0.8781** | **0.0948** | **0.0116** | 3.9518 | 0.0141 |
| `aes_snow`（高调雪景） | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **1.0000** | **0.9961** | 4.1983 | 0.5764 |
| `aes_night_city`（夜景） | 0.0001 | 0.0031 | 0.0000 | 0.0004 | 0.9879 | 0.4476 | 5.1701 | 0.2849 |
| `aes_sunset_silhouette` | 0.0000 | 0.0166 | 0.0000 | 0.0034 | 0.9628 | 0.6677 | 5.2110 | 0.4004 |
| `aes_silhouette2`（逆光剪影） | 0.0000 | **0.3709** | 0.0000 | **0.2875** | 0.4292 | 0.1987 | 4.3636 | 0.1001 |
| `aes_concert`（暗调有人脸） | 0.0022 | 0.0037 | 0.0006 | 0.0002 | 0.7877 | 0.2340 | 4.7782 | 0.1681 |
| `aes_stage_dark`（暗调舞台） | 0.0001 | **0.1282** | 0.0000 | 0.0432 | 0.4796 | 0.0871 | 3.3001 | 0.0800 |
| `ev_snow_+2`（合成 +2EV） | **0.4784** | 0.0000 | **0.2533** | 0.0000 | 0.3433 | 0.0873 | 3.8422 | 0.9515 |
| `ev_snow_+3` | **0.8546** | 0.0000 | **0.7716** | 0.0000 | 0.1051 | 0.0311 | 3.8722 | 0.9821 |
| `ev_night_city_+3` | 0.0714 | 0.0000 | 0.0437 | 0.0000 | 0.8863 | 0.7116 | 5.6489 | 0.6311 |
| `ev_sunset_silhouette_−3` | 0.0000 | 0.1312 | 0.0000 | 0.0991 | 0.7372 | 0.0748 | 3.7221 | 0.1380 |

### 3.2 从这张表能直接读出的四条硬结论

**① `clip_lo` 单指标必然误杀剪影。** `EV−1.82`（真欠曝）= 0.672，`aes_silhouette2`（艺术剪影）= 0.371，`aes_stage_dark`（艺术暗调）= 0.128。三者连续分布，**没有一条 clip_lo 线能把它们分开**。

**② 但 `anchor` 能分开。** `anchor` = 落在 [0.25, 0.85] 亮度带的像素比例（"有没有一块曝光正常的区域"）：
- 真废片 `EV−4.72` = **0.0116**
- 艺术剪影 `aes_silhouette2` = **0.1987**（17 倍）
- 夕阳剪影 `aes_sunset_silhouette` = **0.6677**（57 倍）

**这是本次调研最重要的单一发现。** 艺术性的暗/亮照片，几乎总有一块正常曝光的区域承载视觉重点；真废片没有。

**③ `entropy_keep`（非 clip 区的 64-bin 亮度熵）是第二判据。** 它回答"剩下没被 clip 的部分还有信息吗"：
- `EV−4.72` = 3.95 但 `mass_usable` 只有 0.0948 → **熵高但载体只剩 9%**，所以熵必须跟 mass 联合看
- `aes_stage_dark` = 3.30（最低）→ 这提醒我们**熵不能单独当判据**，暗调舞台照的熵天然低

**④ 最大连通块（`blob_*`）区分"局部反光"和"整片糊掉"。** `aes_concert` 的 `clip_hi` = 0.0022 但 `blob_hi` = 0.0006 → 是分散的舞台灯点，不是一片死白。而 `ev_snow_+3` 的 `blob_hi` = 0.7716 → 单一巨块，77% 的画面连成一片白。**总面积相同的两张图，blob 能差两个数量级。**

### 3.3 EV 阶梯：阈值的量化依据

对 `aes_snow`（high-key 雪景，最脆弱的过曝题材）逐档偏移：

| EV | clip_hi | mass_usable | entropy | **anchor** | 我认为该判 |
|---:|--------:|------------:|--------:|-----------:|-----------|
| −3.0 | 0.0000 | 1.0000 | 2.98 | 0.0954 | ok（欠曝但完全可救） |
| −2.0 | 0.0000 | 1.0000 | 3.36 | 0.8893 | ok |
| **0.0** | 0.0000 | 1.0000 | 4.35 | 0.9908 | ok（原图） |
| +1.0 | 0.0338 | 0.9398 | 4.60 | 0.7774 | ok |
| +1.5 | 0.1956 | 0.7499 | 4.02 | 0.2101 | **maybe** |
| **+2.0** | **0.6823** | **0.1784** | **1.75** | 0.0769 | **reject** |
| +2.5 | 0.9110 | 0.0682 | 0.75 | 0.0363 | reject |

对 `aes_night_city`（夜景）：

| EV | clip_hi | clip_lo | mass_usable | anchor | 我认为该判 |
|---:|--------:|--------:|------------:|-------:|-----------|
| −5.0 | 0.0000 | 0.2223 | 0.3740 | 0.0000 | maybe（极暗，但夜景合理） |
| −3.0 | 0.0000 | 0.0584 | 0.8172 | 0.0326 | ok |
| 0.0 | 0.0002 | 0.0046 | 0.9824 | 0.4320 | ok |
| +3.0 | 0.1360 | 0.0000 | 0.7893 | 0.6240 | ok（夜景提亮 3EV 仍可用！） |
| +5.0 | 0.3595 | 0.0000 | 0.5071 | 0.2736 | maybe |

**读出的阈值**：
- 过曝的**崩塌点在 `clip_hi ≈ 0.20–0.35` 之间**，且必须配 `anchor < 0.10`。雪景 +1.5EV（clip_hi=0.196, anchor=0.21）还救得回；+2.0EV（clip_hi=0.682, anchor=0.077）废了。
- **注意夜景 +3EV 的 `clip_hi` = 0.136 但 `anchor` = 0.624** → 单看 clip_hi 会跟雪景 +1.5EV 混淆，**anchor 是唯一的分离器**。再次印证 §3.2①。
- 欠曝的崩塌点在 `mass_usable < 0.25` **AND** `anchor < 0.05`。

### 3.4 解码分辨率的影响（工程上必须知道）

【实测】同一张图在不同解码长边下的 clip 比例：

| file | hi@full | hi@1024 | hi@512 | hi@256 |
|------|--------:|--------:|-------:|-------:|
| `EV+4.09`（大面积过曝） | 0.1565 | 0.1499 | 0.1408 | 0.1275 |
| `aes_concert`（分散小高光） | 0.0032 | 0.0026 | 0.0022 | 0.0013 |
| `aes_night_city` | 0.0005 | 0.0002 | 0.0001 | 0.0000 |

**结论：降采样系统性低估 clip 比例，但对我们有利。**
- 大面积过曝：512px 相对全尺寸只低 **−10%**（0.1565→0.1408），阈值余量完全吃得下。
- 小面积高光：512px 低 **−31%**，256px 低 **−59%** —— **正好帮我们自动过滤掉 §1 表里的 B 类（局部反光）**，等于免费的一层去噪。
- **推荐用 512px 长边 + PIL `draft()` DCT 缩放解码。** 但阈值必须在**同一分辨率**下校准 —— 换分辨率必须重新跑校准（这是个容易踩的坑，写进代码注释）。

---

## 4. 【实测】数据二：两个"看起来该用"的方案被证伪

### 4.1 CLIP-IQA brightness prompt —— 明确不能用

跑 `openai/clip-vit-base-patch16`（CPU，10 张，1.61 s 推理），4 组 antonym prompt 的正向概率（softmax over pair, logit_scale=100）：

| file | p0 `Bright/Dark photo.` | p1 well-exp vs **overexposed** | p2 well-exp vs **underexposed** | p3 **intentionally dark** vs **accidentally ruined** |
|------|----:|----:|----:|----:|
| `EV+4.09`（严重过曝） | 0.900 | **0.057** | 0.115 | 0.759 |
| `EV+1.51` | 0.705 | 0.170 | 0.260 | 0.892 |
| `EV−1.82` | 0.262 | 0.227 | **0.058** | 0.912 |
| `EV−4.72`（严重欠曝） | 0.016 | 0.320 | **0.003** | 0.947 |
| `aes_snow`（正常高调） | 0.986 | 0.235 | 0.894 | 0.965 |
| `aes_night_city`（正常夜景） | **0.981** | 0.242 | 0.790 | 0.943 |
| `aes_sunset_silhouette` | **0.998** | 0.213 | 0.842 | 0.967 |
| `aes_stage_dark`（暗调舞台） | 0.662 | **0.971** | **0.964** | 0.756 |
| `aes_concert` | 0.817 | 0.300 | 0.952 | 0.488 |

**三个致命问题：**

1. **`brightness` prompt 测的是语义"明亮感"，不是光度学曝光。** `aes_sunset_silhouette`（Ymean=0.40，一张夕阳剪影）拿到 **0.998**，是全场最"亮"，比真·+4EV 全白过曝的 0.900 还高。`aes_night_city`（Ymean=0.28 的夜景）拿 **0.981**。CLIP 认为"有明亮光源/明亮氛围"= bright，跟"直方图偏右"完全是两件事。

2. **符号会反。** `aes_stage_dark` 在 p1（well-exposed vs overexposed）拿 0.971 = "很好曝光"，同时在 p2（well-exposed vs underexposed）拿 0.964 = "也很好曝光"，而它实测 `clip_lo` = 0.128、`anchor` = 0.087 是全场第二暗。**同一张图在两个互补 prompt 上都说"好"，说明模型根本没在回答曝光问题。**

3. **"故意暗 vs 意外毁掉"这个最关键的语义区分（p3）完全无信号。** 全部样本落在 0.49–0.97，且**真废片 `EV−4.72` 拿到最高的 0.947**（最"故意"）。方向是反的。

**判决：CLIP-IQA 的 brightness/quality prompt 不能用于曝光判废。** 这不是 prompt 工程能修的问题 —— CLIP 的图文对齐目标里没有光度学信息，`aes_sunset_silhouette` 那个 0.998 就是铁证。（这也侧面解释了为什么 MDPI 2026 那篇 fine-grained IQA 论文明确批评 "CLIP inherently lacks sensitivity to fine-grained visual cues... exclusive focus on global semantic alignment"。）

> 附带的正面价值：CLIP 依然可以用来做**场景分类**（"a photo of a concert stage" / "a night scene"）以驱动 §6.4 的 scene profile。那是它擅长的语义任务。但别拿它测曝光。

### 4.2 Mertens well-exposedness —— 好指标，错用途

【实测】`E_mean = mean(Π_c exp(−(I_c−0.5)²/0.08))`（σ=0.2，Hessel/IPOL 复现文的标准参数）：

| file | mertens_E_mean | 真实情况 |
|------|---------------:|---------|
| `aes_snow` | **0.6553** | 正常 |
| `EV+1.51` | 0.2706 | 轻过曝 |
| `aes_night_city` | 0.1626 | **正常夜景** |
| `EV+4.09` | **0.0558** | 严重过曝 |
| `aes_sunset_silhouette` | **0.0394** | **正常夕阳（比严重过曝还低！）** |
| `EV−4.72` | 0.0057 | 严重欠曝 |

**问题：Mertens E 的设计目标是"这个像素适合被融合进 HDR 吗"，它奖励一切靠近中灰的像素、惩罚一切偏离的像素。** 所以它把"高对比度的正常照片"（夕阳剪影 0.039）和"信息丢失的废片"（严重过曝 0.056）打成同一档，甚至前者更低。

而且 EV 阶梯上它**非单调**：`aes_snow` 在 −1.0EV 时 E=0.734 > 原图 0EV 的 0.655 —— 因为原图本来就偏亮，压暗反而更"靠近中灰"。**一个在 ground-truth EV 上非单调的指标不能做判据。**

**判决：不作为主判据。** 保留价值：`mertens_frac_wellexp = mean(E > 0.5)` 可以作为 review.html 里给人看的一个辅助数字，或作为**同组内 tie-break**（同一 cluster 内挑最佳帧时，E 高的更"标准曝光"）。这个用途它是对的。

---

## 5. 推荐算法（可直接实现）

### 5.1 特征提取

```python
# 常量。改任何一个都必须重跑校准集（§6）。
HI = 250 / 255.0      # 高光 clip 判定（留 5 级余量给 JPEG 量化 + 色彩转换）
LO = 5 / 255.0        # 阴影 clip 判定
DECODE_LONG_EDGE = 512   # 与阈值绑定，不可随意改（见 §3.4）

def load(path, long_edge=DECODE_LONG_EDGE):
    im = Image.open(path)
    im.draft("RGB", (long_edge, long_edge))   # JPEG DCT 域缩放解码，比 resize 快数倍
    im = im.convert("RGB")
    w, h = im.size
    s = long_edge / max(w, h)
    if s < 1:
        im = im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0

def exposure_features(f):                      # f: HxWx3 float32 in [0,1], sRGB gamma-encoded
    Y = 0.2126*f[...,0] + 0.7152*f[...,1] + 0.0722*f[...,2]   # Rec.709 luma

    # --- clip masks：用"全通道"而非"任一通道" ---
    # 全通道 >= HI 才算真白（任一通道 clip 常见于饱和红花/蓝天，不代表信息丢失）
    hi_m = f.min(axis=2) >= HI
    lo_m = f.max(axis=2) <= LO
    d = {"clip_hi": hi_m.mean(), "clip_lo": lo_m.mean()}

    # --- 最大连通块：区分"局部反光"和"整片糊掉" ---
    for tag, mask in (("hi", hi_m), ("lo", lo_m)):
        mu = mask.astype(np.uint8)
        if mu.any():
            mu = cv2.morphologyEx(mu, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))  # 去孤立噪点
            n, _, st, _ = cv2.connectedComponentsWithStats(mu, connectivity=8)
            a = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
            d[f"blob_{tag}"] = float(a.max() / Y.size) if a.size else 0.0
        else:
            d[f"blob_{tag}"] = 0.0

    # --- 信息存活度 ---
    d["mass_usable"] = ((Y > 0.03) & (Y < 0.97)).mean()    # 还有多少像素在可用带内
    d["anchor"]      = ((Y > 0.25) & (Y < 0.85)).mean()    # ★ 有没有"曝光正常"的锚点区域

    # --- 非 clip 区的亮度熵：剩下的部分还有结构吗 ---
    keep = (~hi_m) & (~lo_m)
    if keep.sum() > 64:
        h, _ = np.histogram(Y[keep], bins=64, range=(0,1))
        p = h / h.sum(); nz = p[p > 0]
        d["entropy_keep"] = float(-(nz*np.log2(nz)).sum())  # 上限 6.0 bits
    else:
        d["entropy_keep"] = 0.0

    d["Ymean"] = float(Y.mean())
    return d
```

### 5.2 判定规则 v0（阈值全部来自 §3 实测）

```python
# 所有 reject 分支都是「多条件 AND」。任何单一指标都不足以判废。
def exposure_label(d, face=None, profile=None):
    ch, cl = d["clip_hi"], d["clip_lo"]
    anc, mu, ek = d["anchor"], d["mass_usable"], d["entropy_keep"]
    bh, bl = d["blob_hi"], d["blob_lo"]
    T = THRESHOLDS[profile or "default"]

    # ---- L3 主体豁免：先看人脸。人脸曝光正常 → 最多 maybe，不 reject ----
    if face is not None and face["ok"]:
        # 人脸 ROI 亮度在合理带内，且脸上没大面积 clip
        if 0.18 <= face["Ymean"] <= 0.85 and face["clip_lo"] < 0.20 and face["clip_hi"] < 0.20:
            return ("ok", "") if (ch < T.hi_maybe and cl < T.lo_maybe) \
                   else ("maybe", f"frame clip hi{ch:.0%}/lo{cl:.0%} but face well-exposed")
        # 人脸自己就废了 → 强证据，直接 reject（这是最安全的一类 reject）
        if face["clip_lo"] > 0.60 or face["clip_hi"] > 0.60:
            return "reject", f"face itself clipped (lo {face['clip_lo']:.0%} / hi {face['clip_hi']:.0%})"

    # ---- 过曝 reject：大面积 AND 连成一片 AND 无锚点 ----
    if ch >= T.hi_reject and bh >= T.blob_reject and anc < T.anchor_reject:
        return "reject", f"blown {ch:.0%} (largest blob {bh:.0%}), anchor {anc:.0%}"

    # ---- 欠曝 reject：大面积死黑 AND 无锚点 AND 剩余结构塌陷 ----
    if cl >= T.lo_reject and anc < T.anchor_reject_dark and ek < T.entropy_reject:
        return "reject", f"black {cl:.0%}, anchor {anc:.0%}, entropy {ek:.1f}"

    # ---- 通用信息丢失：可用色调带几乎空了 ----
    if mu < T.mass_reject and anc < T.anchor_reject_dark:
        return "reject", f"only {mu:.0%} usable tones, anchor {anc:.0%}"

    # ---- maybe 区（送人工/送 review.html） ----
    if ch >= T.hi_maybe:
        return "maybe", f"clip hi {ch:.0%} (blob {bh:.0%}), anchor {anc:.0%}"
    if cl >= T.lo_maybe and anc < T.anchor_maybe:
        return "maybe", f"clip lo {cl:.0%} (blob {bl:.0%}), anchor {anc:.0%}"
    if mu < T.mass_maybe:
        return "maybe", f"usable tones {mu:.0%}"
    return "ok", ""
```

### 5.3 阈值表（v0 起点，**必须按 §6 校准后才能开 reject**）

```yaml
exposure:
  decode_long_edge: 512      # 改这个必须重新校准全部阈值
  hi: 0.9804                 # 250/255
  lo: 0.0196                 # 5/255

  profiles:
    default:
      hi_reject:      0.35   # 来源：ev_snow_+2 (0.478 废) vs ev_snow_+1.5 (0.196 可救)
      blob_reject:    0.20   # 来源：ev_snow_+2 blob=0.253；aes_concert 分散高光 blob=0.0006
      anchor_reject:  0.10   # 来源：ev_snow_+2 anchor=0.087 废 vs night_city_+3 anchor=0.712 好
      lo_reject:      0.60   # 来源：EV−1.82 (0.672) 已明显废；aes_silhouette2 (0.371) 必须活
      anchor_reject_dark: 0.05  # 来源：EV−4.72 anchor=0.0116 vs aes_silhouette2 anchor=0.199
      entropy_reject: 3.00   # 来源：aes_stage_dark ek=3.30 必须活，ev_snow_+2 ek=1.75 该死
      mass_reject:    0.25   # 来源：EV−4.72 mass=0.095 废；EV−1.82 mass=0.269 边界
      hi_maybe:       0.15
      lo_maybe:       0.35
      anchor_maybe:   0.25
      mass_maybe:     0.45

    night:                   # 夜景 / 演出 / 室内暗光：全面放宽暗部
      lo_reject:      0.80
      anchor_reject_dark: 0.02
      entropy_reject: 2.20
      mass_reject:    0.12
      lo_maybe:       0.60
      # 高光侧沿用 default（夜景过曝仍是真缺陷）

    backlit:                 # 逆光 / 剪影 / 日落：放宽暗部 + 容忍天空高光
      lo_reject:      0.85
      anchor_reject_dark: 0.02
      hi_reject:      0.50
      lo_maybe:       0.70
```

**校准前的安全模式**：把所有 `*_reject` 阈值临时设为 `1.01`（不可达），只输出 maybe/ok。跑全库一遍，看 maybe 集合里长什么样，再逐步收紧。

---

## 6. 阈值怎么校准（这是最重要的一节）

### 6.1 三条铁律

1. **不要凭直觉设 reject 阈值。** §3 的表已经证明"看起来合理"的单指标阈值（比如 clip_lo > 0.3）会误杀艺术剪影。
2. **阈值绑定解码分辨率。** 改 `decode_long_edge` 必须全部重跑（§3.4 实测 512 vs full 差 10–31%）。
3. **校准集必须包含"故意的极端曝光"作为负样本。** 只用废片校准出来的阈值一定过杀。

### 6.2 校准集怎么建（150 张，2 小时人工）

从自己的 66,000 张库里**分层抽样**（不是随机抽，随机抽 95% 都是正常照片，学不到边界）：

```
先跑一遍纯特征提取（无判定），把结果落库，然后按下面的桶各抽 N 张人工标注：

桶 A  clip_hi ∈ [0.10, 0.25)             抽 20   ← 过曝边界带，最有信息量
桶 B  clip_hi >= 0.25                     抽 20
桶 C  clip_lo ∈ [0.25, 0.50)             抽 20   ← 剪影 / 欠曝混杂带，最难
桶 D  clip_lo >= 0.50                     抽 20
桶 E  mass_usable < 0.40                  抽 20
桶 F  anchor < 0.15 且 clip_hi/lo 都低    抽 15   ← "低对比雾蒙蒙"，容易漏
桶 G  夜景 / 演出（按 EXIF ISO>1600 或时间 20:00-06:00）抽 20  ← 必须活的负样本
桶 H  完全正常（anchor > 0.6）            抽 15   ← 对照组，验证不过杀
```

**人工只标三档，不给分数**：`废`（我绝对不想留这张）/ `不确定` / `留`。**标的时候不看任何指标数字**，纯看图。标注结果写成 `research/exposure_calib.json`（按 AGENTS.md "多版本项目 = git 或权威指针" 的规则，这个 JSON 就是权威指针，带 `source` / `date` / `decode_long_edge` 字段）。

### 6.3 从标注反推阈值（不训模型，只查表）

```python
# 目标：在校准集上让 reject 的 precision >= 0.98（宁漏不错杀）
# 做法：对每个候选阈值组合，算混淆矩阵，选满足 precision 约束下 recall 最高的那组

for hi_reject in [0.25, 0.30, 0.35, 0.40, 0.50]:
  for anchor_reject in [0.05, 0.08, 0.10, 0.15]:
    for blob_reject in [0.10, 0.15, 0.20, 0.30]:
        pred = [exposure_label(d, ...) for d in calib]
        fp = sum(p == "reject" and gt == "留" for p, gt in zip(pred, gt_labels))
        tp = sum(p == "reject" and gt == "废" for p, gt in zip(pred, gt_labels))
        precision = tp / max(1, tp + fp)
        # 记录所有 precision >= 0.98 的组合，取 recall 最大
```

**硬性收敛准则（不满足就不许开 reject）：**
- `reject` 在标注为「留」的样本上 **FP = 0**（不是"很少"，是 0）。66,000 张里 1% 误杀 = 660 张真照片没了。
- `reject` 在标注为「废」的样本上 recall 不设下限 —— **抓多少算多少，抓不到的进 maybe 由人补**。
- 桶 G（夜景/演出）**全部**不得落 reject。这是硬 gate。

### 6.4 scene profile 怎么定（解决 C 类误杀的正路）

三条可选路线，推荐 ① + ②：

① **EXIF 启发式（零成本，先上）**：`ISO >= 1600` 或 `ExposureTime >= 1/30` 或 EXIF 时间在 20:00–06:00 → `night` profile。命中率不高但零代价。
   ⚠️ §1.1 实测警告：EXIF 缺失率很高，必须有 fallback。

② **无监督分桶（几乎零成本）**：用**已有的 DINOv2 embedding**（stage1 已经在算了！）做 KMeans，人工给几十个簇打 scene 标签，反查每张图的 profile。**这是最划算的路 —— 复用现成特征，不引入任何新模型。**

③ CLIP zero-shot scene 分类（"a photo of a concert" / "a night cityscape" / "a backlit silhouette"）。§4.1 已经证明 CLIP 不能测曝光，但**测场景语义是它的本职**，这个用途是对的。代价是 600 MB 权重 + 一次全库推理。**列为 P1，非必需。**

### 6.5 复校准触发条件（写进代码注释）

任何一条变了就必须重跑 §6.3：
- `decode_long_edge` 改了
- `HI` / `LO` 常量改了
- luma 公式改了（Rec.709 → Rec.601 或改成 L*）
- 库里加入了新相机/新手机（tone curve 不同）
- 加了 RAW 支持（判据完全不同，见 §1.1）

---

## 7. Maybe 机制建议

### 7.1 为什么必须有 maybe

- §3.2① 已证明剪影和真欠曝在单指标上**连续分布，没有干净的分界线**。强行二分必然错杀。
- 行业共识：Aftershoot 的极端筛选另设 Maybe；`photocull-ai` 直接 Keep/Maybe/Reject；`ShutterSift` 用 Keep/Review/Reject。
- Willy 要"全自动"，但**全自动 ≠ 无 maybe**。maybe 的正确含义是"自动不动它，默认保留，另出一张清单供**可选**抽查"。不动手就等于保留，不阻塞流程。

### 7.2 建议的三档语义（跟项目现有机制对齐）

| 标签 | 含义 | 在 pipeline 里的行为 |
|------|------|---------------------|
| `reject` | 高置信废片 | 参与删除候选，但**仍受"每组至少留一张"保护**（见 7.3） |
| `maybe` | 疑似曝光问题 | **不参与删除**。写进 `review.html` 的独立 section，附缩略图 + 触发的具体数字 + reason |
| `ok` | 无曝光问题 | 正常参与后续质量排序 |

### 7.3 四条安全阀（缺一不可）

1. **组内保底**：一个 cluster 里若所有成员都是 `reject`，**强制保留 quality score 最高的一张**并降级为 `maybe`。宁可留一张烂的，不能整组消失。（这条已经在项目 P0 建议里，此处对齐）
2. **人脸优先**：人脸 ROI 曝光正常 → 上限只到 `maybe`（§5.2 L3）。打卡照的人脸就是主体，脸对了就不该判废。
3. **理由必须可读**：每个 reject/maybe 都带一句人能看懂的话（`"blown 48% of frame (largest blob 25%), anchor 9%"`），不能只有一个分数。这是可解释性要求，也是 §2.2 拒绝 IQA 模型的核心理由。
4. **reject 率熔断**：如果全库 `reject` 率 > 3%，**自动降级全部 reject 为 maybe 并告警**。66,000 张里超过 2,000 张判废，一定是阈值错了不是照片错了。

### 7.4 置信度（可选，给 review.html 排序用）

不要拍一个 0–1 的置信度分数（那就变回黑箱了）。用**"超出阈值多少倍"**这种可解释的量：

```python
severity = max(
    (ch - T.hi_reject) / max(1e-6, 1 - T.hi_reject),
    (cl - T.lo_reject) / max(1e-6, 1 - T.lo_reject),
    (T.mass_reject - mu) / T.mass_reject,
)
```
按 severity 降序排 review.html，人从最凶的开始看，看到不像废片就可以停。

---

## 8. 误判边界清单（实现前必读）

| # | 误判场景 | 会发生什么 | 防护 |
|---|---------|-----------|------|
| 1 | **剪影 / 逆光**（`aes_silhouette2` clip_lo=0.371） | 单看 clip_lo 会误杀 | anchor 判据 + `backlit` profile + 人脸豁免 |
| 2 | **夜景 / 演出**（`aes_stage_dark` entropy=3.30 最低） | 熵判据会误杀 | 熵不单独判 + `night` profile（entropy_reject 降到 2.2） |
| 3 | **High-key / 雪景 / 白墙**（`aes_snow` Ymean=0.576） | 高 Ymean + 潜在 clip_hi | anchor=0.996 兜住；blob_hi 判据 |
| 4 | **小面积高光**（金属反光、灯泡、太阳、水面波光） | 总面积可能不小但是分散的 | **blob_hi**（`aes_concert` clip_hi=0.0022 但 blob=0.0006）+ 512px 降采样天然衰减（§3.4） |
| 5 | **手机 HDR / 夜景模式合成图** | 已被相机拉平，指标"太好"看不出单帧缺陷 | **接受**。这不是我们的目标（D 类才是），不要试图检测 |
| 6 | **降采样吃掉小 clip** | 阈值在不同分辨率下不通用 | 阈值与 `decode_long_edge` 绑定，改则重校（§6.5） |
| 7 | **JPEG 块效应把暗部推到 0** | 高压缩低质量图 clip_lo 虚高 | LO 用 5/255 留余量；配合 entropy_keep（块效应会抬熵不会降熵） |
| 8 | **色彩饱和 ≠ 曝光 clip**（大红花、纯蓝天单通道 255） | 用"任一通道 clip"会大量误报 | **全通道 min/max**（§5.1 已处理） |
| 9 | **黑白 / 单色调照片** | 饱和度类指标失效 | 我们的判据全基于 luma，不用饱和度。安全 |
| 10 | **截图 / 表情包 / 文档照** | 纯色大块会触发 clip 判据 | 建议：`entropy_keep` 极低 + `blob` 极大 + 无人脸 → 单独标 `non-photo`，不进曝光判定 |
| 11 | **Motion Photo 的 .MP/.MP4 伙伴** | 单独判会跟主图不一致 | 沿用 `motion_partner_id` 生死绑定，不单独判 |
| 12 | **同一 cluster 全废** | 整组消失 | §7.3① 组内保底 |

---

## 9. 工程集成建议

### 9.1 性能

【实测】43 ms/图（单核，512px draft 解码 + 全部特征）：
- 单核跑 66,000 张 = **0.80 小时**
- 8 worker（`ProcessPoolExecutor`）= **~0.10 小时（6 分钟）**
- **纯 CPU，不占 GPU**，可以和 stage1 的 DINOv2/YuNet GPU 推理并行

时间构成：解码 15–35 ms（HDD 随机读是真瓶颈，不是 CPU），特征计算 ~8 ms。
→ **强烈建议合并进 stage1 的那一次顺序遍历**（DESIGN.md 硬约束 #3："所有涉及原图读取的操作必须一次顺序遍历完成"）。曝光特征在 stage1 里几乎免费，因为图已经解码在内存里了。**不要单独开一个 stage 再遍历一次 HDD。**

### 9.2 DB schema（建议加在 `features` 表）

存**原始特征**，不只存标签 —— 这样改阈值不用重跑全库（这是 §6.5 复校准能便宜的前提）：

```sql
ALTER TABLE features ADD COLUMN exp_clip_hi REAL;
ALTER TABLE features ADD COLUMN exp_clip_lo REAL;
ALTER TABLE features ADD COLUMN exp_blob_hi REAL;
ALTER TABLE features ADD COLUMN exp_blob_lo REAL;
ALTER TABLE features ADD COLUMN exp_mass_usable REAL;
ALTER TABLE features ADD COLUMN exp_anchor REAL;
ALTER TABLE features ADD COLUMN exp_entropy_keep REAL;
ALTER TABLE features ADD COLUMN exp_ymean REAL;
ALTER TABLE features ADD COLUMN exp_face_ymean REAL;      -- NULL = 无人脸
ALTER TABLE features ADD COLUMN exp_face_clip_lo REAL;
ALTER TABLE features ADD COLUMN exp_face_clip_hi REAL;
ALTER TABLE features ADD COLUMN exp_decode_long_edge INTEGER;  -- ★ 记录当时的解码分辨率
-- 标签是派生量，单独一列，可随时按新阈值重算
ALTER TABLE features ADD COLUMN exp_label TEXT;           -- ok|maybe|reject
ALTER TABLE features ADD COLUMN exp_reason TEXT;
```

`exp_decode_long_edge` 这一列很重要 —— 以后混入了不同分辨率跑出来的记录，能查出来。

### 9.3 复用已有组件（零新依赖）

| 需要 | 已有 | 备注 |
|------|------|------|
| 图像解码 | Pillow（已装 12.2.0） | 加 `.draft()` 调用即可 |
| 连通域 | opencv-python 5.0.0（已装，Apache 2.0） | `connectedComponentsWithStats` |
| 人脸 ROI | YuNet ONNX（stage1 已在跑，232 KB，MIT） | **只需把已检出的 bbox 传进来，零额外成本**。已实测 7 ms/图 |
| scene profile | DINOv2 embedding（stage1 已在算） | KMeans 分桶，§6.4② |

**requirements.txt 一行都不用加。** 这是这套方案最大的优点。

### 9.4 人脸 ROI 实测验证

【实测】YuNet（640px 输入，score_threshold=0.6）在测试集上：

| file | faces | face_Ymean | face_clip_lo | face_clip_hi | global_Ymean | ms |
|------|------:|-----------:|-------------:|-------------:|-------------:|---:|
| `aes_concert` | 1 | **0.586** | 0.000 | 0.053 | **0.189** | 7 |
| `aes_stage_dark` | 1 | **0.307** | 0.051 | 0.000 | **0.083** | 7 |

**两张暗调舞台照，全局亮度只有 0.19 / 0.08（看起来该判欠曝），但人脸 ROI 是 0.586 / 0.307 —— 曝光完全正常。** 这正是"审美曝光 vs 真实信息丢失"最典型的分界，人脸 ROI 一测就出来了。§5.2 的 L3 豁免有实测支撑。

（注：其余 8 张风景/建筑无人脸，YuNet 正确返回 0 张，无误检。运行时会打一条 `setPreferableTarget Targets are not supported by the new graph engine` 的 WARN，是 OpenCV 5.0 新推理引擎的已知噪音，不影响结果。）

---

## 10. 明确不推荐的东西（附理由，防以后返工）

| 不做 | 理由 |
|------|------|
| **pyiqa 任何 metric 做曝光判废** | PolyForm **Noncommercial** license（已在本机确认原文）+ 输出不可解释（综合分揉了曝光/锐度/噪声/构图） |
| **CLIP-IQA brightness prompt** | §4.1 实测证伪：夕阳剪影 0.998 > 严重过曝 0.900；互补 prompt 上同时说"好"；"故意暗 vs 意外毁"方向反了 |
| **Mertens E 做主判据** | §4.2 实测：夕阳剪影(0.039) 比严重过曝(0.056) 更低；在 ground-truth EV 上非单调 |
| **SPAQ MT-A** | license 未澄清（issue #37 无回复）+ 权重下载不稳（issue #9）+ 输出是审美 brightness 不是信息丢失 |
| **Q-Align / OneAlign** | 7B LMM，66k 张不现实（≥9 h GPU），且仍是不可解释综合分 |
| **Afifi Exposure Correction (CVPR'21)** | 需 MATLAB 2019b + Deep Learning Toolbox；且是矫正模型不是检测模型；数据集随 MIT-Adobe FiveK license（研究用途） |
| **BuIQA** | 无权重需自训；已有 `BUIQA_ASSESSMENT.md` 结论"不推"，本次一致 |
| **训练自己的曝光分类器** | 66,000 张无标注。要标够训练量（数千张）成本远超阈值校准（150 张）。**阈值法在这个问题上是正解，不是妥协** —— 因为判据本身有明确物理含义 |
| **自动曝光矫正** | 明确超出 P0 范围。我们只出标签 |
| **单独开一个 stage 遍历 HDD** | DESIGN.md 硬约束 #3。合并进 stage1 |

---

## 11. 落地检查清单

- [ ] 特征提取函数合并进 `src/stage1_features.py`（复用已解码的图 + 已检出的人脸 bbox）
- [ ] `src/quality.py` 或新建 `src/exposure.py` 放 `exposure_label()`（注意 AGENTS.md：单文件 <500 行，超了拆）
- [ ] DB schema 迁移（§9.2），含 `exp_decode_long_edge`
- [ ] config.yaml 加 `exposure:` 段（§5.3），**初始所有 `*_reject` 设 1.01（不可达）**
- [ ] 全库跑一遍只出特征，落库
- [ ] 分层抽样 150 张（§6.2）→ 人工标三档 → `research/exposure_calib.json`
- [ ] 网格搜索阈值，要求「留」样本 FP = 0 且桶 G 全不 reject（§6.3）
- [ ] 打开 reject，加 3% 熔断（§7.3④）
- [ ] `review.html` 加 maybe section，按 severity 排序，显示 reason + 缩略图
- [ ] 单元测试：合成 EV 阶梯（§3.3 的数字直接当 fixture，回归检测阈值漂移）
- [ ] 组内保底逻辑（§7.3①）

---

## 附录 A：复现脚本与原始数据

本文所有 `【实测】` 数字由以下脚本产生（在 `/tmp/expcal/`，本机 `photo-dedup/.venv` 运行；如需长期保留应移入 `research/scripts/`）：

| 脚本 | 作用 |
|------|------|
| `expmetrics.py` | 第一轮 20 个候选指标全量 dump |
| `probe2.py` | 第二轮可恢复性指标（shadow lift test、dark level entropy、center-weighted luma）+ 计时 |
| `ladder.py` | 合成 EV 阶梯（sRGB↔线性正确转换），−5…+5 EV |
| `clip_probe.py` | CLIP-IQA 4 组 antonym prompt 探针（§4.1 证伪） |
| `rule_v0.py` | 判定规则 v0 + 吞吐量测试 |
| `dump.py` | 特征表（§3.1） |

测试图（CC 授权，Wikimedia Commons，通过官方 API 取 URL）：
- 包围曝光四连（天然 EV ground truth）：`File:StLouisArchMultExpEV{+4.09,+1.51,-1.82,-4.72}.JPG`
- 审美极端曝光负样本：`Ned Point Lighthouse Sunset Backlit`、`Toona australis in silhouette against backlit rain`、`Château Frontenac at night`、`Kolomenskoe in white - Dec12 - 03 snow`、`Close Your Eyes Kpop dark stage performance live 2026 2`、`Darkness live`

环境：Python 3.12.3 / numpy 2.4.4 / Pillow 12.2.0 / opencv-python 5.0.0 / scikit-image 0.26.0 / torch 2.6.0+cu124 / pyiqa 0.1.16（仅用于 license 核查，未用于判定）。CLIP 探针在 CPU 上跑，1.61 s / 10 图。

---

## 附录 B：来源链接

**算法 / 论文**
- Mertens, Kautz, Van Reeth. *Exposure Fusion*. Pacific Graphics 2007 — well-exposedness 度量原始出处
- Hessel, C. *An Implementation of the Exposure Fusion Algorithm*. IPOL 2018 — σ=0.2 及 σ∈[0.1,0.2] 最优的实证：https://www.ipol.im/pub/art/2018/230/article_lr.pdf
- Fang, Zhu, Zeng, Ma, Wang. *Perceptual Quality Assessment of Smartphone Photography*. CVPR 2020 (SPAQ) — brightness attribute SRCC 0.784/0.704：https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf ｜ repo：https://github.com/h4nwei/SPAQ ｜ license 未澄清 issue：https://github.com/h4nwei/SPAQ/issues/37
- Wang et al. *Exploring CLIP for Assessing the Look and Feel of Images* (CLIP-IQA). AAAI 2023 — antonym prompt pairing：https://ojs.aaai.org/index.php/AAAI/article/view/25353/25125
- *Interpretable Image Quality Assessment via CLIP with Multiple Antonym-Prompt Pairs*. arXiv 2308.13094 — 多 prompt 对（含 light/dark）：https://arxiv.org/pdf/2308.13094
- *Fine-Grained Vision-Language Method with Prompt Tuning for BIQA*. MDPI Information 2026 — 明确指出 CLIP 缺乏 fine-grained 敏感性：https://www.mdpi.com/2078-2489/17/4/316
- Wu et al. *Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels*. ICML 2024：https://arxiv.org/html/2312.17090v1 ｜ repo：https://github.com/q-future/q-align
- Afifi, Derpanis, Ommer, Brown. *Learning Multi-Scale Photo Exposure Correction*. CVPR 2021：https://arxiv.org/abs/2003.11596 ｜ repo（MATLAB）：https://github.com/mahmoudnafifi/Exposure_Correction
- Huang et al. *Exposure Normalization and Compensation for Multiple-Exposure Correction*. CVPR 2022：https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Exposure_Normalization_and_Compensation_for_Multiple-Exposure_Correction_CVPR_2022_paper.pdf

**工程实现 / 工业实践**
- darktable *raw overexposed warning* 手册：https://docs.darktable.org/usermanual/development/en/module-reference/utility-modules/darkroom/raw-overexposed
- darktable issue #6596（clipping 检测在白平衡归一化后 `>1.0`）：https://github.com/darktable-org/darktable/issues/6596
- darktable issue #12193（默认阈值 1.0 只标边缘，需 0.99）：https://github.com/darktable-org/darktable/issues/12193
- RawPedia *Exposure*（Auto Levels `Clip %`，缩略图固定 0.2%）：https://rawpedia.rawtherapee.com/Exposure
- scikit-image `exposure` 模块（含 `is_low_contrast`）：https://scikit-image.org/docs/stable/api/skimage.exposure.html
- OpenCV `MergeMertens` 教程：https://learnopencv.com/exposure-fusion-using-opencv-cpp-python
- Exposure fusion 权重可视化（contrast/saturation/well-exposedness 逐步分解）：https://blog.minhazav.dev/research/exposure-fusion
- torchmetrics CLIP-IQA 内置 prompt 列表（含 `brightness: "Bright photo." / "Dark photo."`）：https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html
- IQA-PyTorch (pyiqa) repo：https://github.com/chaofengc/IQA-PyTorch （license: PolyForm Noncommercial 1.0.0，见 https://polyformproject.org/licenses/noncommercial/1.0.0）

**同类开源挑片项目（分档语言与 profile 设计参考）**
- `ncoevoet/facet`（182★，genre profiles + keeper budget + dry-run）：https://github.com/ncoevoet/facet
- `joneilcaoile/photocull-ai`（Keep/Maybe/Reject）：https://github.com/joneilcaoile/photocull-ai
- `a45674567/fixxer`（burst 分组 + exposure 打分 + XMP）
- GitHub topic 总览：https://github.com/topics/photo-culling

**项目内部**
- `research/BUIQA_ASSESSMENT.md` — BuIQA 评估（结论：不推），本文结论一致
- `DESIGN.md` 硬约束 #3（HDD 一次顺序遍历）、`config.yaml`（已有 `dry_run` 机制）
