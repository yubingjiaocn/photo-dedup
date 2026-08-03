# P0 闭眼检测：技术核查与可编码方案

> 状态：实现建议（不是已校准的生产阈值）
> 核查日期：2026-08-03
> 目标平台：Windows 11 / Python，离线照片批处理
> 结论先行：**MediaPipe Face Landmarker + Face Blendshapes 作为主方案；OCEC 不作为无条件“二次确认器”，只在眼部像素充足且主模型落入灰区时充当独立证据。任何检测缺失、模型冲突或困难条件都输出 `UNKNOWN`，不能把未知当睁眼或闭眼。**

---

## 1. 决策摘要

### 1.1 推荐

1. 使用 MediaPipe Tasks `FaceLandmarker`，`running_mode=IMAGE`、`output_face_blendshapes=True`、`num_faces` 配成可接受的上限（建议默认 10，可配置）。
2. 对每张脸读取 `eyeBlinkLeft`、`eyeBlinkRight`，同时保存 `eyeSquintLeft/Right`、`mouthSmileLeft/Right`、landmarks 和检测覆盖信息；不要只保存最终布尔值。
3. 输出四态而非二态：`OPEN / CLOSED / MAYBE / UNKNOWN`；另存 `WINK_OR_ASYMMETRIC`、`SMILE_SQUINT`、`OCCLUDED_OR_GLASSES` 等 reason code。
4. `CLOSED` 仅是“需要复核/组内降权”的硬缺陷证据，不应单独触发永久删除。多人照只要任一重要人脸闭眼，可把整图标成 `HAS_CLOSED_EYES`，但必须保留逐脸结果。
5. OCEC 可选装。它最适合在 MediaPipe 中间置信区间、脸足够大且双眼 crop 可靠时提供补充证据；**不得**在 MediaPipe 未检测到脸、侧脸、墨镜或极小脸时把 OCEC 的强输出当作补救真值。
6. 上线前以用户真实照片做独立、按人物/连拍组隔离的校准。当前实测所用的大样本来自 OCEC 训练源数据集，不能用其近 100% 数字宣称泛化性能。

### 1.2 为什么主选 MediaPipe

- 一个 Apache-2.0 模型包同时完成多人脸检测、478 landmarks 和 52 个 blendshape；无需依赖 YuNet 的 5 点 landmark 猜眼框。
- `eyeBlinkLeft/Right` 是模型原生输出，而不是用稀疏关键点裁切后再分类。
- 本轮在多人拼图和缩小图上表现稳定；在同源 500 张测试中，缩到眼宽约 7 px 时 MediaPipe 仍明显优于 YuNet→OCEC。但后一个数据集有训练污染，只能说明管线退化趋势，不能作为泛化精度。
- 代价是 CPU 推理比 OCEC 单个眼 crop 慢，但实测整体吞吐仍足以处理 6.6 万张照片。

---

## 2. 官方来源、资产与 license 核查

### 2.1 MediaPipe 代码与模型

| 项目 | 核查结果 | 分发时应做什么 |
|---|---|---|
| MediaPipe 仓库/Tasks Python 代码 | 仓库根 `LICENSE` 及源码头均为 **Apache License 2.0** | 随分发包保留 Apache-2.0 license、版权/归属 notices；改过的文件标注修改。若上游包含 `NOTICE`，一并保留。 |
| Face Landmarker bundle | 官方页面提供 `face_landmarker.task`；本次下载大小 3,758,596 bytes，SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。包内含 `face_detector.tflite`、`face_landmarks_detector.tflite`、`face_blendshapes.tflite` 和 metadata。 | 固定版本/哈希，不要运行时默默跟随 `latest`；在 THIRD_PARTY_NOTICES 中列模型名、URL、下载日期、Apache-2.0。 |
| FaceMesh V2 model card | 明示 **Licensed under Apache License, Version 2.0** | 同上。 |
| Blendshape V2 model card | 明示 **Licensed under Apache License, Version 2.0** | 同上。 |
| Google 文档和示例 | 页面正文通常 CC BY 4.0，代码示例 Apache-2.0；Google Site Policies 明确外链、图片/影音未必自动被页面 license 覆盖 | 不要把“文档页面 license”误当模型 license；本方案的模型 license 依据是模型卡本身。 |

官方模型包：
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`

官方 Face Landmarker 页面：
`https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker`

官方模型卡：

- `https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf`
- `https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf`

MediaPipe license：
`https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE`

**模型卡的重要边界：** Blendshape 模型的原始用途是前置手机相机的实时 AR 表情系数，不是照片质检产品；输入是 FaceMesh 产生的 146 个 landmarks，输出 52 个 `[0,1]` 系数。模型卡列出 `eyeBlinkLeft`、`eyeBlinkRight`，同时指出看离相机超过约 80°、脸可见不足 50%、距离太远而放大造成质量损失、低光/噪声/运动/脸重叠都会降低 landmarks 与 blendshape 的准确度。FaceMesh 模型卡也强调 selfie/单脸居中假设及位置、尺度、方向敏感性。因此必须设计 `UNKNOWN`，不能把每个浮点值都强制二分。

### 2.2 OCEC 代码、ONNX 与训练数据

| 项目 | 核查结果 | 风险/动作 |
|---|---|---|
| PINTO0309/OCEC 仓库 | 根 `LICENSE` 为 **MIT**，Copyright 2025 Katsuya Hyodo | 若分发代码或 ONNX，附 MIT 全文及 copyright。 |
| 官方 ONNX release | `ocec_p/n/s/c/m/l.onnx` 位于同仓库 GitHub Release，README 将系列标为 MIT。模型输入实查为 `images: float32 [batch,3,24,40]`，输出 `prob_open: float32 [batch]`。 | 固定具体 variant 和 SHA-256，不要只依赖可变 URL。推荐试验用 `ocec_s`，而非默认追求 `l`；本轮 `s` 对小 crop 反而更稳。 |
| 训练数据 | README 致谢 `MichalMlodawski/closed-open-eyes`，该数据集页标 **ODC-By 1.0**；数据集说明约 126,560 张、balanced、AI-generated，并提供精确左右眼框。 | 模型仓库 MIT 与训练数据库 ODC-By 是两层事实，不能把 MIT 简化成“所有来源内容均无条件 MIT”。保留数据集 attribution；商业发布前做一次正式 third-party review。 |
| ODC-By 范围 | ODC-By 主要许可数据库权利，并明确不当然覆盖数据库内每张图像的独立 copyright、隐私/人格等权利；公开传播数据库/衍生数据库及公开使用 Produced Work 有 attribution 条款。 | 本项目只消费发布的模型权重，不再分发训练数据库；仍建议在模型 notices 中注明训练数据来源和 ODC-By。不要把训练图片打包进产品。 |

OCEC：`https://github.com/PINTO0309/OCEC`
OCEC MIT：`https://github.com/PINTO0309/OCEC/blob/main/LICENSE`
训练数据：`https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes`
ODC-By 1.0：`https://opendatacommons.org/licenses/by/1-0/`

本次核对的 release 哈希：

- `ocec_p.onnx`: `28fdb1b2782d837a998203d0264f076a00dc501f0bc2318315251a2915dd483c`
- `ocec_s.onnx`: `9a346a08b256ad70725044cd2aa582858e108c6f45d42a9c3415afc604ba9b64`
- `ocec_l.onnx`: `de9b8031f8b521a862d8cff55ba88c2fccab6ac96484ba53154dd12c53c7c7f9`

> 这不是法律意见。上述结论足以指导工程选型和 notices，但产品对外分发前仍应由有资质人员确认训练权重是否被视为 ODC-By 的 Produced Work，以及发行包内第三方 notice 的最终形式。

---

## 3. 已有实测：能证明什么，不能证明什么

### 3.1 多人脸拼图（可作为功能 smoke test）

用 2 张睁眼、2 张闭眼组成 2×2 拼图：

| 配置 | 图像 | 每脸约宽 | 检出 | 结果 |
|---|---:|---:|---:|---|
| `num_faces=4` | 1024×1024 | 约 512 px tile | 4/4 | 闭眼 blink `0.75–0.76`；睁眼 `0.01–0.02`，4/4 正确 |
| `num_faces=4` | 358×358 | 约 179 px tile | 4/4 | 闭眼 `0.70–0.72`；睁眼 `0.01–0.02`，4/4 正确 |
| `num_faces=1` | 同图 | 同上 | 仅 1 张脸 | 证明默认 `num_faces=1` 会漏多人照，不可沿用默认值 |

这是小样本 smoke test，不是准确率评测。

### 3.2 500 张同源数据：仅用于消融，不得当泛化 benchmark

样本来自 `MichalMlodawski/closed-open-eyes`，而 OCEC README 明确该数据是其训练来源；标签和精确眼框也来自数据集。即便测试文件未必逐张进入某个 checkpoint 的 train split，也无法证明 subject/image 去重，存在严重污染和同分布偏乐观。

关键观察：

- 原尺度，MediaPipe 眼框中位宽约 52 px：MediaPipe→OCEC-S、blendshape、EAR 都接近 99–100%。
- 缩放到 0.25，眼框中位宽约 13 px：MediaPipe blendshape balanced accuracy `0.994`；精确 GT 眼框→OCEC-S `0.992`；YuNet 5 点推导眼框→OCEC-S 仅约 `0.898`。
- 缩放到 0.12，眼框中位宽约 7 px：MediaPipe blendshape约 `0.978`；GT 眼框→OCEC-S 约 `0.872`；YuNet→OCEC-S 约 `0.735`。
- 使用 MediaPipe 精细 landmarks 裁眼给 OCEC，在 0.25 缩放时约 `0.996`，但到约 7 px 眼宽时 coverage 仅 `0.594`、balanced accuracy 约 `0.891`。

**可采信的工程结论：** OCEC 对 crop 几何和眼部实际像素极敏感；用 YuNet 五点猜 crop 会显著损失；极小脸时不应强判。
**不可采信的产品结论：** “准确率 99%”“阈值 0.4 已校准”或“优于所有真实照片场景”。

### 3.3 少量 Wikimedia 困难照片

在可成功下载的闭眼/眨眼照片中，MediaPipe 对多张真实闭眼照片给出约 `0.31–0.81` 的最大 blink，且部分照片完全未检到脸；一张大笑眯眼照片同时出现 blink≈`0.51`、squint≈`0.62`、smile≈`0.58`。这说明：

- 单阈值会漏掉低分闭眼或误伤大笑眯眼；
- 无检测必须是 `UNKNOWN`；
- smile/squint 只能触发豁免/复核，不能简单从 blink 中做算术扣分。

样本太少且类别来自网页分类，不足以报告准确率。

### 3.4 侧脸探针

对 60 张睁眼脸做水平压缩模拟 foreshortening：在仍能检出的样本中没有出现 blink 假阳性，但检出率从原图 `1.00` 降到约 46° proxy 的 `0.95`、60° 的 `0.65`、70° 的 `0.05`、76° 的 `0`。这不是物理真实 yaw 测试，只能支持“侧脸首先表现为 coverage 崩溃，应 abstain”，不能证明侧脸误报率为零。

---

## 4. 可编码接口

### 4.1 输入

```python
@dataclass
class EyeAnalysisInput:
    image_rgb: np.ndarray       # uint8, H×W×3, 已按 EXIF 方向转正
    image_id: str
    max_faces: int = 10
```

预处理要求：

- 解码后统一 RGB、应用 EXIF orientation；不要先把长边暴力缩到很小。
- 可为速度将长边缩至 1600–2048 px，但只有在预计最小人脸仍满足像素门槛时；必要时对已知 YuNet face bbox 分 tile 再跑。
- 保留从推理图坐标映射回原图坐标的 scale。

MediaPipe options：

```python
FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task",
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=max_faces,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)
```

Windows Python 上明确使用 CPU：MediaPipe `BaseOptions` 官方源码写明 Python GPU support 当前限 Ubuntu；Windows 请求 GPU delegate 不是本方案支持路径。现有 NVIDIA GPU 不会因此自动加速 Face Landmarker。

### 4.2 逐脸输出

```python
class EyeState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    MAYBE = "MAYBE"
    UNKNOWN = "UNKNOWN"

@dataclass
class FaceEyeResult:
    face_index: int
    bbox_xywh: tuple[float, float, float, float]
    face_width_px: float
    iod_px: float                 # 双眼中心距离
    blink_left: float | None
    blink_right: float | None
    squint_left: float | None
    squint_right: float | None
    smile_left: float | None
    smile_right: float | None
    state: EyeState
    confidence: float | None      # 本地校准后才可称概率；首版仅 rank score
    reasons: list[str]
    ocec_prob_open_left: float | None = None
    ocec_prob_open_right: float | None = None
    model_version: str = "mediapipe-face-landmarker@sha256:64184e..."
```

图级输出：

```python
@dataclass
class ImageEyeResult:
    status: Literal[
        "NO_FACE", "ALL_OPEN", "HAS_CLOSED_EYES", "HAS_MAYBE", "UNKNOWN"
    ]
    faces: list[FaceEyeResult]
    detected_face_count: int
    detector_count_disagreement: bool
```

`NO_FACE` 表示没有检测到脸，不等同 `ALL_OPEN`。如果上游 YuNet 检出 N 张而 MediaPipe 只返回 M<N，图级必须至少 `UNKNOWN`，不能忽略漏掉的人。

---

## 5. 首版规则（shadow mode 的保守起点）

以下阈值是**待校准的初值**，不是官方阈值，也不是由无污染 benchmark 得出：

```python
OPEN_MAX = 0.15
CLOSED_MIN = 0.50
ASYMMETRY_MAX = 0.25
MIN_IOD_PX = 24
MIN_FACE_PX = 96
CAUTION_FACE_PX = 160
SMILE_HIGH = 0.45
SQUINT_HIGH = 0.45
```

逐脸逻辑：

```text
1. 缺少 landmarks / blendshapes                       -> UNKNOWN
2. face短边 < 96 或 IOD < 24 px                        -> UNKNOWN: SMALL_FACE
3. 严重侧脸、脸可见不足、眼部遮挡、检测器数量冲突        -> UNKNOWN
4. 两眼 blink <= 0.15                                  -> OPEN
5. 两眼 blink >= 0.50 且差值 <= 0.25：
     a. 若大笑/高 squint                                -> MAYBE: SMILE_SQUINT
     b. 否则                                            -> CLOSED
6. 只有一眼 >= 0.50 或左右差值 > 0.25                  -> MAYBE: WINK_OR_ASYMMETRIC
7. 其余灰区                                             -> MAYBE
```

为什么闭眼要求“双眼”：它可显著降低 wink、局部遮挡和侧脸远侧眼的误伤。若产品明确要把单眼眨眼也视为坏片，应单独把 `WINK_OR_ASYMMETRIC` 用于组内降权，不要并入高置信 `CLOSED`。

像素门槛是保守工程门，需用真实图库重标定。官方模型卡没有给一个通用最小 face px；它只明确“距离太远、放大到模型输入导致质量损失”会退化。`96/160` 是为了让首版宁可 `UNKNOWN`，不要在极小脸上伪造确定性。

### 5.1 姿态门

优先从 facial transformation matrix 或 landmarks 估计 yaw/pitch/roll：

- `abs(yaw) > 45°`：`UNKNOWN: PROFILE_FACE`；
- `30° < abs(yaw) <= 45°`：最多 `MAYBE`，除非本地侧脸集验证后放宽；
- roll 可先旋正脸 crop，但不要把旋正后的高分当成姿态不存在；
- 若估姿失败，依据左右眼 IOD、脸框和 landmark 可见性设置 `UNKNOWN`。

45° 是产品安全门，不是 MediaPipe 官方极限。模型卡提到约 80° 的失效边界，不代表 79° 仍适合照片自动质检。

### 5.2 大笑眯眼

- `smile=max(mouthSmileLeft, mouthSmileRight) >= 0.45` 且 `squint>=0.45` 时，不自动判硬缺陷；输出 `MAYBE: SMILE_SQUINT`。
- 不要写 `blink - smile` 之类未经校准的公式；blink、squint、smile 不是互斥概率。
- 在同一相似组有相同人物、相近笑容的睁眼替代图时，可把该图组内降权；若这是唯一的大笑高峰图，默认 KEEP/MAYBE。

### 5.3 墨镜、反光镜片、头发/手遮挡

Face Landmarker 没有可靠的“墨镜存在”输出，**高 blink 不代表透过墨镜看到了眼皮**。

首版处理：

- 若项目已有或新增 sunglasses/occlusion classifier，命中后直接 `UNKNOWN: SUNGLASSES_OR_OCCLUSION`；
- 在没有遮挡模型前，利用眼区亮度/纹理只能作为 reason，不应强行推断 OPEN/CLOSED；
- 普通透明眼镜可继续推理，但强反光、深色镜片、眼部 landmark 几何异常时 abstain；
- OCEC 也不能绕过墨镜门。

### 5.4 小脸

- 不要把低分当 OPEN；无法解析眼皮时是 `UNKNOWN`。
- 先保留高分辨率原图或按人脸 tile 推理，而不是把整张多人照统一缩到 512 px。
- OCEC 至少要求两个眼 crop 都有效且裁切前宽建议 `>=16 px`；`<12 px` 禁用。该门来自本轮退化趋势，需真实数据校准。

### 5.5 多人脸

- `num_faces` 必须大于 1；默认建议 10，可按业务提高至 20，但需测吞吐和极端合影覆盖。
- 保存每张脸状态。图级规则：
  - 任一可靠 `CLOSED` → `HAS_CLOSED_EYES`；
  - 无 `CLOSED` 但任一 `MAYBE/UNKNOWN` → `HAS_MAYBE` 或 `UNKNOWN`；
  - 所有检出脸均可靠 `OPEN` 且检测器计数一致 → `ALL_OPEN`。
- 重要人物与背景路人不能仅靠脸面积武断决定。首版可把最大 1–3 张脸标为 `primary_face_candidate`，但任何自动删除仍须遵守组内保底。
- 当人脸数达到 `max_faces` 上限，结果可能被截断，图级标 `UNKNOWN: FACE_LIMIT_REACHED`。

---

## 6. OCEC 的正确角色

### 6.1 不推荐：全量 AND/OR 二次验证

将 MediaPipe 与 OCEC 对所有脸做简单 AND/OR 有三个问题：

1. OCEC 需要可靠 eye crop；YuNet 只有双眼中心等 5 点，实测 crop 偏差把约 99% 的同源表现拉到约 90%，小脸更低。
2. 两者并不真正独立：若用 MediaPipe landmarks 裁 OCEC，landmark 失败会同时污染两条证据。
3. OCEC 的公开高 F1 与本轮近满分都处在其训练数据分布，不能用来校准真实图库阈值。

### 6.2 推荐：灰区证据/冲突检测

只在以下条件全部满足时运行 `ocec_s`：

- MediaPipe 已检出脸且 landmarks 质量门通过；
- 正面或轻微姿态（建议 `abs(yaw)<=30°`）；
- 非墨镜/遮挡；
- 两个 eye crop 裁切前宽均 `>=16 px`；
- MediaPipe 落在 `MAYBE` 灰区，或需要记录模型分歧供人工校准。

用 MediaPipe 眼角 landmarks 33/133 与 263/362 计算中心和宽度，按 OCEC 40:24 比例扩 crop（本轮试验水平宽约眼角距离的 1.35 倍），旋正双眼线后 resize 到 40×24。输入 RGB/BGR 顺序必须通过固定 golden test 锁定；本轮脚本使用 OpenCV BGR，与官方训练代码的实际预处理应在实现时再次逐行核对，不能靠猜。

建议融合：

```text
MP=CLOSED 且 OCEC双眼均强closed -> 保持 CLOSED，evidence += OCEC_AGREE
MP=MAYBE  且 OCEC双眼均强closed -> 仍为 MAYBE，提升 review priority
MP=MAYBE  且 OCEC双眼均强open   -> MAYBE: MODEL_DISAGREEMENT
MP与OCEC冲突 / 单眼冲突          -> MAYBE 或 UNKNOWN，绝不由 OCEC 覆盖 MP
任何质量门失败                    -> 不运行 OCEC
```

OCEC 输出是 `prob_open`，但在用户真实域校准前应称 `score_open`，不要在 UI 中显示“95% 概率”。

---

## 7. 依赖与部署

### 7.1 主方案

```text
mediapipe              # Python Tasks API；锁定实际验证版本
numpy
Pillow / OpenCV        # 解码、EXIF、颜色转换与可选 crop
```

模型资产：`face_landmarker.task`（约 3.6 MiB）。建议 vendor 到受控 model cache，启动时校验 SHA-256；断网也能运行。

### 7.2 可选 OCEC

```text
onnxruntime            # Windows CPUExecutionProvider 即可
opencv-python
numpy
```

`ocec_s.onnx` 约 495 KB。若现有项目已经安装 `opencv-python`，新增主要是 `mediapipe` 和可选 `onnxruntime`。应先在目标 Python 3.11/Windows wheel 环境做安装 smoke test并锁版本；不要把本次 Linux venv 的包版本直接视为 Windows 已验证版本。

### 7.3 线程安全

不要假设同一个 `FaceLandmarker` 实例可被多个 Python worker 并发调用。实现上每个 worker/thread 创建自己的实例，或用受控对象池；批处理启动时预热。多进程会重复占模型内存，先测 4 个 worker，再决定是否增到 8。

---

## 8. 吞吐：实测与估计必须分开

### 8.1 本次实测（Linux CPU，不是目标 Windows 主机）

环境：8 vCPU（AMD EPYC 7R13，4 cores / 8 threads，KVM），MediaPipe CPU/XNNPACK，输入 1600×1600，24 张样本；每 worker 独立 landmarker。

| worker threads | 实测吞吐 | 6.6 万张纯模型时间投影 |
|---:|---:|---:|
| 1 | 58.0 img/s | 0.32 h |
| 4 | 104.2 img/s | 0.18 h |
| 8 | 98.6 img/s | 0.19 h |

另一次 500 张循环实测 MediaPipe 约 `11–13 ms/image`；该测试图尺寸/缓存路径不同，不能与 1600×1600 并发 benchmark 直接等价。4 threads 优于 8，说明过度并发会争抢 XNNPACK/CPU。

OCEC README 的 `0.16–0.80 ms` 是作者报告的不同 variant CPU **单次模型 inference latency**，不含人脸检测、eye crop、解码和 I/O；不能当整图吞吐。本轮在 500 张测试中，YuNet约 `1.2–1.5 ms/img`，三个 OCEC variants 合计约 `3.8–4.2 ms/img`，同样不含完整生产 I/O。

### 8.2 目标机估计（尚未实测）

- Windows CPU 的 Face Landmarker 预计数量级约 **40–100 img/s**（4 个受控 workers、长边约 1600 px、普通单/少人照片）；这是从上述 Linux 实测外推，不是承诺。
- 6.6 万张纯闭眼分析约 **11–28 分钟**；加入磁盘读取、JPEG/HEIC 解码、EXIF、数据库写入、多人 tile 复跑后，保守规划 **20–60 分钟**。
- 多人脸上限、超高分辨率、网络盘/机械盘和 HEIC 解码可明显降低吞吐。
- 目标 RTX 5070 Ti 对 Windows Python MediaPipe 此路径没有可用 GPU delegate 加速；不要把 GPU 算力纳入上述估计。OCEC 虽可用 ONNX CUDA，但模型极小，传输/调度可能抵消收益，CPU 足够。

发布性能数字前必须在目标 Windows 机上用真实图库随机抽样至少 1,000 张，分别记录 decode、Face Landmarker、可选 OCEC、DB write 的 p50/p95 与总 wall time。

---

## 9. 验证与上线门槛

### 9.1 建立无污染验证集

至少 1,000–2,000 张真实照片、逐脸标注，按人物或连拍事件 group split，覆盖：

- 正面睁/闭、单眼眨眼；
- 大笑眯眼、自然小眼；
- 30/45/60° 侧脸、仰俯、roll；
- 透明眼镜、反光镜片、墨镜；
- 遮挡、低光、运动模糊；
- 96/160/256 px 等脸尺寸桶；
- 1、2–5、6–10、10+ 人合影；
- 不同年龄、肤色和拍摄设备。

标注至少区分 `OPEN / CLOSED / WINK / SMILE_SQUINT / OCCLUDED / UNJUDGEABLE`。评测时把 `UNKNOWN` 当 coverage，不要强塞进错误类别。

### 9.2 指标

以“误把好照片标闭眼”为高成本错误：

- `CLOSED precision`（首要）；
- closed recall；
- risk–coverage curve；
- `UNKNOWN/MAYBE` rate；
- 每尺寸、姿态、眼镜、多人桶的分层指标；
- face detection recall 与 detector count disagreement；
- 图级“任一人闭眼”precision/recall。

建议自动硬标签门槛：真实独立集上 `CLOSED precision >= 99%`，并报告置信下界；达不到就只作为 review tag/组内 soft penalty。首版至少 shadow mode 跑完整图库，人工抽查全部高置信 closed + 分层抽样 open/unknown 后再开启任何自动动作。

### 9.3 固定回归样本

加入 golden tests：

1. 4 人拼图原图与 358×358 缩图均检出 4 人，闭/开排序不变；
2. `num_faces=1` 的负向测试，防止配置回退；
3. RGB/BGR 对照，防 OCEC 颜色通道悄然改变；
4. 小脸/侧脸/墨镜必须 `UNKNOWN`；
5. 大笑眯眼必须 `MAYBE` 而非硬 `CLOSED`；
6. 任一模型异常、资产哈希不符、结果列表长度不一致都 fail closed to `UNKNOWN`，不能默认为 OPEN。

---

## 10. 最终实施建议

### P0（现在做）

- 集成 MediaPipe CPU 主路径；保留原始逐脸 blendshape、尺寸、姿态、reason 与模型哈希。
- 按四态和安全门实现，所有异常/漏检走 `UNKNOWN`。
- 对闭眼只做标签、组内降权和人工 review，不单独自动删除。
- 用 4 人拼图与困难样本建 golden tests。

### P0.5（校准后）

- 建真实独立验证集，确定 `OPEN_MAX/CLOSED_MIN`、小脸和姿态门；按风险覆盖曲线选阈值。
- 评估 sunglasses/occlusion detector；未完成前墨镜直接 unknown。

### P1（可选）

- 加 `ocec_s` 作为灰区证据与 disagreement logger；不要全量双跑，也不要允许它覆盖质量门。
- 只有当独立真实集证明融合显著提高 `CLOSED precision` 或降低 `MAYBE` 且不增加误伤，才把它升级为生产二级验证。

**最终答案：主方案是 MediaPipe。OCEC 值得保留为实验性二级证据，但当前不应被称为可靠的“二次确认器”；真正的安全性来自质量门、困难场景豁免、`UNKNOWN` 弃权、多人逐脸聚合，以及真实无污染数据上的本地校准。**
