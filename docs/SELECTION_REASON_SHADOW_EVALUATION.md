# 人类可读选片理由 shadow prototype 评估

日期：2026-09-29。范围：已检入 production-shaped synthetic/annotation smoke fixture；不是外部真实照片准确率。

- 生成器：`deterministic-selection-reason-v1`；纯 Python 确定性模板，模型调用 `0`。
- 证据引用完整性：20/20 = 100.0%（每条事实文本均引用存在的 evidence ID；不等同于真人偏好正确率）。
- 覆盖率：3/4 = 75.0%。
- 弃权：1/4 = 25.0%。
- 合成输入未被函数修改：`True`；前后 digest 均为 `81ef3bea261b20b22f037d78125413ab9be1efc7ffe013fc660b6f377cdc5bd1`。
- 粗略成本：32000 次生成共 1.142s，约 35.7 µs/组；不读图片、不调用模型。

## 证据边界

可陈述：落库 keeper、utility/pair margin、MUSIQ/缓存质量分、CLIP-IQA、人脸局部质量数值、分辨率、BYTE_IDENTICAL、逻辑阶段/姿态保护代码。不可陈述：更好表情、睁眼、无遮挡、主体完整、构图或叙事更佳；这些一律要求人工确认。

## 外部评测集

实现与 smoke 验证不依赖外部大集。冻结真实照片回放由 `scripts/evaluate_frozen_selection_reasons.py` 单独执行，结果见 `docs/SELECTION_REASON_FROZEN_REPLAY.md` 和 `docs/selection-reason-frozen-replay.json`。现有冻结标签为 `model_provisional`，因此真实回放只评估证据忠实度、覆盖率、弃权和潜在误导风险，不报告人工偏好一致率。

## 限制

- Checked-in production-shaped synthetic smoke cases; not a real-photo accuracy estimate.
- Human preference, expression, eye state, occlusion, composition and narrative remain unverified.
- Confidence is explanation sufficiency, not keeper correctness or deletion safety.
