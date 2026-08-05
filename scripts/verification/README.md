# scripts/verification — 真机验证脚本（非 CI）

单元测试表达不了、必须用**真实权重 + 真 GPU** 才有意义的验证。不进 pytest，
需要时手动跑（L40S 或 Windows RTX 5070 Ti）。

| 脚本 | 验什么 |
|---|---|
| `verify_torch_batched.py` | batched IQA 路径与旧 per-image 路径逐图对比 + 数模型调用次数 |
| `e2e_torch.py` | 真 DINOv2/MUSIQ/CLIP-IQA/YuNet 过完整 `stage1_features.run`，serial vs prefetch 必须一致 |
| `adversarial.py` | torch 完全 import 不了（纯 CPU 用户）、batch 大于队列不能死锁、Ctrl+C 中断要提交已完成部分且不留线程 |

跑法：`./.venv/bin/python scripts/verification/<script>.py`

一次性 probe 脚本（IQA batching 是否保分、CPU prefetch 加速比、VRAM vs batch size）
已删除，结论固化在 `docs/STAGE1_THROUGHPUT.md`。
