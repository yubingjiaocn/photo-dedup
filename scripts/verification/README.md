# scripts/verification — 真机验证脚本（非 CI）

单元测试表达不了、必须用**真实权重 + 真 GPU** 才有意义的验证。不进 pytest，
需要时手动跑（L40S 或 Windows RTX 5070 Ti）。

| 脚本 | 验什么 |
|---|---|
| `verify_torch_batched.py` | batched IQA 路径与旧 per-image 路径逐图对比 + 数模型调用次数 |
| `e2e_torch.py` | 真 DINOv2/MUSIQ/CLIP-IQA/YuNet 过完整 `stage1_features.run`，serial vs prefetch 必须一致 |
| `adversarial.py` | torch 完全 import 不了（纯 CPU 用户）、batch 大于队列不能死锁、Ctrl+C 中断要提交已完成部分且不留线程 |
| `album_viewer_fixture.py` | 帮手：生一份合成审阅输出（4 组×3 张 + 2 张未分组 + 1 张缩略图失败，照片刻意大于视口以便验证缩放），并**装入已提交的前端构建**（同 Stage 3），再起本地服务器 |
| `album_viewer_browser.py` | 兼容入口：转发到 `frontend/harness/browser_check.py`（见下） |

审阅 UI 现在是 `frontend/` 下的 Preact 应用（Vite 构建，`dist/` 已提交）。它的真机验证在
`frontend/harness/`，一条命令即可（自带进程内服务器，装入已提交构建并过真 Chromium）：

```bash
python3 frontend/harness/run_all.py
```

它对 **两种查看器**（生产用 Panzoom、`?viewer=osd` 用 OpenSeadragon）都过同一批门槛：
默认落在「未审」队列、居中适应窗口、滚轮/拖动/F/1、双栏同步、按住 C 闪切、精确的
`/api/original` 请求序列 `/2 → /1 → /5`、三次闪切零新增请求、切组释放旧组、A/P/M/U 队列流程、
一次写入护栏、重载恢复、跨站写 403、静态白名单，页面零报错。截图与请求日志写到
`frontend/artifacts/`。状态文件损坏横幅由 `album_viewer_fixture.py <port> --corrupt-state` 手动验。

前端逻辑（reducer/键盘/组件/图片池）由 Vitest 守（`frontend/test/`，`cd frontend && npm test`）；
构建产物与静态白名单由 `tests/test_review_frontend_build.py` 守；队列/分页/状态迁移/undo/scope
由 `tests/test_review_queues.py` 守。跑法：`./.venv/bin/python scripts/verification/<script>.py`。

一次性 probe 脚本（IQA batching 是否保分、CPU prefetch 加速比、VRAM vs batch size）
已删除，结论固化在 `docs/STAGE1_THROUGHPUT.md`。
