# scripts/verification — 真机验证脚本（非 CI）

单元测试表达不了、必须用**真实权重 + 真 GPU** 才有意义的验证。不进 pytest，
需要时手动跑（L40S 或 Windows RTX 5070 Ti）。

| 脚本 | 验什么 |
|---|---|
| `verify_torch_batched.py` | batched IQA 路径与旧 per-image 路径逐图对比 + 数模型调用次数 |
| `e2e_torch.py` | 真 DINOv2/MUSIQ/CLIP-IQA/YuNet 过完整 `stage1_features.run`，serial vs prefetch 必须一致 |
| `adversarial.py` | torch 完全 import 不了（纯 CPU 用户）、batch 大于队列不能死锁、Ctrl+C 中断要提交已完成部分且不留线程 |
| `album_viewer_fixture.py` | 帮手：生一份合成审阅输出（4 组×3 张 + 2 张未分组 + 1 张缩略图失败，照片刻意大于视口以便验证缩放）并起本地服务器，供下面的浏览器验证使用 |
| `album_viewer_browser.py` | 真 Chromium 过审阅工作台：默认落在「未审」队列工作台、A/P/M 自动出队、U 撤销回到原组且回到当时屏幕上那张照片（A/M 也一样）、队列清空后的完成态、队列 tab 计数、列表密度切换与持久化、可折叠「推荐依据」、滚轮缩放/拖动平移/上限与边界/双栏同步、按住 C 闪切与 Shift+C 双栏、非 GROUPS 浏览列表、快速双击/按住键只发一次请求、无 AI 推荐时 A 禁用、跨站写入(form/text-plain)被 403、output 目录只有 review.html 可静态取、状态文件损坏时顶部警告横幅可见且不阻塞审阅（需另起 `album_viewer_fixture.py 18914 --corrupt-state`，不起则该段自动 SKIP），以及「缩略图栏只走 /api/thumb、原图只读当前显示的那张、不预取」的边界 |

跑法：`./.venv/bin/python scripts/verification/<script>.py`

审阅 UI 两步跑（第一步开在后台或另一个终端）：

```bash
./.venv/bin/python scripts/verification/album_viewer_fixture.py &   # 印 URL，Ctrl+C 退出自清
python3 scripts/verification/album_viewer_browser.py               # 默认打 127.0.0.1:18911
python3 scripts/verification/album_viewer_browser.py http://127.0.0.1:18911/review.html /tmp/shots  # 指定截图目录
```

`album_viewer_browser.py` 需要 Playwright 的 Chromium，并按步骤写截图（默认
`/tmp/review-workbench-shots`）。它验的是浏览器真实行为——键盘焦点、CSS grid 回流、滚轮手势后的
transform、实际请求了哪些 URL、以及一整轮审阅的队列迁移——这些 pytest 里的 DOM stub 表达不了；
全部 check 必须 PASS。前端渲染逻辑与安全边界由 `tests/test_review_workbench_ui.py` 在 CI 里守，
队列/分页/状态迁移/undo/scope 由 `tests/test_review_queues.py` 守。

一次性 probe 脚本（IQA batching 是否保分、CPU prefetch 加速比、VRAM vs batch size）
已删除，结论固化在 `docs/STAGE1_THROUGHPUT.md`。
