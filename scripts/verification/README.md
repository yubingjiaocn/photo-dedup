# scripts/verification — 真机验证脚本（非 CI）

单元测试表达不了、必须用**真实权重 + 真 GPU** 才有意义的验证。不进 pytest，
需要时手动跑（L40S 或 Windows RTX 5070 Ti）。

| 脚本 | 验什么 |
|---|---|
| `verify_torch_batched.py` | batched IQA 路径与旧 per-image 路径逐图对比 + 数模型调用次数 |
| `e2e_torch.py` | 真 DINOv2/MUSIQ/CLIP-IQA/YuNet 过完整 `stage1_features.run`，serial vs prefetch 必须一致 |
| `adversarial.py` | torch 完全 import 不了（纯 CPU 用户）、batch 大于队列不能死锁、Ctrl+C 中断要提交已完成部分且不留线程 |
| `album_viewer_fixture.py` | 帮手：生一份合成审阅输出（4 组×3 张 + 2 张未分组 + 1 张缩略图失败）并起本地服务器，供上面的浏览器验证使用 |
| `album_viewer_browser.py` | 真 Chromium 过 GROUPS 相册模式：放大后的响应式网格、单主图+缩略图栏、P/A/M/U/H/L/C/Esc 快捷键、动作后大图保持打开并进下一组、非 GROUPS 大图、以及「缩略图栏只走 /api/thumb、原图只在打开查看器时读」的边界 |

跑法：`./.venv/bin/python scripts/verification/<script>.py`

相册 UI 两步跑（第一步开在后台或另一个终端）：

```bash
./.venv/bin/python scripts/verification/album_viewer_fixture.py &   # 印 URL，Ctrl+C 退出自清
python3 scripts/verification/album_viewer_browser.py               # 默认打 127.0.0.1:18911
```

`album_viewer_browser.py` 需要 Playwright 的 Chromium。它验的是浏览器真实行为——键盘焦点、
CSS grid 回流、实际请求了哪些 URL——这些 pytest 里的 DOM stub 表达不了；全部 check 必须 PASS。
模板本身的渲染逻辑与安全边界由 `tests/test_review_album_ui.py` 在 CI 里守。

一次性 probe 脚本（IQA batching 是否保分、CPU prefetch 加速比、VRAM vs batch size）
已删除，结论固化在 `docs/STAGE1_THROUGHPUT.md`。
