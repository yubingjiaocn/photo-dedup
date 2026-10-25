# 私有 VLM Review UI

## 打开 194 组真实建议

在仓库根目录运行（无需重新调用模型）：

```bash
python -m src.serve_review \
  --output output/private-vlm-review \
  --cloud-batch artifacts/cloud-judge-batch/authorized-scene-run-v3 \
  --cloud-manifest artifacts/cloud-judge-batch/frozen.json
```

`--no-open` 可禁止自动打开浏览器；`--port 8765` 可指定 loopback 端口。默认自动分配端口。打开终端打印的 `review.html?mode=cloud`。需要本机能够读取 manifest 所指的六份原始 inventory；manifest 内路径、DB SHA 和每组 `file_id + content_sha256` 都要匹配。缺失或不匹配时清晰拒绝启动，不修改源数据来迁就输入。

`output` 是独立的审阅工作目录，可以尚不存在，不需要复制 inventory。不能放在 cloud batch 或源 inventory 目录内。编译好的前端直接从仓库 `frontend/dist` 读取，无需运行 Node。修改 UI 后开发者运行 `cd frontend && npm run build`。

## 审阅方式

- **全部 / 待处理 / 已接受AI / 已修改 / AI失败/弃权**：顶部显示总组数、已处理数与剩余数。真实批次共 194 组、510 张照片；185 select、2 abstain、7 manual。
- **蓝色边框、AI 建议标签**是模型推荐；**绿色勾选**是你的待保存选择，两者不混淆。完整组照片始终显示，未勾选不代表删除。
- **接受AI建议**：保存模型的单选或多选 shortlist。
- **手动增删保留**：点击任意照片下方复选框，再点“保存我的选择”。即使 AI 只推荐一张，也能一键加选另一张，保护不同表情或动作阶段。至少保留一张。
- **全部保留**：立即保存全组多选。
- **稍后处理**：标记为稍后，仍计入待处理与剩余；不把跳过计为完成。
- **建议不准确 / 忽略此理由**：记录组级反馈或逐条理由反馈，不改变已经保存的保留选择。可恢复理由、撤回组级反馈。
- **上一组 / 下一组**：按当前筛选浏览；保存后留在当前组便于核对，在会移出当前筛选时显示下一条。可随时到“全部”重新调整已保存选择。
- 点击照片打开大图；点击背景、关闭按钮或 Esc 返回。

最多展示三个区分点。low-confidence 明确标为“弱观察（不支持推荐）”；长文字折叠但原文完整。解释可能有事实错误，结构校验通过不等于判断正确。失败组保留全部照片并显示“AI结果不可用，手工检查”，不展示未通过验证的 raw shortlist。

预览使用 batch 中经 SHA 校验、包含完整画面的每张 2048 长边 JPEG 页，而非重新读取原片；大图是同一审阅分辨率，不宣称原片 100% 像素检查。本轮未接线原片下载或执行操作。

## 保存与身份边界

唯一运行时写入文件是：

```text
<output>/review-state/cloud-review.json
```

采用临时文件、fsync 和原子替换。状态按完整 `file_id + SHA` 成员集合索引；日期组名、Fxx 仅作显示，不参与选片身份。数字 group_id 是当前 HTTP 进程的导航句柄，不作为持久身份。每次动作复核组成员身份及整个批次、inventory 的文件变化戳；启动时对输入做完整 SHA 校验。预览图片每次读取均重新核验 SHA。

重新打开页面或重启命令会恢复已保存选择与反馈。批次、身份或 sidecar 不匹配时拒绝旧状态并禁止覆盖，页面给出警告；使用新的 output 目录重新开始。不会静默丢弃旧记录。不要手动编辑 sidecar 来绕过不匹配检查。

源 inventory 以 SQLite `mode=ro` 打开。review 模式无 `execute_local`、移动、删除、keeper 或 decision 更新接口，旧 `/api/original` 也不开放。没有账号、计费或外部网络调用。成本估算仅在“批次摘要”显示，非实际账单。

## 验证与范围

- `tests/test_cloud_review.py`：194 组真实离线 fixture、510 张完整预览、accept/modify/keep-all/feedback/later、恢复、失败组、错误 SHA、旧状态拒绝、symlink 输出拒绝、raw 变化拒绝。动作前后比较六份 DB、全部 2065 源文件及整个 cloud batch 字节 SHA，证明只写独立 sidecar。
- `frontend/test/cloud-review.test.tsx`：单选改多选、AI 推荐与人工选择分离、反馈、忽略理由、失败/弃权、恢复。
- `scripts/qa_cloud_review_ui.py`：临时 loopback 测试服务器 + 真实 Chromium 的完整交互、重载恢复、G014 多选、失败/弃权及手机尺寸截图；结束关闭服务器，不安装服务。
- 本轮验收记录和截图：`artifacts/vlm-review-ui/`。Parent 独立验收与 Willy 人标仍需单独进行；本 UI 不提供自动清理。
