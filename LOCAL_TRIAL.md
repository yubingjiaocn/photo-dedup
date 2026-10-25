# 本地试用（Windows，小文件夹）

只审阅、不动原图：程序只**读取**照片，从不移动、删除、改写原图或其元数据，也不会执行删除清单。所有结果（数据库、缩略图、审阅记录、导出）只写在你指定的 `-Output` 文件夹里。默认不上传任何东西、不调用任何模型接口、不训练模型；只有你显式加 `-Vlm` / `-Judge` 时，才会把少量分组的缩小图发给**你自己配置的** VLM 接口（见第 5 节）。

## 1. 获取与安装（一次）

直接从 GitHub 拉取（`master` 即可试用；前端已预构建在 `frontend/dist`，不需要 Node）：

```powershell
git clone https://github.com/yubingjiaocn/photo-dedup.git
cd photo-dedup
.\setup_windows.ps1
```

以后更新：在该文件夹 `git pull`；`requirements.txt` 有变化时再跑一次 `.\setup_windows.ps1`。

`setup_windows.ps1` 会创建 `.venv`，安装 PyTorch CUDA 12.8 和依赖，并检查显卡（CUDA 不可用会直接报错）。PowerShell 拒绝执行脚本时，改用同目录的 `setup_windows.bat` / `start-local-review.bat`（参数相同）。

仓库里的 `-Output` 目录、照片、数据库和评估产物都不会进 git；`-Output` 请放在仓库文件夹之外。

## 2. 开始审阅一个新日期文件夹

```powershell
.\start-local-review.ps1 -Root "E:\照片\2026-10-01" -Output "D:\photo-review\2026-10-01"
```

- 默认最多处理 100 张；要改用 `-Limit 300`。
- 路径有空格/中文时照上面加英文双引号即可。
- `-Output` 不能放在照片文件夹里，也不能放在本程序文件夹里；一个 Output 只对应一个 Root（换日期就换一个新 Output）。
- 跑完会自动打开浏览器。审阅按键：`A` 保留 CV（算法）推荐 · `P` 只保留当前一张 · `X` 勾选/取消当前张，`S` 保留所有已勾选（多张） · `G` 全部保留 · `M` 稍后处理 · `U` 撤销。每一步立即保存。
- 关闭：在 PowerShell 窗口按 `Ctrl+C`。已保存的选择不会丢失。

## 3. 下次继续（不重新扫描）

```powershell
.\start-local-review.ps1 -Output "D:\photo-review\2026-10-01" -Reopen
```

（用同样的 `-Root -Output` 再次运行也会直接打开已完成的审阅；确实要重扫请加 `-Rescan`。）

## 4. 导出选择记录（供以后学习用）

```powershell
.\start-local-review.ps1 -Output "D:\photo-review\2026-10-01" -Export
```

生成 `D:\photo-review\2026-10-01\local_trial_export\selections-*.jsonl` 和 `.manifest.json`：每组每张照片一行，记录"选中/未选中"、文件 SHA-256、大小、时间、文件名；**不含照片、不含完整路径**。如果某组照片后来变了（成员或 SHA 不一致），会列为 stale 并跳过。导出只是本地数据文件，目前没有自动学习模型，也不会并入已有的冻结学习数据集。

## 5. 可选：VLM 建议（OpenAI 兼容接口，例如你端口转发过来的 LiteLLM）

先让扫描正常跑完；VLM 只是额外的参考建议，失败也不影响你人工审阅。

地址和模型也可以用环境变量配置，在同一个 PowerShell 窗口启动：

```powershell
$env:PHOTO_VLM_BASE_URL = "http://127.0.0.1:4000/v1"
$env:PHOTO_VLM_MODEL = "<你的支持图片的模型别名>"
# 网关需要 key 时再设置；无认证网关可以不设置：
$env:PHOTO_VLM_API_KEY = "<你的 key>"
.\start-local-review.ps1 -Root "E:\照片\2026-10-01" -Output "D:\photo-review\2026-10-01" -Vlm -VlmMaxGroups 2
```

下面是通过启动参数覆盖地址和模型的例子：

```powershell
# 如果你的网关需要 key，只通过环境变量提供（不会写进任何文件或日志）：
$env:PHOTO_VLM_API_KEY = "<你的 key>"
# 新日期：先扫描，再请求 VLM，最后打开审阅
.\start-local-review.ps1 -Root "E:\照片\2026-10-01" -Output "D:\photo-review\2026-10-01" -Vlm `
    -VlmBaseUrl "http://127.0.0.1:4000/v1" -VlmModel "<你的模型别名>"
# 已扫描过的目录：只补请求 VLM（不扫描），完成后退出，再用 -Reopen 打开
.\start-local-review.ps1 -Output "D:\photo-review\2026-10-01" -Judge `
    -VlmBaseUrl "http://127.0.0.1:4000/v1" -VlmModel "<你的模型别名>"
```

- `http://127.0.0.1:4000/v1` 和模型别名只是**占位示例**：请换成你实际转发的端口和网关里配置的模型名（任意别名都可以）。本程序没有内置任何接口地址或 key。
- 只有本机地址（127.0.0.1/localhost）允许用 `http://`；远程地址必须用 `https://`（会校验证书）。地址里不要带用户名、密码或 query 参数；key 只放环境变量。
- 每次最多请求 `-VlmMaxGroups` 组（默认 8，最多 20），每组只发一次请求，不自动重试、不换模型。只请求 2–20 张的组；超过 20 张的组显示"请人工审阅"，完全相同的文件保留 CV 结果。
- 发送内容：每张照片重新压缩成不含 EXIF/GPS 的整图小 JPEG（标签 P1、P2…），**不含文件名和路径**。原图只读，不会被修改。
- 结果保存在 `-Output\local_vlm\`。再次 `-Reopen` 只显示已保存的建议，**不会再调用模型**；同样参数再跑 `-Judge` 会直接用缓存（要强制重新请求加 `-VlmForce`）。连接失败/超时/认证失败的组下次 `-Judge` 会重新请求。
- 网关不支持 `response_format` 时加 `-VlmNoResponseFormat`（不会自动降级重试）。超时用 `-VlmTimeout 180` 调整。
- 审阅页面里，VLM 建议单独显示在"VLM 建议（仅供参考）"框里，相应照片带 `VLM建议` 标记；只有点 **接受VLM建议** 才会按"保留多张"保存，可再用 `U` 撤销。模型回复格式不对、引用了不存在的照片、接口出错时会明确显示失败，你照常用 `X/S/G/P` 人工决定。
- 格式校验通过只代表回复结构正确，不代表建议一定准确。
- 常见报错：`连不上 VLM 接口` → 端口转发或 LiteLLM 没启动；`认证失败` → 检查 `PHOTO_VLM_API_KEY`；`拒绝请求` → 模型别名不存在或不支持图片。

## 6. GPU 真实结果 vs 冒烟测试

- 默认使用 torch 真实后端（GPU），分组和画质提示来自真实模型。
- `-SmokeStub` 只是冒烟测试（不用模型），分组/画质提示**不代表真实效果**，仅用于确认程序能跑。

## 7. 出错时请发送

1. PowerShell 窗口里的全部文字（截图或复制）；
2. `D:\photo-review\2026-10-01\photo-dedup.log`；
3. `.\setup_windows.ps1` 的输出（如果是安装失败）；
4. 用了 VLM 时再加 `D:\photo-review\2026-10-01\local_vlm\last_run.json`（不含 key 和照片）。

不要发送照片本身。
