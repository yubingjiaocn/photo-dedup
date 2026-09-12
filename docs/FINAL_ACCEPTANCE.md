# Instance recovery：有界最终验收

2026-09-12。**本机工程收尾完成；算法默认启用／产品全面验收未通过。pose、dense均维持default-off、review-only。**

## 数据与冻结

- Mar29：168/168照片，181,628,022 bytes。
- May27：147/147媒体，145照片+2视频，261,706,756 bytes。
- 合计315个文件完整ingest；下载与副本SHA256相同，运行前后原文件SHA/mtime和副本SHA/stat均不变。313照片完整解码且嵌套EXIF DateTimeOriginal日期匹配，两个视频完整解码无错。
- 全日期baseline/panel包含37个非字节相同多图组、143张组内照片；其余原文件保留，视频不充照片，singleton不充评估样本。按120秒间隔，时间连通片不跨日；两个完整日不是连续48h记录，也不保证重复场景的语义独立性。
- 原`holdout/FROZEN.md`保留。新`FROZEN-FINAL-20260912.md`在新预测/像素/标签揭示前冻结：A=原off/f79cc0d，B=原pose-v0/f79cc0d，仅pose packet，C=新dense-only/6d97dc1，仅dense packet。A/B载入原git模块，共享pipeline源码保持一致；不是重建旧依赖环境。
- 无参数、keeper逻辑、模型或依赖变化。所有研究缓存只在新root内；不存在旧dense-pilot，slice路径、ID和输入身份已核验。

## 新完整日期分层结果

标签为**sealed blind model_provisional**，不是人工真值。37组均在隔离上下文看过派生图后封存；May按保留10组checkpoint+两个独立5组批次机械合并，原判断不变。合计48 phases、45 required、4次quality abstention。哈希、分区、coverage及acceptable子集均验证。

下列keeper/错误指标在A、B、C三臂完全相同：

| 分层 | 组/照片 | keepers | 漏保/required phases | 高置信漏保 | 质量错误/已评覆盖 | B/C提案组 |
|---|---:|---:|---:|---:|---:|---:|
| single | 16/93 | 35 | 1/22 | 1 | 1/20 | 1/3 |
| multi | 1/2 | 1 | 0/1 | 0 | N/A（0/0） | 0/0 |
| no_subject | 20/48 | 27 | 1/22 | 1 | 0/22 | 1/2 |
| 合计 | 37/143 | 63 | 2/45 | 2 | 1/42 | 2/5 |

selected unacceptable共5张，均在single；quality abstention为single3、multi1。阶段质量错误与不可接受keeper计数不是同一指标。Mar29/May27分别28/35keepers、各1次高置信漏保；质量错误分别1/25与0/17已评覆盖阶段。

pose输出4条帧级观测提案/2组，dense输出16个含新增观测pair、17个帧级观测/5组；两臂提案组并集5组，不能相加成7组。算法提案计数不等于正确恢复率，未有新增人工确认。keeper变更0，质量/覆盖净改善0，完整语义主体集合确认0。**未证实可重复净收益，因此不晋升默认值。**

multi只有1组、且质量abstain；这一层没有足够泛化证据。no_subject仍沿用全图baseline；未建立文档/票据/屏幕专项支持结论。dense仍可能受到eligible-peer筛选及crop背景上下文影响，见[INSTANCE_RECOVERY.md](INSTANCE_RECOVERY.md)。

## 已通过门禁

- 六个日期×arm组合实际Stage2/3；每个Stage2重复两次（冻结特征回放，不是重新推理重复性），快照精确一致。
- 物理分组、keeper集合/顺序/首选图、utility scores、预算、阶段输出精确相同。所有视觉组REVIEW_REQUIRED，unsafe AUTO_REMOVE=0，六份delete_local.txt均为空，Stage4未运行。
- 旧人标单列回归：10组/13keepers，1phase miss，质量错误2/3已评覆盖阶段，off/review_only相同；没有重标或与模型指标混算。
- `scripts/test-full.sh`：**985 passed，108.77s**；Ruff、compileall、diff-check通过。15个研究Python文件只做formatter/noqa整理，AST与6d97dc1完全相同。
- 原始315文件和315副本运行前后校验通过。自包含总览143张、dense页32张内嵌JPEG均可解码，无脚本/外部资源。
- 上下文隔离源码复核完成，但实际served模型仍为Astra，**不是跨模型独立验收**。

## 未通过与Windows真机剩余项

- **浏览器未验收**：OpenClaw browser对loopback报告导航返回`browser navigation blocked by policy`。未绕过；临时服务已关闭。静态图片检查不是浏览器渲染证明。总览约50MB，dense页约2.4MB，真实加载/交互仍须检查。
- **人标／净收益／multi泛化未通过**：模型标签不能认证删除安全；proposal可行动性、人工耗时收益、完整主体集合及人标安全仍无证据。
- **Windows未验收**：Linux/L40S测试不能替代RTX5070Ti真机。剩余：Torch/CUDA与本地权重、显存/吞吐；盘符/中文/空格及SQLite URI；cache身份；中文编码、图像方向、完整组展示；正式review server分页与keeper顺序。按[WINDOWS_BENCHMARK.md](WINDOWS_BENCHMARK.md)保留真机JSON，不把StubBackend结果当真实Stage1性能。
- Stage4文件变更、sidecar、文件锁、跨卷移动及恢复属于另行授权的专项验收；本次没有执行任何原图移动/删除、云端写入、push或发布。

## 权威产物与重跑边界

实验根：`/home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/`

- `REPORT.md`、`STATUS.md`、`holdout/STATUS.md`：最终结论与版本。
- `holdout/evaluation/summary.json`：逐组、逐日期、分层指标及标签来源。
- `holdout/blind-labels/*/SEAL.json`：盲标输入/输出封存及批次来源。
- `holdout/replay/*/*/replay-complete.json`：实际Stage2/3、模块pin及重复回放证据。
- `holdout/finalization/`：数据、AST/安全/载体验证、测试、浏览器阻塞与git状态；原报告在`archive/`。
- `holdout/review.html`、`holdout/dense-review/review.html`：本地自包含载体，待授权浏览器/Windows打开验证。

本次冻结到此停止，不继续采集或为结果调参。已有输出不可盲目覆盖；`finalization/run_producers.py`、`replay_arms.py`、`evaluate_arms.py`保留精确执行逻辑，恢复前必须核对已完成checkpoint和输入身份。预存`docs/images/`不属于本次写入或清理范围。
