# Photo Dedup — 设计文档

## 背景

Willy 的 Google Photos 已使用 163/202GB。本地全量原图在 Windows 台式机 `E:\Photos\YYYY\MM\` 下，451GB / 66072 文件，主要 JPG，也有 MP4（视频 + 实况照片 Motion Photo 的 MP4 部分）。

Google Photos Library API 2025 年 3 月后大砍：手工上传的照片无法通过 OAuth API 删除。绕路方案：
- **云端删除**：走 GPTK (Google-Photos-Toolkit) Tampermonkey 用户脚本调 Google Photos Web 的私有 API
- **本地删除**：Python 脚本直接操作文件系统

Willy 明确要**全自动**——不接受"每组人工看一眼选最佳"。

## 硬约束 & 特殊情况

### 1. 主体不是"人"，人只是前景

Willy 的典型 case：**打卡照**——景点/建筑/物件是主体，人在前面换姿势。传统 pHash 一定翻车：
- 人一动像素分布变，pHash 认为不重复（漏检）
- 也可能反过来：不同景点前同一个人 pHash 意外接近（误合）

正解：**DINOv2 语义 embedding + 人脸姿态检测**做多条件判定。

### 2. 实况照片（Motion Photo）

小米/华为的实况照片有两种形式：
- **单文件**：JPG 尾部拼接了 mp4 数据，同一文件包含图 + 视频
- **配对文件**：同名 `.jpg` + `.mp4` 或 `.jpg.MP` 存放

处理原则：把配对/嵌入的 mp4 当"附件"，跟随 jpg 生死——jpg 保留则 mp4 保留，jpg 删则 mp4 一起删。识别方法：
- 单文件：检测 JPG EOI (`FFD9`) 之后是否还有 mp4 header (`ftypmp4` / `ftypisom` / `ftypheic`)
- 配对文件：查同目录同 basename 是不是有 `.mp4` / `.MP` / `.MP4`

### 3. HDD 是瓶颈

E:\ 是机械盘，随机读慢。**所有涉及原图读取的操作必须一次顺序遍历完成**，中间结果落 SQLite，后续聚类/打分复用缓存。可断点续跑（每 100 张 commit 一次）。

### 4. 5070S GPU

有 16GB VRAM（RTX 5070 Super）。可跑：
- DINOv2-base（~86M 参数，224x224 输入，batch 32 完全没问题）
- pyiqa 的 MUSIQ / CLIP-IQA（预训练模型，几百 MB）
- YuNet 人脸检测（ONNX，CPU 都够，GPU 更快）

Windows 环境用 `torch` 官方 CUDA wheel 装 CUDA 12.x 版本即可。

## Pipeline

### Stage 0: 环境 & 目录扫描

**产物**：`inventory.sqlite` 表 `files`
```sql
CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,       -- 原始绝对路径
  basename TEXT,          -- 文件名（不含目录）
  size_bytes INTEGER,
  mtime_ns INTEGER,
  exif_datetime TEXT,     -- EXIF DateTimeOriginal（YYYY-MM-DD HH:MM:SS）
  exif_timestamp INTEGER, -- Unix ts（用于时间窗聚类）
  width INTEGER,
  height INTEGER,
  file_kind TEXT,         -- 'jpg' | 'jpg_motion' | 'mp4_paired' | 'mp4_only'
  motion_partner_id INTEGER,  -- 实况的另一半 id（NULL 表示独立）
  scan_status TEXT DEFAULT 'pending',  -- pending/done/error
  scan_error TEXT
);
CREATE INDEX idx_files_kind ON files(file_kind);
CREATE INDEX idx_files_ts ON files(exif_timestamp);
```

**步骤**：
1. `os.walk(E:\Photos)` 遍历所有 `.jpg/.jpeg/.mp4` 文件
2. 每个文件读 stat（size, mtime）
3. JPG 读 EXIF DateTimeOriginal；缺失则 fallback 到文件名里的时间戳（`IMG_YYYYMMDD_HHMMSS.jpg` 常见模式）
4. 检测实况照片：
   - 单文件模式：读 JPG 末尾看有没有 mp4 signature
   - 配对模式：查同 basename 的 mp4/MP 文件，链接到 `motion_partner_id`
5. 全部落库

**性能预期**：只读 stat + EXIF header（~几 KB），6.6 万文件 HDD 上估计 15-30 分钟。

### Stage 1: 特征提取（GPU 密集）

**产物**：`inventory.sqlite` 表 `features`
```sql
CREATE TABLE features (
  file_id INTEGER PRIMARY KEY REFERENCES files(id),
  phash BLOB,              -- 8 字节 64-bit pHash
  dinov2_embedding BLOB,   -- 768-dim float16 = 1536 字节
  quality_score REAL,      -- MUSIQ 分数 0-100
  quality_meta TEXT,       -- JSON: {sharpness, clipiqa, ...}
  face_count INTEGER,
  faces_json TEXT,         -- YuNet 输出：bbox + landmarks + score
  status TEXT DEFAULT 'pending'
);
```

**步骤**（只处理 `file_kind IN ('jpg','jpg_motion')`）：
1. batch 加载图（PIL）→ resize 224x224 → 归一化 → 送 DINOv2 → 拿 embedding（float16 存）
2. 同一批送 pyiqa MUSIQ → quality_score
3. 送 pyiqa CLIP-IQA quality prompt → 辅助分（存 meta）
4. 缩放到 640 宽度送 YuNet → face bbox + landmarks + confidence
5. 每 100 张 commit 一次，支持 Ctrl+C 断点续跑

**性能预期**：DINOv2-base + MUSIQ + YuNet 在 5070S 上估计 20-50 张/秒，6.6 万张 GPU 阶段 30-60 分钟。加上 HDD 顺序读 IO，整个 Stage 1 估计 2-4 小时。

### Stage 2: 聚类（CPU，快）

**产物**：`inventory.sqlite` 表 `groups`
```sql
CREATE TABLE groups (
  id INTEGER PRIMARY KEY,
  group_type TEXT,         -- 'burst' | 'similar_scene' | 'exact_dup'
  keep_file_id INTEGER,
  member_count INTEGER,
  created_at INTEGER
);
CREATE TABLE group_members (
  group_id INTEGER,
  file_id INTEGER,
  is_keep BOOLEAN,
  reason TEXT,             -- 打分或删除原因
  PRIMARY KEY (group_id, file_id)
);
```

**判定规则**（有优先级，一张只属一组）：

**Layer 1 — Exact Duplicate**（pHash 汉明距离 ≤ 2）
- 直接归组，无需再看时间
- 保留最新 mtime 或最大文件（原图 vs 压缩）

**Layer 2 — Burst / Continuous Shot**（30 秒窗口 + DINOv2 相似度）
- 按 exif_timestamp 排序，滑动窗口 30 秒内
- 组内两两 DINOv2 余弦相似度 ≥ 0.92 判为同组
- **打卡照过滤**：如果组内检测到人脸，且人脸 landmark 位置差 ≥ 30% 图像宽度（表明人明显换位置/换姿势），拆组

**Layer 3 — Similar Scene (Loose)**（可选，先默认关）
- 更长时间窗（几分钟）+ 更严 embedding 阈值（≥ 0.96）
- 用于抓"同角度反复拍同一物"的场景
- Willy 打卡照 case 里这个规则容易误合，先关掉，后续再看

**每组选 keep**：
```
score = quality_score * 0.6
      + face_quality * 0.3   # 有脸时看脸质量：清晰度 + 睁眼估计
      + resolution_bonus * 0.1
```

### Stage 3: 审查报告 + 删除清单

**产物**：
1. `review.html` —— 每组一 row：左列大图（keep），右列缩略图（delete），标各种分数
2. `delete_local.txt` —— 待删本地绝对路径（含 motion partner mp4）
3. `delete_cloud.json` —— 云端待删清单：`[{filename, exif_datetime, size_bytes}]`
4. `summary.txt` —— 统计：多少组、能省多少 GB

Willy 用浏览器打开 `review.html` 抽样看几十组，OK 再执行删除。

### Stage 4: 执行

**本地**（脚本）：
```
python execute_local.py --mode move    # 移到 E:\Photos\_trash\YYYY-MM-DD_HHMMSS\
python execute_local.py --mode delete  # 直接删（不推荐首次跑）
```

**云端**（GPTK console 脚本）：
- 读 `delete_cloud.json`
- 遍历云端库（`gptkApi.getItemsByUploadedDate` 分页）
- 按 filename + timestamp 匹配
- `gptkApi.moveItemsToTrash([dedupKey])` 批量丢回收站
- 每批 100 个，慢速执行防速率限制
- 输出匹配率报告

## 目录结构（sub-agent 产出）

```
projects-personal/photo-dedup/
├── DESIGN.md               ← 本文件
├── README.md               ← 使用说明（给 Willy 看）
├── requirements.txt        ← Python 依赖
├── setup_windows.bat       ← 一键装环境（uv + torch cuda + pyiqa）
├── config.yaml             ← 可调参数（阈值、路径）
├── src/
│   ├── stage0_inventory.py
│   ├── stage1_features.py
│   ├── stage2_cluster.py
│   ├── stage3_report.py
│   ├── execute_local.py
│   ├── db.py               ← SQLite schema + helpers
│   ├── motion_photo.py     ← 实况照片检测
│   └── quality.py          ← IQA + 人脸质量启发式
├── scripts/
│   └── gptk_delete.js      ← GPTK console 脚本
└── docs/
    └── ALGORITHM.md        ← 详细算法说明（相似度阈值/评分公式）
```

## 可调阈值（config.yaml）

```yaml
paths:
  root: "E:/Photos"
  db: "./inventory.sqlite"
  trash: "E:/Photos/_trash"

scan:
  extensions: [".jpg", ".jpeg", ".mp4", ".MP"]
  batch_size: 32

cluster:
  burst_window_seconds: 30
  dinov2_threshold: 0.92        # 越大越严
  phash_hamming_threshold: 2
  face_pose_shift_ratio: 0.30   # 人脸位置差超 30% 图宽拆组
  enable_loose_similar: false   # 打开慢/易误合

quality:
  weight_iqa: 0.6
  weight_face: 0.3
  weight_resolution: 0.1

execute:
  mode: "move"   # move | delete
  dry_run: true  # 首次 true，看清单再改 false
```

## 风险 & 回滚

1. **DINOv2 阈值调错误合**：保留 SQLite，重跑 stage2 即可，不用重跑 stage1
2. **实况照片 mp4 遗失**：`motion_partner_id` 保证配对，删 jpg 一定连 mp4 一起删；反向也确认
3. **本地误删**：默认 `mode: move`，只挪到 `_trash/`，用户随时恢复
4. **云端误删**：Google Photos 回收站保留 60 天，可恢复
5. **Windows 中文路径**：全流程用 `pathlib.Path` + UTF-8 open，避免 gbk 坑

## 交付验证

Sub-agent 完成后我需要：
1. 完整代码 + `README.md`（Willy 能照着装环境）
2. `docs/ALGORITHM.md` 讲清楚每个阈值/公式为啥这么定
3. 一个 mini 测试用例：sub-agent 用 10 张假图（自己生成/找几张公开）验证 pipeline 能跑通
4. Windows 环境安装指令验证过（sub-agent 没 Windows 环境但至少 requirements + pip install 命令要对）

Sub-agent 交付后我 review，然后指导 Willy 装 + 跑 Stage 0 试水。
