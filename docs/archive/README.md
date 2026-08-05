# docs/archive — 已完成的调研底稿

这里是 photo-dedup 设计阶段的调研/决策底稿。结论都已经落进代码和
`docs/ALGORITHM.md`，保留是为了追溯"为什么这么定"，日常读代码不需要看这些。

| 文件 | 当时解决什么 | 结论落在哪 |
|---|---|---|
| `P0_DECISION_POLICY.md` | AUTO_REMOVE / REVIEW / KEEP 的判定边界 | `src/decision.py`、ALGORITHM.md 决策章 |
| `P0_EXPOSURE.md` | 曝光评分选型 | `src/exposure.py` |
| `P0_EYES.md` | 闭眼检测选型 | `src/eye_detection.py` |
| `BUIQA_ASSESSMENT.md` | 无参考画质评价（BRISQUE/NIQE 等）选型 | `src/quality.py` |
| `SCENE_GATING_ACADEMIC.md` | 场景门控的学术方案调研 | `src/scene_router.py` 系列 |
| `SCENE_GATING_PRODUCTS.md` | 场景门控的现成产品/模型横评 | 同上 |
| `SCENE_GATING_IMPLEMENTATION_SPEC.md` | 场景门控落地 spec | `src/siglip_*.py`、`src/scene_*.py` |

⚠️ `research/siglip_prompt_bank_v1.yaml` **不在**这里——它是运行时依赖
（`config.yaml` → `scene.prompt.path`），留在 `research/` 原位，别动。
