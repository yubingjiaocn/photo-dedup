# 商业产品调研：Scene Gating、选片与安全回退

> 范围：公开产品说明、帮助中心与隐私说明；检索时间 2026-08-03。`未披露` 不等于产品一定没有该能力，只表示本轮未找到足以支持具体实现结论的公开官方材料。尤其不要把消费图库的搜索能力反推为可用于删除的分类器。

## Executive summary

1. **没有证据显示这些产品会先把整个图库硬分配到“数百个互斥场景”再删除。** 消费图库把语义理解主要暴露为自由文本/人物/地点检索与浏览；专业选片工具主要暴露为相邻相似组内的质量信号。
2. 行业成熟范式是两条正交管线：`语义检索/导航`（人物、物体、地点、文本）与 `组内比较/选片`（同一 burst 或 duplicate group）。不要把前者当后者的删除许可证。
3. Adobe/Apple/Google 均将识别结果包装为可浏览、可搜索、可编辑或可合并的组织功能；Narrative 明确把分数称为“warning lights”，并让用户调整显示阈值。人始终保有确认/纠错权。
4. 对本项目，scene 应是**评分器路由特征**而非删除规则：`PERSON/LIVING_SUBJECT`、`NONHUMAN_CHARACTER_OR_DOLL`、`FOOD/OBJECT`、`FIREWORKS/LOW_LIGHT`、`LANDSCAPE/SCENE`、`MOTION_PHOTO`、`UNKNOWN` 可多标签且允许低置信。
5. `UNKNOWN` 是正常安全状态：缺失、模型低置信、标签冲突、Motion Photo、跨场景组、多人/人偶混淆，都只降低自动化覆盖率，绝不能直接降低保留资格。
6. 自动动作只限于“强重复组内、明确技术失败、质量差距足够大”的 **move-to-quarantine**；scene 标签本身、审美分、语义相似、单张模糊判断均不得单独触发删除。
7. 推荐 UI：按“为什么进入此队列”展示组、默认保留一张、one-click `Keep all / Pick this / Re-run with stricter safety`，并把纠错写成可追溯 override。

---

## 1. 能力矩阵（公开披露，而非逆向模型推断）

| 产品 | taxonomy 是否暴露 / multi-label | 人物、主体、场景分层 | burst / duplicate grouping | 选片依据（公开披露） | 用户可控项与回退 | 本地 / 云端 |
|---|---|---|---|---|---|---|
| **Adobe Lightroom / Lightroom Classic** | 不见完整场景 taxonomy；People View/人脸可命名；搜索、关键词和 AI mask 提供语义入口。一个照片可有多张人脸/多种 mask，不能据此推定内部统一多标签 taxonomy。 | People/人脸组织；AI Mask 可识别 subject、sky、background、objects、people，定位是**局部编辑**而不是删片分类。 | Classic 支持按 capture time 手动/自动 stack；新版公开说明还称可检测 catalog duplicates 并进 stack 复核。 | 公开资料聚焦组织、编辑与用户 rating/flag，不公开将闭眼/审美分用于自动删除的产品规则。 | 人脸命名/确认、stack 展开、flag/rating、非破坏性 mask；删除仍是用户动作。 | Classic 以本机 catalog/文件为中心；Lightroom 生态可云同步。具体处理位置随产品和同步设置而变。 |
| **Apple Photos** | 未向用户公布完整 object/scene 标签表；公开为搜索、People & Pets、地点、媒体类型、Visual Look Up 等入口。可同时按人物、地点、媒体类型/关键词组织，非互斥浏览模型。 | People & Pets 与地点/媒体类型/搜索分开；subject isolation 属编辑/分享。 | iPhone burst 是独立 burst；Mac 的 Duplicates collection 自动呈现候选，用户点 Merge。 | Duplicates 的 Merge 由用户确认；公开资料未把闭眼、审美或场景标签表述为自动删片依据。 | 人物/宠物可命名与纠错；可查看 burst、选择保留；duplicate merge 由人触发，可从 Recently Deleted 恢复。 | macOS/iOS 本地库；若启用 iCloud Photos，会同步至 iCloud。 |
| **Google Photos** | 公开给用户的是自由文本/组合式搜索（人、物、地点、文字等）及 collections；无公开固定 taxonomy。一次查询可组合人/动作/地点，天然不是单标签分类。 | People & pets、Places、Documents、Videos 等 collections；Ask Photos 是用户 opt-in 的对话式检索，不是删片决策。 | “Stack similar photos” 主要整理 Photos view；官方称此设置可选，且不改变存储量。 | 检索按相关性排序；没有官方删除候选/闭眼/美学自动移除机制。 | 选择开启 stacks、展开/在 grid 查看；人宠可标注；Ask Photos 明确 opt-in。 | 以 Google 帐户云图库为中心；有本地设备视图/备份边界。 |
| **Aftershoot** | 本轮未找到官方帮助文档支持其暴露 scene taxonomy 或 multi-label 结构。营销资料通常将定位放在 AI culling/编辑，不能据此写具体内部标签。 | 公开产品定位偏摄影选片；未验证其是否将人物、主体、场景作为可浏览的分层 taxonomy。 | 面向 shoot 的 culling/相似图工作流（需在采购/接入前复核当前版本具体规则）。 | 常见公开卖点是脸部、闭眼、模糊、重复/相似与自定义偏好；本轮没有可引用官方技术细节，故不将其作为实现事实。 | 选片结果应由摄影师复核；具体阈值/override 功能须产品试用验证。 | 桌面工作流；本轮不对推理是否全离线作结论。 |
| **Narrative Select / Narrative** | 不公布通用 scene taxonomy；其官方帮助页描述的是**人脸局部** assessments 与 focus score，不是场景分类。人脸可有多项 assessment，照片可多脸。 | 明确有 face assessments + focus；Key Subject Highlighting 可选开关。未见把通用物体/场景分类用于 culling 的官方说法。 | 为导入 shoot 提供 First Pass/选片；本轮未找到可证实的 duplicate 算法细节。 | 面部 assessment（官方称当前 17+ 种）、脸部/图像 focus；focus 考虑各主体、主体重要性与图像语境。公开措辞承认模型/阈值持续更新。 | hover 看 0–10 分；Preferences 改 indicator colour scales；可关闭 Key Subject Highlighting；官方把指标比作警告灯，邀请反馈错标。 | 桌面选片软件；具体云同步/遥测边界应按其当期 Privacy Policy 另行核验。 |
| **FilterPixel** | 本轮未获得可验证的官方 help/doc 页面；不声称 taxonomy、multi-label 或内部模型细节。 | 公开市场定位为 AI culling；本轮不足以确认人物/主体/场景层级。 | 同上；需以当期官方 trial/help 文档验证。 | 通常宣传闭眼、失焦、重复与 picks；但没有在本报告中当作已核实的产品规则。 | 应保留摄影师审核；具体控制项需验收。 | 桌面产品；本轮不对离线/云端推理作结论。 |
| **Photo Mechanic / Photo Mechanic Plus** | 传统 metadata/关键词/catalog 工作流，不应臆测存在视觉 scene taxonomy 或 multi-label AI classifier。 | 以文件、IPTC metadata、关键词、rating/color class、搜索和 catalog 为组织原语；不等同于 AI 人物/主体/场景层。 | 强项是快速浏览与人工 culling；本轮未核实官方 duplicate/burst AI 自动分组功能。 | 摄影师以星级、颜色、标签、metadata 和人工视觉比较选择；不主张自动闭眼/美学淘汰。 | 高度人工可控（标记、颜色、metadata、选择与导出）；安全性来自不替用户做不可逆语义决定。 | 本地桌面/本地 catalog 为主；具体版本能力需采购时核验。 |

### 矩阵解读与证据强度

- **强证据（官方帮助页直接描述）**：Adobe People/AI masking；Apple Duplicates/People/Burst；Google 搜索与 stacks；Narrative 的 face/focus 评分和可调显示。
- **弱证据/待验证**：Aftershoot、FilterPixel、Photo Mechanic 的本轮可引用官方资料不足。保留在矩阵内是为了避免设计时误把竞品营销说法当产品事实；接入或对标前应做桌面试用与隐私/网络抓包验收。
- 所有“本地/云端”行都描述**产品架构与用户工作流**，并不等于对某项模型实际运行位置的保证。

---

## 2. 行业共性：不是“先分几百场景、再删除”

### 2.1 可观察到的产品分工

1. **消费图库：语义索引 → 找得到、看得到。** Google 把人/物/地点和组合短语放在搜索框，Apple 把人物、地点、media type、documents 等暴露为发现入口，Adobe 把 People 和 AI masks 服务于组织、检索或编辑。它们公开承诺的是可发现性，而非一个稳定、可审计的全库场景分类 API。
2. **专业选片：组内比较 → 选得快。** Narrative 公开的深度信号集中在人脸、焦点和关键主体；这比“food/fireworks/static object”的全局 scene 标签窄得多，也更贴近一次拍摄内挑一张的任务。
3. **重复/相似组与质量分的作用域都很重要。** Google 的 stacks 被明确描述为整理视图且可选；Apple 的 duplicates 是专用候选集合并由用户 Merge；Adobe 的 stack/duplicate 也服务 review。成熟产品把“group”作为减少认知负担的单元，而不是把每个相似项静默删除。
4. **少数任务专用 detector 胜过通用 taxonomy。** 人脸/人宠分组、subject/sky mask、文本/文档、闭眼/焦点等是可解释且任务相关的 detector；没有公开依据支持把模糊的 scene label 提升为删除 gate。

### 2.2 对 66k 本地库的结论

应建设 `embedding/search index + explicit quality detectors + conservative group logic`，而不是穷举 scene 类别：

- embedding/语义标签：仅用于检索、相似候选召回、**路由**；
- 人脸/眼睛/焦点/曝光/分辨率：只在适用时生成可解释质量证据；
- 时间、pHash、embedding、Motion Photo 绑定：生成强/弱 duplicate group；
- 决策：只在同一可信 group 内比较，并保留 `UNKNOWN/MAYBE` 弃权。

---

## 3. 可借鉴与不可照搬

### 可借鉴

| 观察 | 本项目落地原则 |
|---|---|
| Google/Apple 将语义理解用作搜索、collection、stack 浏览 | scene label 写入 `routing_tags`/检索索引；UI 可筛选但不得作为 `AUTO_REMOVE` 条件。 |
| Adobe 将 AI mask 与手工蒙版并列、强调非破坏性 | 任何模型输出均展示来源、置信与区域/适用范围；用户 override 永远优先且可撤销。 |
| Narrative 用每张脸的细粒度 signals，而非一个神秘总分 | 人像组显示“哪张脸、哪项问题、分数/缺失值”；多人脸中任一关键脸不确定即 review。 |
| Google stacks / Apple Duplicates / Lightroom stack 都先组织再复核 | 先做 group 浏览与默认 keeper，再建议 quarantine；不要将“被堆叠”表述成“可删”。 |
| Narrative 允许改 indicator 色阶且承认错标 | 阈值可配置、按来源/场景局部校准；对模型版本和用户覆盖保留 audit trail。 |

### 不可照搬

1. **不可照搬云端“搜索相关性”到本地删除。** 检索排在前面只说明匹配查询，不说明其余照片低价值。
2. **不可照搬单一“best shot/审美”分。** cosplay、人偶、烟花、食物、静物的价值函数不同；审美模型未公开校准且可能偏向常规摄影构图。
3. **不可把人脸 detector 的 absence 解释成低质。** 人偶、面具、背影、强妆、暗光、侧脸都可能漏检；无脸在 food/fireworks/静物中是正常值。
4. **不可把商业 UI 的易用性当安全语义。** “Merge/stack/cull”在不同产品分别表示整理、候选、用户确认；本项目必须区分 `suggestion`、`quarantine`、`purge`。
5. **不可把厂商未公开的模型类别、训练数据或是否全离线写进设计文档。** 未披露即 `unknown`，上线前实验验证。

---

## 4. 本项目推荐 UX：自动路由、解释、UNKNOWN、低成本纠错

### 4.1 可多标签的路由层（不是删除层）

```text
routing_tags[] = {
  PERSON, NONHUMAN_CHARACTER_OR_DOLL, FOOD, OBJECT_OR_STILL_LIFE,
  FIREWORKS_OR_LOW_LIGHT, LANDSCAPE_OR_SCENE, DOCUMENT_OR_SCREENSHOT,
  MOTION_PHOTO, UNKNOWN
}
```

- 标签允许重叠：cosplay 人像可同时为 `PERSON + LOW_LIGHT + EVENT/SCENE`；人偶可为 `NONHUMAN_CHARACTER_OR_DOLL + OBJECT_OR_STILL_LIFE`。
- `UNKNOWN` 由低置信、冲突、模型不可用或新颖内容触发，**不是低质标签**。
- 路由只决定显示何种质量证据与默认安全强度：例如人像显示眼睛/每脸焦点；烟花显示曝光/拖影但不适用“闭眼”；Motion Photo 作为原子资源绑定。

### 4.2 三层决策 UI

1. **Inbox / AUTO-READY**：只显示“强重复 + 高置信技术失败 + 清晰 keeper + 足够 margin”的成员；CTA 为 `Move to quarantine`，不出现一键永久删除。
2. **Review / MAYBE**：按原因分桶：`scene unknown`、`cross-scene group`、`face/eye disagreement`、`quality margin too small`、`feature missing`、`Motion Photo`。每组一屏对照：候选缩略图、时间/相似证据、质量证据、why-not-auto。
3. **Keep all / Exceptions**：任何一键 `Keep all` 都能终止该组本轮自动化；人工 `KEEP`/`REMOVE` 写入持久 override，重跑不得覆盖。

### 4.3 最低成本纠错交互

- 每组默认有一个 keeper，但显示“**建议，不会移动任何文件**”；快捷键 `K`（保留此张）、`A`（全保留）、`Q`（移入隔离候选）、`U`（标 UNKNOWN/不适用）。
- 一次纠错可选作用域：`only this photo` / `this group` / `future groups with this detector`；默认最窄作用域，防止误学习。
- 解释卡固定字段：`group evidence`（时间/pHash/embedding）、`routing tags + confidence`、`applicable metrics`、`not-applicable/missing metrics`、`winner margin`、`safety blocks`、`model/version`。
- 批次执行后提供可恢复清单：原路径、目标 quarantine 路径、组 ID、原因、时间、用户/自动来源；无人工单独确认的 batch 永不 purge。

### 4.4 强制安全不变量

```text
scene tag alone                 -> NEVER AUTO_REMOVE
UNKNOWN / missing / conflict    -> MAYBE or KEEP_ALL
Motion Photo member             -> act on the bound asset-set together, else MAYBE
any group                       -> keep at least one member
cross-scene / low-purity group  -> REVIEW_REQUIRED
AUTO_REMOVE                     -> move/quarantine only, never permanent delete
manual KEEP                     -> highest-priority immutable override until user changes it
```

这与现有 `P0_DECISION_POLICY.md` 的“hard gates → group comparison → reject option → 保底约束 → 可恢复执行”一致；scene gating 只应插在 **quality metric selection** 前，而不是结果删除后。

---

## 5. 参考资料（官方优先）

### Adobe

1. [Lightroom Classic — Intelligent facial recognition](https://helpx.adobe.com/lightroom-classic/desktop/organize-photos-in-lightroom-classic/face-recognition.html) — 人脸索引、相似脸分组、用户命名/确认。
2. [Lightroom — People View](https://helpx.adobe.com/lightroom/desktop/organize-photos/people-view.html) — People 分类组织。
3. [Lightroom Classic — Masking tool](https://helpx.adobe.com/lightroom-classic/help/masking.html) — AI subject/sky/background/people/object mask，编辑定位与非破坏性语义。
4. [Lightroom Classic — What's new](https://helpx.adobe.com/lightroom-classic/desktop/introduction-to-lightroom-classic/whats-new.html) — 官方对 duplicate detection/stack review 的当前说明入口；功能依版本变化，实施前复核。

### Apple

5. [Photos for Mac — Remove duplicate photos and videos](https://support.apple.com/guide/photos/remove-duplicates-pht5a3157c1d/mac) — Duplicates collection 与用户 Merge。
6. [Photos for Mac — Find and name people and pets](https://support.apple.com/guide/photos/find-and-name-people-and-pets-phtad9d981ab/mac) — People & Pets 的命名和纠错入口。
7. [Photos for Mac — View photo bursts](https://support.apple.com/guide/photos/view-photo-bursts-pht8745d2677/mac) — burst 作为可查看、可挑选的媒体组织单元。
8. [Photos for Mac — Search for photos and videos](https://support.apple.com/guide/photos/search-for-photos-and-videos-pht64de33e5a/mac) — 搜索与多种 collection/媒体发现入口。

### Google

9. [Google Photos — Search by people, things & places](https://support.google.com/photos/answer/15235862?hl=en&co=GENIE.Platform%3DDesktop) — 关键词/组合语义搜索、People & pets 标签、按相关性展示、Ask Photos opt-in。
10. [Google Photos — Organize your Photos view](https://support.google.com/photos/answer/14169846?hl=en&co=GENIE.Platform%3DAndroid) — 可选的 similar-photo stacks；整理 view 不改变 storage。
11. [Google Photos — Set up & manage face groups](https://support.google.com/photos/answer/6128838) — face grouping 的用户管理入口。

### Narrative

12. [Narrative — Face and Focus Assessments](https://help.narrative.so/en/articles/7337369-face-and-focus-assessments) — 17+ face assessments、0–10 focus、颜色尺度可调、承认错标/模型阈值会更新。
13. [Narrative — Filter by Focus Score](https://help.narrative.so/en/articles/7337374-filter-by-focus-score) — 焦点分数考虑每个主体、主体重要性与图像语境。
14. [Narrative — Key Subject Highlighting](https://help.narrative.so/en/articles/15454134-key-subject-highlighting) — 可开关的 key-subject 辅助。

## 6. 未决验证清单（不阻塞本地 P0）

- 对 Aftershoot、FilterPixel、Photo Mechanic：用当期 Windows 版本进行离线模式、网络连接、导入外传、元数据写入、撤销/回收站行为的实测；不要依据营销页做安全承诺。
- 以本库分层抽样校准：cosplay 人像、面具/人偶、食物、烟花、静物、场景、Motion Photo 各建立真值组；分别测 group purity、keeper recall、错误 quarantine 率与 UNKNOWN coverage。
- 所有最终阈值应按**强重复组内**校准；任何 scene bucket 数据不足时，自动退化到 `MAYBE`，而非借用其他 scene 的阈值。
