# Step 01: Human-Reviewed Edge Benchmark and Hard Gates

## 1. 为什么要做

`optimization4` 已经建立了 edge mini-benchmark，但它仍然是 `bootstrap_unreviewed` 起步：

- 可以运行
- 可以做相对比较
- 但不够强，尤其不适合作为“最终边缘是否真的变锐”的终局判据

如果没有更可信的人工审核版小基准，后续再强的 alpha、ROI 生成、局部重建方案，也可能因为 benchmark 噪声而被误判。

因此，这一步必须先做，而且必须做扎实。


## 2. 目标

建立一套可重复、可审计、足够小但足够可信的 edge benchmark：

- 覆盖 face / hair / hand / shoulder / cloth / occluded edge
- 同时提供：
  - hard foreground
  - soft alpha
  - trimap
  - boundary type 标签
- 提供 hard gate，用于后续每一步的 pass / fail 判定


## 3. 交付目标

完成后应新增：

1. `docs/optimization5/benchmark/`
   - benchmark README
   - label schema
   - gate policy
   - reviewed manifest
2. `scripts/eval/`
   - keyframe extraction / packaging
   - edge GT metric 计算
   - benchmark summary / gate
3. 小规模人工审核 edge mini-set
4. 相对 `optimization4` baseline 的正式 metrics 快照


## 4. 范围

### In Scope

- 从真实 10 秒 benchmark 视频中抽取 20 到 50 帧关键帧
- 为每帧建立：
  - `hard_mask`
  - `soft_alpha`
  - `trimap`
  - `boundary_type_map`
- 支持 boundary 子类：
  - `face`
  - `hair`
  - `hand`
  - `cloth`
  - `occluded`
- 形成 reviewed benchmark manifest
- 建立 hard gate

### Out of Scope

- 大规模众包标注
- 全视频逐帧真值
- 训练数据构建平台化


## 5. 实施方案

### 5.1 从 optimization4 mini-set 出发，不从零开始

直接复用 `optimization4` 的：

- keyframe extraction
- label schema 基础框架
- GT metric 脚本
- gate 汇总框架

这一步不是重做一套系统，而是把现有小基准升级成“人工审核可信版”。

### 5.2 新标签格式

每个关键帧应包含：

- `rgba` 或 `rgb + alpha`
- `trimap`
- `boundary_type_map`
- `review_metadata`
  - reviewer
  - review status
  - review notes

`boundary_type_map` 至少要支持：

- `0 = non-boundary`
- `1 = face`
- `2 = hair`
- `3 = hand`
- `4 = cloth`
- `5 = occluded`

### 5.3 GT 指标升级

必须支持：

- global `boundary_f1`
- global `alpha_mae`
- global `alpha_sad`
- global `trimap_error`
- category-wise boundary metrics：
  - `face_boundary_f1`
  - `hair_boundary_f1`
  - `hand_boundary_f1`
  - `cloth_boundary_f1`
  - `occluded_boundary_f1`

### 5.4 hard gate 设计

初版 gate 至少包含：

- benchmark 帧数不少于 20
- reviewed 帧占比 100%
- 每个关键 boundary type 至少出现 3 帧
- 指标脚本可稳定重复运行
- 与 `optimization4` baseline 对齐的 reference metrics 已冻结


## 6. 如何验证

### 6.1 基础正确性

- schema 校验通过
- label 文件路径完整
- metric 脚本对整套数据跑通
- summary / gate 文件落盘

### 6.2 可信性检查

- 抽查至少 5 帧，人工确认：
  - alpha 合理
  - trimap 合理
  - boundary type 标注合理

### 6.3 稳定性

同一套 reviewed benchmark，重复跑 3 次 summary / gate，结果一致。


## 7. 强指标要求

这一步不是提升模型质量，而是提升 benchmark 质量。  
因此强指标要求是：

- reviewed 帧覆盖数 `>= 20`
- 每类 boundary 至少 `>= 3` 帧
- reference summary 与 gate 重复运行 3 次结果一致
- relative metric 抖动在容忍阈值内（理论上应接近 0）


## 8. 实现-测评-再改闭环

### Round 1

- 升级 benchmark 目录、schema、scripts
- 形成 first reviewed manifest
- 跑全套 summary / gate

### Round 2

若出现：

- 标签路径问题
- 指标不稳定
- 类别覆盖不足

则修 schema / label / script，再次运行。

### Round 3

若仍不满足：

- reviewed coverage
- category coverage
- summary / gate 稳定性

则补标签或修脚本。  
第 3 轮后若仍未满足，则冻结 best-of-three，并明确记录缺口。


## 9. 风险

- 标注成本会明显上升
- 类别定义不清时，reviewed benchmark 会变得不稳定
- 如果边界类别划分过细，后续统计可能稀疏


## 10. 完成标准

只有当下面条件全部满足时，这一步才算完成：

1. reviewed edge mini-benchmark 已正式落盘
2. GT metrics 与 gate 脚本可稳定重复运行
3. category-wise 边界指标可用
4. findings 文档明确记录：
   - benchmark 规模
   - reviewed 覆盖情况
   - baseline 指标
   - gate 状态
