# Step 02: Production-Grade Alpha / Matting System

## 1. 为什么要做

`optimization4` Step 02 已经证明：

- `alpha_v2` 相关 artifact 能接进系统
- 但当前 heuristic-compatible alpha 路线并没有在 benchmark 上明显打赢 baseline

这说明问题不在“有没有 alpha artifact”，而在于：

> 当前 alpha 质量仍然不够强，不足以支撑真正锐利的头发丝、手部薄边、衣角和半透明边缘。

因此，本步骤必须升级成真正 production-grade 的 alpha / matting 系统。


## 2. 目标

建立一套可以被 generate 和 ROI 重建真正信任的高质量 alpha 系统，输出至少包括：

- `soft_alpha`
- `trimap_unknown`
- `hair_alpha`
- `alpha_confidence`
- `alpha_source_provenance`


## 3. 交付目标

完成后应新增或升级：

1. preprocess 侧 alpha / matting 主模块
2. 与 `SAM2 / parsing / uncertainty / face-hair-hand cue` 的融合逻辑
3. 新 artifact / metadata / contract
4. GT 级 alpha / trimap 指标脚本
5. findings 文档


## 4. 范围

### In Scope

- 高质量 portrait/video matting 接入
- hair-edge 专项增强
- alpha 与 `SAM2 / parsing / uncertainty` 融合
- trimap_unknown 生成
- confidence / provenance 输出

### Out of Scope

- 训练大型 matting 模型
- 整视频逐帧人工 alpha 标注


## 5. 实施方案

### 5.1 不再把 heuristic alpha 当主线

当前 heuristic `soft_alpha` 只能作为 fallback / prior，不再作为本步骤的主策略。

### 5.2 建立 alpha fusion pipeline

alpha fusion 至少融合下面几路：

- raw matting alpha
- `SAM2` hard foreground prior
- parsing-derived semantic edge prior
- uncertainty prior
- face/hair/hand ROI prior

输出：

- `alpha_v3`
- `trimap_unknown_v3`
- `hair_alpha_v3`
- `alpha_confidence_v3`

### 5.3 head / hair 优先

由于最终边缘观感最容易被头发和头肩轮廓拖垮，hair-edge 必须是专项对象：

- 优先在 head ROI 内做更高分辨率 alpha refine
- 输出独立 `hair_alpha`
- 与 face preserve 逻辑配合

### 5.4 兼容策略

active `soft_alpha` 的切换必须采取保守策略：

- 若新路径未达 gate，则保持旧主线
- 若局部指标强但全局指标弱，则允许只输出 artifact 不切默认


## 6. 如何验证

### 6.1 GT 指标

在 reviewed edge mini-benchmark 上至少计算：

- `alpha_mae`
- `alpha_sad`
- `trimap_error`
- `boundary_f1`
- `hair_boundary_f1`

### 6.2 真实 10 秒 smoke

必须跑真实 preprocess smoke，并确认：

- artifact 完整
- contract 正确
- generate 可兼容消费

### 6.3 真实边界 proxy

在真实 replacement benchmark 上测：

- `halo_ratio`
- `band_gradient`
- `edge_contrast`
- `hair_edge_quality_proxy`


## 7. 强指标要求

相对 `optimization4` baseline，目标至少应满足：

- `alpha_mae` 改善 `>= 20%`
- `trimap_error` 改善 `>= 20%`
- `hair_boundary_f1` 提升 `>= 10%`
- 真实 case 上 `halo_ratio` 下降 `>= 10%`
- 真实 case 上 `band_gradient` 不下降


## 8. 实现-测评-再改闭环

### Round 1

- 打通强 alpha 路线
- 输出全套 artifact
- 跑 GT metrics + 真实 smoke

### Round 2

若出现：

- alpha 退化
- trimap_error 明显变坏
- hair_edge 指标弱

则改 fusion 逻辑、head ROI refine、confidence weighting。

### Round 3

若仍未达强 gate，则继续改融合策略或 fallback policy。  
第 3 轮后若仍未达标，则冻结 best-of-three，并明确记录：

- 是否保留为实验路径
- 是否仅保留 artifact 不切 active alpha


## 9. 风险

- 更强 matting 可能带来时序闪动
- hair-edge 强化可能误伤 face boundary
- alpha 过软会让边缘更干净但更不锐


## 10. 完成标准

1. 高质量 alpha / trimap / hair-alpha 路线接通
2. GT 指标与真实 smoke 都完整跑通
3. 至少一套结果在 reviewed benchmark 上显著优于 `optimization4` baseline
4. findings 文档给出是否升默认的明确结论
