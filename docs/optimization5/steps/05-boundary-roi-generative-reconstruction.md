# Step 05: Boundary ROI Generative Reconstruction

## 1. 为什么要做

`optimization4` 的 ROI 路线已经证明：

- deterministic ROI refine 不足以真正重建边缘细节
- 即便 ROI 做高分辨率处理，也仍然更多是在“修边”，不是“生成边缘”

所以本步骤必须升级成：

> boundary ROI 局部生成式重建

而不是继续做更复杂的局部滤波或 sharpen。


## 2. 目标

建立一条专门的 boundary ROI 生成式重建路径，用来恢复：

- 发丝边缘
- 手指边缘
- 头肩轮廓
- 衣角轮廓
- 半透明 / 高不确定边界


## 3. 交付目标

1. ROI extraction / packing / alignment
2. ROI generative reconstruction pipeline
3. alpha-aware paste-back
4. ROI benchmark / gate
5. findings 文档


## 4. 范围

### In Scope

- 窄 boundary ROI 提取
- ROI 级生成或 restoration
- ROI paste-back
- ROI 级 metrics

### Out of Scope

- 全图二阶段生成
- 大规模训练


## 5. 实施方案

### 5.1 ROI 应是“狭窄而高价值”的

ROI 不应覆盖整个人物，而应聚焦：

- boundary band
- hair band
- hand band
- cloth edge band

### 5.2 生成式路线而非滤波路线

本步骤的核心不是：

- sharpen
- unsharp mask
- detail boost

而是：

- ROI 级局部 diffusion / restoration / repaint
- 利用 alpha / trimap / uncertainty / background 作为条件

### 5.3 paste-back 必须 alpha-aware

paste-back 不能只靠硬矩形贴回，必须基于：

- local alpha
- trimap_unknown
- face preserve
- background keep prior


## 6. 如何验证

### 6.1 ROI 专项指标

- ROI `boundary_f1`
- ROI `alpha_mae`
- ROI `gradient`
- ROI `contrast`
- ROI `halo`

### 6.2 全图副作用指标

- seam
- background fluctuation
- face consistency


## 7. 强指标要求

相对 `optimization4` 最优 `legacy` 至少要满足：

- ROI `gradient` 提升 `>= 10%`
- ROI `contrast` 提升 `>= 10%`
- ROI `halo` 下降 `>= 10%`
- 全图 seam / background 不恶化超过 `3%`


## 8. 实现-测评-再改闭环

### Round 1

- 跑通 ROI 生成式重建主链
- 建立 ROI metrics 与真实 AB

### Round 2

若指标仍弱，则重点改：

- ROI 条件构造
- unknown region 扩张策略
- ROI 生成分辨率
- paste-back

### Round 3

若仍未过 gate，则冻结 best-of-three，并明确：

- inference-only 路线是否还有空间
- 是否必须转向 Step 07 trainable path


## 9. 风险

- ROI 生成可能带来 identity 漂移
- ROI paste-back 不当会产生局部闪动或拼接感


## 10. 完成标准

1. ROI 生成式重建链路已打通
2. ROI 指标与全图副作用指标都已建立
3. 至少一套方案在真实 benchmark 上明显优于 `optimization4` 最优基线
4. findings 文档给出是否可升为默认高质量路径
