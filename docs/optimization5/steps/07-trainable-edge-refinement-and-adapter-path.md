# Step 07: Trainable Edge Refinement and Adapter Path

## 1. 为什么要做

如果 Step 02 到 Step 06 之后，纯 inference 路线仍然不能显著提升边缘锐度，那么必须承认：

> 仅靠规则、条件重排和 deterministic / ROI 生成式推理，可能仍不足以打透最终边缘问题。

这时就要转向更重但更可能有效的路线：

- 小规模训练
- adapter / LoRA
- lightweight edge refinement model


## 2. 目标

建立一个可控预算的 trainable path，用于真正学习：

- hair edge reconstruction
- hand edge reconstruction
- cloth fine edge restoration
- alpha-aware edge consistency


## 3. 交付目标

1. 可训练数据打包路径
2. 轻量 edge refinement / adapter 结构
3. H200 上可承受的训练 / finetune 方案
4. benchmark / gate
5. findings 文档


## 4. 范围

### In Scope

- 小模型 / adapter / LoRA
- ROI 级训练样本
- 少量高价值训练轮次

### Out of Scope

- 重训大模型
- 大规模数据平台化


## 5. 实施方案

### 5.1 优先训练 ROI 级小模块

不要尝试直接训练整图 replacement 模型。  
优先做：

- ROI edge refinement adapter
- ROI restoration LoRA
- small local diffusion/refiner

### 5.2 训练目标

训练损失应面向边缘：

- alpha loss
- boundary loss
- contrast-aware loss
- hair-edge / hand-edge 局部 loss

### 5.3 数据来源

优先使用：

- reviewed edge mini-set
- synthetic ROI patches
- preprocess / generate 失败 case


## 6. 如何验证

### 6.1 GT 指标

- ROI `boundary_f1`
- ROI `alpha_mae`
- ROI `trimap_error`

### 6.2 real benchmark

- `halo`
- `gradient`
- `contrast`
- seam/background


## 7. 强指标要求

相对 `optimization4` 最优 baseline：

- ROI `gradient` 提升 `>= 12%`
- ROI `contrast` 提升 `>= 12%`
- ROI `halo` 下降 `>= 10%`
- seam/background 不恶化超过 `3%`


## 8. 实现-测评-再改闭环

### Round 1

- 打通训练数据 -> 模型 -> 推理 -> benchmark

### Round 2

若仍弱，则调整：

- adapter 架构
- loss 配置
- ROI 采样

### Round 3

若仍未过 gate，则冻结 best-of-three，并给出：

- 是否继续训练路线
- 还是转回更简单的生产策略


## 9. 风险

- 训练引入维护成本
- 小数据训练易过拟合
- identity consistency 风险高于纯 inference 路线


## 10. 完成标准

1. trainable edge refinement / adapter 路线已跑通
2. H200 上训练/推理成本可接受
3. 至少一套 trainable 路线在强指标上明显优于 `optimization4` baseline
4. findings 文档给出是否值得进入生产路线
