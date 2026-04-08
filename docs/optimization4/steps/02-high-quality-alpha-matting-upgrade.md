# Step 02: High-Quality Alpha / Matting Upgrade

## 1. 目的

本步骤的目标是把当前 heuristic `soft_alpha` 升级成真正高质量、可稳定驱动边缘提升的 alpha/matting 系统。

这是 `optimization4` 中最优先的技术主攻方向，因为边缘锐度问题最核心的底层约束之一就是 alpha 质量。


## 2. 为什么必须做

当前系统虽然已经有：

- `hard_foreground`
- `soft_alpha`
- `boundary_band`
- `occlusion_band`
- `uncertainty_map`

但现有 `soft_alpha` 仍然偏向“足够可用的边界表达”，而不是“高质量边缘真值逼近”。

所以当前最常见的问题仍然是：

- 发丝不够细
- 衣角边缘不够利
- 手指、袖口等薄边缘容易软化
- 背景高对比边界处容易出现过度保守的柔化


## 3. 交付目标

本步骤结束时，应具备：

1. `alpha_v2` 路径
2. 可选 `trimap_v2`
3. `alpha_uncertainty_v2`
4. hair-edge / fine-edge 辅助 mask
5. metadata / contract / debug 接通
6. 与当前 boundary fusion 主链兼容


## 4. 实施范围

优先修改：

- `wan/modules/animate/preprocess/matting_adapter.py`
- `wan/modules/animate/preprocess/boundary_fusion.py`
- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/utils/animate_contract.py`
- 相关 benchmark 与 debug 脚本

如需要，可新增：

- `wan/modules/animate/preprocess/alpha_refinement.py`
- `scripts/eval/compute_alpha_precision_metrics.py`


## 5. 具体实施方案

### 5.1 先升级 artifact 结构

输出建议新增：

- `alpha_v2`
- `trimap_v2`
- `alpha_uncertainty_v2`
- `fine_boundary_mask`
- `hair_edge_mask`

### 5.2 建立更强的 alpha 生成逻辑

建议从当前系统融合以下信号：

- `hard_foreground`
- parsing 边界
- matting adapter 结果
- `boundary_band`
- `occlusion_band`
- `uncertainty_map`
- face / hair / hand 区域先验

### 5.3 建立 alpha 置信度体系

不要只输出一个 alpha。  
还应输出：

- `alpha_confidence`
- `alpha_uncertainty`
- `alpha_source_provenance`

### 5.4 与现有主链兼容

`soft_alpha` 旧路径要保留兼容，不要一次性破坏已有 bundle。  
新逻辑可以通过：

- `alpha_mode=v2`
- 或 `matting_mode=high_precision_v2`

做显式切换。


## 6. 强指标要求

相对 `optimization3` 最优结果，本步骤至少要争取：

- `halo_ratio` 降低 **15% 以上**
- `band_gradient` 提升 **8% 以上**
- `band_edge_contrast` 提升 **8% 以上**

若 Step 01 的标注集已就绪，则额外要求：

- `boundary F-score` 提升 **10% 以上**
- `trimap error` 降低 **15% 以上**
- `alpha MAE / SAD` 明显下降


## 7. 实现-测评-再改闭环

最多 3 轮。

### Round 1

- 打通 `alpha_v2` artifact
- 接通 metadata / debug / metrics
- 先争取 proxy 正向

### Round 2

- 重点修 hair-edge / thin-edge
- 修正高对比背景边缘的过软问题

### Round 3

- 固定最优融合策略
- 输出 findings，决定是否升默认

若 Round 3 仍无法同时提升：

- halo
- gradient
- contrast

则不得升为默认，只能冻结为实验路径。


## 8. 如何验证

至少执行：

1. synthetic alpha contract checks
2. preprocess smoke with `alpha_v2`
3. boundary proxy AB
4. 若 Step 01 已完成，则跑 labeled edge benchmark
5. 与现有 generate smoke 做兼容性测试


## 9. 风险

### 9.1 alpha 太激进导致伪边

应对：

- 记录 `alpha_uncertainty`
- 用 `fine_boundary_mask` 单独看错误区域

### 9.2 hair-edge 提升但 face-edge 变差

应对：

- 分 face/hair/hand 子区域记录局部指标


## 10. 完成标准

仅当下面条件成立时可结束：

- `alpha_v2` 已完整接通
- 相对旧 `soft_alpha` 有明显客观提升
- generate 兼容通过
- findings 明确写出是否升默认
