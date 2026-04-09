# Step 03: Foreground / Background Decoupled Contract And Compositor Baseline

## 1. 为什么要做

如果系统仍然沿用“整图 replacement 再修边”的问题结构，那么更强 alpha 也很难真正转化成最终边缘质量。

所以这一步的目标不是先提高边缘锐度，而是先改变系统结构：

> 让 foreground、alpha、background、composite 成为显式对象。


## 2. 目标

建立一条正式的 decoupled contract，使 preprocess 和 generate 之间不再只传：

- mask
- bg
- ref

而是显式传：

- foreground representation
- alpha / trimap
- background clean plate
- visible support / unresolved region
- compositor inputs


## 3. 要改哪些模块

建议涉及：

- `wan/utils/animate_contract.py`
- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/modules/animate/preprocess/background_clean_plate.py`
- `wan/animate.py`
- `wan/utils/replacement_masks.py`
- 新增 `wan/utils/compositor.py`
- 新增 `scripts/eval/check_decoupled_contract.py`
- 新增 `scripts/eval/compute_decoupled_composite_metrics.py`


## 4. 具体如何做

### 4.1 contract 扩展

定义并标准化以下 artifact：

- `foreground_rgb`
- `foreground_alpha`
- `foreground_confidence`
- `trimap_unknown`
- `background_clean_plate`
- `background_visible_support`
- `background_unresolved_region`
- `composite_guidance_mask`

### 4.2 建显式 compositor baseline

新增一个 baseline compositor：

- 输入 decoupled artifacts
- 输出 deterministic composite
- 作为后续生成式 ROI reconstruction 的对照基线

### 4.3 generate 侧先做 compatibility，不急着完全改 backbone

第一步不要求立刻把整个 generate 主干改成新的 decoupled architecture，而是要先做到：

- 新 contract 能被 generate 正确读取
- compositing baseline 可独立工作
- 边界 ROI 可从 decoupled artifacts 正确提取


## 5. 如何检验

### 5.1 contract correctness

- artifact 完整
- metadata 完整
- fallback 明确

### 5.2 compositing correctness

- foreground/background 合成结果正确
- alpha-aware 复合不出明显边缘伪影
- unresolved region 语义正确

### 5.3 regression checks

相对当前稳定基线：

- seam 不明显恶化
- background fluctuation 不明显恶化
- face identity 不明显恶化


## 6. 强指标要求

这一步更看结构正确性，但仍建议设硬门槛：

- `seam_degradation_pct <= 3%`
- `background_fluctuation` 不恶化超过 `5%`
- decoupled composite 至少在 `halo` 上不差于当前基线


## 7. 实现-测评-再改闭环

### Round 1

- 建 contract 和 compositor baseline
- 跑 synthetic correctness

### Round 2

- 真实 10 秒 preprocess + compositor smoke
- 跑 regression metrics

### Round 3

- generate compatibility smoke
- 冻结 decoupled baseline


## 8. 成功标准

- decoupled contract 正式建立
- compositor baseline 稳定可用
- regression 在可接受范围内
- 为 Step 04/05 提供可依赖输入
