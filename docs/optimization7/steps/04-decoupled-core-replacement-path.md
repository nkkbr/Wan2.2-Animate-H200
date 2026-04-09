# Step 04: Decoupled Core Replacement Path

## 1. 为什么要做

Step 03 只建立了 decoupled contract 和 compositor baseline，还没有真正让 replacement 主路径“按 decoupled 的方式思考”。

如果 generate 主路径仍然本质上是旧的整图 replacement，只在外围消费新 artifact，那么边缘控制能力依然有限。


## 2. 目标

建立一个真正消费 decoupled inputs 的 core replacement path，使 generate 至少在结构上开始区分：

- foreground synthesis / preservation
- alpha-aware boundary handling
- background preservation / composition guidance


## 3. 要改哪些模块

建议涉及：

- `generate.py`
- `wan/animate.py`
- `wan/utils/rich_conditioning.py`
- `wan/utils/replacement_masks.py`
- `wan/utils/compositor.py`
- 新增 `scripts/eval/run_optimization7_decoupled_core_benchmark.py`
- 新增 `scripts/eval/evaluate_optimization7_decoupled_core_benchmark.py`


## 4. 具体如何做

### 4.1 新增明确的 conditioning mode

建议新增：

- `replacement_conditioning_mode=decoupled_v1`

它必须显式消费：

- `foreground_alpha`
- `trimap_unknown`
- `background_visible_support`
- `background_unresolved_region`
- `foreground_confidence`

### 4.2 边界带构造必须重新定义

不再只从旧 `mask + soft_band` 推导，而应从 decoupled artifacts 推导：

- hard foreground interior
- uncertain transition band
- unresolved composite band

### 4.3 compositing-aware guidance

generate 侧要知道：

- 哪些区域应优先保持 foreground identity
- 哪些区域允许更多 boundary reconstruction
- 哪些区域必须更强 background keep


## 5. 如何检验

### 5.1 correctness

- decoupled mode 输入读取正确
- 没有 silently fallback 到 legacy
- runtime stats 能显示 decoupled artifact availability

### 5.2 quality AB

相对 `legacy`：

- `halo` 不恶化
- `gradient` 至少有正向趋势
- `contrast` 至少有正向趋势
- `seam` 不明显恶化


## 6. 强指标要求

建议 gate：

- `roi_gradient_gain_pct > 0`
- `roi_edge_contrast_gain_pct > 0`
- `roi_halo_reduction_pct >= 0`
- `seam_degradation_pct <= 3%`

这一步不要求直接打到最终目标，但必须至少证明 decoupled core path 没有继续重复旧路线的失败模式。


## 7. 实现-测评-再改闭环

### Round 1

- 接 decoupled mode
- 跑最小 AB

### Round 2

- 根据失败点调整 boundary / composite guidance
- 跑 reviewed + 真实 10 秒 AB

### Round 3

- 冻结 best-of-three
- 明确判断 decoupled core path 是否值得继续进入 ROI reconstruction


## 8. 成功标准

- decoupled core path 正式可运行
- 至少不再复制 `optimization4` 中那种“所有核心指标一起失败”的模式
- 为 Step 05 提供更可靠的 decoupled ROI 输入
