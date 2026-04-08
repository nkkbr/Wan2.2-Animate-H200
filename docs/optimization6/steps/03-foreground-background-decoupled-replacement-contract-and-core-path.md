# Step 03: Foreground / Background Decoupled Replacement Contract and Core Path

## 1. 为什么要做

当前系统虽然已经尝试了前景/背景解耦思路，但 generate 主路径仍然高度依赖整图 replacement 逻辑。

这带来的问题是：

- 边缘仍然只是整图生成的副产物；
- foreground、alpha、background 的语义边界不够硬；
- 即使 preprocess 有更强 alpha / matting，generate 也不一定真正用对。

因此，这一步要做的是：

> 把 replacement 问题正式重写成 foreground / alpha / background / composite 四部分协作的问题。


## 2. 目标

建立一条正式的 decoupled replacement core path，使 generate 能明确消费：

- `foreground_rgb`
- `foreground_alpha`
- `foreground_confidence`
- `background_rgb`
- `background_visible_support`
- `background_unresolved`
- `composite_roi_mask`

并且让这些信号不只用于 debug，而是真正参与核心生成路径。


## 3. 具体要做什么

### 3.1 preprocess 侧

确保正式输出并记录：

- 前景图像
- 前景 alpha
- 前景置信度
- clean plate 背景
- 背景可见性支持图
- 背景 unresolved 区域
- composite ROI mask

### 3.2 contract 侧

metadata / contract 要明确：

- 每个 artifact 的语义
- 来源
- 是否来自外部 alpha/matting 模型
- 是否是 fallback

### 3.3 generate 侧

必须引入：

- foreground-aware conditioning
- background-aware conditioning
- alpha-aware composition hints

这一阶段先不追求最强效果，而是先把主路径彻底改对。


## 4. 如何做

### 4.1 数据结构

建议在 `wan/utils/animate_contract.py` 和 `wan/animate.py` 中增加明确的 decoupled bundle 读取层。

### 4.2 主路径策略

建议先做 `decoupled_v2`：

- `legacy`：老路径
- `decoupled_v1`：artifact-only
- `decoupled_v2`：真正进入 core conditioning

### 4.3 runtime / debug

至少导出：

- `foreground_rgb_preview`
- `foreground_alpha_preview`
- `background_rgb_preview`
- `composite_roi_preview`
- decoupled runtime summary


## 5. 如何检验

### 5.1 correctness

- preprocess bundle contract 正确
- generate 可正常读取
- runtime stats 正常

### 5.2 smoke

- 真实 10 秒 preprocess + generate 跑通
- 不崩，不回退到错误路径

### 5.3 质量 gate

相对当前稳定 baseline：

- `background_fluctuation` 不恶化
- `seam_score` 不恶化
- `roi_gradient` 提升至少 `5%`

如果 3 轮都达不到，说明 decoupled core path 自身还不够强，但 contract 仍可保留。


## 6. 三轮闭环规则

### Round 1

- 把 contract 和读取路径彻底打通
- 跑 preprocess / generate smoke

### Round 2

- 把 decoupled 信号真正推进 core conditioning
- 做 reviewed benchmark + 真实 10 秒 AB

### Round 3

- 对最优实现做最后一次收敛
- 冻结 best-of-three


## 7. 成功标准

- `decoupled_v2` 主路径可跑；
- reviewed benchmark 和真实 smoke 全通过；
- 相对 baseline 至少在 `roi_gradient` 上有真实改善；
- seam/background 不出现不可接受退化。


## 8. 风险与回退

### 风险

- foreground/background 分离后，generate 主体结构不适应；
- 产生新的 seam；
- 过度依赖 foreground alpha，导致人物主体发虚。

### 回退

- 若 Round 3 仍未过 gate，则保留 contract 与 artifact，不升默认核心路径。
