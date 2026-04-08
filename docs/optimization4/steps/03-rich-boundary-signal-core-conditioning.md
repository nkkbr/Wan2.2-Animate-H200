# Step 03: Rich Boundary Signal Core Conditioning

## 1. 目的

本步骤的目标是让 richer boundary signal 真正进入 generate 主 conditioning 路径，而不是继续主要依赖 postprocess。


## 2. 为什么要做

当前系统的一个核心限制是：

- preprocess 已经能输出更强的边界信号
- 但 generate 端对这些信号的消费仍然偏保守

它更像：

- 用这些信号来决定“哪里少犯错”

而不是：

- 用这些信号来主动生成“更好的边缘”


## 3. 交付目标

本步骤结束时，应实现：

1. richer boundary conditioning v1
2. `soft_alpha / boundary_band / occlusion_band / uncertainty_map / background_keep_prior / face_preserve / structure_guard` 更深地下沉到生成主逻辑
3. 明确的 rich-vs-legacy AB benchmark


## 4. 作用范围

优先修改：

- `wan/animate.py`
- `wan/utils/replacement_masks.py`
- `wan/utils/rich_conditioning.py`
- 相关 generate benchmark 脚本

如必要，可扩展：

- `WanAnimateModel` 的条件输入结构
- richer signal 的 feature encoder


## 5. 具体实施

### 5.1 先做低风险版本

第一阶段不要立刻改 backbone。  
先增强以下部分：

- `background_keep`
- latent mask construction
- transition band composition
- face-preserve-aware conditioning

### 5.2 再做更深层条件注入

第二阶段可探索：

- 把 boundary / uncertainty 变成额外 feature channel
- 或把其编码成附加 token / modulation signal

### 5.3 兼容旧 bundle

必须允许：

- `legacy`
- `rich_v1`

并行存在，以支持 AB。


## 6. 强指标要求

相对 `optimization3` 最优 generate baseline：

- `halo_ratio` 降低 **10% 以上**
- `band_gradient` 提升 **6% 以上**
- `band_edge_contrast` 提升 **6% 以上**
- `seam_score` 恶化不超过 **3%**
- `background_fluctuation` 恶化不超过 **3%**


## 7. 实现-测评-再改闭环

最多 3 轮。

### Round 1

- 先把 richer signal 更深接入 existing replacement mask 逻辑
- 看是否能正向拉动 halo / gradient / contrast

### Round 2

- 如果还不够，再做条件表达增强
- 例如附加 latent channel 或 feature encoder

### Round 3

- 固定最优路径
- 决定是否升默认

若 3 轮后仍然只能降低 halo 而不能提升 gradient / contrast，则只保留实验路径，不得升默认。


## 8. 如何验证

至少做：

1. generate smoke
2. `legacy vs rich_v1` 真 AB
3. edge metrics
4. seam/background non-regression


## 9. 风险

### 9.1 richer signal 把边界做得更保守

表现：

- halo 少了
- 但 gradient / contrast 仍然下降

应对：

- 增加“真正鼓励锐化”的信号接法
- 不要只在 `background_keep` 上做更多压制


## 10. 完成标准

只有在真实 AB 下同时满足：

- halo 改善
- gradient 改善
- contrast 改善
- seam/background 无明显回退

本步骤才算真正成功。
