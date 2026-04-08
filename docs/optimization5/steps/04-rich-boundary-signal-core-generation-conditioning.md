# Step 04: Rich Boundary Signal Core Generation Conditioning

## 1. 为什么要做

`optimization4` Step 03 证明：

- richer boundary signal 接到当前 generate 主链里，并没有自然地变成更锐利的边缘

但这并不意味着 richer signal 没价值，更可能意味着：

- 接法不够强
- 信号只在弱约束层起作用
- 没有被组织成更适合生成的 core conditioning

因此，本步骤不是简单重试 `rich_v1`，而是重构 richer boundary signal 的核心 conditioning 方式。


## 2. 目标

让下面这些信号真正成为生成主条件的一部分：

- `soft_alpha`
- `trimap_unknown`
- `hair_alpha`
- `uncertainty_map`
- `occlusion_band`
- `background_keep_prior`
- `face_preserve`


## 3. 交付目标

1. 新 core conditioning 编码路径
2. richer boundary token / map 表达
3. generate debug / runtime / metrics 接通
4. findings 文档


## 4. 范围

### In Scope

- richer boundary signal 的重编码
- core conditioning branch 接入
- 与 decoupled contract 协同

### Out of Scope

- ROI 二阶段局部生成本体
- 小规模训练


## 5. 实施方案

### 5.1 不再只改 replacement mask

本步骤不应再重走 `optimization4` 的老路，即：

- 只改 mask shaping
- 只改 background_keep
- 只靠 boundary postprocess

必须让 richer signal 进入更强的位置：

- feature map conditioning
- token conditioning
- 或更高权重的 auxiliary branch

### 5.2 三类条件分层

建议将 richer signal 分成三层：

1. **hard structure**
   - hard foreground
   - trimap certain region
2. **soft boundary**
   - soft_alpha
   - trimap_unknown
   - hair_alpha
3. **uncertainty / preserve**
   - uncertainty
   - occlusion
   - face preserve
   - background keep prior

不同层应使用不同编码策略，而不是全部塞进同一 mask 通道。


## 6. 如何验证

### 6.1 真实 AB

必须与：

- `legacy`
- `optimization4 rich_v1`

做真实 10 秒 benchmark 对照。

### 6.2 核心指标

- `halo_ratio`
- `band_gradient`
- `edge_contrast`
- `seam_score`
- `background_fluctuation`
- `face_consistency_proxy`


## 7. 强指标要求

相对 `optimization4` 最优 `legacy` 至少要满足：

- `halo_ratio` 改善 `>= 8%`
- `band_gradient` 提升 `>= 8%`
- `edge_contrast` 提升 `>= 8%`
- seam / background 不恶化超过 `3%`


## 8. 实现-测评-再改闭环

### Round 1

- 建 core conditioning 新接法
- 跑真实 AB

### Round 2

若未过 gate，则调整：

- branch 权重
- layer 注入位置
- hard/soft/uncertainty 分层表达

### Round 3

若仍未过 gate，则冻结 best-of-three，并明确：

- richer signal 是否值得继续以 inference 路线推进
- 或是否必须依赖 Step 07 的 trainable adapter


## 9. 风险

- richer conditioning 过强可能伤害 identity
- 多路信号耦合不当可能导致 generate 不稳定


## 10. 完成标准

1. richer boundary signal 的新 core conditioning 路线已落地
2. 真实 AB 与正式 gate 已执行
3. 至少一套方案能显著优于 `optimization4` 的 `legacy`
4. findings 文档明确记录是否升默认
