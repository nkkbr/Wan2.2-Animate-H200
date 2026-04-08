# Optimization5 Steps

## 1. 目标

`optimization5` 的主题，不再是继续挤压 heuristic / deterministic generate-side 边缘增强的残余空间，而是切换到一条更强的新路线：

> 用高质量 alpha / matting、前景背景解耦、边界 ROI 局部生成式重建，以及必要的小规模训练/适配，真正把高精度 preprocess 信号转化成最终锐利边缘。

这套 steps 基于 [wan_animate_replacement_edge_reconstruction_plan.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/wan_animate_replacement_edge_reconstruction_plan.md) 拆出。  
它默认继承 `optimization3` / `optimization4` 已建立的：

- benchmark / gate
- metadata / contract
- candidate search 基础设施
- H200 preprocess/generate profiling

但它不再把“继续调规则”当作主线，而是明确转向更高价值的边缘重建路线。


## 2. 实施原则

### 2.1 先补更可信的边缘真值，再追更强生成路线

`optimization4` 已经有 edge mini-benchmark，但仍是 `bootstrap_unreviewed` 起步。  
`optimization5` 必须先把它升级成更可信、更可依赖的人工审核版小基准，否则后续所有“边缘显著提升”判断都不够扎实。

### 2.2 每一步都必须闭环，最多 3 轮

每一步默认最多 **3 轮**：

1. 实现
2. 全面测评
3. 若未达 gate，则改代码或改方案
4. 再测评
5. 若到第 3 轮仍未达标，则冻结 best-of-three，并明确写入 findings

这不是建议，而是 `optimization5` 的默认执行约束。  
除 benchmark/schema 建设型步骤外，不能只做实现不做测评，也不能只测一次就结束。

### 2.3 以“显著优于 optimization4 最优结果”为目标

本轮不是允许“只好一点点”的优化。  
每一步都应尽量给出相对 `optimization4` 最优配置的清晰指标目标。  
如果只得到轻微改善，应明确判为“未达到本轮目标”。

### 2.4 ROI 生成与 alpha 质量，是本轮最高价值模块

本轮 H200 预算优先给：

- alpha / matting / hair-edge estimation
- boundary ROI 局部生成式重建
- 少量高价值候选搜索
- 必要的小模型训练 / 适配

不再优先给：

- 更多 heuristic postprocess 参数 sweep
- 更多全图 sharpening 实验
- 未被证明有效的低价值 brute-force 路径


## 3. 步骤顺序

### Step 01
[01-human-reviewed-edge-benchmark-and-hard-gates.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/steps/01-human-reviewed-edge-benchmark-and-hard-gates.md)

把 `optimization4` 的 edge mini-set 升级为人工审核版，并建立更强的真值指标与 gate。  
这是后续所有边缘优化能否被可信验证的基础。

### Step 02
[02-production-grade-alpha-matting-system.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/steps/02-production-grade-alpha-matting-system.md)

建立真正可用于边缘重建的高质量 alpha / trimap / hair-alpha 系统。  
这一步是本轮最重要的输入质量升级。

### Step 03
[03-foreground-background-decoupled-replacement-contract.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/steps/03-foreground-background-decoupled-replacement-contract.md)

显式建立 foreground / background / alpha-aware composite contract，让边缘问题从整图 replacement 中独立出来。

### Step 04
[04-rich-boundary-signal-core-generation-conditioning.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/steps/04-rich-boundary-signal-core-generation-conditioning.md)

让 richer boundary signal 真正成为主生成条件，而不再只停留在 mask shaping 与后处理。

### Step 05
[05-boundary-roi-generative-reconstruction.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/steps/05-boundary-roi-generative-reconstruction.md)

在边界 ROI 上做真正的局部生成式重建。  
这是最有希望直接提升“边缘锐度”的主力步骤。

### Step 06
[06-semantic-roi-experts-for-face-hair-hands-cloth.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/steps/06-semantic-roi-experts-for-face-hair-hands-cloth.md)

建立语义化 ROI 专家路径，分别处理 face / hair / hands / cloth / occluded boundary。

### Step 07
[07-trainable-edge-refinement-and-adapter-path.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/steps/07-trainable-edge-refinement-and-adapter-path.md)

若纯 inference 路线仍不足，转向小规模训练 / adapter 路线。  
这是本轮最重的一步，应后置。

### Step 08
[08-h200-high-value-compute-allocation-and-orchestration.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/steps/08-h200-high-value-compute-allocation-and-orchestration.md)

当至少一到两条有效边缘方案已经出现后，再正式做 H200 生产编排、candidate orchestration 与默认 preset 固化。


## 4. 推荐执行策略

推荐顺序就是：

1. Step 01
2. Step 02
3. Step 03
4. Step 04
5. Step 05
6. Step 06
7. Step 07
8. Step 08

不要跳过 Step 01，因为本轮最需要解决的就是：

- 如何更可信地证明“边缘真的变锐了”

同时，也不要把 Step 08 提前。  
在真正有效的边缘方案尚未出现前，先做算力编排意义不大。


## 5. 每一步结束时必须留下什么

每一步结束时，都必须留下：

- findings 文档
- benchmark 运行目录
- gate 结果
- 最优配置快照
- 明确的“是否默认启用”结论

如果一个步骤只留下代码，没有留下量化结果和 gate 结论，则不能算完成。


## 6. 与 optimization4 的关系

`optimization4` 负责证明：

- 当前 heuristic / deterministic generate-side edge enhancement 路线已接近上限

`optimization5` 负责继续往前推进，但要换方法：

- 更强 alpha
- 更强边界真值
- 前景背景解耦
- 局部生成式边缘重建
- 必要的小规模训练

因此，`optimization4` 的 benchmark、artifact、ROI 基础设施与失败结论，都是本轮的输入，不应被重复浪费。
