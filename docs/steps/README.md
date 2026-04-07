# Wan-Animate Replacement 优化实施路线

## 1. 文档目标

本目录把两份上层方案文档：

- `docs/wan_animate_replacement_preprocess_optimization.md`
- `docs/wan_animate_replacement_generate_optimization.md`

收敛成一条可执行的、分步骤的工程实施路线。

这里的目标不是再次罗列想法，而是给后续真正改代码的人提供：

- 清晰的实施顺序
- 每一步的依赖关系
- 每一步的输入、输出和验收标准
- 哪些步骤是基础设施，哪些步骤是质量提升，哪些步骤是高风险探索


## 2. 总体原则

整个优化工作应按下面三条原则推进。

### 2.1 先修 correctness，再做高阶优化

如果颜色空间、参数契约、文件格式、mask 语义都不稳定，那么后续所有调参与质量判断都会失真。

### 2.2 先建立高保真输入输出，再提升控制质量

preprocess 和 generate 之间目前存在明显的 fidelity 损失。  
在把中间信号改成 lossless 之前，不应该过早下结论说某条控制信号“没有价值”。

### 2.3 H200 的预算应优先用于质量，而不是保守省显存

单张 H200 141GB 的价值应优先换成：

- 更高 `sample_steps`
- 更多实验组合
- 更强 continuity
- 更高保真条件和输出

而不是继续沿用“单卡默认 offload”的低显存策略。


## 3. 实施阶段

### 阶段 A：基础设施与契约

- Step 01: 基线、评测与运行清单
- Step 02: correctness 与接口契约修复
- Step 03: lossless 中间结果与输出管线
- Step 04: H200 运行路径与静态条件缓存

### 阶段 B：提升 preprocess 控制信号质量

- Step 05: 稳定 face / pose 控制
- Step 06: 升级 SAM2 mask 生成
- Step 07: 软 mask 与边界感知 replacement
- Step 08: clean plate 背景构建

### 阶段 C：提升 generate 连续性与控制能力

- Step 09: clip continuity 与 seam blending
- Step 10: guidance 解耦与参数搜索

### 阶段 D：高风险探索项

- Step 11: latent temporal memory
- Step 12: 参考图尺度归一化


## 4. 推荐实施顺序

建议严格按下面顺序推进：

1. `01-baseline-evaluation-and-manifest.md`
2. `02-correctness-and-interface-contract.md`
3. `03-lossless-intermediate-and-output-pipeline.md`
4. `04-h200-runtime-and-static-condition-caching.md`
5. `05-stabilize-face-and-pose-controls.md`
6. `06-upgrade-sam2-mask-generation.md`
7. `07-soft-mask-and-boundary-aware-replacement.md`
8. `08-clean-plate-background-pipeline.md`
9. `09-clip-continuity-and-seam-blending.md`
10. `10-guidance-decoupling-and-parameter-search.md`
11. `11-latent-temporal-memory-exploration.md`
12. `12-reference-image-normalization.md`


## 5. 为什么是这个顺序

### 5.1 Step 01 必须先做

没有稳定评测集、统一命名和运行清单，后续所有“优化”都无法可靠比较。

### 5.2 Step 02 和 Step 03 是所有后续工作的前提

这些步骤负责解决：

- 参数与接口不一致
- 颜色空间和 mask 语义不清晰
- lossless 信号无法传递

如果这两步不先完成，后面的 soft mask、背景构建、seam blending 都会受到限制。

### 5.3 Step 04 提前做，是为了提升后续实验效率

H200 相关优化和静态条件缓存本身不一定直接提画质，但它会显著降低后续实验成本，因此应尽早完成。

### 5.4 Step 05 到 Step 08 是 preprocess 主线

这四步对应控制信号质量的核心来源：

- pose
- face
- mask
- background

### 5.5 Step 09 和 Step 10 是 generate 主线

当前 generate 最大瓶颈不是 prompt，而是：

- clip continuity
- overlap 处理
- guidance 耦合

### 5.6 Step 11 和 Step 12 是明确的高风险项

它们可能带来大收益，但不应提前做，否则会在基础问题未解决时把复杂度快速抬高。


## 6. 每一步文档的阅读方式

每个步骤文档都包含：

- 为什么现在做这一步
- 这一步改哪些代码与接口
- 具体如何实施
- 如何验证
- 验收标准
- 风险与回退策略

推荐阅读方式是：

1. 先读本文件，理解顺序和依赖
2. 每次只推进一个步骤
3. 只有当前步骤达到验收标准后，才进入下一个步骤


## 7. 交付标准

整个路线的交付标准不是“文档写完”，而是：

- 每一步都有明确代码变更范围
- 每一步都有可重复的验证方法
- 每一步结束后，都能回答“是否值得继续”

如果某一步无法通过验收，应先停在该步修正，而不是继续往下叠复杂度。
