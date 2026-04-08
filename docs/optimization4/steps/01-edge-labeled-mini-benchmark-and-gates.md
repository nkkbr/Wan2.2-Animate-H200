# Step 01: Edge-Labeled Mini Benchmark And Gates

## 1. 目的

本步骤的目标是建立一个真正面向“最终边缘质量”的小型真实标注基准集，并把它接入现有 benchmark/gate 体系。

这是 `optimization4` 的第一步，因为：

- `optimization3` 的 proxy 指标已经有价值，但仍不足以完全代表真实边缘质量。
- 如果没有真实标注集，后续再多的 generate-side 实验也可能只是围着 proxy 打转。
- 你当前最在意的是“最终轮廓锐度”，这必须有更贴近真值的评价方式。


## 2. 需要解决的问题

当前评测体系的不足主要有：

1. `halo_ratio / band_gradient / edge_contrast` 只能描述边缘 proxy，不能完全代表真实 alpha 质量。
2. `soft_alpha`、`boundary_band`、`uncertainty_map` 是否真的更准确，目前缺少明确真值对照。
3. hair-edge、hand-edge、cloth-edge 这类难边界没有单独评价层。


## 3. 交付目标

本步骤结束时，系统应具备：

1. 一个小型 edge-labeled benchmark mini-set
2. 可重复运行的真实边缘指标脚本
3. 新的 gate policy
4. 与现有 `optimization3` suite 并存、兼容的 benchmark 入口


## 4. 数据设计

### 4.1 样本规模

建议先做一个小而强的集合，而不是追求大而全：

- 视频片段：优先从真实 replacement 场景中选 3 到 5 段
- 每段抽关键帧：6 到 12 帧
- 总关键帧数：20 到 50 帧

### 4.2 关键帧覆盖类型

优先覆盖下列难点：

- 发丝边缘
- 手臂 / 手指边缘
- 肩线 / 脖颈边缘
- 衣摆 / 外套边缘
- 快速动作边缘
- 背景高对比边缘
- 局部遮挡边缘

### 4.3 标注内容

至少包含：

- trimap
- foreground alpha 或简化 alpha
- boundary mask

建议扩展：

- `hair_boundary`
- `hand_boundary`
- `cloth_boundary`
- `face_boundary`


## 5. 具体实施

### 5.1 新增数据与文档结构

建议新增：

- `docs/optimization4/benchmark/`
- `docs/optimization4/benchmark/label_schema.edge_v1.json`
- `scripts/eval/extract_edge_benchmark_keyframes.py`
- `scripts/eval/compute_edge_groundtruth_metrics.py`
- `scripts/eval/summarize_optimization4_validation.py`

### 5.2 数据抽取

基于现有真实素材先抽出关键帧：

- 使用 `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- 结合 preprocess/debug 输出
- 对高难边缘片段做优先抽取

### 5.3 指标定义

最低要求：

- boundary F-score
- trimap error
- alpha MAE / SAD

建议增加：

- hair-edge F-score
- hand-edge F-score
- edge-band gradient proxy
- edge-band contrast proxy

### 5.4 Gate 设计

Step 01 要做的是建立 baseline，而不是追求立刻优化到最好。  
因此本步骤 gate 应以“评测系统建立成功”为主：

- 数据可抽取
- 标注 schema 可验证
- 指标脚本可运行
- 对当前 `optimization3` 最优 bundle 能稳定出分


## 6. 强制验收指标

本步骤必须达成：

1. 至少 20 帧真实关键帧被纳入 mini-set
2. 至少 3 个真实边缘真值指标可跑
3. `optimization3` 最优 bundle 能稳定跑出这些指标
4. 新 gate 结果可自动写入 `gate_result.json`


## 7. 实现-测评-再改闭环

最多 3 轮。

### Round 1

- 建立数据结构
- 跑通 keyframe extraction
- 跑通真值指标脚本

### Round 2

- 修复 schema / 标注 / 路径问题
- 确保 gate 能自动汇总

### Round 3

- 清理指标定义
- 固定 baseline 分数与报告格式

如果到 Round 3 仍无法稳定跑通，则冻结 mini-set v1，记录未完成项，但不得阻塞 Step 02。


## 8. 如何验证

建议至少跑：

1. synthetic contract check
2. keyframe extraction smoke
3. labeled metrics smoke
4. `optimization3` 当前最优 preprocess + generate bundle 的真实出分


## 9. 风险

### 9.1 标注成本高

应对：

- 先做小集
- 优先做关键难边界

### 9.2 指标太复杂导致推进过慢

应对：

- 先保证 `boundary F-score / trimap error / alpha MAE`
- hair/hand 子类指标可延后到 Step 06 强化


## 10. 完成标准

当且仅当下面条件成立时，本步骤可结束：

- edge-labeled mini-set 已存在
- 指标脚本稳定可运行
- baseline 结果被正式记录
- 后续步骤可以直接拿这套数据做 AB 和 gate


## 11. 与后续步骤的关系

这是后续所有“边缘锐度是否真的提升”的判断基础。  
Step 02 到 Step 07 都应优先引用本步骤建立的真实边缘指标，而不是只看 proxy。
