# Step 01: Reviewed Edge Benchmark Expansion And Governance

## 1. 为什么要做

前几轮最重要的教训之一是：

> 如果 benchmark 和旧 baseline 仍然存在较强同源性，那么很多训练型或新模型路线即使失败了，也不一定能说明它们真的没有价值；反过来，即使看起来变好，也未必代表真的超过旧路线。

因此，第二档路线的第一步必须是：

- 扩容 reviewed edge benchmark
- 提高数据独立性
- 把数据治理本身做成正式流程


## 2. 目标

把当前 reviewed edge benchmark 升级成更可信的 `v3` 数据体系，使其：

1. 帧数显著增加
2. 场景更丰富
3. 人工修订更独立
4. semantic boundary 标注更明确
5. split 与版本管理可追踪


## 3. 要改哪些模块

建议至少新增或扩展：

- `docs/optimization8/benchmark/benchmark_manifest.step01.v3.json`
- `docs/optimization8/benchmark/label_schema.edge_reviewed_v3.json`
- `docs/optimization8/benchmark/data_governance.step01.md`
- `scripts/eval/extract_reviewed_edge_keyframes_v3.py`
- `scripts/eval/check_reviewed_edge_dataset_v3.py`
- `scripts/eval/compute_reviewed_edge_metrics_v3.py`


## 4. 具体如何做

### 4.1 扩容关键帧

至少覆盖：

- 静态镜头
- 快动作镜头
- 发丝复杂镜头
- 手部复杂镜头
- 服装复杂纹理镜头
- 遮挡严重镜头

建议总规模至少比当前 reviewed set 增加 `2x`。

### 4.2 强化标注对象

除原有边界外，新增或强化：

- hair edge
- face contour
- ear / neck transition
- hand / finger edge
- cloth boundary
- occlusion boundary
- trimap unknown
- semi-transparent boundary

### 4.3 数据治理

每一帧都应明确：

- 来源视频
- 抽帧方式
- 初始自动标签来源
- 人工修订轮次
- 审核人 / 审核状态
- split 归属


## 5. 如何检验

### 5.1 数据完整性

- schema 正确
- 文件路径完整
- 每一类 semantic boundary 至少有基本覆盖

### 5.2 独立性检查

需要做最小独立性审查：

- 不允许大规模直接复用旧 baseline 输出而不修订
- 至少一部分帧要由人工直接重画或强修

### 5.3 benchmark 稳定性

新的 reviewed set 必须能稳定支撑：

- `boundary_f1`
- `alpha_mae`
- `trimap_error`
- `hair_edge_quality`
- semantic boundary metrics


## 6. 强指标要求

建议 gate：

- 样本量至少达到当前 reviewed set 的 `2x`
- 每类 semantic boundary 至少达到最低覆盖阈值
- 至少 `30%` 以上样本来自人工强修订而非轻微修补


## 7. 实现-测评-再改闭环

### Round 1

- 扩容一批关键帧
- 建 schema 和 governance 文档
- 跑数据完整性检查

### Round 2

- 根据覆盖不足补充类别
- 增加人工强修订样本
- 跑独立性和稳定性检查

### Round 3

- 冻结 `reviewed_edge_v3`
- 输出正式 benchmark manifest


## 8. 成功标准

- `reviewed_edge_v3` 正式形成
- 样本量和类别覆盖显著优于旧版
- 数据治理可追踪、可复现、可审核
- 为后续训练路线提供更强 GT
