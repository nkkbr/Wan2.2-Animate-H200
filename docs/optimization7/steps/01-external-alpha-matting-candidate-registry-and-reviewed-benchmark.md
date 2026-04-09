# Step 01: External Alpha / Matting Candidate Registry And Reviewed Benchmark

## 1. 为什么要做

到 `optimization6` 为止，我们已经比较明确地知道：

- 当前 heuristic-compatible alpha 路线无法稳定打赢现有基线；
- 继续在旧 alpha 体系内细调参数，收益越来越低；
- 如果 `soft_alpha / trimap / hair_alpha` 本身不够强，后面的 decoupled contract 和 ROI reconstruction 很可能也只是建立在不够好的输入上。

因此，`optimization7` 的第一步必须是：

> 先证明有没有更强的外部 alpha / matting 候选，值得成为新主线的输入基础。

这一步不是“先接个模型看看”，而是要正式建立：

- 候选清单
- 版本与许可证记录
- 权重可复现来源
- adapter 接口
- reviewed benchmark 的客观 AB


## 2. 目标

Step 01 的目标是完成以下闭环：

1. 建立外部 alpha / matting 候选 registry
2. 至少接入 `2-3` 个高质量候选
3. 在 reviewed benchmark 上完成客观 AB
4. 选出一个“值得进入 Step 02 生产接线”的赢家

如果没有任何候选赢 baseline，这一步仍然算完成，但结论必须是：

- 冻结 registry/adapter 基础设施
- 不进入 Step 02
- 更新候选池后再重开新轮


## 3. 范围

本步骤只做：

- 候选模型 registry
- adapter 接口
- keyframe / reviewed benchmark AB
- 最小 smoke

本步骤不做：

- 直接覆盖 preprocess 默认主线
- 深度修改 decoupled contract
- ROI reconstruction


## 4. 要改哪些模块

建议至少新增或扩展：

- `docs/optimization7/benchmark/external_model_registry.step01.json`
- `wan/modules/animate/preprocess/external_alpha_base.py`
- `wan/modules/animate/preprocess/external_alpha_<model>.py`
- `scripts/eval/check_external_alpha_candidate.py`
- `scripts/eval/run_optimization7_external_alpha_benchmark.py`
- `scripts/eval/evaluate_optimization7_external_alpha_benchmark.py`

必要时还应扩展：

- `wan/utils/animate_contract.py`
- `wan/modules/animate/preprocess/preprocess_data.py`
- `wan/modules/animate/preprocess/process_pipepline.py`


## 5. 具体如何做

### 5.1 建 registry

每个候选模型必须记录：

- `model_id`
- `task_type`
- `source_repo`
- `release_or_commit`
- `license`
- `weight_url`
- `sha256`
- `required_dependencies`
- `input_semantics`
- `output_semantics`
- `status`

`status` 建议取值：

- `planned`
- `downloaded`
- `adapted`
- `benchmarked`
- `selected`
- `rejected`

### 5.2 建统一 adapter 接口

每个外部模型都必须通过统一 adapter 暴露：

- `load_model()`
- `prepare_inputs()`
- `infer()`
- `to_project_artifacts()`
- `emit_provenance()`

输出必须统一映射到项目语义，例如：

- `soft_alpha_ext`
- `trimap_unknown_ext`
- `hair_alpha_ext`
- `alpha_confidence_ext`
- `alpha_source_provenance_ext`

### 5.3 synthetic correctness

先不跑真实视频，先验证：

- 输入尺寸变化时不崩
- 输出 shape、dtype、range 正确
- provenance 正常写入
- lossless intermediate 可正确存取

### 5.4 reviewed benchmark AB

每个候选都必须在 Step 01 reviewed benchmark 上和当前稳定基线比较：

- `boundary_f1`
- `trimap_error`
- `alpha_mae`
- `alpha_sad`
- `hair_edge_quality`

### 5.5 最小 smoke

对赢 baseline 的候选，做真实 10 秒最小 smoke：

- preprocess 跑通
- contract 检查通过
- 最小 generate smoke 不崩


## 6. 如何检验

### 6.1 correctness gate

必须全部通过：

- adapter 输入输出 shape 正确
- 色彩空间正确
- 输出数值范围正确
- metadata provenance 完整

### 6.2 reviewed benchmark gate

建议强 gate：

- `alpha_mae` 至少下降 `15%`
- `trimap_error` 至少下降 `15%`
- `hair_edge_quality` 至少提升 `10%`

至少满足其中两条，才允许进入真实 smoke。

### 6.3 smoke gate

- 真实 10 秒 preprocess 跑通
- contract 检查通过
- 最小 generate smoke 跑通


## 7. 实现-测评-再改闭环

### Round 1

- 建 registry 与 adapter 基类
- 接第一个最有希望的候选
- 跑 synthetic correctness

### Round 2

- 增加第 2 / 3 个候选
- 做 reviewed benchmark AB
- 淘汰明显不行的候选

### Round 3

- 对最佳候选做真实 10 秒 smoke
- 冻结 winner 或明确宣布“本轮无赢家”


## 8. 成功标准

本步骤完成至少应满足：

- registry 建立完成
- 至少 `2-3` 个候选被实际 benchmark
- 至少一个候选通过 reviewed gate 与 smoke gate
- 或者清楚地证明“当前候选池没有赢家”


## 9. 失败后怎么办

如果到 Round 3 仍无赢家：

- 不允许强推某个候选进入主线
- 保留 registry 和 adapter 基础设施
- findings 中明确写：
  - 哪些候选为什么失败
  - 是 license / interface / quality / runtime 哪类失败
- 下一轮只能通过“更新候选池”继续，而不是继续盲目调旧候选
