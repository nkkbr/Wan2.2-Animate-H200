# Step 02: External Alpha / Matting Model Vetting and Integration

## 1. 为什么要做

`optimization5` 已经证明：

- 当前 heuristic-compatible alpha 路线可以产出丰富 artifact；
- 但它没有能力在 reviewed benchmark 上显著打赢现有高质量 baseline；
- 继续只在现有 heuristic alpha 上调参数，回报非常低。

因此，下一步必须接入真正更强的外部 alpha / matting 模型。

但这里最大的风险不是“模型不好”，而是：

> 选错模型、拿错权重、许可证不兼容、推理接口不可控、结果语义不清晰。

所以这一步不是单纯“接一个模型”，而是要建立 **外部模型选型、下载、验真、接入、回退** 的正式规范，并在此基础上完成第一批候选 AB。


## 2. 目标

建立一个可复用的外部 alpha / matting 模型接入框架，并完成第一批候选模型的 reviewed benchmark AB。

本步骤必须交付：

- 外部模型 candidate registry
- 统一 adapter 接口
- 权重来源与 hash 记录
- synthetic correctness
- reviewed benchmark AB
- 真实 10 秒 smoke
- 通过 gate 的候选，才允许接入主 pipeline


## 3. 候选模型范围

不要在文档阶段绑定单一模型名，而应按任务类型管理候选：

- portrait matting candidate
- video matting candidate
- hair-aware alpha candidate

每类至少准备 `2-3` 个候选。

如果后续实际接入某个具体模型，必须在 benchmark findings 中明确记录：

- 模型名称
- 来源 repo / release / tag / commit
- 权重文件名
- hash
- license


## 4. 具体要做什么

### 4.1 建 registry

建议文件：

- `docs/optimization6/benchmark/external_model_registry.step02.json`

每个模型记录：

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

### 4.2 建统一 adapter 层

每个模型都必须通过独立 adapter 接入：

- `wan/modules/animate/preprocess/external_alpha_<model>.py`

adapter 负责：

- 下载与加载
- 输入预处理
- 推理
- 输出转换成项目统一语义
- runtime 统计
- provenance 写入 metadata

### 4.3 建统一 artifact 语义

接入后统一产出：

- `soft_alpha_v3`
- `trimap_unknown_v3`
- `hair_alpha_v3`
- `alpha_confidence_v3`
- `alpha_source_provenance_v3`

### 4.4 评测流程

每个候选必须依次经过：

1. synthetic correctness
2. reviewed benchmark GT metrics
3. 真实 10 秒 preprocess smoke
4. 与当前 baseline 的 AB 对照


## 5. 如何做

### 5.1 实现顺序

1. 先做 registry 和 adapter 接口
2. 再接第一个 portrait matting 候选
3. 接第二个 candidate 做 AB
4. 只保留赢家进入真实 smoke
5. 通过 gate 后再接入 preprocess 主链开关

### 5.2 命令与运行组织

建议新增：

- `check_external_alpha_model.py`
- `run_external_alpha_benchmark.py`
- `evaluate_external_alpha_benchmark.py`

### 5.3 metadata 记录

metadata 至少新增：

- `external_alpha_model_id`
- `external_alpha_source_url`
- `external_alpha_release`
- `external_alpha_sha256`
- `external_alpha_license`
- `external_alpha_runtime_profile`


## 6. 如何检验

### 6.1 correctness

- adapter 输入输出 shape 正确
- 色彩空间不出错
- 输出数值范围正确
- provenance 写入 metadata

### 6.2 GT 指标

必须在 reviewed benchmark 上比当前 baseline 至少有一项显著更好。

建议硬 gate：

- `alpha_mae` 降低至少 `15%`
- `trimap_error` 降低至少 `15%`
- `hair_edge_quality` 提升至少 `10%`

至少满足其中两条，才允许进入真实 smoke。

### 6.3 smoke

- 真实 10 秒 preprocess 跑通
- contract 检查通过
- generate 最小 smoke 不崩


## 7. 三轮闭环规则

### Round 1

- 建 registry + adapter 接口
- 接第一个候选
- 跑 synthetic correctness

### Round 2

- 增加候选并做 reviewed benchmark AB
- 保留赢者

### Round 3

- 对最优候选做真实 smoke + gate
- 冻结 best-of-three candidate


## 8. 成功标准

Step 02 完成至少要满足：

- 建立外部模型 registry；
- 至少有一个外部 alpha / matting 候选通过 reviewed benchmark gate；
- 至少有一个候选通过真实 10 秒 preprocess smoke；
- 进入主链的路径可显式开关、可回退、可复现。


## 9. 风险与回退

### 风险

- 模型 license 不可接受
- 权重来源不可复现
- 输出语义与项目语义不一致
- 外部模型在 reviewed benchmark 上根本不赢

### 回退

- 若所有候选都失败，则不把任何外部模型接入默认主线；
- registry 和 adapter 层仍保留，为下一批候选继续使用。
