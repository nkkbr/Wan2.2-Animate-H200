# Step 02: Winning Alpha Model Production Integration

## 1. 为什么要做

Step 01 解决的是“谁值得接”，但它没有解决：

- 如何进入 preprocess 主链
- 如何写入标准 artifact
- 如何保证 metadata、runtime、fallback、lossless intermediate 都完整

如果 Step 01 的赢家不能被稳定地、可回退地接进主链，那么它就只是一个离散实验，而不是可持续路线。


## 2. 目标

把 Step 01 的赢家正式接入 preprocess 主链，同时保留：

- explicit feature flag
- legacy fallback
- provenance
- contract 兼容
- runtime 统计

并产出 `optimization7` 后续步骤会依赖的新标准 alpha artifacts。


## 3. 要改哪些模块

建议涉及：

- `wan/modules/animate/preprocess/preprocess_data.py`
- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/utils/animate_contract.py`
- `wan/utils/media_io.py`
- `scripts/eval/check_external_alpha_integration.py`
- `scripts/eval/run_optimization7_alpha_integration_smoke.py`


## 4. 具体如何做

### 4.1 定义正式 feature flag

建议新增：

- `--external_alpha_mode {none,<winner_model_id>}`
- `--external_alpha_profile {default,high_quality,extreme}`

### 4.2 统一主链 artifact 语义

正式主链输出建议包括：

- `soft_alpha_v4`
- `trimap_unknown_v4`
- `hair_alpha_v4`
- `alpha_confidence_v4`
- `alpha_source_provenance_v4`

同时保留 legacy artifacts，方便回退和对照。

### 4.3 metadata 与 runtime 接线

metadata 至少应记录：

- `external_alpha_model_id`
- `external_alpha_profile`
- `external_alpha_weight_sha256`
- `external_alpha_license`
- `external_alpha_runtime_sec`
- `external_alpha_input_resolution`

### 4.4 fallback 设计

任何时候只要外部 alpha 路径失败，应能明确回退到 legacy alpha：

- 不能静默混用
- 不能在无记录情况下 fallback
- 必须在 metadata / runtime stats 中体现


## 5. 如何检验

### 5.1 contract correctness

- 新 artifact 路径存在
- metadata 可正确解析
- `check_animate_contract` 扩展检查通过

### 5.2 preprocess smoke

- 真实 10 秒 preprocess 跑通
- 输出 bundle 完整
- legacy 和 external 两条路径都能成功生成 bundle

### 5.3 generate compatibility

- 最小 generate smoke 跑通
- generate 能识别并记录新的 alpha artifact availability


## 6. 强指标要求

相对 Step 01 reviewed winner，本步骤不要求再赢一次 benchmark，但要求：

- 主链 integration 不引入质量退化
- preprocess runtime 增量在可接受范围内
- generate compatibility 为 `PASS`

如果 integration 后真实 smoke 明显退化，应视为 integration 失败，而不是“模型本身成功”。


## 7. 实现-测评-再改闭环

### Round 1

- 先把 feature flag、artifact、metadata 接通
- 跑 correctness + preprocess smoke

### Round 2

- 补 fallback、runtime stats、lossless 路径
- 跑 legacy vs external AB

### Round 3

- 真实 generate compatibility smoke
- 冻结 best-of-three integration 方案


## 8. 成功标准

- 外部 alpha winner 能稳定接入 preprocess 主链
- metadata / runtime / artifact / fallback 全部完整
- generate smoke 不崩
- 没有显著质量回退
