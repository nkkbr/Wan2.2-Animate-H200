# Step 02 Findings: External Alpha / Matting Model Vetting and Integration

## 1. 结论

Step 02 的工程目标已经完成，但质量 gate 没有通过。

也就是说：

- 外部 alpha / matting 的 registry、adapter、hash 校验、benchmark 脚本都已经做成；
- 官方来源的 `BackgroundMattingV2` 候选已经正确下载、验真、接入；
- synthetic correctness 通过；
- reviewed benchmark AB 已完成；
- best-of-three 候选也已经具备进入真实 preprocess smoke 的实验入口；
- 但 **当前候选没有在 reviewed benchmark 上打赢现有 baseline**，因此不能升默认主线。

## 2. 这一步实际接入了什么

当前正式落地的是 `BackgroundMattingV2` 两个官方 TorchScript 候选：

1. `backgroundmattingv2_mobilenetv2_fp32`
2. `backgroundmattingv2_resnet50_fp32`

来源：

- repo: `https://github.com/PeterL1n/BackgroundMattingV2`
- release: `v1.0.0`
- license: `MIT`

缓存权重：

- `~/.cache/wan2_2_external_models/backgroundmattingv2/torchscript_mobilenetv2_fp32.pth`
- `~/.cache/wan2_2_external_models/backgroundmattingv2/torchscript_resnet50_fp32.pth`

sha256 已锁定在：

- `docs/optimization6/benchmark/external_model_registry.step02.json`

另外，`MODNet` 已进入 registry，但由于当前没有锁定官方 hash，状态保持为待后续候选，不进入本轮主结果。

## 3. 代码交付

本步骤新增：

- `wan/utils/external_alpha_registry.py`
- `wan/modules/animate/preprocess/external_alpha_backgroundmattingv2.py`
- `scripts/eval/check_external_alpha_model.py`
- `scripts/eval/run_external_alpha_benchmark.py`
- `scripts/eval/evaluate_external_alpha_benchmark.py`

本步骤还修正了：

- `wan/modules/animate/preprocess/__init__.py`
  - 改成 lazy import，避免轻量脚本被重型 preprocess 依赖拖死。
- `wan/modules/animate/preprocess/matting_adapter.py`
  - 新增 `matting_mode=external_bmv2` 的实验路径。
- `wan/modules/animate/preprocess/process_pipepline.py`
  - 外部 alpha 实验路径已可在主 preprocess 中显式开关。
- `wan/modules/animate/preprocess/preprocess_data.py`
  - 暴露：
    - `--matting_mode external_bmv2`
    - `--external_alpha_model_id`
    - `--external_alpha_blend`
    - `--external_alpha_delta_clip`
    - `--external_alpha_hair_boost`

## 4. 三轮闭环

### Round 1

候选：

- `backgroundmattingv2_mobilenetv2_fp32`
- `backgroundmattingv2_resnet50_fp32`

结果：

- mobilenet
  - `alpha_mae_mean = 0.13592268185069165`
  - `trimap_error_mean = 0.18839553681512675`
  - `hair_boundary_f1_mean = 0.016017842660178427`
  - `runtime_sec_mean = 0.6607664504942173`
- resnet50
  - `alpha_mae_mean = 0.13939126332600912`
  - `trimap_error_mean = 0.18247298647960028`
  - `hair_boundary_f1_mean = 0.0`
  - `runtime_sec_mean = 0.45785287123483914`

gate：

- mobilenet
  - `alpha_mae_reduction_pct = -609.28%`
  - `trimap_error_reduction_pct = +35.58%`
  - `hair_edge_quality_gain_pct = -97.20%`
  - `gate_passed = false`
- resnet50
  - `alpha_mae_reduction_pct = -627.38%`
  - `trimap_error_reduction_pct = +37.60%`
  - `hair_edge_quality_gain_pct = -100.0%`
  - `gate_passed = false`

### Round 2

尝试对 mobilenet 做轻微 hair boost：

- `--hair_boost 0.12`

结果：

- `alpha_mae_mean = 0.13631193712353706`
- `trimap_error_mean = 0.18811457697302103`
- `hair_boundary_f1_mean = 0.01502057613168724`
- `runtime_sec_mean = 0.764741328273279`

gate：

- `alpha_mae_reduction_pct = -611.31%`
- `trimap_error_reduction_pct = +35.67%`
- `hair_edge_quality_gain_pct = -97.37%`
- `gate_passed = false`

### Round 3

尝试对 resnet50 做轻微 hair boost：

- `--hair_boost 0.08`

结果：

- `alpha_mae_mean = 0.1397006344050169`
- `trimap_error_mean = 0.18238733926167092`
- `hair_boundary_f1_mean = 0.0`
- `runtime_sec_mean = 0.4825716691945369`

gate：

- `alpha_mae_reduction_pct = -628.99%`
- `trimap_error_reduction_pct = +37.63%`
- `hair_edge_quality_gain_pct = -100.0%`
- `gate_passed = false`

## 5. best-of-three 冻结

best-of-three 冻结为：

- `backgroundmattingv2_mobilenetv2_fp32` Round 1

原因：

- 尽管没有通过强 gate，但它是三轮里唯一保住了非零 `hair_boundary_f1_mean` 的候选；
- `alpha_mae` 也是三轮中的最优值；
- 与 Round 2 相比，Round 1 的 mobilenet 更快、hair 指标也略好。

因此：

- registry 中将其标记为 `best_of_three_failed_gate_pending_smoke`
- 其余当前候选只保留为已评估失败项

## 6. 真实 smoke 判定

Step 02 文档的执行顺序是：

1. synthetic correctness
2. reviewed benchmark AB
3. 只保留赢家进入真实 smoke

本轮所有候选都没有通过 reviewed benchmark gate，因此**没有候选被正式晋级到真实 smoke**。

不过，为了验证工程可行性，代码层已经补齐了：

- `matting_mode=external_bmv2`
- `external_alpha_model_id=<registry model id>`

也就是说：

- external alpha 路径已可以在 preprocess 主链中显式打开；
- provenance / source repo / release / sha / license 都能进入 `matting_stats`；
- 但由于当前候选本身未通过 benchmark gate，本轮不把任何外部模型推进到默认 smoke 阶段。

## 7. 为什么失败

当前 reviewed benchmark 的结果非常一致：

- `BackgroundMattingV2` 的 unknown / trimap 行为比 baseline 更激进，因此 `trimap_error` 有明显改善；
- 但现有 baseline 与 reviewed v2 数据在 alpha 形状上高度贴合，导致外部 alpha 在：
  - `alpha_mae`
  - `hair_boundary_f1`
  上明显打不过 baseline；
- 也就是说，这轮不是“外部模型接不上”，而是“当前外部候选并不适合在当前 benchmark 上取代现有 active alpha”。

## 8. 冻结结论

Step 02 的最终状态是：

- registry：完成
- adapter：完成
- hash / license / provenance：完成
- reviewed benchmark AB：完成
- best-of-three candidate：完成
- experimental preprocess switch：完成
- 真实 smoke 晋级：**不允许**
- 默认主线切换：**不允许**

也就是说：

> Step 02 是一次成功的外部模型 vetting 工程步骤，但不是一次成功的主线质量提升步骤。
