# Optimization3 Benchmark

## 1. 目标

本目录用于支撑 `optimization3` 路线中的“高精度 benchmark 与质量 gate”。

与 `optimization2` 的验证目录不同，这里不仅关心：

- 系统能否跑通
- artifact 契约是否正确
- smoke 是否成功

还要为后续步骤提供：

- 高精度 benchmark manifest
- 标注 schema
- gate policy
- baseline 结果的稳定记录方式


## 2. 目录内容

- `benchmark_manifest.example.json`
  - benchmark case 的示例清单
- `label_schema.example.json`
  - 关键帧标注结构定义
- `gate_policy.step01.json`
  - Step 01 的 gate 判定规则
- `01-high-precision-benchmark-and-gates-findings.md`
  - Step 01 的正式执行结论与 baseline 指标
- `02-h200-multistage-multires-preprocess-findings.md`
  - Step 02 的正式执行结论、多轮 benchmark 结果与冻结判断
- `03-boundary-fusion-alpha-uncertainty-system-findings.md`
  - Step 03 的正式执行结论、多轮对照与 uncertainty 冻结判断


## 3. Benchmark 分层

### 3.1 Smoke

用途：

- 快速回归
- 每一步开发后的必跑检查

特点：

- 10 秒以内
- 单人 replacement
- 覆盖基本可运行链路
- 默认采用“可持续运行”的基线配置：
  - 保留 `structure_match`
  - 保留 `soft_band`
  - 保留 `heuristic parsing/matting`
  - `boundary_fusion` 默认采用 `v2`
  - 默认使用 `clean_plate_image`
  - 默认关闭重型 QA 视频写盘，只保留结构化 JSON 指标产物

说明：

Step 01 的目标是建立可靠 gate，而不是默认把最重的高精度配置塞进每次 smoke。  
`clean_plate_video`、更高分析分辨率和更重 QA 可视化应放在扩展 benchmark 或专项对照里运行。

### 3.2 Labeled

用途：

- 真正的高精度评估

特点：

- 提供关键帧标注
- 重点评估：
  - 边界
  - 脸
  - 动作

### 3.3 Stress

用途：

- 检查长时序 drift
- 检查高动作和背景复杂度


## 4. 当前基线来源

当前 `optimization3` 的 smoke proxy 基线，统一参考：

- `runs/optimization3_validation_core_20260407_v3`
- `docs/optimization3/benchmark/01-high-precision-benchmark-and-gates-findings.md`

后续 `optimization3` 步骤若没有特别说明，默认都应至少对齐并尽量超过 Step 01 的 proxy 基线。


## 5. 推荐脚本

### 5.1 构建和提取 benchmark 资产

```bash
python scripts/eval/extract_benchmark_keyframes.py \
  --manifest docs/optimization3/benchmark/benchmark_manifest.example.json \
  --output_dir runs/optimization3_benchmark_bootstrap
```

### 5.2 跑 validation suite

```bash
python scripts/eval/run_optimization3_validation_suite.py \
  --tier core \
  --manifest docs/optimization3/benchmark/benchmark_manifest.example.json \
  --gate_policy docs/optimization3/benchmark/gate_policy.step01.json
```

### 5.3 单独生成 gate

```bash
python scripts/eval/summarize_optimization3_validation.py \
  --summary_json runs/<suite>/summary.json \
  --gate_policy docs/optimization3/benchmark/gate_policy.step01.json \
  --output_json runs/<suite>/gate_result.json
```


## 6. Step 01 验收要求

Step 01 结束时，至少应满足：

1. benchmark manifest 已建立
2. smoke / labeled / stress 三层 case 已登记
3. keyframe extraction 可运行
4. `optimization3` suite 可运行
5. suite 能产出：
   - `summary.json`
   - `summary.md`
   - `gate_result.json`
6. baseline 能在当前代码上成功生成
