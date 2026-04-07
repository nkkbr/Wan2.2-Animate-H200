# Optimization2 Benchmark

本目录包含两类内容：

- 分步骤 findings：
  - `01-sam2-preprocess-stability-findings.md`
  - `02-h200-preprocess-runtime-profile-findings.md`
  - `03-pixel-domain-boundary-refinement-findings.md`
  - `04-parsing-matting-fusion-boundaries-findings.md`
  - `05-video-consistent-clean-plate-findings.md`
  - `06-structure-aware-reference-normalization-findings.md`
- 面向整个 optimization2 阶段的系统验证方案：
  - `optimization2_validation_plan.md`

推荐阅读顺序：

1. 先看 `optimization2_validation_plan.md`，理解验证矩阵与执行层次。
2. 再结合各 step findings，定位某一类回归失败对应的具体改动面。

推荐执行入口：

```bash
python scripts/eval/run_optimization2_validation_suite.py --tier core
```

更全面的回归可使用：

```bash
python scripts/eval/run_optimization2_validation_suite.py --tier extended
```
