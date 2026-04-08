# Optimization5 Benchmark

## 1. 目标

`optimization5` 的 benchmark 重点是把 `optimization4` 的 edge mini-set 升级成更可信的 reviewed edge mini-benchmark，并为后续：

- alpha / matting
- foreground-background decoupling
- ROI generative reconstruction
- trainable edge refinement

提供更硬的 pass / fail gate。


## 2. Step 01 基准

Step 01 使用：

- source video:
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- frozen baseline preprocess:
  - `/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step08_round2_generate_ab_v3/repeat_01/face_focus_highres/preprocess`
- reviewed benchmark manifest:
  - [benchmark_manifest.step01.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/benchmark/benchmark_manifest.step01.json)
- gate policy:
  - [gate_policy.step01.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/benchmark/gate_policy.step01.json)
- label schema:
  - [label_schema.edge_reviewed_v1.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/benchmark/label_schema.edge_reviewed_v1.json)


## 3. 运行入口

### Build + Validate Step 01 benchmark

```bash
python scripts/eval/run_optimization5_validation_suite.py \
  --suite_name optimization5_step01_core
```

### Build reviewed edge dataset only

```bash
python scripts/eval/build_reviewed_edge_benchmark.py \
  --manifest docs/optimization5/benchmark/benchmark_manifest.step01.json \
  --output_dir runs/optimization5_step01_reviewed_edge_dataset
```

### Compute reviewed GT metrics

```bash
python scripts/eval/compute_reviewed_edge_metrics.py \
  --dataset_dir runs/optimization5_step01_reviewed_edge_dataset \
  --prediction_preprocess_dir /home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step08_round2_generate_ab_v3/repeat_01/face_focus_highres/preprocess \
  --output_json runs/optimization5_step01_reviewed_edge_dataset/metrics.json
```


## 4. Reviewed benchmark 输出

reviewed benchmark 每帧至少包含：

- `hard_foreground`
- `soft_alpha`
- `trimap`
- `boundary_mask`
- `occlusion_band`
- `uncertainty_map`
- `boundary_type_map`
- `review_metadata`

`boundary_type_map` 语义：

- `0`: non-boundary
- `1`: face
- `2`: hair
- `3`: hand
- `4`: cloth
- `5`: occluded


## 5. Step 01 Gate

Step 01 至少要求：

- reviewed keyframe count `>= 20`
- reviewed ratio `= 1.0`
- `face / hair / hand / cloth / occluded` 各类覆盖 `>= 3`
- GT metrics / gate 连续 3 次运行稳定一致


## 6. findings

Step 01 的正式结论文档应写在：

- `docs/optimization5/benchmark/01-human-reviewed-edge-benchmark-and-hard-gates-findings.md`

Step 02 的正式结论文档应写在：

- `docs/optimization5/benchmark/02-production-grade-alpha-matting-system-findings.md`

Step 03 的正式结论文档应写在：

- `docs/optimization5/benchmark/03-foreground-background-decoupled-replacement-contract-findings.md`

Step 04 的正式结论文档应写在：

- `docs/optimization5/benchmark/04-rich-boundary-signal-core-generation-conditioning-findings.md`

Step 05 的正式结论文档应写在：

- `docs/optimization5/benchmark/05-boundary-roi-generative-reconstruction-findings.md`

Step 06 的正式结论文档应写在：

- `docs/optimization5/benchmark/06-semantic-roi-experts-for-face-hair-hands-cloth-findings.md`

Step 07 的正式结论文档应写在：

- `docs/optimization5/benchmark/07-trainable-edge-refinement-and-adapter-path-findings.md`

Step 08 的正式结论文档应写在：

- `docs/optimization5/benchmark/08-h200-high-value-compute-allocation-and-orchestration-findings.md`

Step 08 的 tier 配置文件应写在：

- `docs/optimization5/benchmark/quality_tiers.step08.json`
