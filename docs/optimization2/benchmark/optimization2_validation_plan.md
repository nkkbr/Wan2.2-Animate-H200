# Optimization2 Validation Plan

## 1. 目标

本轮 `optimization2` 共实现了 6 个高价值增强项：

1. `SAM2` preprocess 稳定性基线
2. H200 preprocess runtime profile
3. pixel-domain boundary refinement
4. parsing / matting / boundary fusion
5. video-consistent clean plate background
6. structure-aware reference normalization

这些改动横跨：

- preprocess
- metadata / artifact contract
- generate
- debug / runtime stats
- synthetic regression
- 真实 10 秒素材 smoke

因此验证方案必须覆盖两类风险：

- 正确性风险
  - 参数是否接到入口
  - 新 artifact 是否可写、可读、可被下游消费
  - 新增的 metadata / runtime 字段是否真实落盘
- 性能与质量风险
  - H200 safe / aggressive profile 是否仍可运行
  - boundary refinement 是否仍然有效
  - clean plate video 与 structure-aware reference 是否在真实素材上保留收益


## 2. 验证原则

### 2.1 使用分层矩阵，而不是单个“大测试”

本方案把验证拆成：

- `L0`: 环境与 CLI
- `L1`: synthetic / contract / regression
- `L2`: preprocess 真实 smoke
- `L3`: generate 真实 smoke
- `L4`: H200 稳定性与 profile 验证

这样做的原因是：

- 如果真实 smoke 失败，可以先判断是入口、artifact 契约还是模型路径问题
- 如果 generate 失败，但 preprocess 正常，则问题更可能在 replacement 条件消费侧

### 2.2 区分 `core` 和 `extended`

为了兼顾回归频率和覆盖率，验证方案分为两档：

- `core`
  - 每次较大代码修改后都应跑
  - 覆盖必须通过的正确性链路
- `extended`
  - 在版本冻结前、准备做质量对比前跑
  - 额外覆盖 H200 profile matrix 和重复稳定性

### 2.3 优先复用已有检查脚本

本 repo 已经有一批针对单点特性的检查脚本：

- `check_sam_prompting.py`
- `check_control_stability.py`
- `check_boundary_refinement.py`
- `check_boundary_fusion.py`
- `check_clean_plate_background.py`
- `check_video_clean_plate_background.py`
- `check_reference_normalization.py`
- `check_soft_mask_pipeline.py`
- `check_animate_contract.py`
- `compute_boundary_refinement_metrics.py`
- `compute_replacement_metrics.py`

本方案会优先编排这些脚本，而不是重新发明一套新的验证逻辑。


## 3. 验证输入

统一使用：

- checkpoint:
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`
- video:
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- reference:
  - `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`

说明：

- 视频只有 10 秒，适合做高频 smoke
- 已经足够覆盖 preprocess / generate 的完整 replacement 链路


## 4. Core 验证矩阵

### 4.1 L0: 环境与入口

目的：

- 确认最近新增的 CLI 参数没有被入口遗漏
- 确认当前 Python 环境可以完整导入主路径

项目：

1. `python -m compileall` 覆盖本轮 touched modules
2. `python generate.py --help`
3. `python wan/modules/animate/preprocess/preprocess_data.py --help`

通过标准：

- 所有命令返回 0
- `generate.py --help` 中包含：
  - `boundary_refine_*`
  - `replacement_mask_*`
  - `guidance_uncond_mode`
  - `temporal_handoff_*`
- `preprocess_data.py --help` 中包含：
  - `sam_runtime_profile`
  - `bg_inpaint_mode`
  - `reference_normalization_mode`
  - `reference_structure_*`

### 4.2 L1: synthetic / contract / regression

目的：

- 验证每一个子系统的最低正确性
- 确保后续真实 smoke 失败时，不是基础逻辑已经坏了

项目：

1. `check_sam_prompting.py`
2. `check_control_stability.py`
3. `check_boundary_fusion.py`
4. `check_clean_plate_background.py`
5. `check_video_clean_plate_background.py`
6. `check_reference_normalization.py`
7. `check_boundary_refinement.py`
8. `check_soft_mask_pipeline.py`

通过标准：

- 所有脚本返回 0
- 不允许出现 synthetic contract 失败或 artifact roundtrip 失败

### 4.3 L2: preprocess 真实 smoke

目的：

- 用真实素材验证：
  - Step 01/02 的稳定 preprocess 基线没有被后续步骤打坏
  - Step 05 和 Step 06 的新条件链能一起工作

配置：

- `sam_runtime_profile=h200_safe`
- `resolution_area=640x360`
- `fps=5`
- `sam_chunk_len=12`
- `sam_keyframes_per_chunk=3`
- `sam_prompt_mode=mask_seed`
- `--no-sam_apply_postprocessing`
- `--no-sam_use_negative_points`
- `--bg_inpaint_mode video`
- `--reference_normalization_mode structure_match`
- `--lossless_intermediate`
- `--export_qa_visuals`

应检查：

1. preprocess 成功退出
2. contract check 通过
3. `metadata.json` 中存在：
  - `processing.background.stats`
  - `processing.reference_normalization.stats`
4. 产物存在：
  - `src_ref_normalized.png`
  - `reference_normalization_preview.png`
  - `background_clean_plate_video.mp4`

### 4.4 L3: generate 真实 smoke

目的：

- 验证 Step 03/04/05/06 在当前代码上的端到端兼容性

应至少跑两条：

1. `boundary_refine_mode=none`
2. `boundary_refine_mode=deterministic`

统一配置：

- `frame_num=45`
- `refert_num=9`
- `sample_steps=4`
- `output_format=ffv1`

需要检查：

1. 两条 generate 都成功退出
2. runtime stats 中存在：
  - `background_mode = clean_plate_video`
  - `reference_normalization_mode = structure_match`
3. refined run 的 debug 目录下存在：
  - `boundary_refinement/`
  - `wan_animate_runtime_stats.json`

### 4.5 L3.5: 客观指标

目的：

- 不只验证“跑通”，还要量化 boundary refinement 的行为

项目：

1. `compute_replacement_metrics.py`
   - 对 refined output 计算：
     - `seam_score`
     - `background_fluctuation`
     - `mask_area`
2. `compute_boundary_refinement_metrics.py`
   - 对 `none` vs `deterministic` 做 before/after 指标：
     - `halo_ratio`
     - `band_edge_contrast`
     - `band_gradient`

通过标准：

- 指标脚本返回 0
- refined run 的 `halo_ratio_after <= halo_ratio_before`
- 若 `band_gradient_after` 略降，可以接受，但需要在报告中明确写出


## 5. Extended 验证矩阵

### 5.1 H200 preprocess profile matrix

目的：

- 验证 Step 02 的 profile 逻辑仍然成立

项目：

- `run_preprocess_profile_benchmark.py`
  - `profile = h200_safe`
  - `profile = h200_aggressive`
  - `repeat = 1`

通过标准：

- `h200_safe` 必须成功
- `h200_aggressive` 若成功，记为实验 profile 可用
- 若失败，不应影响 `h200_safe` 作为推荐默认

### 5.2 重复稳定性

目的：

- 验证 Step 01 的“可重复通过”没有回归

项目：

- `run_preprocess_stability_smoke.py`
  - `preset = stable_h200_safe`
  - `repeat = 2` 或 `3`

通过标准：

- 所有重复 run 成功
- contract check 全部通过


## 6. 推荐执行方式

### 6.1 Core

```bash
python scripts/eval/run_optimization2_validation_suite.py --tier core
```

### 6.2 Extended

```bash
python scripts/eval/run_optimization2_validation_suite.py --tier extended
```


## 7. 产物要求

每次 suite 执行后，应至少产出：

- `summary.json`
- `summary.md`
- 各子命令的 `stdout/stderr` 日志
- 若执行了真实 preprocess / generate：
  - 对应的 run 目录
  - metrics JSON

推荐落盘位置：

- `runs/<suite_name>/...`


## 8. 成功标准

本轮 optimization2 代码若要判定为“当前版本通过验证”，至少应满足：

1. Core 全绿
2. 真实 preprocess smoke 全绿
3. 真实 generate smoke 全绿
4. `background_mode=clean_plate_video` 与 `reference_normalization_mode=structure_match` 在 runtime 中均有记录
5. `compute_boundary_refinement_metrics.py` 可对 before/after 完整出数

若 Extended 也通过，则可进一步判定：

- 当前版本已经在 H200 上具备较好的 preprocess 稳定性和 profile 可比性


## 9. 当前已知限制

即使验证全部通过，也不意味着：

- 当前代码已经完全榨干 H200
- 当前边缘已经达到最终极质量

验证通过只能说明：

- 当前这 6 个 step 的工程交付是闭环的
- 新增条件链、背景链、结构链和边界链都能正确协同工作
- 后续继续优化时，有可靠的回归基线可以依赖
