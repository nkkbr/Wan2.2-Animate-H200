# Step 01 Findings: High-Precision Benchmark And Gates

## 1. Result

Step 01 is complete.

The optimization3 benchmark foundation now exists and the core validation gate has been executed successfully on the current codebase.

Passing baseline suite:

- `runs/optimization3_validation_core_20260407_v3`

Gate result:

- `all_cases_passed = true`
- `gate_passed = true`

The suite generated all required outputs:

- `summary.json`
- `summary.md`
- `gate_result.json`


## 2. What Was Delivered

This step produced the following benchmark assets:

- `docs/optimization3/benchmark/README.md`
- `docs/optimization3/benchmark/benchmark_manifest.example.json`
- `docs/optimization3/benchmark/label_schema.example.json`
- `docs/optimization3/benchmark/gate_policy.step01.json`
- `scripts/eval/extract_benchmark_keyframes.py`
- `scripts/eval/compute_boundary_precision_metrics.py`
- `scripts/eval/compute_face_precision_metrics.py`
- `scripts/eval/compute_pose_precision_metrics.py`
- `scripts/eval/compute_background_precision_metrics.py`
- `scripts/eval/summarize_optimization3_validation.py`
- `scripts/eval/run_optimization3_validation_suite.py`

The preprocess pipeline was also adjusted so that compact structured diagnostics are written even when heavy QA video overlays are disabled:

- `face_bbox_curve.json`
- `pose_conf_curve.json`
- `mask_stats.json`

This change is important because it decouples benchmark metrics from expensive overlay generation.


## 3. Benchmark Shape

The benchmark manifest now registers three layers:

- `smoke_cases`
- `labeled_cases`
- `stress_cases`

The labeled layer also has a working keyframe extraction path. The core baseline suite successfully extracted labeled keyframes into:

- `runs/optimization3_validation_core_20260407_v3/labeled_keyframes`

This is enough to support the next step, where manual or semi-manual annotations can be added without redesigning the benchmark structure.


## 4. Baseline Execution Profile

The current Step 01 core suite intentionally uses a sustainable smoke configuration instead of the heaviest possible preprocess path.

Current smoke preprocess profile:

- `sam_runtime_profile = h200_safe`
- `soft_mask_mode = soft_band`
- `reference_normalization_mode = structure_match`
- `boundary_fusion_mode = heuristic`
- `parsing_mode = heuristic`
- `matting_mode = heuristic`
- `bg_inpaint_mode = clean_plate_image`
- heavy QA video overlays disabled
- compact JSON diagnostics retained

This is deliberate. Step 01 is about building a reliable benchmark and gate, not about forcing every smoke run through the most expensive path.


## 5. Core Baseline Numbers

The following proxy numbers were produced by the passing suite at:

- `runs/optimization3_validation_core_20260407_v3`

### 5.1 Replacement / Generate

- `frame_count = 50`
- `duration_sec = 1.6667`
- `seam_score.mean = 8.2586`
- `background_fluctuation.mean = 3.6190`
- `mask_area.mean = 0.1179`

### 5.2 Boundary Refinement

- `halo_ratio_before = 0.0672`
- `halo_ratio_after = 0.0606`
- `band_gradient_before_mean = 0.0639`
- `band_gradient_after_mean = 0.0645`
- `band_edge_contrast_before_mean = 0.0366`
- `band_edge_contrast_after_mean = 0.0321`

Interpretation:

- the current refinement path reduces halo
- gradient is roughly preserved or slightly improved
- edge contrast is still not yet clearly better

This confirms the existing optimization2 conclusion: the current refinement path is useful, but not the final answer for extreme edge sharpness.

### 5.3 Boundary Precision Proxy

- `hard_foreground_mean = 0.1179`
- `hard_foreground_std = 0.3225`
- `soft_alpha_available = true`
- `boundary_band_available = true`
- `uncertainty_available = false`
- `soft_alpha_mean = 0.1250`
- `boundary_band_mean = 0.0296`

Interpretation:

- the benchmark already confirms that soft alpha and boundary band are present
- uncertainty is not yet available in the current pipeline
- this creates an explicit gap for Step 03

### 5.4 Face Precision Proxy

- `center_jitter_mean = 2.7700`
- `center_jitter_max = 9.5131`
- `width_jitter_mean = 0.6531`
- `height_jitter_mean = 1.8776`
- `valid_face_points_mean = 67.82`
- `valid_face_points_min = 59.0`

Interpretation:

- face tracking is reasonably stable in the smoke baseline
- but there is still no labeled landmark NME in the baseline results
- this gap is intentional and should be filled in later benchmark rounds

### 5.5 Pose Precision Proxy

- `raw_body_mean_conf = 0.8485`
- `raw_face_mean_conf = 0.9575`
- `raw_hand_mean_conf = 0.8686`
- `smoothed_body_mean_conf = 0.8358`
- `body_conf_delta_mean = 0.0095`

Interpretation:

- the pose stream is stable enough for Step 01
- the next major jump in motion precision will require richer local-refine and labeled evaluation, not just more proxy logging

### 5.6 Background Precision Proxy

- `background_mode = clean_plate_image`
- `metadata_background_mode = clean_plate_image`
- `visible_support_artifact = None`
- `unresolved_region_artifact = None`

Interpretation:

- the current Step 01 gate intentionally uses the cheaper image clean-plate baseline
- visibility-aware background reasoning is not yet part of the smoke gate
- this should be expanded in later optimization3 steps


## 6. Issues Found During Step 01

Two benchmark issues were found and fixed during execution:

1. `compute_boundary_precision_metrics.py` did not accept the repo's current `.npz` mask key layout (`mask`, `fps`).
2. `summarize_optimization3_validation.py` incorrectly marked `gate_result.json` as missing while writing it.

These were benchmark infrastructure issues, not model-quality regressions.


## 7. Quality Judgment

Step 01 should be considered successful.

Why:

- the benchmark is now formalized
- the manifest is layered correctly
- keyframe extraction works
- the suite runs end-to-end
- the gate logic is executable and trustworthy
- the benchmark produces reusable baseline proxy numbers for later comparison

What Step 01 does **not** prove:

- that the current system has already reached extreme precision
- that current edge quality is good enough
- that uncertainty-aware boundaries or full face/pose labeled precision are solved

Step 01 only proves that the evaluation foundation is now strong enough to support the next optimization stages.


## 8. Required Next Move

The next step should treat the following as explicit baseline targets to beat:

- lower `halo_ratio_after`
- better `band_edge_contrast_after_mean`
- lower `center_jitter_mean`
- lower `body_conf_delta_mean`
- richer boundary signals than the current `soft_alpha + boundary_band` pair

Most importantly:

Step 02 and Step 03 should not just “run and pass”.
They should beat the proxy baseline recorded here by a visible margin, and if they do not, they should loop through the planned 3-round improve-and-retest process.
