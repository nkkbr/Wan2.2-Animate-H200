# Step 02 Findings: H200 Preprocess Runtime Profile

## Scope

This note records the concrete findings from executing Step 02 in `docs/optimization2/steps/02-h200-preprocess-runtime-profile.md`.

The purpose of this file is to capture what was actually established by the H200 profile work, not to restate the plan. Subsequent preprocess and boundary-quality work should treat the results in this note as the new operating baseline.

## What Was Added

Step 02 produced five durable engineering outcomes:

1. A first-class preprocess runtime profile mechanism
2. Separation between `analysis_resolution_area` and export `resolution_area`
3. Profile-aware runtime statistics written into preprocess outputs
4. A reusable benchmark driver:
   - `scripts/eval/run_preprocess_profile_benchmark.py`
5. Real benchmark evidence comparing:
   - `legacy_safe`
   - `h200_safe`
   - `h200_aggressive`

These are now the required entry points for any future preprocess throughput or quality profiling work on H200.

## Fixed Inputs Used

- Video:
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- Reference image:
  - `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`
- Process checkpoint root:
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`

## Stable Facts Established

### 1. Preprocess runtime is now explicitly profile-driven

The preprocess entrypoint no longer depends only on scattered flags and hardcoded SAM2 behavior. It now supports:

- `legacy_safe`
- `h200_safe`
- `h200_aggressive`

Each profile controls at least:

- SAM2 runtime profile
- default analysis resolution policy
- chunk length
- keyframe count
- prompt mode
- reprompt cadence
- negative point usage
- SAM2 postprocessing behavior

This is the main structural outcome of Step 02. H200 preprocess behavior is now configurable in a way that can be benchmarked and repeated.

### 2. Analysis resolution is now independent from export resolution

The preprocess pipeline now distinguishes between:

- analysis resolution:
  - the resolution used for pose, face, prompt planning, SAM2, and reference normalization
- export resolution:
  - the resolution used for the saved replacement artifacts

This matters because H200 budget can now be spent on higher-fidelity analysis without blindly increasing final artifact size.

In the successful Step 02 runs:

- export resolution remained `640x352`
- analysis resolution was raised to:
  - `864x480` under `h200_safe`
  - `1280x720` under `h200_aggressive`

That resolution split is now recorded in both:

- `metadata.json`
- `preprocess_runtime_stats.json`

### 3. `h200_safe` is now a reproducibly passing H200 preprocess baseline

The following benchmark passed three consecutive real-video runs:

- profile: `h200_safe`
- analysis resolution target: `832x480`
- actual analysis shape: `864x480`
- export shape: `640x352`
- `fps=5`
- `sam_chunk_len=20`
- `sam_keyframes_per_chunk=4`
- `sam_prompt_mode=mask_seed`
- `sam_reprompt_interval=0`
- `sam_use_negative_points=False`
- `sam_apply_postprocessing=False`

Results were written to:

- `runs/step02_h200_safe_repeat3_v2/summary.json`
- `runs/step02_h200_safe_repeat3_v2/summary.csv`

Observed runtime results:

- run 1 total seconds: `22.58`
- run 2 total seconds: `24.09`
- run 3 total seconds: `34.18`
- peak memory: `2.24 GB`
- mean SAM chunk seconds: approximately `2.09`, `2.26`, `2.27`

Observed QA-side statistics remained stable across the three runs:

- `mask_area_ratio_mean = 0.0833`
- `soft_band_mean = 0.0379`
- `background_diff_mean = 16.06`

This is now the recommended default preprocess profile for H200 quality work.

### 4. `h200_aggressive` can now pass the real 10-second case once

After the first aggressive attempt failed, the runtime profile was adjusted and the following real benchmark passed:

- profile: `h200_aggressive`
- analysis resolution target: `1280x720`
- actual analysis shape: `1280x720`
- export shape: `640x352`
- `fps=5`
- `sam_chunk_len=60`
- `sam_keyframes_per_chunk=6`
- `sam_prompt_mode=mask_seed`
- `sam_reprompt_interval=0`
- `sam_use_negative_points=False`
- `sam_apply_postprocessing=False`

Results were written to:

- `runs/step02_h200_aggressive_once_v2/summary.json`
- `runs/step02_h200_aggressive_once_v2/summary.csv`

Observed runtime results:

- total seconds: `39.99`
- peak memory: `3.40 GB`
- mean SAM chunk seconds: `6.27`

Observed QA-side statistics:

- `mask_area_ratio_mean = 0.0775`
- `soft_band_mean = 0.0386`
- `background_diff_mean = 16.46`

This is enough to treat `h200_aggressive` as a valid experimental profile, but not yet enough to promote it to the default profile.

### 5. The original `legacy_safe` path is no longer the correct H200 baseline

The profile matrix benchmark showed:

- `legacy_safe` failed the real 10-second case
- process exit code: `-11`
- signal: `SIGSEGV`
- manifest status remained `started`

Results were written to:

- `runs/step02_profile_matrix/summary.json`
- `runs/step02_profile_matrix/summary.csv`

This is the most important negative result of Step 02:

The old conservative preprocess path should no longer be treated as the default reference configuration for H200.

For this workload, it is not just lower quality. It is also less reliable.

### 6. The first aggressive kernel attempt failed because the vendored SAM2 attention path could not serve the requested kernel mix

The initial `h200_aggressive` attempt failed with:

- `RuntimeError: No available kernel. Aborting execution.`

The warnings indicated that the desired flash or memory-efficient attention kernels were not available for the actual input dtypes in the vendored SAM2 path.

This is an important boundary:

- the current codebase does not yet support “flip on the most aggressive attention mode and trust H200 to carry it”
- the vendored SAM2 stack still constrains what counts as a safe aggressive runtime profile

The current `h200_aggressive` profile therefore remains more aggressive in analysis geometry and scheduling, but not in raw attention-kernel selection.

That is the correct technical interpretation of the current result.

### 7. Benchmark outputs are now parseable and useful for future sweeps

The benchmark driver now emits:

- per-suite `summary.json`
- per-suite `summary.csv`
- per-run manifest
- per-run preprocess runtime stats
- contract check status
- mask statistics
- soft band statistics
- background difference statistics

This means future sweeps no longer need ad hoc notebooks or manual log scraping just to answer:

- which profile passed
- how long it took
- how much memory it used
- whether the QA artifacts were present

## What Step 02 Achieved

Step 02 has now achieved these concrete outcomes:

1. Preprocess runtime profiles are formally parameterized
2. `h200_safe` and `h200_aggressive` are both implemented
3. Analysis resolution and export resolution are separated
4. Runtime profile selection is written into metadata and runtime stats
5. A reusable preprocess profile benchmark harness now exists
6. `h200_safe` passed the real 10-second case three consecutive times
7. `h200_aggressive` passed the real 10-second case once
8. `legacy_safe` was shown to be the wrong H200 baseline for this workload

## What Step 02 Did Not Yet Achieve

Step 02 did **not** establish that preprocess is now “fully optimized for H200.”

Open items remain:

- `h200_aggressive` has not yet demonstrated repeated stability
- the most aggressive attention-kernel path is still not viable in the vendored SAM2 stack
- no claim has yet been proven that the aggressive profile gives a clearly better subjective boundary result than `h200_safe`
- Step 02 did not attempt to solve vendor-level SAM2 kernel limitations

So Step 02 should now be treated as:

- complete as the H200 preprocess runtime-profile step
- not complete as the final preprocess-performance or final preprocess-quality solution

## Recommended Default Operating Policy

Until stronger evidence exists, the correct policy is:

1. Use `h200_safe` as the default H200 preprocess profile for all subsequent quality experiments
2. Use `h200_aggressive` only for controlled comparison runs
3. Treat `legacy_safe` as a compatibility profile, not as the H200 baseline

In practical terms:

- boundary-quality work should start from `h200_safe`
- future aggressive exploration should be justified by measured quality improvements, not by hardware intuition alone

## Required Next Actions

Subsequent work should start with:

1. Pixel-domain boundary refinement on top of the new `h200_safe` preprocess baseline
2. Subjective and objective comparison of `h200_safe` vs `h200_aggressive` on edge quality
3. If aggressive quality gains appear real, run multi-repeat stability validation before promoting the profile
4. If deeper H200 kernel exploitation is needed, investigate the vendored SAM2 attention path separately from the rest of preprocess

## Usage Notes

For repeated H200-safe validation, use:

```bash
python scripts/eval/run_preprocess_profile_benchmark.py \
  --profile h200_safe \
  --repeat 3 \
  --suite_name step02_h200_safe_repeat3
```

For the current aggressive comparison case, use:

```bash
python scripts/eval/run_preprocess_profile_benchmark.py \
  --profile h200_aggressive \
  --repeat 1 \
  --suite_name step02_h200_aggressive_once
```

For the current matrix comparison, use:

```bash
python scripts/eval/run_preprocess_profile_benchmark.py \
  --profile legacy_safe \
  --profile h200_safe \
  --profile h200_aggressive \
  --repeat 1 \
  --suite_name step02_profile_matrix
```
