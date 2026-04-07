# Step 01 Findings: SAM2 Preprocess Stability

## Scope

This note records the concrete findings from executing Step 01 in `docs/optimization2/steps/01-sam2-preprocess-stability.md`.

The purpose of this file is not to redefine the step. It is to capture what was actually learned so subsequent optimization work starts from an accurate technical baseline.

## What Was Added

The Step 01 execution produced four durable debugging assets:

1. Parameterized SAM2 runtime profiles in preprocess
2. Chunk-level JSON tracing around `init_state_v2`, `add_new_points`, `add_new_mask`, and propagation
3. A real-video preprocess smoke harness:
   - `scripts/eval/run_preprocess_stability_smoke.py`
4. A direct subprocess-based minimal interaction reproducer:
   - `scripts/eval/repro_sam2_interaction_crash.py`

These assets are now the required entry points for any future SAM2 stability work.

## Fixed Inputs Used

- Video:
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- Reference image:
  - `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`
- SAM2 checkpoint:
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint/sam2/sam2_hiera_large.pt`

## Stable Facts Established

### 1. There is now at least one reproducibly passing real-video preprocess configuration

The following configuration passed the real 10-second preprocess smoke three times consecutively:

- profile: `h200_safe`
- `--no-sam_apply_postprocessing`
- `--sam_prompt_mode mask_seed`
- `--no-sam_use_negative_points`
- `--sam_reprompt_interval 0`
- extremely conservative prompt thresholds
- `resolution_area=640x360`
- `fps=5`
- `sam_chunk_len=12`
- `sam_keyframes_per_chunk=3`

Artifacts were fully produced on every run, including:
- `metadata.json`
- `src_mask`
- `src_bg`
- `mask_overlay`
- `sam2_debug/latest_state.json`

In addition:
- a larger low-load H200-safe case at `832x480`, `fps=5`, `sam_chunk_len=20` also completed successfully

This is the main positive result of Step 01. The preprocess chain is no longer “always unstable on real video.”

### 2. The crash is still reproducible outside the conservative stable envelope

The failure is not intermittent enough to be considered flaky noise.

It reproduces under:
- full preprocess smoke
- reduced-load preprocess smoke
- direct two-frame minimal interaction repro

### 3. The crash happens after state initialization in the failing preprocess path

The trace data shows:
- `init_state_v2` completes
- image features are warmed up
- the first interaction call is entered
- the process segfaults before control returns to Python

This eliminates the earlier broad hypothesis that the crash could be anywhere in the preprocess pipeline.

### 4. The crash is not explained only by prompt complexity

The failure reproduces under:
- dense prompt sets
- single-point fallback prompts
- `mask_seed` interaction mode

So the issue is not simply:
- too many positive points
- negative points
- reprompt cadence
- prompt planning bugs

### 5. The crash is not explained only by postprocessing

The direct repro still segfaults when `apply_postprocessing=False`.

This means the issue is not caused only by:
- hole filling
- dynamic multimask fallback
- other predictor postprocessing overrides

### 6. The crash is not explained only by `init_state_v2`

The direct repro also segfaults when using the upstream vendor predictor path:
- `sam2.build_sam.build_sam2_video_predictor`
- `predictor.init_state(<jpeg_dir>)`
- first `add_new_mask(...)`

This is an important reduction:

- the issue is not specific to the custom `init_state_v2` wrapper
- the issue survives when using the upstream predictor entrypoint

### 7. The crash is not explained only by frame offload mode

The direct repro segfaults with:
- `offload_video_to_cpu=True`
- `offload_video_to_cpu=False`

So CPU frame offload is not the primary trigger.

### 8. The crash is not explained only by the H200-safe profile alone

The direct repro also segfaults under `h200_aggressive`.

This means the current instability is not simply “the safe profile is wrong.” The vendor interaction path is still unstable across multiple runtime modes.

## Current Root-Cause Boundary

The strongest statement supported by current evidence is now:

- there exists a conservative, H200-safe preprocess configuration that is reproducibly stable on the 10-second real video smoke case
- richer or more isolated SAM2 interaction paths can still segfault in the vendored stack

For the failing cases, the native crash boundary remains:

The crash is inside the vendored SAM2 interaction/inference path that runs after state initialization and before the first interaction returns to Python.

In practical terms, the fault boundary is now:
- downstream of preprocess prompt planning
- downstream of custom chunk scheduling
- downstream of custom `init_state_v2`
- upstream of any successful Python-level interaction result

That is the correct technical baseline for Step 02 work.

## What Step 01 Achieved

Step 01 has now achieved these concrete outcomes:

1. A real reproducible failing case
2. Per-chunk JSON tracing and session-level last-known-good state
3. Runtime profile parameterization
4. A direct minimal SAM2 interaction repro script
5. A reproducibly passing H200-safe preprocess smoke configuration
6. Three consecutive successful real-video smoke runs under that stable configuration

## What Step 01 Did Not Yet Achieve

Step 01 still did **not** fully achieve the broader end-state of “problem solved for all practical preprocess loads.”

Open items remain:

- the underlying vendor-level crash mechanism is not yet fully root-caused
- the isolated minimal interaction repro can still segfault
- a fully completed high-load H200-safe end-to-end preprocess run was not confirmed in this step
- the stable configuration is intentionally conservative and is not yet the final quality-oriented preprocess profile

So Step 01 should now be treated as:

- complete as a stability triage and smoke-gate step
- not complete as the final SAM2 robustness solution

## Required Next Actions

Subsequent work should start with:

1. Vendor-level SAM2 investigation
   - inspect the exact SAM2 build / extension state loaded at runtime
   - compare against a known-good environment
2. Minimal reproduction outside the full preprocess stack
   - continue using `scripts/eval/repro_sam2_interaction_crash.py`
3. Version / build compatibility audit
   - torch
   - CUDA runtime
   - compiled SAM2/native extensions
4. If a known-good vendor revision exists, validate it first in the minimal repro before reintroducing the full preprocess path

## Usage Notes

For future debugging, start with:

```bash
python scripts/eval/repro_sam2_interaction_crash.py \
  --backend upstream \
  --interaction mask \
  --profile h200_safe \
  --no-apply_postprocessing \
  --summary_path runs/sam2_repro_upstream_mask.json
```

For stable real-video preprocess validation, use:

```bash
python scripts/eval/run_preprocess_stability_smoke.py \
  --preset stable_h200_safe \
  --repeat 3
```
