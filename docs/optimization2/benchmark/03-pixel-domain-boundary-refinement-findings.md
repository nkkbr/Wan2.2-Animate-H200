# Step 03 Findings: Pixel-Domain Boundary Refinement

## Scope

This note records the concrete findings from executing Step 03 in `docs/optimization2/steps/03-pixel-domain-boundary-refinement.md`.

The purpose of this file is to document what the first pixel-domain refinement pass actually achieved, what it did not achieve yet, and what should be treated as the correct baseline for future edge-quality work.

## What Was Added

Step 03 produced four durable engineering outcomes:

1. A new deterministic pixel-domain boundary refinement module:
   - `wan/utils/boundary_refinement.py`
2. A new optional generate-time refinement path controlled by:
   - `--boundary_refine_mode`
   - `--boundary_refine_strength`
   - `--boundary_refine_sharpen`
   - `--boundary_refine_use_clean_plate`
3. Debug artifacts for before/after comparison, alpha maps, boundary masks, and keyframe crops
4. A reusable metric script:
   - `scripts/eval/compute_boundary_refinement_metrics.py`

These are now the required entry points for any future pixel-domain edge-quality iteration.

## Stable Facts Established

### 1. Generate baseline remains cleanly switchable

When `--boundary_refine_mode none` is used, the current generate path completes without entering the new refinement branch.

This preserves the existing Wan-Animate replacement baseline and satisfies the Step 03 requirement that refinement be strictly opt-in.

### 2. The first refinement implementation is deterministic and localized

The current refinement implementation applies:

- outer-band alpha compositing against a background source
- inner-band unsharp masking

It does **not** rewrite the full frame.

The actual implementation operates on:

- `person_mask`
- `soft_band`
- a selected background source
- the decoded generated frames

This is consistent with the Step 03 design requirement that the first version be deterministic and localized rather than another learned black-box stage.

### 3. Background-source fallback is required in real bundles

On the real mini replacement bundle used for validation, preprocess metadata reported:

- `background_mode = none`

As a result, `boundary_refine_use_clean_plate=True` could not use a clean-plate artifact and the refinement branch correctly fell back to:

- `background_source = source_video`

This is an important practical result:

The new refinement path must not assume that a clean-plate background is always available, even when boundary refinement is enabled.

### 4. The real mini generate smoke passed for both baseline and refined modes

Using the existing mini preprocess bundle:

- `runs/system_validation_generate_20260407_mini_bundle/preprocess`

Two real generate runs completed successfully with `ffv1` output:

1. Baseline:
   - `runs/step03_boundary_refine_none_smoke`
2. Deterministic refinement:
   - `runs/step03_boundary_refine_deterministic_smoke`

Both runs used:

- `frame_num=45`
- `refert_num=9`
- `sample_steps=2`
- `quality_preset=hq_h200`
- `temporal_handoff_mode=hybrid`
- `output_format=ffv1`

This is the main execution result of Step 03:

The pixel-domain refinement path is now integrated into the real generate pipeline and is no longer only a synthetic prototype.

### 5. Debug artifacts are now complete enough for actual edge review

The deterministic refinement run produced a dedicated debug subtree:

- `runs/step03_boundary_refine_deterministic_smoke/debug/generate/boundary_refinement/`

Artifacts include:

- comparison video
- outer band visualization
- inner band visualization
- sharpen alpha visualization
- target foreground alpha visualization
- cropped before/after/background keyframes
- refinement metrics JSON

This satisfies the Step 03 requirement that boundary refinement be visually inspectable rather than only numerically scored.

### 6. Synthetic validation passed and showed the intended behavior

The dedicated synthetic validation script:

- `scripts/eval/check_boundary_refinement.py`

passed the following checks:

- zero-strength passthrough
- edge metric improvement on the synthetic boundary case
- debug artifact export

This establishes that the core refinement operator behaves as designed in a controlled setting.

### 7. The first real-case outcome is mixed, not uniformly positive

Using the before/after metric script:

- `scripts/eval/compute_boundary_refinement_metrics.py`

on the real mini outputs produced:

- `band_gradient_before_mean = 0.3443`
- `band_gradient_after_mean = 0.3396`
- `band_edge_contrast_before_mean = 0.0628`
- `band_edge_contrast_after_mean = 0.0614`
- `halo_ratio_before = 0.2315`
- `halo_ratio_after = 0.2239`

The correct interpretation is:

- halo ratio improved
- edge contrast softened slightly
- boundary-band gradient also softened slightly

This means the current deterministic refinement is doing more **halo suppression / gentler compositing** than **true edge sharpening** on the real mini case.

That is the most important quality finding from Step 03.

### 8. The first implementation has a meaningful runtime cost

The refined mini run reported:

- total generate time: `76.49s`
- boundary refinement time: `31.45s`

By comparison, the baseline mini run reported:

- total generate time: `44.92s`

So the current pixel-domain refinement branch is not yet cheap.

The main reason is that it currently:

- loads aligned source-video frames when clean plate is unavailable
- writes multiple debug artifacts

This is acceptable for an experimental quality path, but it is not yet a low-overhead production default.

## What Step 03 Achieved

Step 03 has now achieved these concrete outcomes:

1. A formal pixel-domain boundary refinement path now exists
2. The path is fully optional and baseline-safe
3. Real generate integration is complete
4. Debug artifacts are good enough for actual edge review
5. A reusable before/after metric script now exists
6. Real `ffv1` before/after validation has been completed

## What Step 03 Did Not Yet Achieve

Step 03 did **not** yet prove that the first deterministic refinement is the final answer to edge quality.

Open items remain:

- the current real-case metrics do not show a net edge-gradient increase
- refinement is still expensive when it falls back to aligned source-video loading
- the current operator is better at reducing halo than at increasing apparent sharpness
- no claim has yet been proven that this version is ready as the default replacement path

So Step 03 should now be treated as:

- complete as the first integrated pixel-domain refinement step
- not complete as the final edge-quality solution

## Recommended Default Operating Policy

Until further tuning exists, the correct policy is:

1. Keep `--boundary_refine_mode none` as the default
2. Use `--boundary_refine_mode deterministic` for controlled comparison runs
3. Judge the current deterministic path primarily as a halo-reduction pass, not as a definitive sharpening pass

## Required Next Actions

Subsequent work should start with:

1. Tune the deterministic operator so that halo reduction does not come with a gradient drop
2. Re-run the real mini benchmark after tuning
3. Compare against future parsing/matting fusion inputs from Step 04
4. If clean plate is available, measure whether `clean_plate` background sources outperform `source_video` fallback

## Usage Notes

For the current synthetic regression:

```bash
python scripts/eval/check_boundary_refinement.py
```

For the current real before/after metric comparison:

```bash
python scripts/eval/compute_boundary_refinement_metrics.py \
  --before runs/step03_boundary_refine_none_smoke/outputs/replacement_smoke_none.mkv \
  --after runs/step03_boundary_refine_deterministic_smoke/outputs/replacement_smoke_refined.mkv \
  --src_root_path runs/system_validation_generate_20260407_mini_bundle/preprocess \
  --output_json runs/step03_boundary_refine_deterministic_smoke/metrics/boundary_refinement_metrics.json
```
