# Recommended Wan-Animate Replacement Inference

This document describes the recommended way to run this fork for
Wan2.2-Animate character replacement on an H200-class machine.

The commands are intentionally path-agnostic. Replace the variables in the
setup block with paths from your own machine.

## Practical Recommendation

Use the stable path:

- high-quality preprocess
- `replacement_conditioning_mode=legacy`
- `boundary_refine_mode=none`
- `replacement_mask_mode=soft_band`
- `overlap_blend_mode=mask_aware`
- `temporal_handoff_mode=pixel`
- `ffv1`/MKV output first, then MP4 only for preview

Do not enable the experimental Optimization7/8/9 branches by default. They are
kept in the repository for reproducibility and research, but they did not pass
the promotion gates for stable edge-quality improvement.

## Inputs You Need

You need:

- the repository root
- the Wan2.2-Animate-14B checkpoint directory
- the Wan2.2-Animate preprocess checkpoint directory
- a driving video
- a reference character image

Expected checkpoint layout:

```text
/path/to/Wan2.2-Animate-14B/
  process_checkpoint/
  ...
```

## Full Recommended Script

Run this from any shell after editing the path variables.

```bash
set -euo pipefail

# Optional: activate your environment.
# conda activate wan

# Optional: set CUDA paths if your environment needs them.
# export CUDA_HOME=/usr/local/cuda-12.4
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Choose one GPU. Change this if you want a different device.
export CUDA_VISIBLE_DEVICES=0

# Repository root. If you are already inside the repo, this is enough.
export REPO_ROOT="$(pwd)"

# Change these paths for your machine.
export GEN_CKPT="/path/to/Wan2.2-Animate-14B"
export PRE_CKPT="$GEN_CKPT/process_checkpoint"
export VIDEO="/path/to/driving_video.mp4"
export REF="/path/to/reference_image.png"

export RUN="infer_$(date -u +%Y%m%d_%H%M%S)"
export OUT="$REPO_ROOT/runs/$RUN"

mkdir -p "$OUT/outputs" "$OUT/debug/generate"

python wan/modules/animate/preprocess/preprocess_data.py \
  --ckpt_path "$PRE_CKPT" \
  --video_path "$VIDEO" \
  --refer_path "$REF" \
  --save_path "$OUT/preprocess" \
  --replace_flag \
  --fps -1 \
  --resolution_area 1280 720 \
  --analysis_resolution_area 1280 720 \
  --analysis_min_short_side 720 \
  --preprocess_runtime_profile h200_extreme \
  --multistage_preprocess_mode h200_extreme \
  --sam_runtime_profile h200_safe \
  --sam_chunk_len 20 \
  --sam_keyframes_per_chunk 4 \
  --sam_prompt_mode mask_seed \
  --no-sam_use_negative_points \
  --sam_reprompt_interval 0 \
  --no-sam_apply_postprocessing \
  --face_analysis_mode heuristic \
  --pose_motion_stack_mode v1 \
  --soft_mask_mode soft_band \
  --boundary_fusion_mode v2 \
  --parsing_mode heuristic \
  --matting_mode heuristic \
  --bg_inpaint_mode video_v2 \
  --reference_normalization_mode structure_match \
  --lossless_intermediate

python scripts/eval/check_animate_contract.py \
  --src_root_path "$OUT/preprocess" \
  --replace_flag \
  --skip_synthetic

python generate.py \
  --task animate-14B \
  --ckpt_dir "$GEN_CKPT" \
  --src_root_path "$OUT/preprocess" \
  --save_file "$OUT/outputs/replacement_best.mkv" \
  --output_format ffv1 \
  --replace_flag \
  --use_relighting_lora \
  --offload_model False \
  --frame_num 77 \
  --refert_num 5 \
  --sample_solver unipc \
  --sample_steps 20 \
  --sample_shift 5.0 \
  --sample_guide_scale 1.0 \
  --replacement_mask_mode soft_band \
  --replacement_conditioning_mode legacy \
  --replacement_mask_downsample_mode area \
  --replacement_boundary_strength 0.5 \
  --replacement_transition_low 0.1 \
  --replacement_transition_high 0.9 \
  --overlap_blend_mode mask_aware \
  --temporal_handoff_mode pixel \
  --boundary_refine_mode none \
  --log_runtime_stats \
  --save_debug_dir "$OUT/debug/generate"

echo "Output: $OUT/outputs/replacement_best.mkv"
```

## Optional MP4 Preview

The recommended primary output is `ffv1`/MKV because it avoids adding extra
compression blur to the replacement boundary. If you need a convenient preview
file, transcode from the MKV:

```bash
ffmpeg -y -i "$OUT/outputs/replacement_best.mkv" \
  -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \
  "$OUT/outputs/replacement_best_preview.mp4"
```

## Why These Settings

The preprocess settings select the strongest stable pipeline that survived the
benchmarking work:

- `h200_extreme` multistage preprocess for better analysis coverage
- `h200_safe` SAM runtime profile for stability
- `boundary_fusion_mode=v2` for richer replacement masks
- `bg_inpaint_mode=video_v2` for video-consistent clean plate background
- `reference_normalization_mode=structure_match` for better reference-driver
  structure alignment
- `lossless_intermediate` to avoid avoidable compression artifacts in
  intermediate artifacts

The generation settings intentionally avoid experimental edge-refinement paths:

- `replacement_conditioning_mode=legacy`
- `boundary_refine_mode=none`

The experimental branches remain useful for research, but they did not produce
a stable promotable improvement in the reviewed gates.

## Settings To Avoid As Production Defaults

Do not enable these by default unless you are intentionally reproducing a
specific experiment:

- `--replacement_conditioning_mode rich_v1`
- `--replacement_conditioning_mode semantic_v1`
- `--replacement_conditioning_mode decoupled_v1`
- `--replacement_conditioning_mode decoupled_v2`
- `--replacement_conditioning_mode core_rich_v1`
- `--boundary_refine_mode deterministic`
- `--boundary_refine_mode v2`
- `--boundary_refine_mode roi_v1`
- `--boundary_refine_mode semantic_v1`
- `--boundary_refine_mode semantic_experts_v1`
- `--boundary_refine_mode local_edge_v1`
- `--boundary_refine_mode roi_gen_v1`
- `--boundary_refine_mode roi_gen_v2`
- `--matting_mode production_v1`
- `--matting_mode external_bmv2`

These paths are documented and reproducible, but they are not the recommended
inference path.
