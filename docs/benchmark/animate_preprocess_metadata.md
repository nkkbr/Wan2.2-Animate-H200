# Wan-Animate Preprocess Metadata Contract

`metadata.json` 是 `preprocess_data.py` 在输出目录中写入的正式接口描述文件。
它的目标不是替代 run manifest，而是定义 preprocess 和 generate 之间的稳定 I/O 契约。

## Version

- `version`: `1`
- `storage_format`: `wan_animate_preprocess_v1`

当前版本兼容现有 `mp4 + png` 目录结构，也支持 Step 03 引入的 `png_seq / npz` lossless 中间格式。

## Top-Level Fields

- `pipeline`: 固定为 `wan_animate_preprocess`
- `mode`: `replacement` 或 `animation`
- `replace_flag`: preprocess 是否按 replacement 模式执行
- `retarget_flag`: preprocess 是否启用 retarget
- `use_flux`: preprocess 是否启用 Flux 编辑
- `frame_count`: 处理后视频帧数
- `fps`: 处理后帧率
- `height`: 处理后视频高度
- `width`: 处理后视频宽度
- `channels`: 当前固定为 `3`
- `color_space`: 当前固定为 `rgb`
- `mask_semantics`: replacement 模式下固定为 `person_foreground`
- `derived_mask_semantics.background_keep_mask`: 当前固定为 `1 - person_foreground`
- `processing.save_format`: preprocess 请求的中间格式
- `processing.lossless_intermediate`: 是否启用了推荐的高保真中间格式布局
- `processing.control_stabilization`: face / pose 稳定化参数快照
- `processing.sam2_mask_generation`: replacement 模式下 SAM2 chunk / prompt / negative-point 参数快照
- `processing.soft_mask`: soft mask / boundary band 参数快照
- `qa_outputs`: 可选的 QA overlay 与曲线产物
  - 例如 `face_bbox_overlay`、`pose_overlay`、`mask_overlay`、`soft_band_overlay`、`sam_prompts_overlay`、`mask_stats`

## `src_files`

`src_files` 描述 generate 侧要消费的产物，以及每个产物的逻辑语义。

每个 artifact 现在都应显式记录：

- `path`
- `type`
- `format`
- `dtype`
- `shape`

如果 artifact 是视频或时序数组，还应记录：

- `frame_count`
- `fps`
- `height`
- `width`
- `channels`

### Required for all animate runs

- `pose`
  - `path`: 例如 `src_pose.mp4` 或 `src_pose/`
  - `type`: `video`
  - `format`: `mp4`、`png_seq` 或 `npz`
  - `frame_count/height/width/channels/color_space/dtype/shape/fps`
- `face`
  - `path`: 例如 `src_face.mp4` 或 `src_face/`
  - `type`: `video`
  - `format`: `mp4`、`png_seq` 或 `npz`
  - `frame_count/height/width/channels/color_space/dtype/shape/fps`
- `reference`
  - `path`: `src_ref.png`
  - `type`: `image`
  - `format`: `png`
  - `height/width/channels/color_space/dtype/shape`
  - `resized_height/resized_width`: generate 在内存中会将参考图 resize 到这个尺寸

### Additional required artifacts for replacement mode

- `background`
  - `path`: 例如 `src_bg.mp4`、`src_bg/` 或 `src_bg.npz`
  - `type`: `video`
  - `format`: `mp4`、`png_seq` 或 `npz`
  - `frame_count/height/width/channels/color_space/dtype/shape/fps`
- `person_mask`
  - `path`: 例如 `src_mask.mp4`、`src_mask/` 或 `src_mask.npz`
  - `type`: `video`
  - `format`: `mp4`、`png_seq` 或 `npz`
  - `frame_count/height/width`
  - `channels`: `1`
  - `stored_channels`: `3` for `mp4`, `1` for `png_seq/npz`
  - `dtype/shape/fps`
  - `value_range`: `[0.0, 1.0]`
  - `stored_value_range`: `[0, 255]` for `mp4/png_seq`, `[0.0, 1.0]` for `npz`
  - `mask_semantics`: `person_foreground`
- `soft_band` (optional)
  - `path`: 例如 `src_soft_band.mp4`、`src_soft_band/` 或 `src_soft_band.npz`
  - `type`: `video`
  - `format`: `mp4`、`png_seq` 或 `npz`
  - `frame_count/height/width`
  - `channels`: `1`
  - `stored_channels`: `3` for `mp4`, `1` for `png_seq/npz`
  - `dtype/shape/fps`
  - `value_range`: `[0.0, 1.0]`
  - `mask_semantics`: `boundary_transition_band`

## Color Contract

- preprocess 内部的 `numpy` 图像/视频数组统一按 `RGB` 理解
- `cv2.imread` 读到的参考图必须立即转换到 RGB
- `decord` 读回的视频帧按 RGB 处理
- generate 阶段的 `prepare_source` / `prepare_source_for_replace` 必须遵守同一契约

## Mask Contract

- `src_mask` 的逻辑语义是 `person_foreground`，无论它被保存成 `mp4`、`png_seq` 还是 `npz`
- generate 内部会基于它派生 `background_keep_mask = 1 - person_foreground`
- 后续如果引入 soft mask，仍然必须沿用同一方向的语义，不允许把 `src_mask` 的定义改成背景区域

## Backward Compatibility

generate 当前仍支持没有 `metadata.json` 的旧目录，但只作为兼容 fallback。

- 如果 `metadata.json` 缺失，generate 会打印 warning
- 新产物目录应始终带 `metadata.json`
- 后续步骤新增的 reader 或存储格式，都必须先扩展 metadata contract
