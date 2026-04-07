# Wan-Animate Preprocess Metadata Contract

`metadata.json` 是 `preprocess_data.py` 在输出目录中写入的正式接口描述文件。  
它的目标不是替代 run manifest，而是定义 preprocess 和 generate 之间的稳定 I/O 契约。

## Version

- `version`: `1`
- `storage_format`: `wan_animate_preprocess_v1`

当前版本兼容现有 `mp4 + png` 目录结构，并为后续 lossless/soft-mask 扩展预留了空间。

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

## `src_files`

`src_files` 描述 generate 侧要消费的产物，以及每个产物的逻辑语义。

### Required for all animate runs

- `pose`
  - `path`: `src_pose.mp4`
  - `type`: `video`
  - `frame_count/height/width/channels/color_space`
- `face`
  - `path`: `src_face.mp4`
  - `type`: `video`
  - `frame_count/height/width/channels/color_space`
- `reference`
  - `path`: `src_ref.png`
  - `type`: `image`
  - `height/width/channels/color_space`
  - `resized_height/resized_width`: generate 在内存中会将参考图 resize 到这个尺寸

### Additional required artifacts for replacement mode

- `background`
  - `path`: `src_bg.mp4`
  - `type`: `video`
  - `frame_count/height/width/channels/color_space`
- `person_mask`
  - `path`: `src_mask.mp4`
  - `type`: `video`
  - `frame_count/height/width`
  - `channels`: `1`
  - `stored_channels`: `3`
  - `value_range`: `[0.0, 1.0]`
  - `stored_value_range`: `[0, 255]`
  - `mask_semantics`: `person_foreground`

## Color Contract

- preprocess 内部的 `numpy` 图像/视频数组统一按 `RGB` 理解
- `cv2.imread` 读到的参考图必须立即转换到 RGB
- `decord` 读回的视频帧按 RGB 处理
- generate 阶段的 `prepare_source` / `prepare_source_for_replace` 必须遵守同一契约

## Mask Contract

- `src_mask.mp4` 的逻辑语义是 `person_foreground`
- generate 内部会基于它派生 `background_keep_mask = 1 - person_foreground`
- 后续如果引入 soft mask，仍然必须沿用同一方向的语义，不允许把 `src_mask` 的定义改成背景区域

## Backward Compatibility

generate 当前仍支持没有 `metadata.json` 的旧目录，但只作为兼容 fallback。

- 如果 `metadata.json` 缺失，generate 会打印 warning
- 新产物目录应始终带 `metadata.json`
- 后续步骤新增的 reader 或存储格式，都必须先扩展 metadata contract
