# Step 05 Findings: 视频一致性的 Clean Plate Background

## 1. 本步骤完成了什么

本步骤已经把 replacement preprocess 里的 background 构建链，从“只有 `hole` 和逐帧 `image` inpaint”升级成了正式的三模式体系：

- `hole`
- `clean_plate_image`
- `clean_plate_video`

其中第一版 `clean_plate_video` 不是重型视频修复模型，而是一个保守但可落地的视频一致性背景构建方案：

1. 用 `hard_foreground + soft boundary + background_keep_prior` 共同定义 inpaint region
2. 在局部时间窗口内聚合“当前像素可见的背景观测”
3. 对可见背景不足的区域回退到 `clean_plate_image`
4. 最后仅在 inpaint region 内做轻量时间平滑

这一步的目标不是“完美恢复人物后面所有被遮挡的真实背景”，而是：

- 让背景条件更稳定
- 减少边缘附近的脏边与闪烁
- 为 generate 提供比逐帧 image inpaint 更一致的背景先验


## 2. 代码落点

本步骤新增或修改的关键模块如下：

- 新增：
  - `wan/modules/animate/preprocess/background_clean_plate.py`
    - 重写为正式三模式 background pipeline
  - `scripts/eval/check_video_clean_plate_background.py`
    - synthetic 的 image vs video consistency 检查
- 修改：
  - `wan/modules/animate/preprocess/process_pipepline.py`
  - `wan/modules/animate/preprocess/preprocess_data.py`
  - `wan/animate.py`


## 3. CLI、artifact 与 metadata 变化

### 3.1 preprocess 新参数

本步骤新增：

- `--bg_inpaint_mode {none,image,video}`
- `--bg_video_window_radius`
- `--bg_video_min_visible_count`
- `--bg_video_blend_strength`

其中：

- `image`
  - 保留为低风险 baseline
- `video`
  - 第一版视频一致 clean plate

### 3.2 QA artifact 扩展

现在 replacement preprocess QA 可导出：

- `background_hole`
- `background_clean_plate`
- `background_clean_plate_image`
- `background_clean_plate_video`
- `background_diff`
- `background_temporal_diff`
- `background_inpaint_mask`
- `background_visible_support`
- `background_unresolved_region`

### 3.3 metadata / runtime

`metadata.json` 现在会同时记录：

- `processing.background.bg_inpaint_mode`
- `src_files.background.background_mode`
- `processing.background.stats`

generate runtime stats 现在也会记录实际 background artifact 来源，例如：

- `background_mode = clean_plate_video`

此外，真实 rerun 已确认 `preprocess_runtime_stats.json` 里也会直接写出：

- `background.mode = clean_plate_video`
- 对应的 `temporal_fluctuation_mean / band_adjacent_background_stability / support_ratio_mean / unresolved_ratio_mean`


## 4. synthetic 检查结果

执行：

```bash
python scripts/eval/check_clean_plate_background.py
python scripts/eval/check_video_clean_plate_background.py
```

结果：

- `Synthetic clean-plate background improvement: PASS`
- `Synthetic clean-plate metadata contract: PASS`
- `Synthetic clean-plate video consistency: PASS`
- `Synthetic clean-plate video metadata contract: PASS`

这说明：

1. 旧 `hole -> image` 的改进链没有被破坏。
2. 新 `clean_plate_video` 路线在 synthetic case 上，确实比逐帧 image 路线更稳定。
3. `clean_plate_video` 的 metadata 和 contract 已经完整打通。


## 5. 真实 10 秒 preprocess smoke

### 5.1 输入

- checkpoint:
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`
- video:
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- reference:
  - `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`

### 5.2 配置

为避免 Step 01 已知的 SAM2 稳定性问题，本步骤统一复用安全 preprocess 基线：

- `sam_runtime_profile=h200_safe`
- `--no-sam_apply_postprocessing`
- `--sam_prompt_mode mask_seed`
- `--no-sam_use_negative_points`
- `--sam_reprompt_interval 0`
- 极保守 prompt 阈值
- `resolution_area=640x360`
- `fps=5`
- `sam_chunk_len=12`
- `sam_keyframes_per_chunk=3`
- `--lossless_intermediate`
- `--export_qa_visuals`

并在此基础上做 image / video AB：

- image run:
  - `runs/step05_clean_plate_image_preprocess_smoke`
- video run:
  - `runs/step05_clean_plate_video_preprocess_smoke`

### 5.3 contract 检查

执行：

```bash
python scripts/eval/check_animate_contract.py \
  --src_root_path runs/step05_clean_plate_image_preprocess_smoke/preprocess \
  --replace_flag --skip_synthetic

python scripts/eval/check_animate_contract.py \
  --src_root_path runs/step05_clean_plate_video_preprocess_smoke/preprocess \
  --replace_flag --skip_synthetic
```

结果均为 `PASS`。

这说明：

- 老的 preprocess/generate artifact 契约没有被破坏
- 新 `clean_plate_video` bundle 可以被当前 replacement 系统无歧义识别
- runtime-check rerun 也已确认新的 `preprocess_runtime_stats.json` 会带上 `background` 字段


## 6. 真实 image / video 客观对照

直接读取两个真实 preprocess bundle 的 metadata，得到：

### 6.1 image

- `background_mode = clean_plate_image`
- `temporal_fluctuation_mean = 6.0311`
- `band_adjacent_background_stability = 19.2530`
- `inpainted_area_ratio_mean = 0.2645`
- `support_ratio_mean = 0.0`
- `unresolved_ratio_mean = 0.0`

### 6.2 video

- `background_mode = clean_plate_video`
- `temporal_fluctuation_mean = 1.5511`
- `band_adjacent_background_stability = 0.4464`
- `inpainted_area_ratio_mean = 0.2645`
- `support_ratio_mean = 0.1271`
- `unresolved_ratio_mean = 0.8729`

### 6.3 解读

这一组结果说明：

1. 在真实 10 秒 case 上，`clean_plate_video` 的时序波动明显小于 `clean_plate_image`。
2. 边界带附近的背景稳定性指标也显著更好。
3. `inpainted_area_ratio_mean` 完全一致，说明两条路径需要处理的人物遮挡区域规模相同。
4. `support_ratio_mean / unresolved_ratio_mean` 只在 `video` 路线里有意义：
   - 当前真实 case 中，真正有足够时序可见背景支撑的视频重建区域只占一部分
   - 大部分长期被遮挡区仍然需要回退到 image clean plate

因此，这一步的真实结论不是：

- `video` 已经完整替代 `image`

而是：

- `video` 已经能显著提升背景时序一致性
- 但当前第一版仍然 heavily 依赖 image fallback 去填补长期遮挡区域


## 7. 真实 generate smoke

为了确认这条新 background 路线不只是 preprocess 产物增强，而是下游真的按 metadata 消费，又额外跑了一次最小 generate smoke：

- source bundle:
  - `runs/step05_clean_plate_video_preprocess_smoke/preprocess`
- output:
  - `runs/step05_clean_plate_video_generate_smoke/outputs/clean_plate_video_smoke.mkv`
- 配置：
  - `frame_num=45`
  - `refert_num=9`
  - `sample_steps=4`
  - `output_format=ffv1`

结果：

- generate 成功完成
- runtime stats 已写出：
  - `runs/step05_clean_plate_video_generate_smoke/debug/generate/wan_animate_runtime_stats.json`

其中关键信息为：

- `background_mode = clean_plate_video`
- `replacement_mask_mode = soft_band`
- `soft_band_available = True`
- `hard_foreground_available = True`
- `soft_alpha_available = True`
- `boundary_band_available = True`
- `background_keep_prior_available = True`

这说明：

1. generate 现在确实按 preprocess metadata 识别并记录了新的 background artifact 语义。
2. `clean_plate_video` 与 Step 04 的边界融合信号可以共同工作。


## 8. 当前版本的优点与边界

### 8.1 当前优点

当前第一版 `clean_plate_video` 已经具备明确价值：

- 比逐帧 image inpaint 更稳定
- 对边缘附近的背景闪烁更友好
- 仍保留 image fallback，因此不容易因为“视频背景支撑不足”而彻底失败
- 和当前 replacement / boundary refinement / metadata 契约完全兼容

### 8.2 当前边界

当前版本仍有明显上限：

1. 它不是光流对齐或显式视频修复模型。
2. 长期遮挡区域并没有真正“恢复出真实背景”，只是更稳地回退到了 image clean plate。
3. `unresolved_ratio_mean` 仍然较高，说明第一版 video prefill 的覆盖范围有限。

所以这一步应被理解为：

- 一条可落地、低风险、比 image 更稳定的视频一致 background 路线

而不是：

- 已经把 clean plate 做到最终极限


## 9. 默认策略建议

基于当前实现和真实 AB 结果，推荐策略是：

1. 若环境与 case 允许，replacement preprocess 默认优先使用 `clean_plate_video`
2. 若对 preprocess 时延更敏感，或后续 case 证明视频聚合收益不明显，可回退 `clean_plate_image`
3. 若需要做极简保底 smoke，仍可回退 `hole`

更准确地说，当前策略应该是：

- 生产默认优先：
  - `clean_plate_video`
- 保底 fallback：
  - `clean_plate_image`
- 极简兼容：
  - `hole`


## 10. 对下一步的意义

这一步为后续优化打下了两个关键基础：

1. generate 侧已经能稳定区分并记录 background artifact 来源。
2. Step 03 的像素域边界 refinement、以及后续更强的 boundary-aware compositing，都可以基于更稳定的背景条件继续推进。

因此，从边缘质量角度看，Step 05 的意义是：

- 它本身不会直接让人物发丝变成 matting 级别
- 但它明显改善了边界附近背景条件质量，是后续继续抠边、减脏边、减闪烁的重要支撑层
