# Step 06 Findings: 结构级 Reference Normalization

## 1. 本步骤完成了什么

本步骤已经把 reference normalization 从“bbox 级缩放与重心对齐”扩展成了正式的双路径体系：

- `bbox_match`
- `structure_match`

其中 `structure_match` 的第一版不是复杂的非刚性人物重建，而是一个 **可解释的结构感知 warp**：

1. 从 reference pose 提取人物结构锚点：
   - `top / shoulder / hip / foot`
   - `shoulder_width`
   - `head / torso / leg` 段长
2. 从驱动视频前若干帧统计目标结构先验：
   - `target_structure`
   - `target_bbox`
3. 使用结构先验驱动：
   - 横向宽度缩放
   - 头/躯干/腿的分段纵向变形
4. 再用 replacement region 预算做 clamp：
   - `reference_structure_width_budget_ratio`
   - `reference_structure_height_budget_ratio`

这一步的目标不是把参考图直接变成驱动视频当前姿态，而是让参考人物的：

- 宽高占比
- 肩宽
- 上下身比例
- 在画面中的空间预算

更接近驱动目标，从而减少后续 replacement 时的体型失配和边缘挤出问题。


## 2. 代码落点

本步骤新增或修改的关键模块如下：

- 修改：
  - `wan/modules/animate/preprocess/reference_normalization.py`
    - 新增结构统计提取
    - 新增 driver target structure 聚合
    - 新增 `normalize_reference_image_structure_aware`
    - 扩展 preview overlay
  - `wan/modules/animate/preprocess/process_pipepline.py`
    - 接入 `structure_match`
    - reference 结构统计写入 metadata
  - `wan/modules/animate/preprocess/preprocess_data.py`
    - 新增 structure-aware CLI 参数
- 修改：
  - `scripts/eval/check_reference_normalization.py`
    - 扩展为 bbox + structure 双路径 synthetic 检查


## 3. CLI 与 metadata 变化

### 3.1 新增 mode

`--reference_normalization_mode` 现在支持：

- `none`
- `bbox_match`
- `structure_match`

### 3.2 新增结构约束参数

新增：

- `--reference_structure_segment_clamp_min`
- `--reference_structure_segment_clamp_max`
- `--reference_structure_width_budget_ratio`
- `--reference_structure_height_budget_ratio`

### 3.3 metadata 扩展

`metadata.json` 中的 `processing.reference_normalization.stats` 现在可以包含：

- `target_structure`
- `normalized_structure`
- `raw_segment_scales`
- `applied_segment_scales`
- `width_budget_triggered`
- `height_budget_triggered`
- `replacement_region_budget_bbox`

因此，这一步不只是“写出了一张新的 reference 图”，而是把结构级 normalization 的统计与决策也显式保存下来了。


## 4. synthetic 检查结果

执行：

```bash
python scripts/eval/check_reference_normalization.py
```

结果：

- `Synthetic reference bbox normalization: PASS`
- `Synthetic structure-aware reference normalization: PASS`
- `Synthetic reference normalization metadata contract: PASS`

synthetic case 的设计要点是：

- reference 的肩宽更窄、头部更高、躯干更长
- driver target 的肩更宽、头更紧凑、下半身更长

在这个 case 上：

- `bbox_match` 只能做整图缩放，无法真正改善结构失配
- `structure_match` 能把目标 bbox 误差进一步压低，并且显著降低结构误差

因此，synthetic 检查已经证明：

- `structure_match` 不只是“能运行”
- 它确实在结构对齐层面比 `bbox_match` 更接近目标


## 5. 真实 10 秒 preprocess smoke

### 5.1 输入

- checkpoint:
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`
- video:
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- reference:
  - `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`

### 5.2 配置

沿用 Step 01 / Step 02 / Step 05 已验证的稳定 preprocess 基线：

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
- `--bg_inpaint_mode video`
- `--lossless_intermediate`
- `--export_qa_visuals`

并将 reference normalization 切到：

- `--reference_normalization_mode structure_match`
- `--reference_structure_segment_clamp_min 0.8`
- `--reference_structure_segment_clamp_max 1.25`
- `--reference_structure_width_budget_ratio 1.05`
- `--reference_structure_height_budget_ratio 1.05`

真实 smoke run：

- `runs/step06_structure_reference_preprocess_smoke`

### 5.3 结果

真实 preprocess 已成功完成，且 contract 检查通过：

```bash
python scripts/eval/check_animate_contract.py \
  --src_root_path runs/step06_structure_reference_preprocess_smoke/preprocess \
  --replace_flag --skip_synthetic
```

结果：

- `PASS`

实际产物中确认存在：

- `src_ref_normalized.png`
- `reference_normalization_preview.png`

metadata 中确认：

- `reference_normalization_mode = structure_match`
- `applied = True`
- `reason = ok`
- `target_structure` 存在
- `normalized_structure` 存在


## 6. 真实 bundle 与 Step 05 的 bbox 基线对照

为了避免只看 synthetic case，本步骤还把真实 `structure_match` bundle 和 Step 05 已有的真实 `bbox_match` bundle 做了 metadata 级对照：

- bbox baseline:
  - `runs/step05_clean_plate_video_preprocess_smoke/preprocess`
- structure-aware:
  - `runs/step06_structure_reference_preprocess_smoke/preprocess`

### 6.1 结果

- bbox run:
  - `reference_normalization_mode = bbox_match`
  - `target_bbox width/height ratio = 0.3795`
  - `normalized_bbox width/height ratio = 1.0212`

- structure run:
  - `reference_normalization_mode = structure_match`
  - `target_structure width_height_ratio = 0.3540`
  - `normalized_structure width_height_ratio = 0.3840`
  - `applied_segment_scales = {'head': 0.8, 'torso': 0.8, 'legs': 0.8}`
  - `width_budget_triggered = True`
  - `height_budget_triggered = True`

### 6.2 解读

这个真实 case 说明了一个很重要的事实：

- 仅靠 `bbox_match`，reference 会被缩放成一个“整体占位合适但结构依旧偏宽”的人物。
- `structure_match` 会主动把 reference 的宽高占比往 driver 目标结构拉近。

换句话说：

- `bbox_match` 更像“把人放到差不多的位置和面积”
- `structure_match` 更像“把人的结构预算压回到更接近 driver 的身体模板里”

这正是本步骤真正要解决的问题。


## 7. preview 与可解释性

真实 preprocess 已产出：

- `reference_normalization_preview.png`

尺寸为：

- `1280 x 352`

这张 preview 会同时展示：

- 原始 letterbox reference
- target bbox / target structure
- normalized reference
- normalized structure

因此，这一步不是黑盒改图，而是有可视化可解释性的：

- 可以直接看出 target 结构线和 normalized 结构线是否接近
- 可以判断是否触发了 budget clamp


## 8. 真实 generate smoke

为了确认新 mode 不只是 preprocess 写出来，而是 generate 也能无歧义消费，又额外跑了一次最小 generate smoke：

- source bundle:
  - `runs/step06_structure_reference_preprocess_smoke/preprocess`
- output:
  - `runs/step06_structure_reference_generate_smoke/outputs/structure_reference_smoke.mkv`

结果：

- generate 成功完成
- runtime stats 已写出：
  - `runs/step06_structure_reference_generate_smoke/debug/generate/wan_animate_runtime_stats.json`

关键字段：

- `background_mode = clean_plate_video`
- `reference_normalization_mode = structure_match`
- `total_generate_sec = 38.58`

这说明：

1. `structure_match` 已经贯穿 preprocess -> metadata -> generate
2. generate 侧没有把它误当成旧的 `bbox_match` 或 `none`


## 9. 当前版本的优点与边界

### 9.1 当前优点

当前第一版 `structure_match` 已经有明确价值：

- 不再只用单一 bbox 决定 reference 缩放
- 已把头/躯干/腿三段结构显式建模
- 已把 replacement region 预算通过 width/height budget 接入 normalization
- 已具备 preview、metadata、generate runtime 全链路可解释性

### 9.2 当前边界

当前版本仍然是“结构感知的一维/二维可解释 warp”，不是最终极的人体形变系统：

1. 它不会重建 reference 的真实视频姿态。
2. 它主要调的是结构比例，不是服装体积的复杂非刚性形变。
3. 当前真实 case 中三个 segment scale 都触发到下限 `0.8`，说明参考图与 driver 结构差异较大，且预算 clamp 已经在强约束它。

因此，这一步的准确定位是：

- 它已经显著优于纯 bbox 路线
- 但仍属于“可解释、保守、低风险”的第一版结构级 normalization


## 10. 结论

Step 06 已达到设计目标：

- `structure_match` 已正式实现
- preprocess 真实 10 秒素材 smoke 已通过
- metadata、preview、generate runtime 都能识别并记录结构级 normalization
- 相比 Step 05 的 `bbox_match`，真实 case 上 reference 的宽高结构已经明显更接近 driver 目标

从后续优化价值看，这一步的意义是：

- 它不会直接解决所有边缘问题
- 但它已经减少了“reference 结构失配导致的人物被挤宽/挤扁”的前置条件问题
- 因而会对后续 replacement 的形体一致性和边缘预算更友好
