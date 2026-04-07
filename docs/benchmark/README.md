# Wan-Animate Replacement Benchmark 规范

## 1. 目标

本目录定义 replacement 优化阶段的最小 benchmark 规范。

它的目标不是存放所有媒体素材，而是统一：

- case id
- 素材标签
- 目录约定
- 人工评分维度
- 自动评测入口


## 2. 建议覆盖的 case 类型

正式 benchmark 至少应覆盖下面这些场景：

- 正脸慢动作
- 快速肢体动作
- 头发和衣摆丰富
- 强遮挡
- 镜头运动明显
- 光照变化强
- 背景复杂


## 3. repo 内置示例

当前 repo 自带一个 replacement 示例：

- 视频: `examples/wan_animate/replace/video.mp4`
- 参考图: `examples/wan_animate/replace/image.jpeg`

它可以作为 benchmark manifest 的演示 case，但不应成为唯一 benchmark。


## 4. benchmark manifest

建议 benchmark case 使用统一 manifest 描述。  
示例见：

- `docs/benchmark/benchmark_manifest.example.json`

推荐字段包括：

- `case_id`
- `status`
- `driver_video`
- `reference_image`
- `tags`
- `expected_challenges`
- `notes`


## 5. 人工评分模板

人工评分模板见：

- `docs/benchmark/manual_score_template.csv`

建议至少记录：

- identity consistency
- expression following
- motion naturalness
- background stability
- edge quality
- seam visibility


## 6. preprocess/generate 契约

replacement 的 preprocess 输出目录现在包含正式的 `metadata.json`。
字段说明见：

- `docs/benchmark/animate_preprocess_metadata.md`

lossless roundtrip 检查脚本见：

- `scripts/eval/check_lossless_roundtrip.py`

控制信号稳定性检查脚本见：

- `scripts/eval/check_control_stability.py`

SAM2 mask prompt / QA 检查脚本见：

- `scripts/eval/check_sam_prompting.py`

soft mask / boundary-aware replacement 检查脚本见：

- `scripts/eval/check_soft_mask_pipeline.py`

clean plate 背景构建检查脚本见：

- `scripts/eval/check_clean_plate_background.py`

## 7. 后续建议

后续应把外部 benchmark 素材逐步补充到统一 manifest 中，并且每个新增素材都附带：

- 使用授权状态
- case 标签
- 适用阶段

否则 benchmark 本身会再次失去可维护性。
