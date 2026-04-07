# Wan-Animate Replacement 系统测试方案

## 1. 背景

本 repo 在前面 12 个步骤里，已经对 `Wan-Animate replacement` 做了系统性改造。

主要变化不再是单点参数，而是横跨两条主链路：

- `preprocess_data.py`
  - manifest / run 目录
  - metadata 契约
  - lossless 中间格式
  - pose / face 稳定化
  - SAM2 prompt 升级
  - soft mask / boundary band
  - clean plate background
  - reference image normalization
- `generate.py -> wan/animate.py`
  - H200 质量优先 preset
  - 静态条件缓存
  - mask-aware seam blending
  - decoupled guidance
  - temporal handoff prototype
  - runtime stats / debug 导出

因此，这一轮验证不能只靠单个 smoke test。


## 2. 测试目标

本轮测试要同时回答四类问题：

1. 代码是否仍然正确。
2. preprocess 与 generate 的接口是否稳定。
3. 新增功能是否真的被实际路径消费，而不只是停留在 synthetic test。
4. 在真实素材上，端到端流程是否能完整跑通，并产出可回溯的 debug / metrics。


## 3. 测试输入

本方案固定使用下面素材：

- preprocess checkpoint:
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`
- animate 主 checkpoint:
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B`
- 驱动视频:
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- 参考图:
  - `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`

说明：

- 10 秒视频足以覆盖多 clip continuity、mask、background、reference normalization 和 debug 导出。
- 真实 benchmark 仍应包含更多 case，但这条 10 秒样例适合作为系统验证 smoke case。


## 4. 分层测试矩阵

### 4.1 L0: 环境与依赖

目标：

- 确认运行环境具备真实测试条件

检查项：

- GPU 与显存
- `torch` / `decord` / `easydict` / `diffusers` / `peft` / `transformers`

通过标准：

- 依赖全部可导入
- 至少有一张可用 H200


### 4.2 L1: 静态与 CLI smoke

目标：

- 确认入口、参数和语法没有被后续步骤改坏

检查项：

- `python -m compileall`
- `python generate.py --help`
- `python wan/modules/animate/preprocess/preprocess_data.py --help`

通过标准：

- 不报语法错误
- 新参数全部可见


### 4.3 L2: Synthetic / contract / regression suite

目标：

- 用 repo 内置脚本验证每个步骤引入的核心逻辑仍然成立

建议全量执行：

- `scripts/eval/check_animate_contract.py`
- `scripts/eval/check_lossless_roundtrip.py`
- `scripts/eval/check_control_stability.py`
- `scripts/eval/check_sam_prompting.py`
- `scripts/eval/check_soft_mask_pipeline.py`
- `scripts/eval/check_clean_plate_background.py`
- `scripts/eval/check_clip_blending.py`
- `scripts/eval/check_guidance_and_parameter_search.py`
- `scripts/eval/check_temporal_handoff.py`
- `scripts/eval/check_reference_normalization.py`

通过标准：

- 全部 PASS


### 4.4 L3: preprocess 真实素材 smoke

目标：

- 在真实视频上走通增强后的 preprocess 主路径
- 验证 metadata、lossless artifact、QA 导出和 reference normalization 不是纸上逻辑

建议参数：

```bash
python ./wan/modules/animate/preprocess/preprocess_data.py \
  --ckpt_path /home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint \
  --video_path /home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4 \
  --refer_path /home/user1/wan2.2/test1/Gemini_Generated_Image_2.png \
  --run_dir <RUN_DIR> \
  --save_manifest \
  --resolution_area 1280 720 \
  --fps 15 \
  --save_format mp4 \
  --lossless_intermediate \
  --replace_flag \
  --export_qa_visuals \
  --iterations 2 \
  --k 7 \
  --w_len 4 \
  --h_len 8 \
  --sam_chunk_len 60 \
  --sam_keyframes_per_chunk 6 \
  --sam_reprompt_interval 20 \
  --soft_mask_mode soft_band \
  --soft_mask_band_width 24 \
  --soft_mask_blur_kernel 5 \
  --bg_inpaint_mode image \
  --bg_inpaint_method telea \
  --bg_inpaint_mask_expand 16 \
  --bg_inpaint_radius 5 \
  --reference_normalization_mode bbox_match \
  --reference_target_bbox_source median_first_n \
  --reference_target_bbox_frames 8 \
  --reference_bbox_conf_thresh 0.35 \
  --reference_scale_clamp_min 0.75 \
  --reference_scale_clamp_max 1.6
```

后续检查：

- `scripts/eval/check_animate_contract.py --src_root_path <RUN_DIR>/preprocess --replace_flag`
- 检查 `metadata.json`
- 检查 `src_ref_normalized.png`
- 检查 QA 文件：
  - `reference_normalization_preview.png`
  - `mask_overlay`
  - `soft_band_overlay`
  - `background_clean_plate`

通过标准：

- preprocess 成功退出
- contract 检查通过
- metadata 中出现：
  - `processing.control_stabilization`
  - `processing.sam2_mask_generation`
  - `processing.soft_mask`
  - `processing.background`
  - `processing.reference_normalization`
- `src_files.reference.path` 指向 normalized reference
- `src_files.reference_original` 保留原图


### 4.5 L4: generate 真实素材 smoke

目标：

- 在真实 preprocess 结果上走通 generate 主路径
- 覆盖本轮增强里最关键的 runtime 逻辑

建议参数：

```bash
python generate.py \
  --task animate-14B \
  --ckpt_dir /home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B \
  --src_root_path <RUN_DIR>/preprocess \
  --run_dir <RUN_DIR> \
  --save_manifest \
  --save_file <RUN_DIR>/outputs/replacement_smoke.mp4 \
  --quality_preset hq_h200 \
  --replace_flag \
  --use_relighting_lora \
  --sample_solver dpm++ \
  --sample_steps 6 \
  --sample_shift 5.0 \
  --sample_guide_scale 1.05 \
  --guidance_uncond_mode decoupled \
  --face_guide_scale 1.05 \
  --text_guide_scale 1.05 \
  --frame_num 77 \
  --refert_num 9 \
  --overlap_blend_mode mask_aware \
  --overlap_background_current_strength 0.25 \
  --temporal_handoff_mode hybrid \
  --temporal_handoff_strength 0.6 \
  --replacement_mask_mode soft_band \
  --replacement_mask_downsample_mode area \
  --replacement_boundary_strength 0.5 \
  --replacement_transition_low 0.1 \
  --replacement_transition_high 0.9 \
  --base_seed 123456 \
  --log_runtime_stats
```

说明：

- 这里不是最终画质 benchmark，而是端到端系统 smoke。
- `sample_steps=6` 的目的，是在可接受时间里覆盖：
  - static condition caching
  - soft mask consumption
  - mask-aware blending
  - decoupled guidance
  - hybrid temporal handoff
  - runtime stats

后续检查：

- 最终视频是否生成
- `manifest.json` 是否新增 generate stage
- `debug/generate/wan_animate_runtime_stats.json`
- `debug/generate/seams/`
- `debug/generate/temporal_handoffs/`

通过标准：

- generate 成功退出
- 输出视频存在
- runtime stats 存在
- seam / temporal handoff debug 至少生成一部分样本


### 4.6 L5: 基础客观指标

目标：

- 用已实现的统计脚本对真实 smoke run 做最小客观量化

命令：

```bash
python scripts/eval/compute_replacement_metrics.py --run_dir <RUN_DIR>
```

关注字段：

- `frame_count`
- `fps`
- `seam_score`
- `background_fluctuation`
- `mask_area`

通过标准：

- 脚本成功写出 metrics json
- manifest、runtime stats、metrics 三者能互相对齐


## 5. 推荐执行顺序

建议按下面顺序执行：

1. L0 环境检查
2. L1 静态与 CLI smoke
3. L2 synthetic / contract suite
4. L3 preprocess 真实素材 smoke
5. L4 generate 真实素材 smoke
6. L5 metrics 统计

原因：

- 一旦 L0-L2 失败，不应直接进入耗时的真实素材 run。
- L3 失败时，不应继续跑 L4。


## 6. 本轮修改的重点验收项

针对这 12 个步骤，本轮系统测试至少要显式确认下面这些点：

- preprocess / generate 都能写 run manifest
- lossless intermediate 真正被 generate 消费
- `person_mask` / `soft_band` 语义一致
- clean plate background 能进入 replacement 主路径
- clip overlap blending 不再只是 hard cut
- decoupled guidance 能跑通
- temporal handoff 原型能跑通
- normalized reference 会被 generate 作为主 `reference` artifact 消费


## 7. 已知边界

这套系统测试仍然不是最终 benchmark。

它主要验证：

- correctness
- interface stability
- runtime wiring
- debug observability

它不能单独回答：

- 最终画质是否已达到最佳
- latent handoff 是否优于 pixel baseline
- 参数空间的全局最优点在哪里

这些问题仍需依赖后续 benchmark case 和参数搜索。
