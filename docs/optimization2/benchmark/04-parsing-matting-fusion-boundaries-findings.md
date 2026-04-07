# Step 04 Findings: Parsing / Matting 融合边界建模

## 1. 本步骤完成了什么

本步骤已经把 `SAM2 + soft_band` 的单一路径，扩展成一套可运行、可回退、可被 generate/refinement 消费的边界融合框架。当前实现不是外部大模型版 parsing / matting，而是第一阶段的 **heuristic adapter**：

- `parsing_adapter.py`
  - 从 `pose_metas + face_bboxes + 图像梯度` 构造语义边界先验
- `matting_adapter.py`
  - 从 `hard_mask + trimap + 颜色/距离启发式` 构造软 alpha
- `boundary_fusion.py`
  - 以 `SAM2 hard mask` 为主体锚点
  - 融合 `soft_band + parsing prior + matting alpha`
  - 输出：
    - `hard_foreground`
    - `soft_alpha`
    - `boundary_band`
    - `background_keep_prior`

同时，generate 侧已经完成兼容消费：

- 若新 artifact 存在：
  - 优先消费 `hard_foreground / soft_alpha / boundary_band / background_keep_prior`
- 若新 artifact 不存在：
  - 回退到旧 `person_mask + soft_band`

像素域 `boundary_refinement` 也已经接入 `soft_alpha`，不是继续只用 `person_mask + soft_band`。


## 2. 代码落点

本步骤新增或修改了以下关键模块：

- 新增：
  - `wan/modules/animate/preprocess/parsing_adapter.py`
  - `wan/modules/animate/preprocess/matting_adapter.py`
  - `wan/modules/animate/preprocess/boundary_fusion.py`
  - `scripts/eval/check_boundary_fusion.py`
- 修改：
  - `wan/modules/animate/preprocess/process_pipepline.py`
  - `wan/modules/animate/preprocess/preprocess_data.py`
  - `wan/utils/animate_contract.py`
  - `wan/utils/replacement_masks.py`
  - `wan/utils/boundary_refinement.py`
  - `wan/animate.py`


## 3. 新增 CLI 与 metadata 能力

### 3.1 preprocess 新参数

已新增：

- `--boundary_fusion_mode {none,heuristic}`
- `--parsing_mode {none,heuristic}`
- `--matting_mode {none,heuristic}`
- `--parsing_head_expand`
- `--parsing_hand_radius_ratio`
- `--parsing_boundary_kernel`
- `--matting_trimap_inner_erode`
- `--matting_trimap_outer_dilate`
- `--matting_blur_kernel`

### 3.2 preprocess 新 artifact

replacement preprocess 现在可输出：

- `person_mask`
  - 兼容旧语义，仍保留
- `soft_band`
  - 旧 baseline band，仍保留，方便 AB 对照
- `hard_foreground`
  - 新的强主体区域
- `soft_alpha`
  - 新的柔性 alpha 先验
- `boundary_band`
  - 新的融合边界带
- `background_keep_prior`
  - 新的背景保留先验

### 3.3 metadata

`metadata.json` 已支持：

- `src_files.hard_foreground`
- `src_files.soft_alpha`
- `src_files.boundary_band`
- `src_files.background_keep_prior`
- `processing.boundary_fusion`


## 4. synthetic 检查结果

执行：

```bash
python scripts/eval/check_boundary_fusion.py
```

结果：

- `Synthetic parsing/matting fusion boundary gain: PASS`
- `Synthetic fused boundary metadata contract: PASS`

这说明：

1. 新融合逻辑在构造的“发丝/手部细边” synthetic case 上，能让融合边界比旧 `soft_band` 更有针对性地覆盖细结构区域。
2. 新 artifact 的写入、读取、metadata 契约与 generate 侧 bundle 校验链是通的。


## 5. 真实 preprocess smoke

### 5.1 输入

- ckpt:
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`
- video:
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- reference:
  - `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`

### 5.2 配置

为了复用 Step 01 / Step 02 已证明稳定的 H200 基线，本次采用：

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

并叠加：

- `--boundary_fusion_mode heuristic`
- `--parsing_mode heuristic`
- `--matting_mode heuristic`

### 5.3 结果

真实 preprocess smoke 已成功完成：

- run dir:
  - `runs/step04_boundary_fusion_preprocess_smoke`
- contract check:
  - `python scripts/eval/check_animate_contract.py --src_root_path runs/step04_boundary_fusion_preprocess_smoke/preprocess --replace_flag --skip_synthetic`
  - 结果：`PASS`

metadata 中确认存在：

- `hard_foreground`
- `soft_alpha`
- `boundary_band`
- `background_keep_prior`

实际输出格式：

- `hard_foreground`: `npz`
- `soft_alpha`: `npz`
- `boundary_band`: `npz`
- `background_keep_prior`: `npz`


## 6. 真实 generate smoke

### 6.1 配置

为了验证新 artifact 不只是 preprocess 写出来，而是后续真的被消费，额外跑了一次小型 generate smoke：

- source bundle:
  - `runs/step04_boundary_fusion_preprocess_smoke/preprocess`
- `frame_num=45`
- `refert_num=9`
- `sample_steps=4`
- `boundary_refine_mode=deterministic`
- `output_format=ffv1`

### 6.2 结果

真实 generate smoke 已成功完成：

- output:
  - `runs/step04_boundary_fusion_generate_smoke/outputs/fused_generate_smoke.mkv`
- runtime stats:
  - `runs/step04_boundary_fusion_generate_smoke/debug/generate/wan_animate_runtime_stats.json`

runtime stats 明确表明新 artifact 全部被识别为可用：

- `soft_band_available = True`
- `hard_foreground_available = True`
- `soft_alpha_available = True`
- `boundary_band_available = True`
- `background_keep_prior_available = True`

且 boundary refinement 已实际执行：

- `boundary_refinement.applied = True`

debug 目录中也确实落出了：

- `hard_foreground.mp4`
- `soft_alpha.mp4`
- `boundary_band.mp4`
- `background_keep_prior.mp4`
- `boundary_refinement/soft_alpha.mp4`


## 7. 真实 bundle 的客观统计

对真实 preprocess smoke bundle 的边界 artifact 做了直接统计，得到：

### 7.1 原始均值

- `soft_band_mean = 0.0365`
- `boundary_band_mean = 0.0296`
- `hard_foreground_mean = 0.1179`
- `background_keep_prior_far_bg_mean = 1.0000`

### 7.2 边界集中度

- `soft_band_near_shell_mean = 0.8024`
- `soft_band_far_shell_mean = 0.6364`
- `soft_band_concentration_ratio = 1.2608`

- `boundary_band_near_shell_mean = 0.8024`
- `boundary_band_far_shell_mean = 0.5730`
- `boundary_band_concentration_ratio = 1.4004`

### 7.3 如何解释

这里最关键的不是 `boundary_band` 的全局均值是否更大，而是它是否 **更集中在真正靠近主体的窄边界壳层**。

真实统计显示：

- 新 `boundary_band` 在 near shell 上几乎保持了与旧 `soft_band` 相同的强度
- 但在 far shell 上明显更弱
- 所以 concentration ratio 从 `1.26` 提升到了 `1.40`

这说明当前 heuristic fusion 的真实行为更像：

- **让边界更薄、更集中、更保守**
- 而不是简单把 band 做得更宽

这对减少 halo 和“边缘一圈糊”的问题是合理方向。


## 8. 当前版本的准确判断

### 8.1 这一步已经成立的部分

本步骤已经满足以下目标：

1. preprocess 已能输出更丰富的融合边界信号
2. metadata 和 debug 体系已完整支持这些新 artifact
3. generate / refinement 已能兼容消费这些新信号
4. synthetic case 上能看到比旧 `soft_band` 更强的细结构补强
5. 真实 case 上边界分布更集中，说明新 band 更偏“细边界建模”

### 8.2 还没有达到的部分

这一步还不能宣称已经实现了“真正外部模型级 parsing / matting”：

- 当前 parsing 是 heuristic prior，不是独立人体 parsing 网络
- 当前 matting 是 trimap + color/spatial alpha，不是高质量 video matting 模型

因此，这一步更准确的定位是：

- **完成了 Step 04 的框架与第一版保守实现**
- **还没有完成 Step 04 的最终上限版本**


## 9. 风险与观察

### 9.1 风险：当前 heuristic 版本偏保守

真实统计里，`boundary_band` 相比旧 `soft_band` 更窄、更集中。  
这有利于减少 halo，但也意味着：

- 它未必会在所有素材上让“肉眼边缘更锐”
- 更像是在为后续 Step 03 refinement 提供更干净的边界先验

### 9.2 风险：`soft_alpha` 不是外部 matting 模型输出

当前 `soft_alpha` 的质量取决于：

- SAM2 主体锚点是否可信
- trimap 是否合理
- 局部颜色对比是否足够明显

对于复杂发丝、透明材质、快速运动模糊，后续仍有继续升级空间。


## 10. 结论

Step 04 已经通过，前提是“通过”定义为：

- 把 parsing/matting fusion 的工程框架和新边界语义真正接进 preprocess、metadata、generate、boundary refinement
- 并在 synthetic 与真实 smoke 上验证其可运行、可消费、可回退

但也要明确：

- 当前版本是 **heuristic first pass**
- 它已经把边界建模从“单一 soft band”推进到了“hard / alpha / band / background prior 四层结构”
- 还没有把外部 parsing/matting 模型的质量上限全部吃进来

后续如果继续做 Step 05/06 或进一步的 boundary work，这套四层语义和消费链可以直接复用，不需要再推翻重来。
