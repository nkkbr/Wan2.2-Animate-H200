# Wan-Animate Replacement 预处理优化方案

## 1. 文档目标

本文档只讨论 `Wan-Animate replacement mode` 的预处理阶段，也就是：

- `python ./wan/modules/animate/preprocess/preprocess_data.py ... --replace_flag`

目标不是解释“当前代码能跑”，而是系统性回答下面几个问题：

1. 当前预处理链路到底在做什么，信息是如何从原视频流向后续生成模型的。
2. 当前实现里，哪些地方决定了最终替换质量的上限。
3. 在现有 H200 141GB 单卡硬件条件下，如何同时从“参数侧”和“代码侧”提升质量。
4. 哪些优化应当优先做，哪些优化应当延后做。
5. 后续优化工作应当如何验证，不至于只凭主观感觉来调。

本文档面向后续工程优化，强调“可执行性”和“优先级”。


## 2. 当前预处理链路的真实职责

当前 replacement 预处理并不直接“生成新人物”，它只是把驱动视频拆成一组供 `generate.py --replace_flag` 使用的条件数据。

当前输出产物为：

- `src_pose.mp4`
- `src_face.mp4`
- `src_bg.mp4`
- `src_mask.mp4`
- `src_ref.png`

这 5 个文件在后续生成阶段的角色分别是：

- `src_pose.mp4`: 身体动作控制信号
- `src_face.mp4`: 表情和局部脸部运动控制信号
- `src_bg.mp4`: 需要保留的场景背景条件，后续会被直接编码进 replacement 的强条件 `y_reft`
- `src_mask.mp4`: 指定“哪里保背景、哪里允许重生成人物”，但在当前 `generate` 实现里会先取反，再下采样到 latent 分辨率后作为背景保留先验使用
- `src_ref.png`: 目标人物身份、服装、外观、风格来源

这意味着：预处理阶段的任务不是“抠像做得好看”，而是“给后续生成提供尽可能稳定、准确、时序一致的控制条件”。


## 3. 当前实现的逐步数据流

当前主实现位于：

- `wan/modules/animate/preprocess/preprocess_data.py`
- `wan/modules/animate/preprocess/process_pipepline.py`

replacement 分支的实际数据流如下。

### 3.1 视频读取与重采样

代码使用 `decord.VideoReader` 读取整段视频，然后：

- 读取总帧数
- 读取原始 fps
- 如果 `fps == -1`，则直接沿用原视频 fps
- 调用 `get_frame_indices()` 按时间等间隔取帧

注意：

- 当前实现没有做自适应降采样，也没有基于运动复杂度动态调 fps。
- 如果输入视频较长且 `fps=-1`，则会保留整段视频的全部帧。
- 这会显著增加后续 pose 和 SAM2 的处理量。

### 3.2 分辨率处理

每一帧会进入 `resize_by_area(image, target_area, divisor=16)`。

当前逻辑不是“强制 resize 到某个宽高”，而是：

- 保持原始长宽比
- 总面积不超过 `resolution_area[0] * resolution_area[1]`
- 输出宽高对齐到 `16` 的倍数

这件事非常重要，因为当前代码把“分析分辨率”和“导出分辨率”耦合在了一起：

- pose 检测在这个分辨率上做
- mask 跟踪在这个分辨率上做
- 最终导出视频也是这个分辨率
- 后续生成也会吃这个分辨率的预处理结果

这是一处设计局限。对于高质量场景，分析分辨率和导出分辨率应当解耦。

并且这里还有一个必须从全链路看待的事实：

- `export_resolution_area` 不只是预处理输出尺寸
- 它会直接决定后续 `generate` 阶段的 latent 空间尺寸、token 数量和每步采样成本

所以在 H200 场景下，也不应机械地把导出分辨率一路拉高，而应把一部分预算留给生成阶段的 `sample_steps`、overlap 和 continuity 优化。

### 3.3 Pose2d 检测

当前 `Pose2d` 实现由两部分组成：

- YOLO ONNX 人体检测
- ViTPose whole-body ONNX 关键点估计

执行方式是逐帧独立处理：

- 先对单帧做 YOLO 检测
- 默认只保留一个“主人物框”
- 再对单帧做 ViTPose
- 输出 133 个 whole-body 点
- 再转换成 repo 内部使用的 `meta`

后续会用到的关键点子集包括：

- body
- hands
- face

当前实现的特点是：

- 单人优先
- 逐帧独立
- 没有显式 tracking
- 没有时序平滑
- 对低置信点的处理非常弱

### 3.4 face 视频提取

`src_face.mp4` 的生成逻辑是：

- 从每帧 `keypoints_face` 求一个脸框
- 使用固定比例放大
- 直接裁剪
- resize 到 `512x512`
- 逐帧写成视频

当前实现的问题是：

- face bbox 计算不使用关键点置信度
- 遇到遮挡、侧脸、模糊时容易抖动
- 没有时间平滑
- 没有回退策略

### 3.5 pose 视频生成

`src_pose.mp4` 的生成逻辑是：

- 把每帧 whole-body meta 转为 `AAPoseMeta`
- 在黑底画布上绘制身体、头部、手部骨架
- 导出为视频

当前 replacement 模式不会做 pose retarget。

这意味着：

- 它保留了原视频里人物与场景的空间关系
- 也继承了原人物的比例和动作分布
- 如果参考人物和原人物体型差异很大，生成时更容易变形

### 3.6 SAM2 mask 跟踪

`src_mask.mp4` 的生成不是逐帧分割，而是：

- 按固定长度切 chunk
- 当前固定 `th_step = 400`
- 每个 chunk 只选少数关键帧作为提示帧
- 每个提示帧只喂少量正点给 SAM2
- 然后用 `propagate_in_video()` 向整段传播

当前使用的正点来源是 8 个 body 关键点，等价于：

- 头部中心附近
- 颈部
- 肩
- 髋
- 脚踝

当前实现没有使用：

- 负点
- face bbox 辅助点
- hands/feet 补充点
- 动态增加提示帧
- 基于运动变化的再提示

因此这条链本质上是“稀疏提示 + 单目标视频跟踪”，不是高精度人体分割。

### 3.7 背景与 mask 后处理

SAM2 输出的 raw mask 之后还要经过两步处理：

1. `get_mask_body_img()`
   - 对 mask 做膨胀
   - 由 `iterations` 和 `k` 控制

2. `get_aug_mask()`
   - 对 bbox 内的 mask 做网格化填充
   - 由 `w_len` 和 `h_len` 控制

然后：

- `src_bg.mp4 = frame * (1 - aug_mask)`
- `src_mask.mp4 = aug_mask`

注意：

- 当前 `src_bg.mp4` 不是 clean plate
- 它只是“人物区域被置零后的原背景”
- 也就是说，人物原本挡住的背景并没有被修复

这是 replacement 质量的重要上限之一。

### 3.8 中间结果存储

当前中间结果主要使用 `moviepy.ImageSequenceClip(...).write_videofile(...)` 直接导出 mp4。

这会带来几个现实问题：

- mask 会被有损压缩
- pose 控制视频会被有损压缩
- face crop 会被有损压缩
- 下游 `generate.py` 读取到的并不是“精确信号”，而是“被编码器污染过的信号”

对于高质量替换任务，这是明显的 fidelity 损失。

并且要注意：

- 即使 preprocess 端输出了更细的边界或软 mask
- 当前 `generate` 端仍会把 `src_mask` 读成单通道标量 mask
- 再用 `nearest` 下采样到 latent 分辨率

所以“提升 mask fidelity”必须与 `generate` 端的 mask 消费方式一起升级，否则收益会被下游再次吃掉。


## 4. 当前实现的误差预算

为了系统优化，需要明确误差从哪里来。

当前 replacement 预处理阶段的误差可分为 6 类。

### 4.1 人体定位误差

来源：

- YOLO 只做单帧检测
- 单人选择策略偏简单
- 没有跨帧一致性约束

后果：

- bbox 抖动
- 人物大小变化
- 后续 pose、face、mask 全部跟着抖

### 4.2 pose 关键点误差

来源：

- ViTPose 逐帧独立
- 没有时序平滑
- 没有低置信点修复
- 没有遮挡回退

后果：

- `src_pose.mp4` 抖动
- SAM2 提示点漂移
- 动作边缘错分

### 4.3 face crop 误差

来源：

- bbox 由 face 点直接求 min/max
- 不看 confidence
- 不做 temporal smoothing

后果：

- 人脸区域缩放不稳
- 脸部运动向量抖动
- 下游表情控制不稳定

### 4.4 mask 误差

来源：

- SAM2 提示过少
- chunk 太长
- 没有动态再提示
- 没有人体解析或 matting 融合
- mask 最终仍是粗二值

后果：

- 头发、衣摆、手臂边缘漏出
- 旧角色残留
- mask 内外边界不稳

### 4.5 背景误差

来源：

- `src_bg.mp4` 不是 inpainted 背景
- 人物区域直接被抹黑

后果：

- 背景边界闪烁
- replacement 区域内部背景需由模型自行补全
- 长视频更容易出现局部脏纹理

### 4.6 存储误差

来源：

- mp4 有损压缩
- mask 和 pose 都被编码

后果：

- 条件信号被污染
- 细边界丢失
- 软 mask 无法保真传递


## 5. 总体优化原则

在 H200 141GB 单卡条件下，预处理优化应遵循下面 5 条原则。

### 5.1 把“分析质量”和“导出质量”解耦

当前代码把：

- 检测分辨率
- mask 跟踪分辨率
- 导出分辨率
- 后续生成分辨率

绑在同一个 `resolution_area` 上，这不合理。

建议改成两级：

- `analysis_resolution_area`
- `export_resolution_area`

高质量模式下，允许在更高分析分辨率上做 pose 和 mask，再导出到目标生成分辨率。

### 5.2 优先修 correctness，再追额外复杂度

高质量任务中，下面这些属于 P0：

- 颜色空间一致性
- face bbox 稳定性
- pose 时序平滑
- mask 提示质量
- lossless 中间结果

这些问题不修复，单纯堆更高分辨率和更多计算，收益有限。

### 5.3 让 mask 从“二值框”变成“高质量控制信号”

当前 mask 更像“给模型留一块人物重画区域”，而不是“精细人物边界控制”。

高质量替换任务里，mask 应当升级为：

- 更稳的主体区域
- 更可靠的头发/服装边界
- 支持软边界
- 最好支持 trimap 或 confidence

### 5.4 让背景从“挖洞视频”变成“可保留背景”

如果不做背景修复，replacement 只能在“人物区域重新生成一切”，这对长视频是不利的。

高质量模式建议把 `src_bg` 升级为：

- inpainted background
- 或至少近似 clean plate

原因不只是视觉直觉，而是因为当前 `generate` 实现里，`src_bg` 会被直接 VAE encode 并进入 `y_reft`，属于 replacement 的强条件之一。

### 5.5 用 H200 换“更密的控制”，不只是换“更大的尺寸”

H200 的价值不应只体现在：

- 更高分辨率
- 更高 fps

还应体现在：

- 更多关键帧提示
- 更短 chunk
- 更高密度的 SAM2 再提示
- batched detector/pose 推理
- 多阶段 refine


## 6. 参数侧优化方案

本节只讨论“不改代码时，当前版本能做什么”。

### 6.1 `resolution_area`

当前建议：

- 若追求稳定性优先，建议以模型更熟悉的 720p 档位为主
- 若确认后续生成对 1080 级输入稳定，才使用 `1920 1080`

原因：

- 预处理变高分辨率，不等于生成一定更好
- 当前模型主用示例和默认设置更偏 720p
- 在不改代码的前提下，更高分辨率只会同时放大计算量和 mask/pose 抖动
- 并且会直接放大后续 `generate` 阶段的 latent 尺寸与 token 数量，使单步采样成本上升

建议：

- 当前代码不改时，把 `resolution_area` 当作“最终生成分辨率”来选，不要只因为显存充足就盲目拉高
- 如果最终目标是最高质量，优先通过“提高控制质量”而不是“只拉大画幅”

### 6.2 `fps`

当前建议：

- 以原视频 24/25/30fps 为优先
- 对人物动作很快、表情密集的素材，优先保持原 fps
- 对人物动作较缓的视频，才考虑降 fps

原因：

- `src_face.mp4` 和 `src_pose.mp4` 都直接受 fps 影响
- 降 fps 会减少控制密度
- 对 replacement，控制密度通常比速度更重要

你的当前选择 `--fps -1` 在高质量目标下是合理的。

### 6.3 `iterations` 与 `k`

这两个参数控制 mask 膨胀力度。

经验方向：

- 增大 `iterations` / `k`
  - 更容易罩住头发、衣摆、快速挥动的手臂
  - 旧角色残留更少
  - 背景保留变差
  - 生成区域变大，边界处更容易重画背景

- 减小 `iterations` / `k`
  - 背景保留更好
  - 但更容易漏出原人物边缘

建议：

- 高质量 replacement 的首要目标通常是“不要漏旧人物”
- 因此建议从略保守的较大 mask 起步，再往回收

当前起始建议：

- 稳定保守档: `iterations=2~3`, `k=7~9`
- 细边界档: `iterations=1~2`, `k=5~7`

你当前的 `iterations=1, k=5` 偏激进保背景，容易导致人物边缘漏出。

### 6.4 `w_len` 与 `h_len`

这两个参数控制 coarse mask 的网格细度。

经验方向：

- `w_len/h_len` 越小
  - 轮廓越粗
  - mask 越像“整块人物区域”
  - 更稳
  - 更容易吃掉背景

- `w_len/h_len` 越大
  - 轮廓越细
  - 背景保留更好
  - 更容易出现边缘漏人

当前起始建议：

- 稳定保守档: `w_len=2~4`, `h_len=4~8`
- 折中档: `w_len=4~6`, `h_len=8~12`
- 细边界档: `w_len=8~12`, `h_len=12~20`

你当前的 `w_len=8, h_len=16` 明显偏细边界档。

### 6.5 当前代码下的推荐起始参数

如果当前代码完全不改，建议先用“更稳”的起始参数，而不是一开始就追求背景极限保留。

推荐起始档：

```bash
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
    --video_path <video> \
    --refer_path <ref_image> \
    --save_path <save_dir> \
    --resolution_area 1280 720 \
    --fps -1 \
    --iterations 2 \
    --k 7 \
    --w_len 4 \
    --h_len 8 \
    --replace_flag
```

如果明确要保 1080 级输出，可在代码未改时改为：

```bash
--resolution_area 1920 1080 \
--iterations 2 \
--k 7 \
--w_len 4 \
--h_len 8
```

然后再根据症状微调，而不是直接用极细 mask。


## 7. 代码侧优化方案

下面是建议的代码级优化路线，按优先级排序。

### 7.1 P0: 必须优先做的正确性与 fidelity 修复

#### P0-1. 分离分析分辨率和导出分辨率

建议新增参数：

- `--analysis_resolution_area`
- `--export_resolution_area`

策略：

- pose、mask 在更高分辨率上分析
- 最终 `src_pose/src_mask/src_bg` 导出到目标生成分辨率

收益：

- 提高关键点和 mask 边界精度
- 控制最终生成分辨率不超出模型稳定区间

#### P0-2. 替换 mp4 中间结果为 lossless 方案

建议：

- `src_mask` 改为 PNG 序列或 `npz/npy`
- `src_pose` 改为 PNG 序列或 lossless 视频
- `src_face` 改为 PNG 序列或高质量无损编码
- `src_bg` 改为 PNG 序列或高质量无损编码

原因：

- 当前 mp4 会污染条件信号
- mask 的软边界无法保真

但这里要强调：

- 仅把 preprocess 输出改成 lossless，并不自动等于 soft mask 能在最终生成中生效
- 还必须同步修改 `generate.py / wan/animate.py` 的读取与 mask 使用逻辑

建议同时修改 `generate.py` 对输入的读取方式，使其支持目录或 `npz`。

#### P0-3. 统一颜色空间

需要重点检查 `Pose2d.load_images()` 与 `decord` 输出的颜色空间是否一致。

风险点：

- `decord` 通常输出 RGB
- 当前 `load_images()` 对 ndarray/list 直接执行 `cv2.COLOR_BGR2RGB`
- detector 和 pose 分支的通道翻转路径并不完全一致

行动：

- 明确全链路统一为 RGB
- 写单元测试确认 `decord -> pose2d -> detector/vitpose` 的通道输入一致

这是 correctness 问题，不是“可选优化”。

#### P0-4. face bbox 改为 confidence-aware + temporal smoothing

当前 face bbox 逻辑应升级为：

- 只使用高置信 face 点
- 点不足时回退到上一帧 bbox
- 使用 EMA / One-Euro / Kalman 做平滑
- 限制 bbox 尺寸变化率

收益：

- `src_face.mp4` 稳定性明显提升
- 下游表情控制更稳

#### P0-5. pose 关键点时序平滑

建议：

- 对 body、face、hands 关键点分别做 confidence-aware smoothing
- 对缺失点做邻域插值或前后帧补偿
- 对异常跳点做速度约束和 clamping

收益：

- `src_pose.mp4` 更稳
- SAM2 提示点更稳

#### P0-6. 提升 SAM2 提示质量

当前提示过于稀疏。

建议至少改为：

- 使用更多正点
- 使用 face bbox 中心、肩、肘、腕、髋、膝、踝等高置信点
- 在人体 bbox 外围自动生成负点
- 提示点先做坐标 clamp

收益：

- 更少漏分
- 边界更稳

#### P0-7. 缩短 chunk 并增加提示帧

当前 `th_step=400` 对长视频过大。

建议新增参数：

- `--sam_chunk_len`
- `--sam_keyframes_per_chunk`

高质量模式建议起点：

- `sam_chunk_len = 80~160`
- `sam_keyframes_per_chunk = 8~16`

策略：

- 短 chunk
- 多关键帧
- 对高运动片段增加再提示

#### P0-8. 支持 soft mask

当前最终保存的是硬二值 mask。

建议：

- 保留 SAM2 logits 或概率图
- 在边界形成 soft band
- 下游 replacement 使用软 mask 而不是纯二值

收益：

- 降低硬边和闪边
- 让背景保留和人物重绘之间形成更平滑过渡

注意：

- 这一项必须与生成阶段联动实现
- 因为当前 `generate` 端会把 `src_mask` 读成单通道后再 `nearest` 下采样
- 如果不改生成端，preprocess 端单独输出 soft mask，收益会非常有限

### 7.2 P1: 质量提升型升级

#### P1-1. 引入人体解析或 matting 融合

建议把 SAM2 与下列之一融合：

- human parsing
- portrait matting
- video matting

推荐融合逻辑：

- SAM2 提供主体时序稳定性
- parsing/matting 提供头发、衣摆、手指等细边界
- 最终合成主 mask + soft boundary

收益：

- 边缘质量显著提升
- 旧角色泄漏显著减少

#### P1-2. 构建真正的背景视频

建议在 preprocess 中新增：

- video inpainting
- 或至少 image inpainting + temporal stabilize

目标是把 `src_bg` 从“挖洞视频”升级成“近 clean plate 背景视频”。

收益：

- replacement 区域内背景更自然
- 边界更容易与原视频衔接

#### P1-3. 参考图人体尺度归一化

当前 `src_ref.png` 只是简单拷贝。

建议新增参考图预处理：

- 检出参考图人体 bbox
- 估计全身比例
- 结合驱动视频第一帧或统计 bbox 估计目标尺度
- 自动生成参考图缩放与画布摆放方案

收益：

- 降低 replacement 后的体型错配
- 减少肢体和衣物变形

#### P1-4. 生成 QA 中间结果

建议在 preprocess 中额外导出：

- `mask_overlay.mp4`
- `pose_overlay.mp4`
- `face_bbox_overlay.mp4`
- 每段 chunk 的关键帧提示图

收益：

- 人可以快速定位问题在哪一层
- 后续调参不再盲目

### 7.3 P2: 充分利用 H200 的工程优化

#### P2-1. Pose2d 批处理

当前 detector 和 pose 基本按单帧循环。

建议：

- detector 支持 batch preprocess + batch ONNX forward
- vitpose 支持 batch crop + batch forward

收益：

- 不只是提速
- 也允许在同样时间预算内跑更高分析分辨率和更多消歧步骤

#### P2-2. H200 专用高质量模式

建议新增 `--quality_preset hq_h200`，默认包含：

- 高分析分辨率
- 短 chunk
- 多关键帧
- 软 mask
- lossless 中间结果
- 更保守的 face/pose smoothing

#### P2-3. 验证 SAM2 是否可开启更强算子

当前 preprocess 代码强制：

- `USE_FLASH_ATTN = False`
- `OLD_GPU = True`

在 H200 上应重新验证：

- 是否可安全启用更强 attention kernel
- 是否可使用 bf16 / fp16 稳定推理

这主要影响效率，但效率提升能反过来支持更密的高质量分析。


## 8. 推荐新增的 CLI 与配置项

为了让后续优化可控，建议新增下面这些参数。

### 8.1 分辨率与存储

- `--analysis_resolution_area`
- `--export_resolution_area`
- `--save_format {mp4,png_seq,npz}`
- `--lossless_intermediate`

### 8.2 pose 与 face

- `--pose_batch_size`
- `--pose_conf_thresh`
- `--pose_smooth_method`
- `--pose_smooth_strength`
- `--face_conf_thresh`
- `--face_bbox_smooth_strength`
- `--face_bbox_expand_scale`

### 8.3 mask / SAM2

- `--sam_chunk_len`
- `--sam_keyframes_per_chunk`
- `--sam_use_negative_points`
- `--sam_positive_point_mode`
- `--sam_negative_margin`
- `--mask_soft_edge_px`
- `--mask_confidence_floor`

### 8.4 背景

- `--bg_inpaint_mode {none,image,video}`
- `--bg_inpaint_model`


## 9. 推荐实施顺序

### 阶段 A: 先修正确性和 fidelity

优先完成：

- 颜色空间统一
- lossless 中间结果
- face bbox 稳定
- pose smoothing
- mask 提示增强

这一步完成后，哪怕不引入新模型，质量通常也会明显提升。

### 阶段 B: 再提升 mask 和背景质量

优先完成：

- 更短 chunk + 更多关键帧
- soft mask
- 背景 inpainting
- QA 可视化输出

### 阶段 C: 最后再引入复杂融合

包括：

- parsing/matting 融合
- 参考图尺度归一化
- 多阶段 refine


## 10. 针对 H200 的建议工作模式

H200 的优势不应只用来“把分辨率堆大”，而应当转化为更高质量的控制信号。

推荐思路：

- 先在更高分析分辨率上做检测和 mask
- 用更多关键帧给 SAM2 提示
- 使用更短 chunk
- 导出 lossless 中间结果
- 把节省下来的时间预算用在 background inpainting 和 QA 可视化上
- 不要把全部预算都堆到 `export_resolution_area`，因为生成阶段真正昂贵的部分还包括更高的 `sample_steps` 和更稳的 clip continuity

简单说：

- H200 的价值 = 更密、更稳、更保真的条件提取
- 不只是更大的 `resolution_area`


## 11. 建议的实验设计

后续优化不建议“凭感觉一把改很多”。

建议采用固定评测集和分阶段对比。

### 11.1 固定评测集

至少覆盖：

- 正脸慢动作
- 快速肢体运动
- 头发和衣摆丰富
- 强遮挡
- 镜头运动明显
- 明暗变化强

### 11.2 每轮只改一类变量

顺序建议：

1. 先改保存格式
2. 再改 pose/face 稳定
3. 再改 SAM2 提示策略
4. 再改背景构建
5. 最后改高阶融合

### 11.3 评测维度

建议至少记录：

- 旧角色残留程度
- 头发和衣摆边界质量
- 背景保留一致性
- 脸部稳定性
- 大动作四肢质量
- 长视频后段是否漂移
- 运行时间
- 显存占用

### 11.4 建议增加的客观信号

可以在 QA 脚本中自动统计：

- mask 面积时间曲线
- face bbox 中心和面积时间曲线
- pose 关键点速度异常点比例
- chunk 间 mask 统计差异


## 12. 当前任务的直接建议

如果目标是尽快把现有 replacement 质量往上提，并且暂时只讨论 preprocess，建议按下面顺序推进。

### 第一优先级

- 修正或确认颜色空间一致性
- 改用 lossless 中间结果
- 提升 face bbox 稳定性
- 给 pose 加时序平滑
- 把 SAM2 的 chunk 改短、关键帧改多

### 第二优先级

- 让 mask 支持 soft edge
- 用更高质量的 mask 构造 `src_bg`
- 输出 overlay QA 视频

### 第三优先级

- 引入背景 inpainting
- 引入 parsing/matting 融合
- 做参考图尺度归一化


## 13. 一句话结论

要把 replacement 做到高质量，最重要的不是单纯提高 `resolution_area`，而是把预处理从“粗条件打包器”升级为“高保真控制信号生成器”。

对于当前代码，最值得优先投入的方向是：

- 更稳的 pose
- 更稳的 face
- 更强的 SAM2 提示
- 更高保真的 mask / bg 存储
- 更合理的分析分辨率设计

在 H200 上，这些优化都具备实施条件，且比单纯堆更大分辨率更有价值。
