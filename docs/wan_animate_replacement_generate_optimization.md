# Wan-Animate Replacement 生成阶段优化方案

## 1. 文档目标

本文档只讨论 `Wan-Animate replacement mode` 的生成阶段，也就是：

- `python generate.py --task animate-14B ... --replace_flag`

本文档不重复解释预处理阶段，而是系统性回答下面几个问题：

1. 当前 `generate.py -> wan/animate.py -> WanAnimateModel` 这条链路，真实在做什么。
2. 当前实现里，哪些地方决定了 replacement 结果的质量上限。
3. 在单张 H200 141GB GPU 条件下，应该如何同时从“参数侧”和“代码侧”提升质量。
4. 哪些优化属于低风险高收益，适合优先落地；哪些属于高风险项，需要单独验证甚至再训练。
5. 后续优化工作应当如何评测，避免只凭主观观感做判断。

本文档强调“指导后续程序优化工作”，所以重点是：

- 识别真实瓶颈
- 给出明确优先级
- 给出可实施的改动方向
- 给出实验和评估建议


## 2. 当前生成链路的真实职责

当前 replacement 生成并不是一句“按 prompt 生成视频”可以概括的。

它真正做的事情是：

- 读取预处理产物 `src_pose.mp4 / src_face.mp4 / src_bg.mp4 / src_mask.mp4 / src_ref.png`
- 把这些产物重新编码成一组 latent 条件
- 按 `clip_len=77` 的短视频片段逐段采样
- 在人物 mask 内重生成新角色
- 在人物 mask 外尽量保留背景
- 用上一段的最后少量帧做时序续接
- 最后把所有 clip 拼起来输出整段视频

这条链路的核心文件是：

- `generate.py`
- `wan/animate.py`
- `wan/modules/animate/model_animate.py`
- `wan/modules/animate/face_blocks.py`
- `wan/modules/animate/motion_encoder.py`
- `wan/modules/animate/clip.py`
- `wan/utils/fm_solvers.py`
- `wan/utils/fm_solvers_unipc.py`
- `wan/utils/utils.py`


## 3. 当前实现的逐步数据流

这一节只保留对后续优化决策最有价值的部分。

### 3.1 `generate.py` 入口行为

当前 `generate.py` 对 animate 任务的关键行为如下：

- 如果没有显式传 `--prompt`，则默认使用 `视频中的人在做动作`
- 如果没有显式传 `--frame_num`，则默认使用配置里的 `77`
- 如果没有显式传 `--offload_model`，单卡默认会自动设为 `True`
- `--sample_solver` 的 CLI 默认值是 `unipc`
- `--sample_steps`、`--sample_shift`、`--sample_guide_scale` 都支持覆盖配置

这意味着：

- 当前仓库默认对单卡用户更偏向“能跑起来、少占显存”，而不是“充分利用高端显卡追求最高质量”
- 对 H200 141GB 这种卡，这个默认策略并不合适

### 3.2 clip 切分

当前 animate 推理不是整段做，而是分 clip 做。

默认逻辑：

- `clip_len = 77`
- `refert_num = 1 or 5`
- 实际步长 = `clip_len - refert_num`

例如 `refert_num = 5` 时：

- 每段生成 77 帧
- 下一段拿上一段最后 5 帧做 temporal guidance
- 输出时把后一段的前 5 帧裁掉
- 然后直接拼接

注意：

- 当前代码只允许 `refert_num == 1 or 5`
- 但 argparse 默认值写成了 `77`
- 这是一个明确的接口设计问题

如果用户忘记显式传 `--refert_num`，当前实现会直接触发断言失败。

### 3.3 条件构造

每个 clip 都会构造成下面一组张量：

- `conditioning_pixel_values`: 骨架视频
- `face_pixel_values`: 人脸视频
- `refer_pixel_values`: 单张参考图
- `bg_pixel_values`: 背景视频
- `mask_pixel_values`: 人物区域 mask
- `refer_t_pixel_values`: 上一段生成结果的最后若干帧

然后通过 VAE / CLIP / T5 编成多路条件：

- `y_ref`: 静态参考图条件
- `y_reft`: 背景 + temporal guidance 条件
- `pose_latents`: 动作条件
- `clip_context`: 参考图 CLIP 视觉 token
- `context`: 文本 token
- `motion_vec`: face motion token

### 3.4 当前 replacement 的强弱条件层级

从代码注入方式看，当前条件强弱大致是：

1. `y_ref + y_reft`
   直接与噪声 latent 按通道拼接，最强
2. `pose_latents`
   在 patch embedding 之后直接加到主 token 上，次强
3. `CLIP` 图像 token
   在每层 cross-attn 中参与，强身份条件
4. `face motion`
   每隔 5 层用 `FaceAdapter` 注入，局部细节条件
5. `text prompt`
   在 replacement 里相对最弱

因此，对 replacement 质量最敏感的，不是 prompt，而是：

- `y` 的构造方式
- overlap 续接方式
- mask 的使用方式
- pose / face 信号的稳定性


## 4. 当前生成实现的主要质量瓶颈

下面这些问题，不是“某次推理偶然发生”，而是代码结构天然会导致的风险点。

### 4.1 段间续接过短，且拼接方式过硬

当前段间 continuity 只依赖：

- 上一段最后 `refert_num` 帧
- 后一段前 `refert_num` 帧直接裁掉

当前没有：

- overlap blending
- seam smoothing
- boundary-aware compositing
- optical-flow-based merge
- latent-level temporal memory

这会带来：

- 段间接缝
- 背景局部闪烁
- 人物边缘跳变
- 长视频误差累计

### 4.2 temporal handoff 发生了 decode -> re-encode

这一点非常关键。

当前每段生成完以后：

- 先把 latent `x0` decode 成 `out_frames`
- 下一段再把 `out_frames` 取最后几帧，拼回 `refer_t_pixel_values`
- 再 VAE encode 成新的 `y_reft`

也就是说，段间传递不是 latent-to-latent，而是：

- latent -> pixel -> latent

这会带来两类损失：

- VAE 解码再编码的细节损失
- temporal handoff 中的表观抖动和颜色漂移

这一点是当前长期时序一致性的硬瓶颈之一。

### 4.3 `refert_num` 限制过死

当前代码强制只允许 `1` 或 `5`。

这意味着：

- 低重叠时段间续接弱
- 无法探索 `9 / 13 / 17` 等更强 temporal guidance
- 无法针对高动作视频和低动作视频做差异化配置

从代码结构看，大部分逻辑本身是可以支持更一般的 `refert_num` 的，当前限制更像是“经验值硬编码”，不是结构硬限制。

### 4.4 mask 在生成阶段被二值化并用 nearest 直接下采样

当前 replacement 的 mask 使用方式是：

- 读取 `src_mask.mp4`
- 变成单通道 `[0,1]`
- 取反得到“背景保留区域”
- 用 `nearest` resize 到 latent 分辨率
- 再通过 `get_i2v_mask()` 重排成 4 通道时间打包格式

这会带来：

- 边界块状化
- 细轮廓损失
- 人物边缘区域过硬
- 背景与人物交界处更容易闪

如果 preprocess 侧已经输出了较好的 mask，这一层的硬下采样会进一步浪费信息。

### 4.5 `guide_scale` 把 face 和 text 绑在了一起

当前 unconditional 分支不是完全无条件，而是：

- 文本换成 negative prompt
- `face_pixel_values` 变成全 `-1`
- 其余条件保持不变

这意味着当前 `guide_scale` 实际上是把：

- face expression guidance
- text guidance

绑成一个开关。

问题在于 replacement 任务中：

- 文本通常不是主要控制来源
- face control 才更需要细调

当前实现无法单独增强或减弱 face influence。

### 4.6 参考图编码重复计算

当前每个 clip 都会重复做：

- `src_ref.png` 的 VAE encode
- `src_ref.png` 的 CLIP visual encode

这对质量没有直接帮助，但会显著浪费推理预算。

在 H200 上，最合理的做法不是继续 offload，而是把这部分改成预先缓存，然后把省下来的时间和显存预算用于：

- 更高的 `sample_steps`
- 更多的实验组合
- 更复杂的段间融合逻辑

### 4.7 输出保存是有损 mp4

当前 `save_video()` 使用：

- `libx264`
- `quality=8`

这意味着：

- 最终输出文件本身就是有损压缩
- 肉眼上看到的边缘、细节、闪烁，部分可能来自编码器
- 不利于客观评估不同代码版本和参数组合的真实差异

如果目标是“高质量 replacement 优化”，最终输出必须支持 lossless 或近无损评测格式。


## 5. 优化原则

后续优化必须先遵守两条原则。

### 5.1 不轻易改模型输入结构

当前 `WanAnimateModel` 的输入通道数、条件注入方式、face adapter 结构，都是和预训练权重绑定的。

因此：

- 不要轻易改 `in_dim`
- 不要轻易改变 `y` 的通道定义
- 不要轻易给主干额外拼新条件通道

否则很容易进入“必须重新微调甚至重训”的路径。

优先做的，应当是：

- 不改变模型权重含义的推理逻辑优化
- 不改变输入维度的条件组织优化
- 输出和拼接逻辑优化
- 缓存和精度策略优化

### 5.2 H200 的价值应当换成“更高质量预算”，不是省显存

对你们的单卡 H200 141GB 场景，生成阶段最值得换来的东西是：

- `offload_model=False`
- 更高的 `sample_steps`
- 更多的 overlap
- 更精细的段间融合
- 预编码缓存
- 更高质量输出格式

而不是继续沿用“单卡默认 offload”的保守策略。


## 6. 参数侧建议

下面先给出不改代码时的参数建议。

### 6.1 建议的基础 profile

作为 H200 单卡高质量 baseline，建议先固定下面这组生成参数：

- `--offload_model false`
- `--use_relighting_lora`
- `--sample_solver dpm++`
- `--sample_steps 60`
- `--sample_shift 5.0`
- `--sample_guide_scale 1.0`
- `--refert_num 5`

理由：

- `offload_model false`
  H200 没必要默认来回搬运模型，这主要是速度问题，但速度提升会直接换来更多高质量实验预算。
- `use_relighting_lora`
  replacement 场景下建议默认打开，提升人物与场景光照协调性。
- `dpm++`
  在更高步数下通常更适合作为高质量优先方案，建议作为 HQ baseline。
- `sample_steps 60`
  40 步已经是中高档位，H200 上完全可以把 baseline 提到 60 步。
- `sample_shift 5.0`
  先以官方默认作为中心点做小范围搜索。
- `guide_scale 1.0`
  当前 CFG 主要影响 expression，不建议上来就放大。
- `refert_num 5`
  目前不改代码时，5 是更稳妥的 continuity 选项。

### 6.2 参数搜索建议

如果只改参数，建议用下面这个优先级做网格。

第一组：求解器与步数

- `sample_solver`: `unipc`, `dpm++`
- `sample_steps`: `40`, `60`, `80`

目的：

- 找到在你们素材分布上“质量/时间”最合适的 solver

第二组：噪声日程

- `sample_shift`: `4.5`, `5.0`, `5.5`, `6.0`

目的：

- 找到人物纹理稳定性、边缘收敛和动态自然度更好的区域

第三组：expression guidance

- `sample_guide_scale`: `1.0`, `1.1`, `1.2`, `1.4`

目的：

- 只在确实需要更强表情跟随时再提高
- 避免过高导致 face motion 副作用放大

### 6.3 参数层面的明确结论

- `offload_model`
  对 H200，不建议继续使用默认 `True`
- `sample_steps`
  高质量 replacement 不建议长期停留在 40
- `guide_scale`
  当前实现下不是越高越好，应当保守搜索
- `refert_num`
  当前只允许 1 或 5，限制太强，建议进入代码改造项


## 7. 代码侧优化建议

下面按风险和收益分层。

## 7.1 P0：低风险高收益，优先落地

这些改动不改变模型结构，不依赖再训练，应当优先做。

### 7.1.1 修改 H200 场景下的默认资源策略

建议修改：

- `generate.py`
- `wan/animate.py`

目标：

- 对 animate replacement 场景，把单卡默认 `offload_model=True` 改为显式推荐 `False`
- 文档和 CLI 示例都以“高显存卡高质量模式”为主

理由：

- 当前默认更适合低显存卡
- 对 H200 来说，这个默认值不合理

### 7.1.2 缓存静态参考条件

建议修改：

- `wan/animate.py`

当前可缓存项：

- `context`
- `context_null`
- `ref_latents`
- `mask_ref`
- `y_ref`
- `clip_context`

这些量在整段视频中基本不变，没有必要每个 clip 重新计算。

收益：

- 降低每段开销
- 节省 VAE/CLIP/T5 重复工作
- 为更高 `sample_steps` 留出预算

这是典型的质量间接提升项。

### 7.1.3 改进段间拼接：从“硬裁切”升级为“重叠融合”

建议修改：

- `wan/animate.py`

当前逻辑：

- 后一段前 `refert_num` 帧直接丢掉

建议改为：

- 保留 overlap 区间
- 在 overlap 帧上做 alpha blending
- alpha 可按时间线性变化
- 背景区域和人物区域可以使用不同 blending 策略

更推荐的方式：

- 背景区域优先沿用上一段
- 人物区域在 overlap 中做时间 ramp 混合

收益：

- 明显减轻段间接缝
- 降低人物边缘跳变
- 对长视频稳定性提升明显

### 7.1.4 增加高质量输出格式

建议修改：

- `wan/utils/utils.py`
- `generate.py`

新增能力：

- 输出 PNG sequence
- 输出无损或近无损视频
- 输出 clip 级中间结果

理由：

- 当前 H.264 质量 8 不适合做客观评估
- 优化阶段必须能区分“模型问题”和“编码器问题”

### 7.1.5 增加调试产物导出

建议修改：

- `wan/animate.py`

建议支持导出：

- 每个 clip 的 `y_reft` 可视化
- overlap 区间前后帧对比
- 每段最后 `refert_num` 帧与下一段最前若干帧对比
- solver 使用的实际 timestep

理由：

- 目前很多问题只能靠最终视频猜
- 缺乏可视化中间产物，不利于定位是 seam 问题、mask 问题，还是 face guidance 问题


## 7.2 P1：中风险高价值，建议第二阶段做

### 7.2.1 放开 `refert_num` 限制并系统评估

建议修改：

- `generate.py`
- `wan/animate.py`

当前断言只允许 1 或 5，这过于死板。

建议改为：

- 支持任意正整数 `refert_num`
- 约束条件为 `0 < refert_num < clip_len`
- 优先实验 `5 / 9 / 13`

预期收益：

- 更长 overlap 往往有助于 continuity

风险：

- overlap 太长可能让新段过度依赖旧段，导致漂移积累
- 需要结合 seam 融合一起做

### 7.2.2 解耦 face guidance 与 text guidance

建议修改：

- `wan/animate.py`

当前 `guide_scale` 同时作用于：

- 负文本上下文
- 空白 face condition

建议增加：

- `face_guide_scale`
- `text_guide_scale`

或者至少先做：

- 只 null 掉 face，不改 text
- 或者只改 text，不动 face

这样可以把 replacement 场景里真正有价值的控制项单独调出来。

收益：

- 表情跟随可以单独增强
- 不会顺带放大无关 prompt 影响

### 7.2.3 支持 soft mask / feathered mask

建议修改：

- `wan/animate.py`
- 必要时联动 preprocess 输出格式

当前 mask 用法过硬：

- 二值
- nearest resize

建议升级为：

- 支持浮点 mask
- 背景强约束区、人物自由区、中间过渡边界区三段式 mask
- latent resize 使用更平滑的策略

收益：

- 边缘过渡更自然
- 降低背景和人物交界闪烁

这是 replacement 质量提升非常值得做的一项。

### 7.2.4 把段间 handoff 从“像素重编码”升级为更稳定的 memory 方案

建议修改：

- `wan/animate.py`

当前问题：

- temporal handoff 依赖 decode -> re-encode

建议探索两条路径：

1. 低风险版本
   保留像素 handoff，但增加 overlap 并配合 seam blend
2. 高收益版本
   设计显式的 latent memory 传递，而不是单纯拿最后几帧 RGB 再编码

第二条路径更难，但它是长视频稳定性的关键突破口之一。

注意：

- 这一项属于“中高风险”
- 因为当前 VAE 时序压缩和 causal conv 会让简单的 latent splice 不一定等价

因此建议先做原型验证，再决定是否深入。


## 7.3 P2：高风险项，只在前两阶段完成后再推进

### 7.3.1 更长 clip 或自适应 clip

理论上更长 clip 可能改善 continuity，但当前模型配置和默认推理路径都围绕 `77` 帧组织。

建议：

- 不要把这项作为第一优先级
- 若要探索，先小步尝试
- 重点验证是否真的优于“77 帧 + 更长 overlap + better blending”

### 7.3.2 参考图多视角或多裁剪增强

可以设想：

- 同时使用全身图、脸部 crop、上半身 crop
- 让参考图条件更强

但当前 cross-attn 对图像 token 的处理是按固定 `257` 个 token 设计的。

如果要引入多视角参考：

- 需要重新设计 token 组织方式
- 风险较高

### 7.3.3 直接加入新条件分支

例如：

- 原始视频 RGB 额外分支
- 边界带 mask 分支
- 光流分支

这类改动基本都会触碰模型输入结构或语义分布。

如果没有再训练或微调计划，不建议在当前阶段推进。


## 8. 针对 H200 141GB 的明确建议

### 8.1 默认运行策略

对 H200，建议默认：

- 不 offload
- 全程走 GPU
- 以高质量为第一目标，不以“勉强省显存”为目标

### 8.2 H200 应当优先换来的能力

优先级从高到低建议是：

1. 更高 `sample_steps`
2. 更多试验组合
3. 更长 overlap
4. 更复杂的 seam blending
5. 更高质量输出格式
6. 条件缓存和调试导出

而不是：

- 继续默认 offload
- 只为了让代码更保守地跑

### 8.3 推荐的 H200 高质量基线命令

在当前代码不改的前提下，建议先用：

```bash
python generate.py \
    --task animate-14B \
    --ckpt_dir ./Wan2.2-Animate-14B/ \
    --src_root_path /path/to/process_results \
    --refert_num 5 \
    --replace_flag \
    --use_relighting_lora \
    --sample_solver dpm++ \
    --sample_steps 60 \
    --sample_shift 5.0 \
    --sample_guide_scale 1.0 \
    --base_seed 123456 \
    --offload_model false \
    --save_file /path/to/output.mp4
```

然后围绕下面三个轴做搜索：

- `sample_steps`
- `sample_shift`
- `sample_guide_scale`


## 9. 建议的实施顺序

### 第一阶段：先把工程地基补好

优先做：

1. 修正 `refert_num` 默认值与断言冲突
2. 对 H200 场景默认关闭 offload
3. 缓存参考图和文本相关编码
4. 增加 lossless 输出能力
5. 增加 clip 调试导出

这一步的目标不是直接让画质飞跃，而是：

- 让后续优化效率更高
- 让问题可观测
- 让推理预算真正花在采样和融合上

### 第二阶段：解决 continuity

优先做：

1. overlap blending
2. 放开 `refert_num`
3. 做 `5 / 9 / 13` 的 continuity 实验

这一阶段通常会带来最直观的主观质量提升。

### 第三阶段：解决边界与表情控制

优先做：

1. soft mask / boundary band
2. face/text guidance 解耦

### 第四阶段：探索高风险 memory 方案

仅在前三阶段做完之后，再考虑：

- latent-level temporal memory
- 更激进的长时序方案


## 10. 评测建议

如果没有评测规范，优化工作很容易陷入“看起来这次好像更顺眼”的主观陷阱。

建议固定以下评测维度。

### 10.1 主观评测维度

- 身份一致性
- 表情跟随度
- 动作自然度
- 背景稳定性
- 人物边缘自然度
- 段间接缝可见度

### 10.2 客观评测维度

建议至少保存以下指标：

- clip seam score
  对每个拼接点前后若干帧做差异统计
- background consistency
  在 mask 外区域做帧间差异统计
- identity similarity
  用独立人脸/人物 embedding 模型评估目标人物一致性
- temporal flicker
  对人物区域和背景区域分别统计高频闪烁

### 10.3 实验管理建议

每次实验应固定记录：

- git commit
- preprocess 参数版本
- generate 参数版本
- 是否开启 relighting lora
- solver
- steps
- shift
- guide scale
- refert_num
- 输出格式

否则后续无法追责是哪一项改变带来了好坏变化。


## 11. 明确结论

当前 replacement 生成质量的核心瓶颈，不在“显存不够”，而在：

- 段间 temporal handoff 太弱
- 拼接方式太硬
- mask 在 latent 中的使用过于粗糙
- face guidance 与 text guidance 耦合
- 很多静态条件被重复编码，浪费了高端 GPU 的预算

对 H200 141GB 来说，最值得优先做的不是“继续保守 offload”，而是：

1. 关闭 offload，释放高质量推理预算
2. 把参考条件缓存起来
3. 先做 overlap blending
4. 放开 `refert_num`
5. 增加高质量输出和调试能力
6. 再做 soft mask 与 face/text guidance 解耦

如果只允许我给一个最重要的判断，那就是：

当前 generate 侧最值得优先投入工程资源的，不是换 solver，不是继续调 prompt，而是“修复 clip 级连续性机制”。

从实际收益看，段间 continuity 优化通常会比单纯把 `sample_steps` 从 40 拉到 60 更值。
