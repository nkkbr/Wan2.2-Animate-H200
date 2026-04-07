# Wan-Animate Replacement 极致精度优化方案

## 1. 文档目标

本文档基于当前仓库已经完成的两轮优化工作，面向下一阶段的核心诉求：

- 进一步降低人物替换后的边缘模糊
- 进一步提高人物轮廓、动作、脸部相关识别精度
- 尽可能把单张 H200 141GB 的算力和显存转化为更高质量的 `preprocess_data` 产物
- 为后续代码改造提供一条明确、可实施、可验收的工程路线

本文档不再讨论“能否跑通”，而是专门讨论“如何逼近极致精度”。


## 2. 当前系统的准确定位

### 2.1 当前系统已经具备的能力

当前代码已经拥有一套比较完整的高质量 replacement 工程骨架：

- `preprocess` 侧：
  - SAM2 稳定性修复与 H200 profile
  - face / pose 稳定化
  - soft mask / boundary band
  - parsing / matting heuristic fusion
  - clean plate image / clean plate video
  - structure-aware reference normalization
  - lossless intermediate pipeline
- `generate` 侧：
  - soft-band replacement
  - clip continuity / seam blending
  - decoupled guidance
  - temporal handoff
  - pixel-domain boundary refinement
  - runtime stats / debug / metrics

这说明当前系统已经明显优于最初版本，且具备继续向上优化的基础。

### 2.2 当前系统仍未达到的目标

当前系统仍未达到“极致识别精度”和“极致边缘精度”，主要原因有三类：

1. 边缘控制的核心仍然主要停留在 latent replacement
2. 当前 parsing / matting 仍是 heuristic 第一版，不是高上限边界模型
3. preprocess 虽然已经更稳，但还没有进入“多模型、多尺度、多候选”的高精度范式

因此，下一阶段不能再只做局部打补丁，而必须把 `preprocess` 正式升级成“高分辨率、多阶段、多模型融合”的系统。


## 3. 下一阶段的总原则

### 3.1 用 H200 换取识别精度，而不是只换取生成耗时

下一阶段中，H200 的预算应优先用于：

- 更高分析分辨率
- 多模型融合
- 多候选 preprocess
- 更强的时序一致性分析
- 更精细的脸部、身体、边界局部重跑

不应主要浪费在：

- 继续沿用保守的低分辨率 preprocess
- 只增加 diffusion steps，而不提高输入控制信号质量
- 只做最终输出分辨率放大，而不提升识别与边界建模质量

### 3.2 preprocess 应成为极致精度的主战场

对于 replacement 任务，真正决定最终上限的，不只是模型采样本身，而是：

- mask 有多准
- soft alpha 有多准
- background keep prior 有多准
- pose/face 轨迹有多稳
- reference 结构归一化有多合理

因此，下一阶段的核心思路应是：

**把 preprocess 从“生成所需条件准备”升级为“高精度视觉分析系统”。**

### 3.3 生成阶段负责消费更细信号，而不是替 preprocess 补课

下一阶段 generate 的职责应是：

- 精确消费 richer signals
- 保证时序一致性
- 在边界带内做像素级 refinement

而不是继续依赖 generate 去弥补输入识别精度不足。


## 4. 下一阶段的目标形态

目标形态不是“再多几个参数”，而是以下结构：

### 4.1 轮廓系统

输出不再只是 `src_mask` 和 `soft_band`，而是至少包含：

- `hard_foreground`
- `soft_alpha`
- `boundary_band`
- `occlusion_band`
- `uncertainty_map`
- `background_keep_prior`

这些信号应区分：

- 高置信人物主体
- 软边界过渡区域
- 遮挡和半透明区域
- 无法可靠判断的区域

### 4.2 动作系统

输出不再只是骨架视频，而应包含：

- 全身关键点
- 局部 ROI refine 关键点
- 时序平滑后的轨迹
- 关键点速度
- 可见性与置信度
- 遮挡状态

### 4.3 脸系统

输出不再只是 `src_face.mp4`，而应包含：

- tracking-aware face bbox
- dense landmarks
- head pose
- expression / motion code
- face parsing
- face alpha
- face uncertainty

### 4.4 背景系统

输出不再只是 `clean_plate`，还应显式给出：

- visible support map
- unresolved region
- temporal confidence
- background source provenance

### 4.5 生成消费系统

generate 不再只使用“硬人物区域 + 背景保留”两级语义，而是应消费：

- soft alpha
- boundary band
- uncertainty map
- face pose/expression
- refined structure prior


## 5. 核心优化方向

## 5.1 高精度轮廓系统

### 目标

让人物轮廓从“分割级别可用”提升到“接近 alpha matting 级控制”。

### 建议方案

1. 把当前 `SAM2 + parsing + matting` 扩展成真正的多信号融合系统
2. 对人物 ROI 单独进行高分辨率边界分析
3. 同时输出：
   - 硬主体
   - 软 alpha
   - 边界过渡带
   - 遮挡带
   - 不确定区域
4. 用前向传播、反向传播、双向融合和跨 chunk 约束减少边界抖动
5. 对头发、手指、衣角、袖口等薄边结构单独建立局部 refine 流程

### 新增建议

这是本轮之前没有充分展开的一点：

**应引入“边界不确定性”作为一等公民。**

很多最差的边缘不是因为“模型没画出来”，而是因为系统把低置信区域错误当成高置信边界。  
如果 preprocess 能显式输出 `uncertainty_map`，generate 和 boundary refinement 才能在这些区域更保守地处理。


## 5.2 高精度动作系统

### 目标

让动作识别不只是“骨架大致正确”，而是达到：

- 关键点更准
- 时序更稳
- 遮挡下更鲁棒
- 局部肢体更可信

### 建议方案

1. 保留全身 pose 主链
2. 新增局部人体 ROI 分析：
   - 上半身
   - 下半身
   - 双手
   - 头肩
3. 对高运动、高遮挡片段做局部重跑
4. 引入双向时序平滑，不只用过去帧，也用未来帧约束
5. 输出结构化轨迹数据，而不是只输出骨架视频

### 新增建议

应考虑把动作系统拆成：

- `global pose`
- `limb refinement`
- `hand refinement`

手部、肘部、膝部和脚踝这几类边缘最容易在 replacement 后变形。  
如果这些局部点位不准，即使大体骨架正确，最终的轮廓仍会看起来软、抖或错位。


## 5.3 高精度脸系统

### 目标

让脸相关条件从“可驱动”提升到“可精确约束”。

### 建议方案

1. 从逐帧 face crop 升级到 tracking-aware face system
2. 输出 dense landmarks
3. 显式估计：
   - head pose
   - expression
   - face confidence
   - face occlusion
4. 增加 face parsing 和 face alpha
5. 对高运动、大转头、部分遮挡帧做专门 rerun

### 新增建议

应把脸系统的识别目标分成三类：

- 几何：landmark, pose
- 语义：face parsing
- 光学：face alpha / occlusion / uncertainty

当前系统更偏向 motion；下一阶段必须补齐 geometry 和 optical 两块，否则：

- 发际线边缘不准
- 下颌与脖子连接不准
- 侧脸与耳边轮廓不稳


## 5.4 视频一致背景系统 2.0

### 目标

让背景不只是“尽量干净”，而是能明确区分：

- 确实可恢复的背景
- 只能估计的背景
- 根本不可恢复的背景

### 建议方案

1. 对 clean plate video 明确输出：
   - visible support map
   - unresolved region
   - temporal confidence
2. 对长期遮挡区域保留“低置信”标记，而不是伪装成高置信 clean plate
3. 在 generate 和 boundary refinement 中按背景置信度采用不同融合策略

### 新增建议

应把背景系统分为：

- `conditioning background`
- `evaluation background`

前者用于生成约束，后者用于 debug/评估。  
这样可以避免把纯 debug 逻辑误用为高置信生成条件。


## 5.5 结构感知 reference normalization 2.0

### 目标

让参考图与驱动视频人物的对应关系，从“bbox 或粗结构对齐”进一步提升到“人体局部比例对齐”。

### 建议方案

1. 将结构拆成：
   - 头部
   - 肩胸
   - 躯干
   - 骨盆
   - 双腿
2. 分段计算目标预算
3. 对 face / torso / lower-body 使用不同的约束
4. 将 reference normalization 输出的结构信息显式传给 generate，而不只是保存归一化后的图像

### 新增建议

这一步应加入 `body proportion confidence`。  
如果驱动视频人物被遮挡严重，结构预算不应被错误地强制执行；否则会把 reference warp 到错误比例。


## 5.6 生成消费升级

### 目标

让 generate 真正吃到 preprocess 提供的更细语义，而不是继续把一切压缩成简单 mask。

### 建议方案

1. 在 replacement conditioning 中正式引入：
   - `soft_alpha`
   - `boundary_band`
   - `uncertainty_map`
   - `background_keep_prior`
2. face 分支接入：
   - head pose
   - expression
   - face confidence
3. boundary refinement 改成 uncertainty-aware，而不是单一 deterministic blending
4. 对高不确定区域使用更保守的背景/人物融合策略

### 新增建议

未来可考虑两阶段生成：

- 第一阶段：主体 replacement
- 第二阶段：局部边界和高频细节 refinement

这是高风险高收益项，但如果目标是“极致边缘清晰度”，这一方向迟早要进入候选路线。


## 6. H200 141GB 的极致使用策略

## 6.1 H200 应优先花在哪里

优先级如下：

1. 更高分析分辨率
2. 更多局部 ROI rerun
3. 多模型融合
4. 双向时序分析
5. 多候选 preprocess 自动选优
6. 更高质量无损中间产物

### 具体建议

- 全局分析分辨率可提升到短边 `1280` 或 `1536`
- 人物 ROI 可提升到 `1536` 到 `2048`
- face ROI 可提升到 `1024` 或更高
- 支持同一片段跑 2 到 4 套 preprocess 候选，然后自动选最优

## 6.2 不应优先花在哪里

- 只增加 diffusion `sample_steps`
- 只提升最终导出分辨率
- 继续使用为旧 GPU 兼容而保守的 preprocess 配置

如果识别精度不提升，更多生成算力的边际收益会迅速下降。


## 7. 建议的实施顺序

建议下一阶段按下面顺序推进：

1. 建立“高精度 benchmark 子集”
2. 重构 preprocess 为多阶段、多分辨率管线
3. 升级轮廓系统为多模型融合边界系统
4. 升级 face system 为完整脸系统
5. 升级动作系统为多尺度 pose + 时序优化系统
6. 升级背景系统为可见性驱动 clean plate 2.0
7. 升级 generate 消费 richer signals
8. 建立 H200 多候选 preprocess 自动选优流程


## 8. 验收标准

下一阶段不应只以“肉眼感觉更好”作为验收，而应至少满足：

### 8.1 轮廓

- boundary F-score 提升
- trimap error 降低
- halo ratio 降低
- 边界带 gradient 不下降，最好提升

### 8.2 动作

- 关键点 jitter 降低
- 遮挡片段轨迹稳定性提高
- limb-local metrics 改善

### 8.3 脸

- landmark NME 降低
- head pose 稳定性提高
- face boundary consistency 提高

### 8.4 背景

- temporal fluctuation 降低
- band-adjacent background stability 提高
- unresolved region 被显式隔离，不再污染高置信背景

### 8.5 系统

- preprocess 可在 H200 上稳定重复通过
- 允许高分辨率分析与多候选运行
- metadata/debug/metrics 覆盖新增加的全部高精度信号


## 9. 结论

如果目标只是“比现在再好一点”，继续微调当前参数就可以。  
但如果目标是你要求的：

- 极致轮廓精度
- 极致动作识别精度
- 极致脸部识别精度
- 极致 preprocess data 质量
- 尽量榨干 H200 141GB

那么下一阶段必须接受一个事实：

**仅靠当前的单层 preprocess + latent replacement + heuristic boundary refinement，是不够的。**

下一阶段的正确方向，是把 `preprocess` 升级为高分辨率、多模型、多阶段、多候选的视觉分析系统，再让 generate 精确消费这些 richer signals。

这才是把 H200 真正换成最终画质和边缘精度的路线。
