# Step 03: 像素域边缘 Refinement

## 1. 步骤目标

本步骤的目标是新增一层 **pixel-domain boundary refinement**，让当前 replacement 系统不再只依赖 latent 空间中的 mask / soft band 控制，而是在生成完成后，对人物边缘带做更高精度的像素域精修。

这一步的核心出发点非常明确：

- 当前系统已经有 soft band 和 boundary-aware replacement
- 但这些能力主要仍工作在 latent 级
- 这足以减少粗糙接缝，但不足以把头发丝、衣角、袖口、快速运动边缘做得足够锐利

本步骤就是要把边缘质量的主战场，从 latent 级进一步推进到 pixel 级。


## 2. 为什么要做这一步

### 2.1 当前边缘质量的主要上限来自架构，而不是单一参数

当前 `Wan-Animate replacement` 的边界控制仍然建立在：

- VAE 8x 空间下采样
- latent mask 控制
- latent-space background keep

之上。  
因此，哪怕 preprocess 再准、H200 再强，最终边缘仍然容易出现：

- 轻微软化
- 毛边
- halo
- 背景色渗入
- 细部轮廓不够锐利

### 2.2 继续只调 latent 侧参数，边际收益会越来越小

当前系统已经做过：

- soft mask
- boundary band
- overlap blending
- temporal handoff

这些改动都有效，但都没有真正突破 latent 级边界建模上限。  
所以这一轮必须承认：只继续调 latent 逻辑，已经不够了。

### 2.3 这一步直接针对用户最在意的问题

用户最在意的是：

- 替换后人物边缘有时有些模糊

而 pixel-domain refinement 正是最直接针对这个问题的工程手段。


## 3. 本步骤的交付结果

本步骤完成后，应至少交付：

1. 一条可开关的边缘精修链
2. 仅作用于 boundary band 的像素域融合策略
3. 对边缘相关 debug artifact 的可视化导出
4. 用无损输出对比 refinement 前后的边缘质量
5. 一套可回归的 edge-quality 检查方案


## 4. 设计原则

本步骤的设计必须遵守下面几条原则。

### 4.1 不重写整帧，只精修边缘带

人物主体的大部分区域已经来自 diffusion 结果。  
如果整帧再做一次强处理，容易引入：

- 纹理漂移
- 背景变脏
- 时间抖动

因此，本步骤应坚持：

- 主体区域保持生成结果
- 背景区域尽量保持原像素
- 只在 boundary band 做 refinement

### 4.2 先做确定性精修，再考虑学习型 refinement

第一版应优先采用可解释、可调试的像素域融合逻辑，而不是直接再接一个新的黑盒模型。

建议先做：

- 基于 hard mask + soft band 的 alpha 组合
- 前景锐化 / 边界局部平滑 / 颜色过渡控制
- 与 clean plate / original background 的局部合成

### 4.3 必须支持无损评估

边缘 refinement 的判断不能只看 mp4 输出。  
这一步必须优先用：

- `ffv1`
- `png_seq`

来做对照，不然编码器本身会把边缘重新抹软。


## 5. 代码改动范围

本步骤预计会改动如下模块：

- `wan/animate.py`
- `generate.py`
- `wan/utils/replacement_masks.py`
- 新增 `wan/utils/boundary_refinement.py`
- `wan/utils/media_io.py`
- `scripts/eval/` 下新增 edge-quality smoke / metric 脚本


## 6. 具体实施方案

### 6.1 子步骤 A：定义像素域 refinement 的输入与输出契约

目标：

- 明确 refinement 阶段到底吃什么、输出什么

建议输入：

- 生成结果视频
- 原始驱动视频
- clean plate background
- `person_mask`
- `soft_band`
- `background_keep_mask`
- `replacement_strength`

建议输出：

- refinement 后的视频帧
- 中间 alpha / edge map / local blend mask
- debug 对照图

为什么必须先定义契约：

- 没有清晰输入输出，后面很容易把 refinement 写成一团隐式逻辑

### 6.2 子步骤 B：实现第一版确定性边缘融合

目标：

- 给当前系统增加一条确定性、可解释的像素域边缘精修路径

建议逻辑：

1. 背景区域：
   - 优先保留原背景或 clean plate
2. 主体区域：
   - 保留 diffusion 生成结果
3. 边界带区域：
   - 用 `soft_band` 作为 alpha 过渡控制
   - 允许加入轻量锐化或局部平滑
   - 必要时对边界颜色差做约束

可以把这一步理解为：

- 不是“重新生成边缘”
- 而是“更高质量地把生成主体和背景重新接起来”

### 6.3 子步骤 C：增加局部 edge-aware filter

目标：

- 处理边界带中的轻微软化和色边

建议方向：

- 仅在 transition band 内做局部处理
- 不动 free replacement 主区域
- 不动 hard background keep 区域

可考虑的处理：

- 局部 unsharp mask
- guided filter 风格的边缘保真平滑
- 小半径 bilateral / joint bilateral
- 基于背景差异的颜色泄漏抑制

第一版应保持简单，不要一次堆太多操作。

### 6.4 子步骤 D：把 refinement 纳入 generate debug 体系

目标：

- 让 refinement 结果可视、可比较、可调试

建议至少导出：

- refinement 前后对比视频
- alpha / band 可视化
- 边缘局部 crop 对照图
- 关键帧的 before/after PNG

这样后续调参时，能快速看出：

- 是边缘更清晰了
- 还是只是引入了 halo / 伪锐化

### 6.5 子步骤 E：把 refinement 做成严格可开关功能

目标：

- 确保 refinement 不会污染 baseline

建议提供参数，例如：

- `--boundary_refine_mode {none,deterministic}`
- `--boundary_refine_strength`
- `--boundary_refine_sharpen`
- `--boundary_refine_use_clean_plate`

必须确保：

- 关闭时输出与当前版本一致
- 打开时才进入 pixel refinement 路径


## 7. 推荐实施顺序

建议按下面顺序推进：

1. 定义 refinement 输入输出契约
2. 做最简单的 deterministic alpha-based boundary composite
3. 加入局部 edge-aware filter
4. 补完整 debug / lossless 对比输出
5. 再做参数搜索

顺序原则：

- 先有可工作的基本版
- 再逐步提高边缘质量
- 不要一开始就把 refinement 逻辑堆得太复杂


## 8. 验证与测试方案

### 8.1 必做验证

1. refinement 开关关闭时，与当前 generate baseline 输出一致
2. refinement 开关开启时，输出视频和 debug 产物都完整写出
3. 在 `ffv1` 或 `png_seq` 下完成前后对照
4. 长视频 clip seam 没有因为 refinement 明显恶化

### 8.2 建议增加的客观指标

应增加更专门的边缘指标，例如：

- boundary-band gradient strength
- foreground-background edge contrast
- refinement 前后 band 区域 MAD
- boundary halo ratio

这些指标不应替代主观评估，但能帮助快速筛掉明显坏配置。

### 8.3 主观评估重点

建议重点看：

- 头发边缘
- 手部边缘
- 衣摆 / 袖口
- 快速运动时的人物轮廓
- 边缘与背景交界处是否有亮边 / 暗边

### 8.4 成功标准

本步骤算通过，应满足：

- 在无损输出下，边缘清晰度相对当前版本有可见提升
- 不明显增加 halo、闪烁、色边
- seam 与 temporal continuity 不明显变差


## 9. 风险

### 9.1 风险一：边缘更锐，但更假

这是最常见风险。  
如果 refinement 过强，容易出现：

- 伪锐化
- 黑边 / 白边
- alpha 过渡过于生硬

应对方式：

- 只在 boundary band 内作用
- 默认强度保守
- 必须保留 baseline 对照

### 9.2 风险二：边缘更清晰，但时序更抖

边缘处理如果逐帧独立，可能会导致：

- 帧间 band 区域亮度抖动
- 衣角边缘抖动
- 细节闪烁

应对方式：

- 只使用轻量局部操作
- 后续可再补时间一致性约束，但第一版先保持确定性和可控性


## 10. 回退策略

本步骤必须支持完全回退：

- refinement 通过单独参数开关控制
- 关闭时完全回到当前 generate baseline
- debug 与评估脚本应支持同时比较 baseline / refined 两路输出


## 11. 完成标志

本步骤完成的标志是：

1. 当前 generate 之后多出一条正式的 pixel-domain boundary refinement 路径
2. 该路径只作用于边界带，不污染主体和背景主区域
3. refinement 前后可通过无损输出稳定对照
4. 用户最关心的边缘模糊问题，在真实素材上有肉眼可见改善
