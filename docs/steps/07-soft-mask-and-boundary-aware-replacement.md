# Step 07: Soft Mask 与边界感知 Replacement

## 1. 目标

把当前“硬二值人物区域”升级为“端到端可传递的软边界控制信号”，并让 generate 真正利用这些信息改善人物边缘与背景过渡。

这一步的目标是解决当前 replacement 中最典型的边界问题：

- 硬边
- 闪边
- 头发边缘不自然
- 人物与背景交界处跳变


## 2. 为什么现在做

当前链路的核心问题之一是：

- preprocess 端的 mask 已经偏硬
- generate 端还会进一步把它二值化并 `nearest` 下采样到 latent 分辨率

也就是说，当前系统几乎没有真正的“边界过渡区”。

如果不打通 soft mask：

- 背景保留区域和人物重生成区域之间只能硬切
- 很多边缘问题只能靠扩大 mask 或重画更多背景去弥补


## 3. 本步骤的范围

本步骤横跨 preprocess 和 generate：

- preprocess 输出 soft mask 或 boundary band
- generate 读取浮点 mask
- generate 在 latent 中使用更合理的 mask 下采样与组合方式
- 引入边界感知的 replacement 逻辑

本步骤不做：

- parsing/matting 融合
- latent memory


## 4. 当前问题

### 4.1 只有硬二值区域

当前人物区域和背景区域之间没有显式过渡带。

### 4.2 generate 端处理过于粗糙

当前 generate 端会：

- 读取单通道 mask
- 取反
- 用 `nearest` resize 到 latent 分辨率
- 重排成 4 通道时间打包 mask

这会导致边界量化严重。

### 4.3 无法表达不同强度的约束

当前 mask 只能表达：

- 这里保背景
- 这里重画人物

不能表达：

- 这里背景必须保留
- 这里允许柔性过渡
- 这里人物必须自由生成


## 5. 设计原则

### 5.1 soft mask 必须端到端落地

只在 preprocess 输出 soft mask，但 generate 仍按硬 mask 使用，是无效优化。

### 5.2 边界带应独立看待

推荐把区域分成三类：

- hard background keep
- transition band
- free replacement area

### 5.3 先做低风险版，再做更复杂版

第一版不必重写模型结构，只需先改：

- mask 存储格式
- mask 读取方式
- latent 下采样策略
- 条件构造逻辑


## 6. 具体实施方案

### 6.1 preprocess 输出 soft mask

建议输出至少一种下面的结构：

- 原始概率图
- logits
- boundary band

第一版建议最实用的形式是：

- `hard_mask`
- `soft_band`

其中：

- `hard_mask` 负责主体区域
- `soft_band` 负责边界过渡

### 6.2 generate 读取浮点 mask

generate 端不再把 `src_mask` 直接视为硬二值单通道，而是：

- 读取浮点 mask
- 保持 `[0,1]` 或 logits 语义
- 根据 metadata 明确它表示的是人物区域还是背景区域

### 6.3 改进 latent 下采样

当前 `nearest` 不适合 soft mask。  
建议替换为：

- `bilinear`
- `area`
- 或更明确的 boundary-aware 聚合方式

目标是保留边界强弱，而不是把一切硬量化。

### 6.4 设计三段式 replacement 约束

建议在 generate 端将人物 mask 派生为三类区域：

- 区域 A：硬保留背景
- 区域 B：边界带，允许渐变
- 区域 C：人物自由生成区

第一版可先通过 mask 值范围近似实现，例如：

- `> 0.9`
- `0.1 ~ 0.9`
- `< 0.1`

### 6.5 debug 导出

建议生成以下调试产物：

- soft mask 可视化
- boundary band 可视化
- latent 尺寸下的 mask 可视化
- final replacement mask 可视化

否则很难判断问题出在 preprocess，还是出在 generate 的下采样和使用。


## 7. 需要改动的文件

建议涉及：

- preprocess mask 输出逻辑
- `wan/animate.py`
- metadata schema

必要时新增：

- mask resize 工具
- mask debug 可视化工具


## 8. 如何实施

建议顺序如下：

1. 先定义 soft mask 的存储格式。
2. preprocess 产出 soft mask。
3. generate 读取浮点 mask。
4. 替换 latent 下采样逻辑。
5. 增加边界带逻辑。
6. 输出 debug 可视化。

这个顺序里，读取与下采样要早于边界带策略，否则后面看起来像“边界带没用”，实际可能是输入已经被量化掉了。


## 9. 如何检验

### 9.1 边界视觉检验

重点看：

- 头发边缘
- 手臂外轮廓
- 衣摆
- 人物与背景接触区域

### 9.2 稳定性检验

对长视频比较：

- 边缘闪烁是否减少
- 背景交界是否更自然

### 9.3 消融检验

至少比较三组：

- 硬 mask baseline
- soft mask only
- soft mask + boundary band


## 10. 验收标准

只有满足下面条件，才算本步骤完成：

- soft mask 能从 preprocess 传递到 generate
- latent 下采样后仍保留边界强度
- 边界视觉质量明显提升
- 没有显著扩大背景污染


## 11. 风险与回退

### 风险

- soft band 太宽，导致背景区域被过度重绘
- 下采样策略不当，反而引入模糊边界

### 回退策略

- hard mask 路径保留为对照
- transition band 宽度可配置
- 分阶段开启，不要一次把所有策略绑死


## 12. 本步骤完成后的收益

这一步是 replacement 边界质量的关键节点。  
它会显著降低“为了遮住旧人物，只能把 mask 越做越大”的被动局面。
