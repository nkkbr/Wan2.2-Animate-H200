# Step 09: Clip Continuity 与 Seam Blending

## 1. 目标

系统性改善当前 animate replacement 的 clip 级连续性机制，减少长视频中的：

- 段间接缝
- 背景局部闪烁
- 人物边缘跳变
- continuity 误差累计

这一步是 generate 侧最重要的质量优化步骤之一。


## 2. 为什么现在做

当前 generate 的连续性机制非常朴素：

- 下一段拿上一段最后 `refert_num` 帧做 temporal guidance
- 输出时把重叠帧直接裁掉
- 然后硬拼接

这会导致：

- overlap 区域完全没有融合
- 接缝只能靠模型自己“恰好接上”
- 长视频中误差不断累积

相比继续单纯拉高 `sample_steps`，改善 continuity 往往更划算。


## 3. 本步骤的范围

本步骤包括：

- 放开 `refert_num` 限制
- 引入 overlap blending
- 设计人物区域与背景区域不同的融合策略
- 增加 seam 级 debug 导出

本步骤不做：

- latent memory
- 改模型结构


## 4. 当前问题

### 4.1 overlap 过短且被硬裁切

当前 `refert_num` 只允许 `1` 或 `5`，而且生成后直接把后一段 overlap 丢掉。

### 4.2 没有显式 seam 处理

当前没有：

- 时间 alpha blending
- 区域感知 blending
- seam 质量评估

### 4.3 长视频累计误差明显

每一段的轻微漂移，都会进入下一段的 temporal guidance。  
没有 seam 处理时，这种误差更容易堆积。


## 5. 设计原则

### 5.1 先做低风险的像素级 overlap blending

第一版不建议直接进入 latent-level 拼接。  
更可控的路线是：

- 先在解码帧层做 overlap blending

### 5.2 背景和人物应区别对待

对 replacement 来说：

- 背景区域更适合保守融合
- 人物区域更适合渐变融合

### 5.3 放开 `refert_num`，但不要一次性瞎扩

建议先支持任意正整数，再重点验证：

- `5`
- `9`
- `13`


## 6. 具体实施方案

### 6.1 放开 `refert_num`

建议把当前硬编码约束：

- `1 or 5`

改为：

- `0 < refert_num < clip_len`

并在 CLI 文档中明确推荐区间。

### 6.2 引入统一 overlap blending

基础方案：

- 保留 overlap 区间帧
- 在 overlap 上做时间 alpha 融合
- alpha 从上一段到下一段线性过渡

第一版即可显著减少硬接缝。

### 6.3 引入 mask-aware blending

在 replacement 场景下，推荐进一步区分：

- 背景区域
- 人物区域
- 边界带

建议策略：

- 背景区域：更偏向上一段，避免闪动
- 人物区域：按时间渐变融合
- 边界带：使用更平滑的 alpha

如果 Step 07 已经完成 soft mask，这里应直接复用 soft/band 信息。

### 6.4 seam debug 导出

建议每个拼接点导出：

- seam 前后帧对比
- overlap blending 前后对比
- seam score 统计

否则很难精确判断 continuity 改动有没有价值。


## 7. 需要改动的文件

建议涉及：

- `generate.py`
- `wan/animate.py`

必要时新增：

- overlap blending 工具
- seam 评测工具


## 8. 如何实施

建议顺序如下：

1. 先放开 `refert_num` 参数约束。
2. 再做最基础的时间 alpha blending。
3. 再加入 mask-aware blending。
4. 最后补 seam debug 导出与评测。

这个顺序很重要。  
先做最简单的 blending，验证“接缝问题确实能降”，再逐步增加复杂度。


## 9. 如何检验

### 9.1 seam 视觉检验

重点看：

- 相邻 clip 连接处
- 背景边缘
- 人物轮廓

### 9.2 seam score

对每个拼接点前后若干帧统计差异，比较：

- baseline
- 纯 blending
- mask-aware blending

### 9.3 长视频检验

重点看：

- 长视频后半段是否更稳定
- 是否减少错误累计


## 10. 验收标准

只有满足下面条件，才算本步骤完成：

- `refert_num` 可配置范围放开
- overlap blending 能稳定运行
- benchmark 上 seam 可见度明显下降
- 长视频 continuity 更稳


## 11. 风险与回退

### 风险

- overlap 太长导致新段过度依赖旧段
- blending 不当导致 ghosting

### 回退策略

- overlap 长度做可配置
- 先仅启用时间 alpha blending
- 人物/背景分开调参


## 12. 本步骤完成后的收益

这一步通常会带来最直观的画质提升之一。  
它直接针对当前 generate 侧最明显的结构性短板：clip 之间接得太硬。
