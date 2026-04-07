# Step 05: 视频一致性的 Clean Plate Background

## 1. 步骤目标

本步骤的目标是把当前 preprocess 中的 `clean plate background` 从“逐帧 image inpaint 的保底实现”，升级为更适合视频 replacement 的 **时序一致背景构建链**。

这一步的目标不是重新定义 replacement 主体生成逻辑，而是提高背景条件质量，从而间接改善：

- 人物边界附近的背景稳定性
- 背景保留区域的可信度
- 边缘融合时的脏边与闪烁


## 2. 为什么要做这一步

### 2.1 当前 clean plate 已经优于挖洞背景，但还不是最佳方案

当前系统已经比原版更进一步：

- 不再只把人物区域挖掉后直接拿去生成
- 已经支持 image inpaint 方式构建 clean plate

这一步已经有价值，但仍存在两个明显限制：

1. 逐帧处理容易带来帧间不一致
2. 被人物长期遮挡的区域，逐帧 inpaint 往往不稳定

### 2.2 背景质量会直接影响边缘观感

人物边缘是不是“脏”“糊”“闪”，不只取决于人物本身，还取决于：

- 背景是否稳定
- 背景与人物边缘颜色是否协调
- 边界带附近是否存在 inpaint 伪影

因此，clean plate 不是附属功能，而是影响最终边缘质量的重要 supporting condition。

### 2.3 当前系统还没有真正的视频一致性 background

当前 clean plate 仍更接近：

- 每帧单独补洞

而不是：

- 利用时序信息构建更稳定的视频背景先验

这也是下一轮值得继续投入的方向。


## 3. 本步骤的交付结果

本步骤完成后，应至少交付：

1. 一条正式的 video-consistent clean plate 路线
2. 与现有 image 路线并存的 background mode 机制
3. 更完整的 QA artifact 与背景质量对照
4. 在 generate/runtime/metadata 中记录 clean plate 来源


## 4. 设计原则

### 4.1 image 路线保留为保底路径

当前 `image` clean plate 路线虽然不够强，但具备：

- 实现简单
- 易调试
- 可作为保底 baseline

因此，不应删除当前 image 路线，而应在其上新增：

- `video` 模式

并保持两者可对照。

### 4.2 video clean plate 的目标是“稳定”，不是“完美复原所有遮挡区”

第一版不应把目标设为：

- 完全恢复人物后面所有被遮挡的真实背景

这在很多真实场景中本来就不可能。  
更合理的目标是：

- 让背景条件更平滑、更时序一致、更少伪影

### 4.3 clean plate 的使用场景要明确

需明确区分：

- 用于 generate 条件的 background
- 用于 debug/评估参考的 background

这样可以避免把“看起来更干净的展示图”误当成真正进入生成模型的条件输入。


## 5. 代码改动范围

本步骤预计会改动如下模块：

- `wan/modules/animate/preprocess/background_clean_plate.py`
- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/utils/animate_contract.py`
- `wan/animate.py`
- `scripts/eval/` 下新增 background consistency 检查脚本


## 6. 具体实施方案

### 6.1 子步骤 A：把 background mode 机制扩成正式双路径

目标：

- 不再把 `image` clean plate 当作唯一增强路径

建议 mode：

- `hole`
- `clean_plate_image`
- `clean_plate_video`

要求：

- metadata 明确记录模式
- QA artifact 中明确区分来源
- generate 侧 runtime stats 记录 background mode

### 6.2 子步骤 B：设计第一版 video-consistent clean plate

目标：

- 利用时间信息，让 clean plate 背景比逐帧 inpaint 更稳定

建议方向：

1. 基于相邻帧可见背景区域的聚合
2. 对长期被人物遮挡的区域，采用时序平滑重建
3. 对每一帧的 clean plate，不只看当前帧，而看局部时间窗口

第一版可以采用保守策略：

- 优先复用前后帧可见背景
- 对真正不可见区域再做 inpaint
- 最后做轻量时间平滑

### 6.3 子步骤 C：把人物 mask / soft band 与 inpaint region 解耦

目标：

- 避免 clean plate 构建只机械依赖单一硬 mask

建议做法：

- inpaint 区域不一定等于最终 replacement 区域
- 可使用：
  - hard foreground
  - soft boundary
  - background keep prior
  共同决定 clean plate 需要补的范围

这样做的好处是：

- 边缘附近的背景重建更稳
- 减少不必要的大面积 inpaint

### 6.4 子步骤 D：增加背景质量评估与 debug

目标：

- 不再只看背景是否“生成出来了”，而是系统评估它是否适合 replacement

建议导出：

- `background_hole`
- `background_clean_plate_image`
- `background_clean_plate_video`
- `background_diff`
- `background_temporal_diff`
- 关键帧 crop 对照

建议指标：

- temporal fluctuation
- inpainted area ratio
- band-adjacent background stability

### 6.5 子步骤 E：形成面向 generate 的推荐 background policy

目标：

- 明确什么情况下 generate 应优先使用哪种 background

建议策略：

- 默认：
  - 若 `clean_plate_video` 可用，则优先使用
  - 否则回退 `clean_plate_image`
  - 再否则回退 `hole`

并在文档中写清：

- 什么时候值得使用 `video`
- 什么时候为了速度使用 `image`


## 7. 推荐实施顺序

建议按下面顺序推进：

1. 扩展 background mode 机制
2. 实现第一版 `clean_plate_video`
3. 输出更完整的 QA artifact
4. 跑 image vs video 的真实对照
5. 给 generate 写正式默认策略

顺序原则：

- 先让模式和契约稳定
- 再提升背景质量
- 最后决定默认策略


## 8. 验证与测试方案

### 8.1 必做验证

1. 老 bundle 与当前 `clean_plate_image` 路径继续兼容
2. 新 `clean_plate_video` artifact 可正常写出
3. generate 可无歧义读取并记录新的 background mode
4. 在真实 10 秒素材上 image / video 可完成 AB 对照

### 8.2 应重点看的指标

- background fluctuation
- seam 附近背景差异
- 边界带附近背景稳定性
- clean plate 覆盖率
- 关键帧主观对照

### 8.3 成功标准

本步骤算通过，应满足：

- `clean_plate_video` 相较当前 `image` 路线更稳定
- 不显著增加 preprocess 总失败率
- 在边缘附近能减少背景闪烁和脏边


## 9. 风险

### 9.1 风险一：video clean plate 成本过高

视频一致性背景比逐帧 inpaint 更耗时、更耗显存。  
如果实现过重，可能影响整个 preprocess 实用性。

应对方式：

- 第一版先限制时间窗口大小
- 先做局部稳定，不追求全局最强复原

### 9.2 风险二：背景更平稳，但细节更假

背景时序平滑过强时，可能出现：

- 背景细节被抹平
- 纹理变塑料
- 边界区域与真实背景不匹配

应对方式：

- 把“稳定”与“细节保真”同时纳入评估
- 不接受只看 temporal score 的单一结论


## 10. 回退策略

必须完整保留当前 `clean_plate_image` 路线作为稳定回退。  
任何时候如果 `clean_plate_video`：

- 成本过高
- 不稳定
- 质量无明显收益

都应允许系统回退到 image 路线。


## 11. 完成标志

本步骤完成的标志是：

1. preprocess 正式具备 `clean_plate_video` 路线
2. metadata / runtime / debug 全部支持新背景模式
3. image vs video 可被系统对照
4. 在真实素材上，背景条件质量对最终边缘与稳定性有可见正向收益
