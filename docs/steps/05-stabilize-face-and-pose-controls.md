# Step 05: 稳定 Face / Pose 控制信号

## 1. 目标

显著提升 `src_face` 与 `src_pose` 的时序稳定性，让它们从“可用控制信号”升级为“高质量控制信号”。

这一步的目标非常明确：

- 降低 face crop 抖动
- 降低 pose 骨架抖动
- 降低低置信关键点对后续链路的污染


## 2. 为什么现在做

在当前链路里：

- face crop 直接影响 `motion_encoder` 的输入
- pose 关键点既影响 `src_pose.mp4`，也影响 SAM2 提示点

所以 face / pose 的不稳定，不是局部问题，而是会向后放大到：

- 表情控制不稳
- 骨架视频抖动
- mask 跟踪提示漂移
- 边缘分割质量下降

因此它必须先于 mask 高级优化落地。


## 3. 本步骤的范围

本步骤包括：

- face bbox 的 confidence-aware 计算
- face bbox 的时序平滑与回退
- whole-body 关键点的置信度过滤
- body / hand / face 关键点的时序平滑
- 简单异常点修正与插值
- 对应的 QA 可视化

本步骤不做：

- SAM2 提示策略升级
- soft mask
- background inpainting


## 4. 当前问题

### 4.1 face bbox 抖动

当前 face bbox 基本是：

- 直接取 face keypoints min/max
- 固定比例放大

这会导致：

- 遮挡时 bbox 大小突变
- 侧脸时 crop 抖动
- 模糊帧或置信度低时 bbox 飘移

### 4.2 pose 关键点逐帧独立

当前 pose pipeline 本质是逐帧 whole-body 检测，没有显式 temporal smoothing。

这会导致：

- 手部和脸部点抖动明显
- 低置信点在局部突然跳跃
- 骨架渲染结果不稳定

### 4.3 没有显式回退与异常约束

当前逻辑缺少：

- 点缺失时的上一帧回退
- 速度异常点裁剪
- bbox 尺寸变化率限制


## 5. 设计原则

### 5.1 confidence-aware 优先于盲平滑

不能对所有点一视同仁地平滑。  
应优先使用高置信点，低置信点走回退或补偿逻辑。

### 5.2 先稳，再求灵敏

对 replacement 来说：

- 控制信号稳定

通常比：

- 控制信号极度灵敏

更重要。

### 5.3 body / face / hand 分开处理

三类关键点的动态特征不同：

- body 较稳定
- hand 高频变化大
- face 细节密且易受遮挡

因此平滑参数不应完全共用。


## 6. 具体实施方案

### 6.1 face bbox: 高置信点筛选

建议 face bbox 不再直接使用所有 face 点，而是：

1. 先过滤低于阈值的 face points
2. 点数足够时再计算 bbox
3. 点数不足时回退到上一帧 bbox

建议新增参数：

- `--face_conf_thresh`
- `--face_min_valid_points`

### 6.2 face bbox: 时序平滑

建议对 bbox 的以下量分别做平滑：

- 中心点 `(cx, cy)`
- 尺寸 `(w, h)`

可选方法：

- EMA
- One-Euro
- 简化 Kalman

第一版建议先用 EMA，原因是：

- 易实现
- 稳定
- 易调参

建议新增参数：

- `--face_bbox_smooth_method`
- `--face_bbox_smooth_strength`
- `--face_bbox_max_scale_change`

### 6.3 face bbox: 异常回退

建议增加三类保护：

- 面积变化率过大时回退到上一帧或限制变化
- 中心点位移过大时做 clamp
- 连续低置信帧时启用短时保持策略

### 6.4 pose: 关键点级平滑

建议对关键点坐标单独做时序平滑。

可按类别分别设定强度：

- body：中等平滑
- hand：偏保守平滑，避免把真实快速动作磨掉
- face：中高平滑，但需结合置信度

建议新增参数：

- `--pose_smooth_method`
- `--pose_smooth_strength_body`
- `--pose_smooth_strength_hand`
- `--pose_smooth_strength_face`

### 6.5 pose: 低置信点修复

建议对低置信点使用：

- 前后帧插值
- 上一帧回退
- 邻近骨架点几何约束

第一版可以先做：

- 上一帧回退
- 短缺失插值

### 6.6 pose: 异常速度裁剪

对关键点的帧间位移引入速度上限：

- 超出阈值时不直接采用新值
- 先做裁剪或回退

这一步尤其适合抑制偶发错误点。

### 6.7 QA 可视化

建议新增以下调试产物：

- `face_bbox_overlay.mp4`
- `pose_overlay.mp4`
- `face_bbox_curve.json`
- `pose_conf_curve.json`

这样后续能快速判断“问题到底是 bbox 抖了，还是后续模型没跟上”。


## 7. 需要改动的文件

建议涉及：

- `wan/modules/animate/preprocess/pose2d.py`
- `wan/modules/animate/preprocess/utils.py`
- `wan/modules/animate/preprocess/process_pipepline.py`

必要时新增：

- smoothing 工具模块
- QA 可视化工具


## 8. 如何实施

建议顺序如下：

1. 先加 face bbox 的高置信筛选。
2. 再加 bbox 平滑与异常回退。
3. 再加 pose 关键点平滑。
4. 再加低置信点修复。
5. 最后导出 QA 可视化。

这个顺序很重要。  
如果先做复杂平滑，但 bbox 计算本身就不稳定，收益会打折。


## 9. 如何检验

### 9.1 face bbox 稳定性

至少检查：

- bbox 中心点曲线
- bbox 面积曲线
- 快速动作处的视觉稳定性

### 9.2 pose 稳定性

至少检查：

- 骨架视频是否明显减少抖动
- 手部与脸部点是否仍保留必要动态
- 异常跳点比例是否下降

### 9.3 对后续链路的连带收益

在固定 benchmark 上，比较改动前后：

- `src_face` 的稳定性
- `src_pose` 的稳定性
- SAM2 mask 是否更稳


## 10. 验收标准

只有满足下面条件，才算本步骤完成：

- face bbox 在 benchmark 上明显更稳
- pose 骨架抖动明显下降
- 没有因过度平滑造成明显动作迟滞
- QA 可视化能清楚展示改进效果


## 11. 风险与回退

### 风险

- 过度平滑导致表情和动作变钝
- hand 点平滑过强，抹掉真实细节

### 回退策略

- face / body / hand 分别调节
- 所有平滑强度均做可配置
- 先保守启用，再逐步加大


## 12. 本步骤完成后的收益

这一步完成后，后续两条链都会直接受益：

- generate 中的 face motion 更稳
- preprocess 中的 SAM2 提示点更靠谱

它属于典型的“低风险高收益”基础步骤。
