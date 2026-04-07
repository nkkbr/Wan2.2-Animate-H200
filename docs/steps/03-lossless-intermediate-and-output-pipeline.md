# Step 03: Lossless 中间结果与输出管线

## 1. 目标

把 replacement 链路中的关键中间信号，从当前默认的有损视频格式升级为 lossless 或近无损格式，并让 generate 与最终输出都能消费这些高保真信号。

这一步的目标是提升信号保真，不是直接改模型。


## 2. 为什么现在做

当前链路里至少有两处明显的 fidelity 损失：

- preprocess 输出的 `src_mask/src_pose/src_face/src_bg` 主要是 mp4
- generate 最终输出也是 `libx264` 有损编码

这会导致几个现实问题：

- mask 边界被编码污染
- soft mask 无法真正传递
- pose / face 控制视频在回读时已经失真
- 做前后版本比较时，很难区分“模型差异”和“编码器差异”

如果不先解决这一层，后面即使做了 soft mask、better seam blending，收益也可能被 I/O 吃掉。


## 3. 本步骤的范围

本步骤包括：

- preprocess 中间结果的 lossless 存储方案
- generate 对多种输入格式的读取支持
- 最终输出增加 lossless / png_seq 选项
- 中间结果与最终输出的格式 metadata

本步骤不做：

- mask 算法升级
- background inpainting
- continuity 改造


## 4. 设计原则

### 4.1 优先选择结构清晰、调试友好的格式

对 replacement 优化阶段，最有价值的格式通常是：

- `png` 序列
- `npz/npy`

原因：

- 易于检查
- 易于局部读取
- 易于和 metadata 配合

### 4.2 区分“训练/调试保真格式”和“交付展示格式”

工程调试阶段应优先保真。  
用户交付阶段可以再额外导出 mp4。

### 4.3 preprocess 与 generate 必须联动升级

只改 preprocess 存储格式而 generate 端不支持读取，没有意义。  
只改最终输出而中间仍然有损，也不足以支撑后续高质量优化。


## 5. 建议的格式设计

### 5.1 `src_mask`

建议优先格式：

- `npz` 或 `npy`

原因：

- mask 是最不适合有损压缩的视频信号
- 后续 soft mask 需要保留浮点值

建议保存内容：

- `mask`: `[T, H, W]` 浮点或 uint8
- 可选 `logits`
- 可选 `soft_band`

### 5.2 `src_pose`

建议优先格式：

- `png` 序列
- 或 lossless 视频

原因：

- pose 可视化本身就是图像信号
- 逐帧调试时 png 最直观

### 5.3 `src_face`

建议优先格式：

- `png` 序列
- 或高质量无损编码

原因：

- face crop 直接决定 motion encoder 输入质量
- 有损编码容易在嘴型和眼部细节上引入噪声

### 5.4 `src_bg`

建议优先格式：

- `png` 序列
- 或高质量无损视频

原因：

- `src_bg` 在 generate 中属于强条件
- 其细节和边界对背景稳定性影响很大

### 5.5 最终输出

建议 `generate.py` 同时支持：

- `mp4` 展示输出
- `png_seq` 调试输出
- 近无损或无损视频输出


## 6. 具体实施方案

### 6.1 preprocess 增加 `save_format`

建议增加参数：

- `--save_format {mp4,png_seq,npz}`
- `--lossless_intermediate`

默认可以保持兼容旧行为，但高质量模式应切到 lossless。

### 6.2 metadata 写入格式

在 Step 02 的 metadata 基础上，进一步写清：

- 每个产物的格式
- 路径
- dtype
- shape

例如：

- `src_mask.format = "npz"`
- `src_face.format = "png_seq"`

### 6.3 generate 端 reader 抽象

建议不要在 `wan/animate.py` 里直接写死“总是用 VideoReader 读 mp4”。  
应抽象成统一 reader：

- `load_frames_from_mp4`
- `load_frames_from_png_seq`
- `load_mask_from_npz`

然后根据 metadata 分派。

### 6.4 最终输出 writer 抽象

建议在 `wan/utils/utils.py` 之上增加统一 writer 层：

- `save_video_mp4`
- `save_video_png_seq`
- `save_video_lossless`

避免把所有逻辑都继续堆在一个 `save_video()` 里。


## 7. 需要改动的文件

建议涉及：

- preprocess 入口与 pipeline
- `wan/animate.py`
- `generate.py`
- `wan/utils/utils.py`

必要时新增：

- 格式 reader / writer 工具模块


## 8. 如何实施

建议顺序如下：

1. 先定义 metadata 中的格式字段。
2. preprocess 新增 lossless 写出能力。
3. generate 新增 reader 抽象。
4. 最终输出新增 writer 抽象。
5. 在 benchmark case 上做 roundtrip 对比。

这个顺序里，reader 必须在 writer 之后吗？不是。  
真正的关键是：先有格式契约，再有读写实现。


## 9. 如何检验

### 9.1 像素一致性检验

对 `png_seq / npz` 的 roundtrip，检查：

- 原始数组与读回数组完全一致
- 对 mask，软边界数值不变

### 9.2 输入输出一致性检验

对同一组 preprocess 结果：

- 分别用 `mp4` 和 `lossless` 喂 generate
- 比较最终输出差异

目的不是要求画面完全相同，而是确认 lossless 路径没有功能回归。

### 9.3 可视化检验

随机抽取几帧比较：

- mask 边界
- face crop 细节
- bg 边缘细节

### 9.4 性能检验

记录：

- 磁盘占用
- 写入速度
- 读取速度

高保真格式允许更大，但不能大到难以工程落地。


## 10. 验收标准

只有满足下面条件，才算本步骤完成：

- preprocess 可输出至少一种 lossless 中间格式
- generate 可读取该 lossless 格式
- 最终输出支持至少一种高保真调试格式
- metadata 能完整描述格式信息
- roundtrip 检验通过


## 11. 风险与回退

### 风险

- `png_seq` 占用过大，管理成本高
- `npz` 虽保真，但不够直观
- 多格式支持让代码路径变复杂

### 回退策略

- 高质量模式默认使用 lossless
- 普通模式仍可保留 mp4
- 先只支持少数几种格式，不要一口气接入很多编码器


## 12. 本步骤完成后的收益

这一步完成后：

- 后续 soft mask 才真正有落地空间
- pose / face / bg 的质量判断更可信
- 评测结果更接近模型真实表现
- preprocess 与 generate 的边界会更清晰
