# Step 08: Clean Plate 背景构建

## 1. 目标

把当前的 `src_bg` 从“人物区域被抹掉的挖洞视频”，升级为更接近真实背景的 clean plate 条件。

这一步的目标不是为了让背景单独更好看，而是为了提升 replacement 的整体稳定性：

- 减少人物区域内部的背景脏纹理
- 降低边界接缝压力
- 降低长视频中的背景闪烁


## 2. 为什么现在做

当前 generate 侧把 `src_bg` 作为强条件之一编码进 `y_reft`。  
这意味着：

- `src_bg` 质量差，不只是局部问题
- 它会直接影响整个 replacement 的背景先验

当前的 `src_bg = frame * (1 - mask)` 本质上只是挖洞。  
这会让模型在人物区域内不得不同时做两件事：

- 生成人物
- 自行猜测被挡住的背景

这对长视频和复杂背景尤其不利。


## 3. 本步骤的范围

本步骤包括：

- clean plate 背景构建方案
- `src_bg` 的新产物定义
- preprocess 中的 background inpaint 路径
- generate 对新背景条件的兼容

本步骤不做：

- latent memory
- 多视角参考图


## 4. 当前问题

### 4.1 `src_bg` 不是背景，只是挖洞结果

它不包含人物遮挡后的真实背景内容。

### 4.2 背景边界与人物边界耦合

一旦人物边界控制不稳，当前挖洞背景很容易在交界处产生脏纹理和局部闪烁。

### 4.3 长视频误差更容易积累

因为当前背景区域在人物周围并不完整，后续每段生成都会在这一区域继续“猜”背景。


## 5. 设计原则

### 5.1 先有可用 clean plate，再追极致

第一版不必一开始就做到完美 video inpainting。  
只要比“直接挖洞”更稳定，就已经有价值。

### 5.2 clean plate 应作为独立产物

不要直接覆盖旧 `src_bg` 的定义。  
建议通过 metadata 区分：

- `bg_mode = "hole"`
- `bg_mode = "clean_plate"`

### 5.3 与 mask 同步设计

background clean plate 的质量高度依赖 mask。  
因此这一步必须建立在前面 mask 质量提升之后。


## 6. 具体实施方案

### 6.1 定义背景模式

建议新增参数：

- `--bg_inpaint_mode {none,image,video}`

三种模式含义：

- `none`
  当前挖洞逻辑
- `image`
  逐帧或关键帧 image inpainting，再做时间稳定
- `video`
  真正 video inpainting

第一版建议先落 `image` 模式，工程风险更低。

### 6.2 image inpainting 路线

低风险版本建议：

1. 先依据高质量 mask 构建 inpaint 区域
2. 对关键帧或所有帧做 image inpainting
3. 对结果做时间稳定或局部传播

这条路线不完美，但比纯挖洞更容易快速验证收益。

### 6.3 video inpainting 路线

作为进阶方案，可后续引入真正的视频修复模型。

适合在下面条件满足时推进：

- 有稳定 benchmark
- 已经确认 clean plate 确实带来收益
- 愿意接受更高工程成本

### 6.4 generate 侧兼容

generate 不应假设 `src_bg` 一定是挖洞背景。  
应通过 metadata 读取背景模式，并在 debug 日志中打印。

第一版不需要改变 `y_reft` 的构造方式，只需兼容新的背景来源即可。

### 6.5 QA 产物

建议导出：

- 原始帧
- 挖洞背景
- clean plate 背景
- 背景差分图

这样很容易看出 inpainting 是否真的更稳，还是只是换了一种脏纹理。


## 7. 需要改动的文件

建议涉及：

- preprocess background 构建逻辑
- metadata schema
- `wan/animate.py` 的 background 读取与日志

必要时新增：

- inpainting 封装
- 背景时间稳定工具


## 8. 如何实施

建议顺序如下：

1. 先定义新的背景产物和 metadata 字段。
2. 实现 `bg_inpaint_mode=image`。
3. 让 generate 兼容读取新背景。
4. 输出 QA 对比产物。
5. 在 benchmark 上判断是否值得继续做 `video` 路线。

不要一开始就直接上视频修复模型。  
先验证“clean plate 思路本身是否值”。


## 9. 如何检验

### 9.1 背景视觉检验

重点看：

- 人物区域附近是否更少脏纹理
- 边界处是否更自然
- 背景结构是否更连续

### 9.2 长视频稳定性

重点看：

- 人物周围局部背景是否更少闪烁
- 后半段是否更少出现脏纹理积累

### 9.3 对照实验

至少比较：

- 挖洞背景
- clean plate 背景

并在相同参数、相同 seed 下对比。


## 10. 验收标准

只有满足下面条件，才算本步骤完成：

- preprocess 能稳定产出 clean plate 背景
- generate 能兼容消费该背景
- benchmark 上背景稳定性明显优于挖洞版本
- 没有明显引入新伪影


## 11. 风险与回退

### 风险

- inpainting 自身引入明显伪影
- 逐帧 image inpainting 时序不稳定

### 回退策略

- 保留 `bg_inpaint_mode=none`
- 先在复杂背景 case 上验证收益
- 若 image 路线收益不足，再考虑 video 路线


## 12. 本步骤完成后的收益

clean plate 一旦可用，replacement 的强条件质量会大幅提升。  
这通常会同时改善：

- 人物区域内部背景自然度
- 边界衔接
- 长视频时的局部稳定性
