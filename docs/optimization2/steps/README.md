# Wan-Animate Replacement 下一阶段实施路线

## 1. 文档目标

本目录基于上层文档：

- `docs/optimization2/wan_animate_replacement_next_phase_optimization.md`

把“下一阶段还需要继续优化什么”收敛成一条可执行的、分步骤的工程路线。

这条路线不再覆盖已经完成的 12 个步骤，而是专门解决当前系统剩下的最关键问题：

- 真实 SAM2 preprocess 不稳定
- H200 在 preprocess 侧明显没有被用满
- replacement 边缘控制仍然主要停留在 latent 空间
- 背景与参考图条件虽然已有升级，但仍不足以支撑“尽可能高精度”的目标


## 2. 下一阶段的总目标

下一阶段的目标不是“再做一些优化”，而是把当前系统从“已经明显优于初始版本”推进到“更接近高精度 production replacement”的状态。

为此，后续工作必须围绕三件事展开：

1. 让 preprocess 真正稳定地产生高质量人物区域
2. 让边缘控制从 latent 级逼近 pixel 级
3. 让 H200 的硬件预算更直接地转换成最终画质


## 3. 与上一轮路线的区别

上一轮 12 个步骤主要解决的是：

- 工程契约
- lossless pipeline
- mask / background / continuity / guidance 的系统化增强

这一轮则更聚焦在“高精度上限”的问题，尤其是用户最关心的边缘质量。

因此，这一轮的顺序不再是“从基础设施开始全量梳理”，而是改成：

- 先修当前最阻塞的真实问题
- 再释放 H200 的有效算力
- 再直接攻击边缘质量
- 最后再提升背景与参考图条件的精细度


## 4. 实施阶段

### 阶段 A：修复当前最关键的阻塞问题

- Step 01: 真实 SAM2 preprocess 稳定性修复
- Step 02: H200 preprocess 运行 profile 重建

### 阶段 B：把边缘质量从 latent 级推进到 pixel 级

- Step 03: 像素域边缘 refinement
- Step 04: parsing / matting 融合边界建模

### 阶段 C：完善 supporting conditions

- Step 05: 视频一致性的 clean plate background
- Step 06: 结构级 reference normalization


## 5. 推荐实施顺序

建议严格按下面顺序推进：

1. `01-sam2-preprocess-stability.md`
2. `02-h200-preprocess-runtime-profile.md`
3. `03-pixel-domain-boundary-refinement.md`
4. `04-parsing-matting-fusion-boundaries.md`
5. `05-video-consistent-clean-plate.md`
6. `06-structure-aware-reference-normalization.md`


## 6. 为什么是这个顺序

### 6.1 先修 Step 01

如果真实视频上的 preprocess 不能稳定跑完，那么任何“更高精度 mask”“更好背景”“更好 reference”都没有稳定输入基础。

### 6.2 Step 02 紧随其后

当前 preprocess 中对 SAM2 的运行方式明显偏保守。  
在真实稳定性问题收敛之后，应该立即重新 benchmark H200 配置，否则后续所有高分辨率分析都会继续浪费硬件。

### 6.3 Step 03 和 Step 04 是真正决定边缘上限的主线

当前系统已经有 soft mask、soft band、boundary-aware replacement，但这些主要还是 latent 级控制。  
真正继续提升边缘质量，必须引入像素域 refinement，并让 preprocess 能提供更细的边界建模信号。

### 6.4 Step 05 和 Step 06 放在后面

clean plate 和 reference normalization 当然重要，但它们不是当前边缘问题的第一主因。  
它们应在 preprocess 稳定、H200 运行路径和边缘建模增强完成后再继续抬高上限。


## 7. 每个步骤文档包含什么

每个步骤文档都必须回答下面这些问题：

- 为什么要做这一步
- 这一步具体改哪些模块
- 如何分阶段实施
- 做完后如何客观验证
- 什么情况下算通过
- 有哪些风险
- 失败后如何回退

文档目标不是“写建议”，而是让后续实施者在打开文档后可以直接开始改代码和做实验。


## 8. 推荐执行方式

建议采用下面的执行纪律：

1. 每次只推进一个步骤
2. 当前步骤完成前，不提前引入下一个步骤的大改动
3. 每一步都必须有真实素材 smoke 和回归验证
4. 每一步结束后，都重新评估是否还值得继续推进下一步


## 9. 交付标准

这一轮路线的交付标准不是“文档齐全”，而是：

- 真实 preprocess 能稳定通过
- H200 上 preprocess profile 更积极且可重复
- 边缘质量在无损输出下可见提升
- 背景与参考图条件的增强不会引入新的不稳定性

只有达到这些条件，才能说当前系统开始逼近“高精度 replacement”目标。
