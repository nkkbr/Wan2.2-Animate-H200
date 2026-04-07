# Step 06: 结构级 Reference Normalization

## 1. 步骤目标

本步骤的目标是把当前的 reference normalization 从“bbox 级画幅与尺度对齐”，提升到 **structure-aware 的人物结构对齐**。

这一步的目的不是单纯让参考图“放得更合适”，而是减少因为参考人物与驱动视频人物在结构上不匹配而带来的问题，例如：

- 肩宽不自然
- 腿长比例不对
- 人物在画面中的占比失真
- 肢体边缘更容易挤出 replacement 区域

这一步是 supporting condition 的高价值增强项，放在后面做，是因为它建立在：

- preprocess 已稳定
- H200 preprocess profile 已建立
- 边缘建模和背景条件已进一步增强

之上。


## 2. 为什么要做这一步

### 2.1 当前 reference normalization 已经解决了“明显不合适”，但还没解决“结构不匹配”

当前系统已经有：

- bbox 级尺度匹配
- 画幅与落位调整

这已经可以解决：

- 参考图是竖图、视频是横图
- 参考人物过大或过小
- 人物落位明显偏离视频主体

但仍然无法解决：

- 头身比差异
- 肩宽差异
- 上下身比例差异
- 宽松服装和紧身服装的体积差异

### 2.2 结构失配最终会放大到边缘质量问题

reference 结构不匹配不仅影响“像不像”，还会影响：

- replacement 区域是否足够覆盖人物
- 生成人物是否被迫压缩/拉伸
- 边缘是否更容易漏出 mask 外

因此，这一步虽然不直接处理边缘 band，但对最终边缘质量仍然有实质影响。

### 2.3 这一步是前面几步之后的条件优化

如果在 preprocess 不稳定、边缘建模还不够细之前就做结构级 normalization，容易把问题混在一起。  
因此，这一步应放在后面，作为条件质量的进一步提升。


## 3. 本步骤的交付结果

本步骤完成后，应至少交付：

1. 一套 structure-aware reference normalization 策略
2. 比当前 bbox 级归一化更细的对齐信息
3. 新的 metadata 和 QA preview
4. reference normalization 与 replacement region 的联动机制


## 4. 设计原则

### 4.1 不追求“完全重建参考人物姿态”

reference 图是静态全身像，而驱动视频里的人物会持续运动。  
因此，本步骤的目标不是让参考图在 preprocess 阶段就“变成视频姿态”，而是：

- 让参考图的尺度、结构和身体占比更接近驱动目标

### 4.2 结构级归一化应建立在人体关键区域上

建议至少区分：

- 头脸区域
- 上半身 / torso
- 下半身 / legs

如果仍然只用单一 bbox 统一缩放，结构问题无法真正改善。

### 4.3 第一版要可解释、可回退

结构级归一化不应一开始就做成复杂黑盒变形。  
第一版应优先做：

- 可解释的分区域尺度约束
- 结构比例估计
- 可视化 preview


## 5. 代码改动范围

本步骤预计会改动如下模块：

- `wan/modules/animate/preprocess/reference_normalization.py`
- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/utils/animate_contract.py`
- `wan/animate.py`
- `scripts/eval/` 下新增 reference normalization 检查脚本


## 6. 具体实施方案

### 6.1 子步骤 A：把 reference normalization 的结构信息显式建模

目标：

- 不再把 reference 只看成一个 bbox

建议增加的结构信息：

- 头脸 bbox
- 肩线宽度
- torso 高度
- 下半身长度
- 全身高度与宽度比例

这些信息可以由：

- reference pose
- 参考图上的人体关键点
- 驱动视频前若干帧的 pose 统计

共同得到。

### 6.2 子步骤 B：建立驱动目标的结构先验

目标：

- 不只知道“目标人物大致在画面哪里”，还知道“目标人物大致长什么比例”

建议做法：

- 从驱动视频前若干帧统计：
  - 头部占比
  - 肩宽
  - torso 高度
  - 下半身长度
- 对这些统计做中位数或鲁棒平均

这样可以得到：

- 一个更适合 reference 对齐的目标结构模板

### 6.3 子步骤 C：实现分区域尺度约束，而不是整图统一缩放

目标：

- 让 reference 对齐从“单一比例缩放”升级成“结构感知的画面归一化”

建议方向：

- 头脸区域：
  - 更关注脸部大小不要明显失真
- torso：
  - 更关注肩宽与躯干占比
- 下半身：
  - 更关注整体长度与落点

第一版不建议做复杂非刚性 warping。  
可以先做：

- 结构比例驱动的统一缩放与平移
- 对 extreme mismatch case 做 clamp

### 6.4 子步骤 D：让 normalization 和 replacement region 联动

目标：

- 参考图归一化结果不再是独立处理，而是和最终可生成区域相匹配

建议做法：

- 归一化时读取当前 replacement region 的空间预算
- 避免 reference 人物经过 normalization 后：
  - 宽度明显超出可替换区域
  - 高度严重挤压或漂移

这一点很重要，因为 reference normalization 做得再漂亮，如果和 replacement mask 空间预算不匹配，最终还是会放大边缘问题。

### 6.5 子步骤 E：补足 preview 与 metadata

目标：

- 让结构级 normalization 可被观察、可被 debug

建议至少导出：

- 原始 reference
- bbox normalization preview
- structure-aware normalization preview
- 关键结构线 overlay

metadata 中应记录：

- normalization mode
- 目标结构统计
- scale clamp 是否触发
- 各关键区域的目标比例


## 7. 推荐实施顺序

建议按下面顺序推进：

1. 定义结构信息与 metadata 扩展
2. 建立驱动目标结构先验
3. 实现分区域尺度约束
4. 让 normalization 与 replacement region 联动
5. 增加 preview / QA / regression

顺序原则：

- 先有结构描述
- 再有结构对齐
- 最后再让它真正影响 replacement


## 8. 验证与测试方案

### 8.1 必做验证

1. `none` 模式继续兼容当前 reference 路径
2. 新模式下 `src_ref_normalized` 正常输出
3. generate 侧可无歧义读取新的 reference artifact
4. preview 中能清楚看出结构对齐差异

### 8.2 应重点对照的 case

建议选下面几类 reference：

- 体型接近的 reference
- 体型明显更瘦 / 更宽的 reference
- 长裙 / 大衣等外轮廓差异明显的 reference
- 构图非常紧或非常松的 reference

重点看：

- 人物落位
- 头身比
- 肩宽
- 下半身比例
- 最终 replacement 后的边缘是否更少挤出

### 8.3 成功标准

本步骤算通过，应满足：

- structure-aware normalization 相较 bbox-only 有更稳定的构图与体型匹配
- 不显著增加 reference 预处理失败率
- 在真实素材上减少由参考图结构失配导致的边缘和形体问题


## 9. 风险

### 9.1 风险一：结构估计本身不稳定

如果 reference pose 或驱动前几帧 pose 估计不稳，structure-aware normalization 可能建立在错误先验上。

应对方式：

- 使用鲁棒统计
- 对极端值做 clamp
- 提供回退到 bbox-only 的开关

### 9.2 风险二：过度纠正 reference，破坏原始身份感

如果 normalization 过强，可能导致：

- 脸部比例被拉坏
- 人物整体观感失真
- reference 身份特征被削弱

应对方式：

- 优先保住头脸和主体视觉身份
- 对 torso / legs 的修正强度更保守


## 10. 回退策略

必须完整保留当前 bbox-only normalization 路线。  
任何时候如果 structure-aware 路径：

- 不稳定
- 质量无明显收益
- 明显破坏身份感

都应允许系统回退到当前实现。


## 11. 完成标志

本步骤完成的标志是：

1. reference normalization 已从 bbox-only 升级到 structure-aware 可选模式
2. metadata 与 QA 可完整表达新模式
3. generate 可正确消费归一化后的 reference
4. 在真实素材上，因 reference 结构失配造成的形体与边缘问题有明确下降
