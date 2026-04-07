# Step 04: Parsing / Matting 融合边界建模

## 1. 步骤目标

本步骤的目标是在 preprocess 阶段引入 **人体 parsing / 人像 matting 与现有 SAM2 mask 的融合建模**，让 replacement 边界条件不再只来自单一 segmentation mask，而是形成更细粒度的三层结构：

1. 硬主体区域
2. 软边界过渡区域
3. 背景保留区域

这一步是当前系统继续提升边缘质量的关键 preprocess 补强项，目的是为 Step 03 的像素域 refinement 和 generate 侧的 boundary-aware replacement 提供更高质量输入。


## 2. 为什么要做这一步

### 2.1 当前 soft band 仍然来自单一 mask 推导

当前系统虽然已经有：

- `person_mask`
- `soft_band`
- `background_keep_mask`

但 `soft_band` 本质上仍然是从 segmentation mask 经过膨胀、边界带构造和模糊推导出来的。  
这种方式已经比纯硬 mask 好，但它仍然缺少对下面这些细节的理解：

- 头发丝
- 衣服薄边
- 手指与手掌细部
- 半透明或运动模糊边缘

### 2.2 复杂边界要靠更细的语义或 alpha 信息

对于人物替换来说，最难的不是主体区域，而是主体与背景交界处的像素过渡。  
这类问题通常需要：

- parsing 提供语义部件分解
- matting 提供更接近 alpha 的软边界

把两者与现有 SAM2 主体分割融合，才能真正把边缘质量继续向上推。

### 2.3 这一步是 Step 03 的输入增强，而不是替代

要特别强调：

- Step 03 是在像素域精修边缘
- Step 04 是在 preprocess 侧生成更好的边界控制信号

两者不是二选一，而是前后配合。


## 3. 本步骤的交付结果

本步骤完成后，应至少交付：

1. 一套 parsing / matting / SAM2 融合策略
2. 明确的输出语义：
   - hard foreground
   - soft boundary alpha / band
   - background keep prior
3. 新的 metadata 记录与 debug artifact
4. 可与当前单一 mask 路径做 AB 对照


## 4. 设计原则

### 4.1 SAM2 仍然是主体区域锚点

第一版不建议推翻当前 SAM2 管线，而应让 SAM2 继续担任：

- 单人主体范围的主锚点
- 时序跟踪的主结构

然后让 parsing / matting 负责提升边界细节，而不是替换整条主体跟踪链。

### 4.2 Parsing 和 matting 的职责要分开

建议职责如下：

- parsing：
  - 提供人体部件和衣物语义结构
  - 帮助识别头发、脸部、衣物、手等细粒度区域
- matting：
  - 提供更接近 alpha 的边界过渡
  - 帮助构建更自然的 soft band / transition alpha

### 4.3 第一版先解决“边界更细”，不是“全系统最复杂”

第一版不建议一开始就做过于复杂的多模型联合时序系统。  
应先建立稳定的融合框架，让系统能回答一个核心问题：

- 边界是不是明显更细、更可信了


## 5. 代码改动范围

本步骤预计会改动如下模块：

- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/utils/replacement_masks.py`
- `wan/utils/animate_contract.py`
- 新增：
  - `wan/modules/animate/preprocess/parsing_adapter.py`
  - `wan/modules/animate/preprocess/matting_adapter.py`
  - `wan/modules/animate/preprocess/boundary_fusion.py`
- `scripts/eval/` 下新增边界融合检查脚本


## 6. 具体实施方案

### 6.1 子步骤 A：定义新的边界语义

目标：

- 不再把所有边界信息都压缩成单一 `soft_band`

建议输出的标准语义：

- `hard_foreground`
  - 强主体区域
- `soft_alpha`
  - 0 到 1 的柔性边界/透明度先验
- `boundary_band`
  - 重点边界过渡区
- `background_keep_prior`
  - 背景保持强约束区域

为什么要先定义这些语义：

- 后续 generate 与 refinement 需要稳定消费这些信号
- 如果语义不清楚，融合结果会越来越难调试

### 6.2 子步骤 B：接入 parsing 模型作为语义边界补充

目标：

- 把头发、衣物、手等部件从“单纯连通区域”提升到“有语义结构的区域”

建议做法：

1. 为 parsing 模型建立独立 adapter
2. 在 preprocess 中以可开关方式运行
3. 从 parsing 结果中提取：
   - 头发区域
   - 头脸区域
   - 手部区域
   - 衣物外轮廓
4. 与 SAM2 主体区域做空间对齐

第一版目标不是“用 parsing 替代主体分割”，而是：

- 用 parsing 补强边界细节理解

### 6.3 子步骤 C：接入 matting 模型生成软边界 alpha

目标：

- 获取比当前 soft band 更接近真实边界过渡的柔性信号

建议做法：

1. matting 只在局部人物区域运行，避免整帧成本过高
2. 使用 SAM2 / parsing 给出人物 ROI
3. matting 输出 soft alpha
4. 将 alpha 与现有 hard mask 对齐到统一空间

第一版不要求完美时序一致，但必须保证：

- 边界更细
- 不出现大面积主体缺失

### 6.4 子步骤 D：设计融合逻辑，而不是简单相加

目标：

- 让三个来源各司其职，而不是互相打架

建议融合策略：

- SAM2：
  - 给出主体主范围
- parsing：
  - 修正部件边界，提供语义先验
- matting：
  - 提供最终软边界过渡

可以先采用保守逻辑：

- `hard_foreground` 主要由 SAM2 主体区给出
- `boundary_band` 由 SAM2 边界与 parsing / matting 边缘联合确定
- `soft_alpha` 主要来自 matting，但在明显越界时受 SAM2 主体约束

### 6.5 子步骤 E：把融合结果输出到现有 replacement pipeline

目标：

- 让新信号能够真正被后续 generate/refinement 消费

建议做法：

1. preprocess 输出新增 artifact
2. metadata 中显式记录：
   - boundary source
   - matting/parsing 是否启用
   - 各 artifact 的语义说明
3. generate 侧先保持兼容：
   - 若新 artifact 存在，则优先使用
   - 若不存在，则回退到当前 soft band 逻辑


## 7. 推荐实施顺序

建议按下面顺序推进：

1. 定义新边界语义与 metadata 契约
2. 接入 parsing adapter
3. 接入 matting adapter
4. 实现保守融合逻辑
5. 与当前 soft band 路径做 AB 对照
6. 接入 generate / refinement 消费

顺序原则：

- 先把输出语义设计清楚
- 再逐项增加信号来源
- 最后才改变后续消费逻辑


## 8. 验证与测试方案

### 8.1 必做验证

1. 新增 artifact 能正确写入 metadata
2. 老 bundle 仍能被 generate 正常消费
3. 新 bundle 在 generate/refinement 路径上可正常读取
4. QA overlay 能明显看出边界比当前版本更细

### 8.2 建议新增 QA artifact

- parsing overlay
- matting alpha preview
- fused boundary preview
- hard foreground / soft alpha / boundary band 三路分开展示

### 8.3 应重点比较的对象

建议至少对比以下两个版本：

1. 当前 `SAM2 + soft_band`
2. `SAM2 + parsing + matting + fused boundary`

重点看：

- 发丝
- 衣摆
- 手指边缘
- 快速动作的轮廓完整性

### 8.4 成功标准

本步骤算通过，应满足：

- 新的边界信号在真实素材上比当前 soft band 更细
- generate / refinement 消费新信号时不显著破坏主体区域
- QA 结果显示复杂边界确实得到补强


## 9. 风险

### 9.1 风险一：多模型输出彼此冲突

parsing、matting、SAM2 的边界可能不一致。  
如果融合逻辑过激，可能导致：

- 主体区域缺口
- 边界局部抖动
- 头发或衣角被裁掉

应对方式：

- 第一版让 SAM2 继续当主体锚点
- parsing / matting 只做边界补强，不主导主体删除

### 9.2 风险二：成本显著增加

如果 parsing 和 matting 都在整帧上高分辨率运行，preprocess 成本会显著上升。

应对方式：

- 限制在人物 ROI 内运行
- 优先让 H200 预算换到“更细边界”，而不是整帧重算


## 10. 回退策略

必须保留当前 `SAM2 + soft_band` 路径作为回退基线。  
任何时候都应允许：

- 关闭 parsing
- 关闭 matting
- 回退到旧 soft band 逻辑

这样可以保证新边界建模失败时，不会影响已有流程。


## 11. 完成标志

本步骤完成的标志是：

1. preprocess 已能输出融合后的更细边界信号
2. metadata 和 debug 体系完整支持这些新 artifact
3. generate/refinement 能兼容消费这些新信号
4. 在真实素材上，复杂边界的质量相较当前版本有明确提升
