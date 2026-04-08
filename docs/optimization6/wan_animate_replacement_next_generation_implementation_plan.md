# Wan-Animate Replacement Next-Generation Implementation Plan

## 1. 背景与结论

`optimization2` 到 `optimization5` 已经把系统推进到了一个非常重要但也很清晰的阶段：

- `preprocess` 的识别能力已经显著增强：
  - 更稳定的 `SAM2` 预处理
  - 多阶段多分辨率分析
  - 完整 face stack
  - 多尺度 pose / motion stack
  - visibility-aware clean plate background
  - richer boundary artifacts
- `generate` 侧也已经做了大量实验：
  - richer signal conditioning
  - ROI refine
  - semantic boundary specialization
  - local edge restoration
  - generate-side candidate search
  - trainable edge adapter
  - H200 quality tiers

但最终结论已经比较明确：

> 当前这条“在既有 backbone 上做 heuristic / deterministic / 轻量 adapter 式边缘增强”的路线，已经接近收益上限，无法稳定把最终替换人物边缘做得显著更锐。

因此，`optimization6` 不应继续在这条线上做更多小修小补，而应切换到下一代路线：

1. 用更可信的人工审核边缘数据建立更强监督基准。
2. 接入真正高质量的外部 alpha / matting 模型。
3. 明确建立前景 / 背景解耦式 replacement contract。
4. 让 richer boundary signal 成为生成主条件，而不是只做后处理。
5. 将边界问题升级为局部生成式重建问题，而不是后处理修边问题。


## 2. 本阶段的总目标

`optimization6` 的目标不是“再试一次 sharpen”，而是：

> 建立一条能够真正把高质量边界信号转化为最终锐利边缘的生产级路线。

本阶段理想结果应满足：

- 在人工审核边缘基准上，至少一条方案明显优于 `optimization5` 最优结果：
  - `boundary_f1`
  - `trimap_error`
  - `alpha_mae`
  - `alpha_sad`
  - `hair_edge_quality`
  - `roi_gradient`
  - `roi_edge_contrast`
  - `halo_ratio`
- 在真实 10 秒 benchmark 上：
  - 边缘锐度出现肉眼可见的等级差异
  - seam 不显著恶化
  - background fluctuation 不显著恶化
  - face identity 不显著恶化
- H200 141GB 的预算主要用于：
  - 高质量 alpha / matting
  - 边界 ROI 局部生成式重建
  - 少量高价值候选搜索
  - 必要的训练型边缘模型


## 3. 你担心的核心问题：真的能找到、下载并正确使用高质量外部模型吗？

这个担心是对的，而且必须被写成正式工程约束。

`optimization6` 里，**外部模型接入本身就是一个独立的工程问题**，不能靠“临时搜一个 repo，然后直接接上”。

### 3.1 模型接入必须遵守的硬规则

任何外部模型在进入主线前，必须依次通过以下 8 个检查：

1. **任务适配性**
   - 模型是否真的解决当前问题，而不是名义上类似。
   - 例如：
     - portrait matting
     - video matting
     - human parsing
     - hair segmentation
     - ROI restoration

2. **许可证可接受**
   - 必须确认 license 与当前项目使用方式兼容。
   - 不允许引入用途受限或商业使用不明确的模型进入默认主线。

3. **推理接口可控**
   - 必须能被本项目当前环境以脚本方式稳定调用。
   - 不接受只能在作者 notebook 手工跑通、无法工程化接入的模型。

4. **权重来源可验证**
   - 必须记录：
     - 下载来源
     - commit / tag / release
     - 文件 hash
     - 依赖版本
   - 不能只写“从某仓库下载”。

5. **输入输出语义明确**
   - 必须明确：
     - 输入张量格式
     - 色彩空间
     - 尺寸限制
     - 输出是否是 alpha / trimap / logits / parsing labels
   - 不允许“先接进来看看结果再说”。

6. **在 reviewed benchmark 上有客观价值**
   - 不能只看 demo 图好看。
   - 必须在 `optimization6` 的 reviewed benchmark 上比当前基线更优，至少在一组关键指标上显著胜出。

7. **可回退**
   - 新模型接入后，必须保留：
     - legacy path
     - fallback path
     - explicit feature flag

8. **运行成本可接受**
   - 即使 H200 资源很大，也不能把主链做成不可重复生产的巨型实验。
   - 必须有：
     - default profile
     - high-quality profile
     - extreme profile

### 3.2 外部模型选型方法

对每一类外部模型，都不先假定某个单一模型一定最好，而采用“候选清单 -> 小规模接入 -> reviewed benchmark AB -> 冻结”的流程。

正式流程：

1. 列出候选模型清单
2. 逐个做最小接入适配层
3. 先跑 synthetic correctness
4. 再跑 reviewed benchmark 关键帧
5. 只保留客观上明显优于当前基线的候选
6. 进入真实 10 秒 smoke
7. 通过 gate 后再允许接入主线

### 3.3 不能接受的做法

以下做法在 `optimization6` 中视为不合格：

- 仅凭主观感觉选模型
- 不记录权重来源和 hash
- 直接把外部模型接成默认路径
- benchmark 没赢就继续硬调参数
- 为了让新模型看起来更好，修改 benchmark 定义本身


## 4. optimization6 的总体设计原则

### 4.1 先强数据，再强模型

没有更强的人工审核边缘数据，后续再强的 alpha / reconstruction / training 路线，都可能被现有基准的偏置误导。

### 4.2 把边缘当作独立对象，而不是整图生成后的副产物

当前问题已经证明：

- 整图生成 + 修边，不够
- 更好的策略应该是：
  - 前景
  - alpha
  - 背景
  - 边界 ROI

分别建模，再复合。

### 4.3 优先做“真正会长细节”的模块

后续优先级应向：

- matting
- ROI generative reconstruction
- trainable edge model

倾斜，而不是继续向 deterministic 后处理倾斜。

### 4.4 H200 应该花在高价值节点，而不是低价值 sweep

H200 的额外预算应优先给：

1. 高质量 alpha / matting
2. ROI 局部生成式重建
3. 小规模可训练边缘模型
4. 高价值少量候选搜索

而不是给：

- 更多 heuristic 候选
- 更多 sharpen 参数 sweep
- 更多无差别全图采样

### 4.5 每一步都必须做“实现 -> 测评 -> 再改”闭环

`optimization6` 每个工作流默认最多 **3 轮**：

1. 实现
2. 全面测评
3. 若未达目标，则改方法或改参数
4. 再测评
5. 到第 3 轮仍未达标，则冻结 best-of-three，并明确写入 findings


## 5. optimization6 的五条主工作流

下面这五条工作流，正对应我们当前最值得继续的方向。

### Workstream A: 升级人工审核 edge benchmark

对应目标：

- 把 `optimization5` 的 reviewed mini-set 升级成更强、更多样、更独立的边缘基准

为什么必须先做：

- 当前 benchmark 与历史 baseline 仍有较强同源性
- 这会让很多“新方案没有超过 baseline”这个结论变得不够彻底可信

应新增的标注对象：

- `soft_alpha`
- `trimap_unknown`
- `hair_edge`
- `face_contour`
- `hand_edge`
- `cloth_boundary`
- `occluded_boundary`

最低要求：

- 30 到 50 帧关键帧
- 多样场景：
  - 头发散开
  - 侧脸
  - 手遮挡
  - 衣摆
  - 快动作
  - 强背景对比

输出：

- 更强 reviewed benchmark
- 更严格 hard gate
- 更适合训练的小规模 supervised set


### Workstream B: 高质量外部 alpha / matting 模型接入

对应目标：

- 用真正更强的外部模型，把 `soft_alpha / trimap_unknown / hair_alpha / alpha_confidence` 的上限抬高

为什么必须做：

- 当前 heuristic-compatible alpha 路线已经证明上限有限
- 边缘模糊很多时候是 alpha 不准，而不是 sharpen 不够

实施分层：

1. 候选搜集
   - portrait matting
   - video matting
   - hair-aware alpha
2. 候选适配层
   - 单独 adapter，不能直接污染主 pipeline
3. reviewed benchmark AB
4. 真值通过后再接入 preprocess 主链

必须产出的 artifact：

- `soft_alpha_v3`
- `trimap_unknown_v3`
- `hair_alpha_v3`
- `alpha_confidence_v3`
- `alpha_source_provenance_v3`

成功标准：

- 相对 `optimization5` 最优 baseline：
  - `alpha_mae` 降低至少 15%
  - `trimap_error` 降低至少 15%
  - `hair_edge_quality` 提升至少 10%


### Workstream C: 真正的前景 / 背景解耦式 replacement

对应目标：

- 将当前整图 replacement 主导的逻辑，升级成显式的：
  - foreground
  - alpha
  - background
  - composite ROI

为什么必须做：

- 只要前景和背景还隐式耦合，边缘问题就永远只是整图生成的副产物
- 真正的边缘控制，必须依赖显式前景 / 背景 contract

应建立的 contract：

- `foreground_rgb`
- `foreground_alpha`
- `foreground_confidence`
- `background_rgb`
- `background_visible_support`
- `background_unresolved`
- `composite_roi_mask`

generate 端应做的变化：

- 允许 foreground-aware conditioning
- 允许 background-aware conditioning
- 不再只把边界当成 mask shaping 的结果

成功标准：

- decoupled contract 在真实 smoke 中可稳定运行
- 相对 legacy：
  - `background_fluctuation` 不恶化
  - `seam_score` 不恶化
  - `roi_gradient` 至少改善 5%


### Workstream D: 边界 ROI 局部生成式重建

对应目标：

- 把边界问题从 deterministic refine 升级成局部生成式重建问题

为什么必须做：

- `optimization4` 和 `optimization5` 已经证明：
  - deterministic ROI refine 只能轻微调边
  - 不能真正恢复边界细节

真正要做的不是：

- sharpen
- feather
- blend

而是：

- ROI 局部重生成
- ROI 局部 restoration
- alpha-aware paste-back

关键设计：

1. ROI 提取器
   - hair / face / hand / cloth / generic boundary
2. ROI 条件构造器
   - alpha
   - trimap unknown
   - uncertainty
   - background patch
   - foreground patch
3. ROI 生成器
   - 可先从 lightweight restoration / patch model 做起
4. ROI 复合器
   - 高质量 alpha paste-back

成功标准：

- 相对当前最佳 baseline：
  - `roi_gradient` 提升至少 12%
  - `roi_edge_contrast` 提升至少 10%
  - `halo_ratio` 降低至少 10%
  - `seam_score` 恶化不超过 3%


### Workstream E: 更强的训练型边缘模型

对应目标：

- 在更强 benchmark 和更强 alpha 基础上，引入训练型 edge model，而不是只靠 heuristic adapter

为什么不应过早做：

- 如果没有更强数据和更好的任务定义，训练出来的模型只会继续拟合旧 baseline

前提条件：

1. reviewed benchmark 升级完成
2. production-grade alpha 路径已建立
3. decoupled contract 已落地
4. ROI generative reconstruction 已有可行原型

训练型方向可包括：

- edge alpha refinement model
- hair-edge refinement model
- ROI expert patch model
- LoRA / adapter / small restoration model

成功标准：

- 在 reviewed benchmark 上，训练型模型必须客观优于：
  - heuristic baseline
  - deterministic ROI baseline
- 否则不进入主线


## 6. 推荐执行顺序

推荐顺序如下：

1. **升级人工审核 edge benchmark**
2. **接更强的外部 alpha / matting**
3. **做前景/背景解耦式 contract 和生成路径**
4. **做边界 ROI 局部生成式重建**
5. **在更强数据上再考虑训练型 edge model**
6. **最后再做 H200 orchestration**

为什么这样排：

- Step 1 决定后面所有评测是否可信
- Step 2 决定边缘信号本身是否可信
- Step 3 决定生成问题是否被正确建模
- Step 4 才是真正直接打边缘锐度
- Step 5 是高风险高回报项，必须建立在前面更强的数据和路径上
- Step 6 最后做，避免过早优化错误方向


## 7. 每条工作流的统一执行协议

所有 `optimization6` 工作流，都必须遵守以下统一协议。

### 7.1 Round 1

- 做最小可运行实现
- 打通 artifact / metadata / runtime / debug / benchmark
- 跑 synthetic correctness
- 跑真实 10 秒 smoke

### 7.2 Round 2

- 根据 Round 1 指标退化点改方法
- 不允许只调单个参数而不解释原因
- 跑 reviewed benchmark + 真实 smoke AB

### 7.3 Round 3

- 在已有方法上做最后一次收敛
- 明确比较 Round 1 / 2 / 3
- 冻结 best-of-three

### 7.4 收口规则

- 若 Round 3 后仍未过强 gate：
  - 不再继续死磕
  - 冻结 best-of-three
  - 明确写出失败原因
  - 进入下一工作流


## 8. 外部模型接入的工程规范

这是 `optimization6` 的关键新增要求。

### 8.1 每个外部模型必须有独立 adapter

禁止做法：

- 在主 pipeline 中直接散落外部模型调用逻辑

正确做法：

- 一个外部模型，一个 adapter 文件
- adapter 负责：
  - 权重加载
  - 输入标准化
  - 输出语义转换
  - runtime 统计
  - fallback

### 8.2 每个外部模型必须配套 provenance

至少记录：

- model_name
- source_url / source_repo
- release / commit / tag
- weight_file_name
- sha256
- license
- required_dependencies

### 8.3 下载和缓存必须可复现

建议：

- 统一下载目录
- 本地缓存目录
- hash 校验
- 第一次下载后写清单文件

禁止：

- 不经校验直接使用临时下载权重

### 8.4 主线接入前必须先过三道门

1. synthetic correctness
2. reviewed benchmark
3. 真实 10 秒 smoke

任一道门不过，都不能进默认主线。


## 9. H200 141GB 在 optimization6 中应如何使用

H200 的价值不在于继续暴力增加旧方案的采样步数，而在于：

1. **高分辨率 alpha / matting / hair-edge estimation**
2. **高分辨率 ROI generative reconstruction**
3. **少量高价值候选并行**
4. **小规模训练型模型**

具体策略：

- preprocess：
  - 优先给 alpha / matting / parsing / hair-edge
- generate：
  - 优先给 boundary ROI reconstruction
  - 不优先给整图 brute force
- training：
  - 优先给小而强的 edge module


## 10. 这条路线的主要风险

### 10.1 外部模型许可证或依赖不适配

缓解：

- 把许可证检查前置到候选筛选阶段

### 10.2 reviewed benchmark 仍然不够独立

缓解：

- 在 Step 1 中把人工审核比例提高
- 明确记录哪些标签是人工修订，哪些是 bootstrap

### 10.3 ROI generative reconstruction 成本过高

缓解：

- 先做窄边界 ROI
- 只在高价值区域启用

### 10.4 训练型模型过拟合 benchmark

缓解：

- 必须有 holdout frame
- 必须有真实 10 秒 smoke
- 不能只看训练集指标


## 11. 阶段成功标准

`optimization6` 成功的最低标准，不是“把更多实验跑通”，而是：

1. 至少一条新的 alpha / reconstruction 路线，在 reviewed benchmark 上显著优于 `optimization5` baseline。
2. 至少一条 generate 路线，在真实 10 秒素材上让边缘观感出现明显等级差异。
3. seam / background / face identity 不发生不可接受退化。
4. 外部模型接入是可追踪、可复现、可回退的。

如果这些做不到，就说明：

> 问题已经不是“工程不够扎实”，而是当前选用的模型族本身不够强，需要继续升级模型层级。


## 12. 最后的判断

`optimization6` 不是继续延长 `optimization4/5` 的 heuristic 余波，而是一次真正的路线切换。

它的本质是：

- 承认旧路线已经接近上限
- 用更强数据重新定义目标
- 用更强 alpha / matting 和 ROI 生成重建重做边缘问题
- 在确有必要时引入训练型边缘模型

如果执行得当，`optimization6` 才是最有可能第一次真正把：

> “替换后的人物边缘有时模糊”

这个核心问题拉开明显等级差异的一轮。
