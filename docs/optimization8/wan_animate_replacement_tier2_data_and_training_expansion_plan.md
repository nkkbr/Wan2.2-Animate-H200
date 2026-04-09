# Wan-Animate Replacement Tier-2 Data And Training Expansion Plan

## 1. 文档定位

本文档定义的是下一阶段“第二档”路线，也就是：

> 明显有价值，但必须建立在更强数据、更清晰任务定义和更稳主路径之上的增强方向。

第二档路线不应先于第一档主线盲目启动。  
它的作用不是替代第一档，而是：

- 放大第一档成功路线的收益
- 提供更强的验证和训练支撑
- 为未来真正可持续的边缘模型打基础

因此，第二档的关键词是：

- 更强人工审核数据
- 更清晰任务定义
- 更合理训练目标
- 更可靠的模型验证


## 2. 为什么第二档不能直接当主线

前面已经发生过的失败，给出了非常清楚的教训：

1. benchmark 与旧 baseline 同源性太强时，训练型路线很难证明自己真的更好
2. 在主路径还不清楚的情况下，训练模型很容易只是拟合当前不够好的系统输出
3. 如果没有更强数据约束，更多训练不一定带来更好边缘，反而会把错误目标学得更稳

所以第二档必须明确定位为：

- **支撑性增强**
- **放大器**
- **验证器**

而不是当前最先打的突破点。


## 3. 第二档的总体目标

第二档路线的目标，是让系统具备下面这些能力：

1. 有一套更独立、更强、更细粒度的 reviewed edge 数据
2. 有一套真正适合训练边缘模型的任务定义，而不是“拟合当前 baseline”
3. 能对不同类型边界分别建模，而不是把所有边界都压进一个统一任务
4. 能对第一档路线中真正有效的部分进行放大，而不是对无效路线继续堆训练


## 4. 第二档包含哪些工作

### 4.1 reviewed edge benchmark 扩容与独立化

#### 为什么重要

如果继续用小而偏同源的数据集做训练和验证，会不断出现这样的假象：

- 模型看似学到了东西
- 实际上只是绕着 baseline 转

#### 这一项要做的事

- 扩大关键帧数量
- 提高场景多样性
- 引入人工直接修订，而不是继续从旧产物派生
- 明确标注边界语义类别

建议扩展的标注对象：

- hair edge
- face contour
- ear / neck transition
- hand / finger edge
- cloth boundary
- occlusion boundary
- trimap unknown region
- semi-transparent boundary

#### 最低要求

- 关键帧规模至少扩到原来的 2 倍以上
- 至少覆盖：
  - 静态镜头
  - 快动作镜头
  - 发丝复杂镜头
  - 手部复杂镜头
  - 服装复杂纹理镜头


### 4.2 训练型 edge model 的重新定义

#### 为什么之前不够好

前面训练型路线失败，不是“训练永远不行”，而是：

- GT 不够独立
- 输入不够强
- 目标太含混
- 任务边界不清楚

#### 现在应如何重定义

训练对象不应再是一个模糊的“边缘增强器”，而应拆成更具体的任务：

1. alpha refinement model
2. boundary uncertainty refinement model
3. ROI matte completion model
4. semantic boundary expert model
5. compositing-aware edge correction model

每个任务都应有清晰输入输出。

例如：

- 输入：
  - foreground patch
  - background patch
  - soft alpha
  - trimap unknown
  - boundary ROI mask
  - semantic boundary tag
- 输出：
  - refined alpha
  - refined matte
  - boundary correction residual


### 4.3 semantic experts 的训练化升级

#### 为什么值得做

我们已经在 heuristic 路线上验证过：

- `face / hair / hand / cloth` 边界不是同一个问题

所以第二档值得做的是：

- 不再只是 semantic 分区
- 而是训练独立或半独立的 semantic experts

#### 推荐思路

- face contour expert
- hair matte expert
- hand boundary expert
- cloth edge expert

这些专家不一定必须是完全独立模型，也可以是：

- shared backbone + semantic adapters
- ROI routing + expert heads


### 4.4 compositing-aware loss 和 edge-aware loss 体系

#### 为什么值得做

以前很多训练型路线的问题在于：

- loss 只盯局部像素
- 没盯最终 composite 效果

而边缘问题最关键的失败恰恰体现在 composite 后。

#### 推荐 loss 方向

- alpha MAE / SAD
- trimap-focused loss
- boundary F-score proxy loss
- gradient preservation loss
- contrast preservation loss
- compositing reconstruction loss
- semantic boundary weighted loss

目标不是堆 loss，而是保证 loss 真正对应最终边缘质量。


## 5. 第二档的推荐执行顺序

### Phase 1：先做数据升级

没有更强 reviewed 数据，不要进入训练型主线。

### Phase 2：再做任务定义与数据管线

先把：

- 样本切分
- ROI 采样
- semantic tagging
- train/val split
- hard negative cases

做清楚。

### Phase 3：最后才做训练

否则只是在模糊任务上浪费 GPU。


## 6. 第二档实施方法

### 6.1 不该做的事

第二档执行中，不应继续做：

- 用旧 mini-set 继续训练更多小模型
- 只看 val loss，不看 reviewed benchmark 和最终 composite 指标
- 在任务定义没清楚前就做大规模 sweep
- 在基础数据仍不独立时，用结果解释为“模型不行”

### 6.2 推荐采用的工程方式

- 数据版本化
- 明确 split 规则
- 每个 semantic 类别有独立覆盖统计
- 每次训练 run 都绑定：
  - dataset version
  - label schema version
  - loss config
  - ROI policy
  - base preprocess bundle lineage


## 7. 第二档的成功标准

第二档不以“训练出了多少模型”为成功，而以以下结果为成功：

1. reviewed dataset 质量显著提升
2. 至少一个训练型子任务在 reviewed benchmark 上显著优于当前非训练基线
3. 这种提升能在真实 10 秒 smoke 上部分转化为可见质量收益

更严格地说：

- 如果训练型路线只是在 loss 上变好，而在 reviewed benchmark 或真实 composite 上没有改善，就不算成功


## 8. 第二档的主要风险

### 风险 1：数据扩容慢，导致主线等待过久

应对：

- 先做重点类别小规模高质量扩充
- 不要求一次性扩很大

### 风险 2：任务定义过多，训练线变得分散

应对：

- 先只选 1 到 2 个最关键任务
- 不同时开太多专家

### 风险 3：模型学会了 benchmark，但没有改善真实视频

应对：

- 每轮训练都必须跑真实 smoke
- reviewed benchmark 和真实 smoke 双重 gate 才算成功


## 9. 第二档适合什么时候启动

以下条件至少满足两条后，第二档才适合进入主线阶段：

1. 第一档至少有一条路线在 reviewed benchmark 上客观优于基线
2. reviewed dataset 已经完成扩容与独立化
3. decoupled contract 已经足够稳定
4. ROI generation / reconstruction path 已有明确正向信号

否则第二档应保持为准备态，而不是主执行态。


## 10. 结论

第二档路线不是“次要”，而是“时机要求更高”。

它的核心价值是：

- 让未来的边缘模型建立在更强数据和更清晰问题定义上
- 防止我们再次陷入“训练了很多，但没在正确问题上训练”的循环

因此，第二档最应该被当作：

> 第一档成功后的放大器，而不是第一档失败时的情绪性替代方案。
