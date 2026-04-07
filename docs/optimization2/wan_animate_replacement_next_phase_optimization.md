# Wan-Animate Replacement 后续优化评估与路线

## 1. 文档目标

本文档基于当前仓库已经完成的 12 个优化步骤，对现状做一次新的工程判断，回答三个问题：

1. 当前代码离“高精度 replacement”还有多远
2. 单张 H200 141GB 是否已经被充分利用
3. 下一阶段最值得投入的优化工作是什么

这份文档不重复已有实施细节，而是为下一轮优化提供方向性判断、优先级和验收标准。


## 2. 当前状态判断

### 2.1 当前代码已经明显优于初始版本

当前代码已经具备一套完整的高质量 replacement 工程骨架：

- preprocess 侧已经有：
  - face / pose 稳定化
  - 改进后的 SAM2 prompt 规划
  - soft boundary band
  - clean plate background
  - reference normalization
  - lossless intermediate pipeline
- generate 侧已经有：
  - H200 质量预设
  - 静态条件缓存
  - mask-aware overlap blending
  - decoupled guidance
  - temporal handoff
  - runtime stats 与 debug 导出

这些改动意味着当前系统已经不再是“勉强能跑”的原始实现，而是一套可追踪、可测试、可调优的 replacement 系统。

### 2.2 但当前代码还没有达到“边缘尽可能高精度”的目标

最重要的原因不是单一参数，而是当前架构本身仍然把核心 replacement 控制放在 latent 空间：

- Wan-Animate 的 VAE stride 仍然是 `(4, 8, 8)`，见 `wan/configs/wan_animate_14B.py`
- replacement mask 在生成前会被下采样到 latent 分辨率，见 `wan/animate.py`
- mask 还会按 VAE 时间结构重新打包，见 `wan/animate.py`

这带来的后果是：

- 头发丝
- 衣角
- 袖口
- 四肢边缘
- 运动模糊边界

这些最敏感的位置，仍然不是像素级 alpha matting 控制，而是 latent 级近似控制。  
因此，当前代码虽然已经能比原版更稳地处理边界，但还不能从架构上彻底消除边缘模糊。

### 2.3 当前代码还没有充分利用 H200 141GB

当前实现已经开始有“质量优先”的 H200 路线，但仍然不能说已经把这张卡用满：

- generate 侧：
  - `hq_h200` 预设已经会关闭 offload，并默认走更高质量的 solver / steps
  - 这比原版合理得多
  - 但仍然只是单卡质量优先 baseline，不是“把 141GB 最大程度转成最终画质”的终点
- preprocess 侧：
  - SAM2 路径仍被强制设成保守模式
  - 当前实现里显式关闭了 flash attention
  - 这和 H200 的硬件能力并不匹配

因此，当前对 H200 的利用是“已经改善，但远未充分”。


## 3. 对当前系统的准确评价

### 3.1 可以做到什么

当前代码已经适合做下面这些事：

- 构建较高质量的 replacement 输入包
- 用 lossless 中间格式减少质量损失
- 通过 soft band、background keep、mask-aware blending 降低明显接缝
- 通过 decoupled guidance 和 temporal handoff 改善 clip continuity
- 在 H200 上以质量优先方式运行单卡 Wan-Animate

### 3.2 还做不到什么

当前代码还不能被视为已经满足下面这些更高要求：

- 对真实素材稳定地产生最高质量的 SAM2 人物 mask
- 在所有镜头里都提供接近抠像级的人物边缘
- 在真实 10 秒以上素材上，以 full-size replacement 稳定、快速地完成高质量生成
- 把 H200 的大显存和新架构优势充分转成最终边缘质量

### 3.3 当前最大的风险点

当前最现实的两个风险点是：

1. preprocess 的真实 SAM2 视频链路仍不够稳定  
2. replacement 边界控制仍然主要停留在 latent 空间

前者影响“能否稳定做高精度输入”，后者影响“即使输入正确，边缘质量能否真正拉高”。


## 4. 下一轮优化的核心目标

下一轮优化不应该再平均发力，而应该围绕下面三个目标集中投入：

1. 让 preprocess 真正稳定地产生高精度人物区域
2. 让边缘控制从 latent 级进一步逼近 pixel 级
3. 让 H200 的显存和吞吐真正换成更高的最终画质


## 5. 优先级建议

### P0：必须优先解决的问题

#### P0-1. 修复真实 SAM2 preprocess 路径的稳定性

这是当前最优先的问题。  
如果真实视频 preprocess 不能稳定完成，那么后续所有高质量 replacement 都没有稳定基础。

建议方向：

- 定位真实视频上 native crash 的触发条件
- 最小化复现 case
- 分离“模型内核问题”和“chunk/prompt/shape 问题”
- 为 H200 单独建立一套经过验证的 SAM2 runtime profile

验收标准：

- 10 秒素材 preprocess 可重复连续运行通过
- 不再出现中途 native crash
- manifest 中 preprocess stage 能稳定写入 `completed`

#### P0-2. 重新审视 preprocess 中对 H200 过于保守的设置

当前 `process_pipepline.py` 中对 SAM2 transformer 的强制保守设置，应重新 benchmark：

- 是否必须 `USE_FLASH_ATTN = False`
- 是否必须 `OLD_GPU = True`
- 在 H200 上哪些配置组合最稳、最快、最不损质量

验收标准：

- 至少得到一套“对 H200 安全可用”的高性能 profile
- 明确哪些设置是为了兼容旧卡，哪些设置在 H200 上可以解除

#### P0-3. 把边缘质量的主优化目标明确转向“像素域 refinement”

只继续在 latent mask 上做文章，收益会越来越有限。  
下一轮必须承认这一点，并增加一层像素域边缘精修。

建议方向：

- 生成后只在 boundary band 内做局部 refinement
- 保持背景像素尽可能不动
- 保持人物主体尽可能来自模型生成结果
- 用更高精度 alpha / matte 只处理边缘过渡区域

验收标准：

- 头发、衣角、袖口等边界的主观清晰度明显提升
- 不引入新的明显 halo、色边、抖动


### P1：高价值、低于 P0 的质量优化

#### P1-1. 引入 parsing / matting 融合的 preprocess 边界建模

当前 soft band 是从 segmentation mask 推导出来的过渡带。  
这已经比硬 mask 好，但仍然不足以处理复杂边界。

建议方向：

- 人体 parsing 与 SAM2 结果融合
- 人像 matting 模型与当前 person mask 融合
- 将“硬主体区域”和“软边界区域”分开建模

预期收益：

- 边缘更细
- 手部、头发和衣物边缘更自然
- replacement_strength 的空间分布更合理

#### P1-2. 升级 clean plate background

当前 clean plate 是 image inpaint 路线，已经优于“挖洞背景”，但仍不是时序一致的最佳方案。

建议方向：

- 先保持当前 image 路线为保底
- 再增加视频一致性的 clean plate 构建路径
- 明确区分用于生成条件的背景和用于 debug/评估的背景

预期收益：

- 人物边界附近背景更稳定
- 边缘融合时更少出现脏边和闪烁

#### P1-3. 把参考图归一化从 bbox 级提升到结构级

当前 reference normalization 已经解决了明显的画幅和尺度失配问题。  
下一步应该从“bbox 对齐”提升到“人体结构对齐”。

建议方向：

- 头身比、肩宽、腿长分开估计
- 对 face / torso / lower-body 使用不同的缩放约束
- 将参考图归一化和 replacement region 的空间预算联动

预期收益：

- 形体失真减少
- 肢体边缘更少挤压出 mask 外


### P2：高风险探索项

#### P2-1. 更强的 temporal memory

当前已经有 temporal handoff 探索，但仍处于 prototype 阶段。  
可以继续探索更强的 latent continuity 机制，但这不是边缘问题的第一优先级。

#### P2-2. 多阶段生成

如果单阶段 Wan-Animate replacement 始终受限于 latent resolution，那么可以考虑：

- 第一阶段：主体 replacement
- 第二阶段：边缘与局部高频 refinement

这是高风险高收益方向，但实现成本高，应放在 P0/P1 之后。


## 6. 对 H200 的建议使用策略

### 6.1 H200 应优先换取什么

下一轮优化中，H200 的预算应优先换成：

- 更高质量的 preprocess 分析分辨率
- 更高质量的 mask / matte 生成
- 更高的 `sample_steps`
- 更高保真的中间与最终输出
- 更多参数搜索和对照实验

### 6.2 不应把 H200 主要浪费在哪里

不建议把主要预算浪费在：

- 盲目增大最终导出分辨率，但不提升分析质量
- 继续沿用兼容旧 GPU 的保守运行模式
- 只增加采样步数，却不升级边缘建模方式

如果边缘建模本身仍是粗粒度 latent mask，那么仅靠更多采样步数，边际收益会越来越小。


## 7. 推荐的下一轮实施顺序

建议按下面顺序推进：

1. 修复真实 SAM2 preprocess 稳定性
2. 为 H200 建立更积极但稳定的 preprocess runtime profile
3. 引入 parsing / matting 融合边界建模
4. 在 generate 后增加像素域 boundary refinement
5. 升级 clean plate 为更强的视频一致性方案
6. 再做新一轮 sample steps / guidance / continuity 搜索

这个顺序的原则是：

- 先把输入做对
- 再把边缘做细
- 最后再继续放大 H200 的算力投入


## 8. 下一轮优化的验收原则

下一轮优化不能只靠主观观感判断，建议同时看下面几类信号：

- preprocess 是否稳定通过
- mask / soft band / clean plate 的 QA 产物是否更可信
- seam score 是否继续下降
- background fluctuation 是否继续下降
- 边缘主观观感是否更清晰
- 在 `ffv1` 或 `png_seq` 输出下，边缘是否仍然明显优于当前版本

尤其要强调：

边缘质量评估应优先基于无损或近无损输出，而不是只看 mp4，因为 mp4 编码本身会进一步抹软边缘。


## 9. 最终判断

当前代码已经走在正确方向上，而且已经比初始版本更接近“高质量 replacement”。  
但是，它还不能被视为已经完成了以下目标：

- 最高精度的人物动作与边界检测
- 对 H200 141GB 的充分利用
- 对人物边缘模糊问题的根治

要真正接近这个目标，下一轮优化必须从“继续修参数”升级为“进一步升级边缘建模和像素域精修能力”。  
只有这样，H200 的硬件优势才会更直接地转化为你最在意的最终边缘质量。
