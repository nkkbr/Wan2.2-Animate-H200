# Step 02: H200 Preprocess Runtime Profile 重建

## 1. 步骤目标

本步骤的目标是为 `Wan-Animate replacement preprocess` 建立一套真正面向 `H200 141GB` 的运行 profile，使 preprocess 侧不再停留在“兼容旧 GPU 的保守模式”，而是可以：

- 在 H200 上更积极地利用可用算力与显存
- 提升分析分辨率、chunk 规模和 prompt 密度
- 在不牺牲稳定性的前提下，提高人物区域、边界和背景建模质量

本步骤默认建立在 Step 01 已经完成的前提上，也就是：

- 真实 preprocess smoke 已能稳定通过
- crash triage 能力已经建立


## 2. 为什么要做这一步

### 2.1 当前 preprocess 明显没有充分利用 H200

当前代码中，SAM2 preprocess 路径仍然包含明显保守的运行配置。  
这意味着系统虽然已经有高质量 replacement 的工程能力，但在 preprocess 侧，硬件潜力还没有转化成：

- 更高分辨率分析
- 更高 prompt 密度
- 更细粒度 mask 与边界信号

### 2.2 当前 H200 的主要价值没有被优先花在 preprocess 分析质量上

在 high-quality replacement 场景中，H200 的首要价值不应只体现在 generate 步数上。  
对你最关心的边缘质量来说，更高质量的 preprocess 往往收益更直接，因为它决定：

- 人物主体区域是否足够准确
- 背景保留区域是否足够可信
- soft band 是否足够细
- clean plate 的构建区域是否合理

### 2.3 如果不单独建立 H200 preprocess profile，后续优化会长期被保守路径绑住

如果没有 profile 化的运行方式，后续每次实验都只能在“老 GPU 保守路径”上调参数。  
这样会导致：

- 调参空间小
- H200 的真正收益看不出来
- 一旦出现问题，很难判断是 profile 问题还是算法问题


## 3. 本步骤的交付结果

本步骤完成后，应至少交付以下内容：

1. 一套显式的 H200 preprocess runtime profile 机制
2. 至少两套可对照 profile：
   - `h200_safe`
   - `h200_aggressive`
3. 一套 preprocess benchmark 方案，用于比较：
   - 运行时间
   - 峰值显存
   - 关键 QA 指标
   - 真实 smoke 成功率
4. 推荐的 H200 默认 preprocess 配置
5. profile 选择结果写入 manifest / metadata / runtime debug


## 4. 代码改动范围

本步骤预计会改动如下模块：

- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/modules/animate/preprocess/sam_utils.py`
- `wan/modules/animate/preprocess/video_predictor.py`
- `wan/modules/animate/preprocess/preprocess_data.py`
- `wan/utils/experiment.py`
- `scripts/eval/` 下新增 preprocess benchmark 与 profile sweep 脚本


## 5. 具体实施方案

### 5.1 子步骤 A：把 preprocess runtime 配置显式抽象成 profile

目标：

- 不再依赖散乱参数和硬编码开关决定运行方式

建议新增 profile 概念，例如：

- `legacy_safe`
- `h200_safe`
- `h200_aggressive`

每个 profile 至少需要控制：

- SAM2 kernel 相关开关
- 默认分析分辨率范围
- 默认 chunk 长度
- 默认 keyframe 数量
- 默认 reprompt 策略
- 默认并发 / stream 策略

为什么要先做这个：

- 没有 profile，后续 benchmark 结果不可比较
- 同一台 H200 上也需要区分“保守稳定”和“质量优先”两条路径

### 5.2 子步骤 B：分离 analysis resolution 与 export resolution

目标：

- 把 H200 的额外算力优先换成 preprocess 分析质量，而不是盲目扩大最终导出尺寸

建议做法：

1. 现有 `resolution_area` 继续保留，用于导出尺寸
2. 新增一套 analysis 相关参数，例如：
   - `analysis_resolution_area`
   - `analysis_min_short_side`
3. preprocess 内部执行：
   - 检测 / prompt 规划 / SAM2 / matting 在 analysis resolution 上做
   - 导出 artifacts 再根据 export 需求决定保存尺寸

为什么这一步重要：

- 你最关心的是检测和边界质量
- 这一步比直接堆输出分辨率更有价值

### 5.3 子步骤 C：建立 H200 preprocess benchmark 脚本

目标：

- 让 profile 选择不靠感觉，而靠可重复对照

建议 benchmark 维度：

- profile
- analysis resolution
- chunk length
- keyframe count
- reprompt interval
- negative points 开关

每次 run 至少记录：

- 总耗时
- 每 chunk 耗时
- 峰值显存
- smoke 是否成功
- QA artifact 是否完整
- mask / soft band / clean plate 的基础统计

建议 benchmark 输出：

- JSON summary
- CSV 摘要
- 每组配置的 run manifest 引用

### 5.4 子步骤 D：引入 H200 优先的内存与吞吐策略

目标：

- 在不破坏正确性的前提下，提升 preprocess 速度和分析质量

建议方向：

- 合理启用更积极的 attention/kernel profile
- 避免不必要的 CPU/GPU 反复拷贝
- 对 chunk 间重复使用的静态结构做缓存
- 若底层库支持，评估 stream/pipeline 化的机会

注意：

本步骤的目标不是极限压榨吞吐，而是：

- 在 H200 上把更多预算转成更高质量分析
- 同时保持真实 smoke 可重复通过

### 5.5 子步骤 E：写入正式推荐 profile

目标：

- 给后续工程使用者一个明确的默认入口

建议至少给出：

- `h200_safe`
  - 真实素材长期 smoke、批量 preprocessing 的默认配置
- `h200_aggressive`
  - 面向高质量对照实验、允许更高耗时的配置

每个 profile 文档中要写清楚：

- 适用场景
- 预期耗时
- 风险
- 建议搭配的分辨率与 chunk 策略


## 6. 推荐实施顺序

建议按下面顺序推进：

1. profile 参数化
2. analysis / export resolution 分离
3. benchmark 脚本建立
4. 跑第一轮 `safe vs aggressive` 对照
5. 形成推荐 H200 profile

这个顺序的目的是：

- 先让配置空间可描述
- 再让实验结果可比较
- 最后才确定默认值


## 7. 验证与测试方案

### 7.1 必做验证

1. Step 01 的真实 smoke regression 继续通过
2. `h200_safe` 至少连续通过 3 次
3. `h200_aggressive` 至少能在真实 10 秒 case 上通过 1 次
4. 新 profile 写入 manifest / metadata
5. benchmark 输出结构稳定、可重复解析

### 7.2 应重点关注的指标

- preprocess 总耗时
- 峰值显存
- 每 chunk 平均耗时
- mask 覆盖率稳定性
- soft band 宽度统计
- clean plate 缺陷区域统计
- QA overlay 的主观可用性

### 7.3 成功标准

本步骤算通过，应同时满足：

- H200 上已有明确、可复用的 preprocess profile
- 至少一个 profile 比当前保守路径更积极
- 真实 smoke 稳定性没有因为 profile 升级而明显恶化
- 质量指标或 QA 结果显示有实际提升空间


## 8. 风险

### 8.1 风险一：更积极的 profile 重新引入不稳定性

这是最现实的风险。  
如果启用更积极的 kernel / chunk / resolution 组合后又回到 native crash，就说明 profile 过激。

应对方式：

- `safe` 和 `aggressive` 分开维护
- 任何更激进配置都必须先过最小真实 smoke

### 8.2 风险二：性能提升但质量没有改善

H200 profile 的目标不只是更快。  
如果只是更快，但人物边界和 mask 质量没有提升，这一步就没有真正达成目标。

应对方式：

- benchmark 中必须同时记录质量侧指标
- 不接受纯速度导向的 profile 结论


## 9. 回退策略

如果 H200 新 profile 验证失败，应始终保留：

- `legacy_safe`
- `h200_safe`

其中 `h200_safe` 应作为后续所有质量实验的保底 profile。  
只有在 `h200_aggressive` 连续通过后，才考虑把它用于主实验。


## 10. 完成标志

本步骤完成的标志是：

1. preprocess runtime profile 已正式参数化
2. `h200_safe` / `h200_aggressive` 已建立
3. benchmark 脚本和结果目录可重复使用
4. H200 上的 preprocess 不再被单一路径硬编码绑死
5. 后续边缘质量优化可以建立在“更高质量分析 profile”之上
