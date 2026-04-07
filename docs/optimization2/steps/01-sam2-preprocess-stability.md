# Step 01: 真实 SAM2 Preprocess 稳定性修复

## 1. 步骤目标

本步骤的目标是修复当前 `Wan-Animate replacement` 在真实视频 preprocess 阶段的稳定性问题，尤其是 `SAM2` 视频分割链在真实素材上出现的 native crash / segmentation fault。

只有这一步完成，后续所有“高精度边缘建模”“更积极利用 H200”“更高分析分辨率”的工作才有现实意义。  
否则，系统虽然在 synthetic 和局部 smoke 上可用，但仍然不能作为稳定的高质量 replacement pipeline。


## 2. 为什么必须先做这一步

### 2.1 当前最严重的问题不是画质，而是 preprocess 真实链不稳定

上一轮验证中，真实 10 秒视频的 preprocess run 没有正常完成，而是停在 manifest 的 `started` 状态。  
这说明当前问题不是“参数需要再调”，而是流程存在 native 级不稳定点。

如果这一步不先处理，后续继续做：

- 更高 `sam_chunk_len`
- 更高分析分辨率
- 更积极的 H200 kernel profile
- parsing / matting 融合

都会建立在一个不稳定基础上，风险会被继续放大。

### 2.2 当前 system validation 已经给出了明确信号

现有验证结果说明：

- synthetic contract / regression 全部通过
- mini generate smoke 可以完成
- 但真实 preprocess 仍然无法稳定跑通

这意味着：

- 工程框架不是主要问题
- 真实素材下的 SAM2 preprocess 路径才是最优先的 blocking issue


## 3. 当前问题假设

在真正改代码之前，应先明确当前问题可能来自哪几类原因。

### 3.1 候选原因 A：SAM2 runtime/kernel 配置与当前环境不匹配

当前 `process_pipepline.py` 会在导入时强制设置：

- `USE_FLASH_ATTN = False`
- `MATH_KERNEL_ON = True`
- `OLD_GPU = True`

这说明当前 preprocess 路径实际上在以非常保守的方式运行。  
这类强制配置有两类风险：

- 为了兼容旧环境，反而触发当前环境里的非预期路径
- 在大分辨率或长 chunk 下走到更慢、更占内存、更容易崩的实现分支

### 3.2 候选原因 B：chunk / prompt / frame tensor 组合触发底层问题

当前 SAM2 replacement preprocess 具有如下特征：

- chunk 处理
- 多关键帧 prompt
- 正负点组合
- 再提示逻辑
- 逐 chunk 的 `init_state` / `propagate`

如果其中任一环节产生异常形状、空 prompt、全零 mask、越界点，底层实现可能不会抛 Python 异常，而是直接在 native 层崩溃。

### 3.3 候选原因 C：真实视频的 shape / fps / mask 提示与 synthetic case 差异太大

synthetic 检查无法覆盖：

- 真实分辨率
- 真实帧数
- 真实人体尺度变化
- 真实光照和遮挡

因此，很可能是“流程逻辑看上去没问题，但某些真实条件组合会把底层实现推到危险区”。


## 4. 本步骤的交付结果

本步骤完成后，应至少交付以下结果：

1. 可以稳定复现当前 crash 的最小真实 case
2. 可以明确 crash 属于哪一层：
   - Python 逻辑
   - wrapper 层
   - SAM2 native / CUDA 内核
3. 至少一套在 H200 上可重复通过的 preprocess 配置
4. 新的 smoke 脚本与调试产物
5. 明确写入文档的稳定性验收标准


## 5. 代码改动范围

本步骤预计会改动如下模块：

- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/modules/animate/preprocess/sam_prompting.py`
- `wan/modules/animate/preprocess/sam_utils.py`
- `wan/modules/animate/preprocess/video_predictor.py`
- 如有必要，`src/sam-2/` 中当前 repo vendor 的 wrapper 层
- `scripts/eval/` 下新增真实 preprocess smoke / crash triage 脚本
- `docs/optimization2/benchmark/` 或 `docs/optimization1/benchmark/` 下补充稳定性记录文档


## 6. 具体实施方案

### 6.1 子步骤 A：先把 crash 变成可复现、可比较的最小 case

目标：

- 把“偶发崩溃”变成一个固定可重放的 case

建议做法：

1. 固定输入素材
   - 视频：`/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
   - 参考图：沿用当前测试图
   - checkpoint：`/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`
2. 固定两套已失败配置
   - 当前高负载失败配置
   - 当前低负载失败配置
3. 新增一个“最小真实 preprocess smoke”脚本
   - 自动建 run 目录
   - 自动记录 stdout/stderr
   - 自动记录最后一次成功写出的 chunk / frame index
4. 对 crash 前的关键状态落盘
   - chunk 索引
   - 提示帧索引
   - prompt 点数量
   - 当前 frame tensor shape
   - 当前 chunk 的原始 fps / resolution

为什么先做这个：

- 如果没有可复现 case，后续每次修改都只能靠猜
- native crash 最大的问题不是“难修”，而是“无法稳定确认是否真的修好了”

### 6.2 子步骤 B：为 SAM2 包一层更细的运行时诊断

目标：

- 把现在“直接 segmentation fault”的黑盒行为，尽可能往前转化成可定位的显式失败

建议做法：

1. 在调用 `init_state_v2` 前后加诊断
2. 在 `add_new_points` / `propagate_in_video` 前后加诊断
3. 为每个 chunk 写一份轻量 JSON：
   - chunk 起止帧
   - 关键帧索引
   - 正点数 / 负点数
   - 点坐标范围
   - 是否存在空 prompt
   - mask logits 输出是否为空
4. 在 Python 层加更强的输入校验：
   - 点坐标是否越界
   - prompt labels 是否为空
   - chunk 帧数是否非零
   - 视频 resize 后 shape 是否一致

为什么要这么做：

- 这一步不一定直接修掉 crash
- 但可以先把“native 崩溃前到底发生了什么”记录清楚

### 6.3 子步骤 C：把 SAM2 runtime 配置改成可配置而不是硬编码

目标：

- 不再把 runtime kernel 选择写死在导入阶段

建议做法：

1. 将下面这些配置显式参数化：
   - `USE_FLASH_ATTN`
   - `MATH_KERNEL_ON`
   - `OLD_GPU`
2. 提供 profile 级参数而不是散乱布尔值：
   - `legacy_safe`
   - `h200_safe`
   - `h200_aggressive`
3. 在 manifest / metadata / runtime debug 中记录实际采用的 SAM2 runtime profile

为什么这么做：

- 当前写死的行为既难 debug，也难 benchmark
- 对 H200 来说，必须先把 profile 切换能力建立起来

### 6.4 子步骤 D：把 chunk 和 reprompt 调度逻辑收敛到更可控的状态

目标：

- 排除是“某些 chunk / reprompt 组合”触发崩溃

建议做法：

1. 先引入一个最保守模式：
   - 更短 chunk
   - 更少 keyframe
   - 暂时关闭负点或关闭 reprompt
2. 逐项恢复：
   - 开启负点
   - 开启更多 keyframe
   - 开启 reprompt
3. 用最小真实 case 做 AB 对照

为什么这么做：

- 这一步的目标不是立刻拿到最高画质
- 而是先确认 crash 是由哪一类复杂特性触发的

### 6.5 子步骤 E：为真实 preprocess 建立稳定 smoke gate

目标：

- 后续任何改动都能快速知道自己有没有把稳定性重新改坏

建议做法：

1. 新增真实素材 smoke 脚本
2. smoke 的输出必须包含：
   - `metadata.json`
   - `src_mask`
   - `src_bg`
   - `qa_outputs` 中至少一份 mask overlay
3. smoke 成功后自动执行：
   - contract check
   - roundtrip check
   - QA artifact existence check


## 7. 推荐的实施顺序

建议严格按以下顺序推进：

1. 固定最小真实复现 case
2. 增加更细的 runtime 诊断与 JSON trace
3. 参数化 SAM2 runtime profile
4. 先用最保守 profile 跑通
5. 在可重复通过的前提下，逐步恢复更复杂 prompt / reprompt 逻辑

顺序原则是：

- 先稳定
- 再定位
- 再释放性能


## 8. 验证与测试方案

### 8.1 必做验证

1. Synthetic regression 全通过
2. 真实 10 秒 preprocess smoke 至少连续成功 3 次
3. 高负载配置和低负载配置都完成，不再停在 `started`
4. manifest 中 preprocess stage 正常写 `completed`
5. QA 产物完整存在

### 8.2 推荐额外记录的指标

- 每个 chunk 的耗时
- 每个 chunk 的峰值显存
- 每个 chunk 的 prompt 点数量
- 每个 chunk 的 mask 覆盖率统计
- 每次 reprompt 的触发点和触发原因

### 8.3 判定标准

本步骤算通过，必须同时满足：

- 真实素材 preprocess 不再 native crash
- 至少有一套配置可在 H200 上重复通过
- 失败时能给出明确诊断信息，而不是无信息崩溃


## 9. 风险

### 9.1 风险一：问题在第三方 vendor 代码里

如果问题在 `src/sam-2/` 的底层实现里，repo 自身可能只能做绕过，不能根治。

应对方式：

- 先把 wrapper 层输入严格化
- 尽量缩小到能稳定触发问题的最小条件
- 必要时对 vendor 层做最小补丁，而不是大规模魔改

### 9.2 风险二：为了稳定性把质量降得太多

在稳定性修复阶段，很容易通过“更小 chunk、更少 prompt、更保守 profile”先跑通。  
但这不能直接当最终配置。

应对方式：

- 稳定性配置和高质量配置分开管理
- smoke gate 先验证能跑通
- 后续 Step 02 再专门做 H200 高性能 profile


## 10. 回退策略

如果在修复过程中引入了更大范围的不稳定性，应保留：

- 当前可工作的 synthetic 路径
- 当前 lossless pipeline
- 当前 QA 输出能力

并允许通过开关退回“最保守 preprocess profile”。

回退目标不是回到原版，而是保住当前已完成的工程能力，同时继续定位真实 crash。


## 11. 完成标志

本步骤完成的标志是：

1. 真实 10 秒素材 preprocess smoke 连续通过
2. preprocess manifest 正常闭环
3. SAM2 runtime profile 已参数化
4. crash triage 信息完整可复用
5. 后续步骤可以在“稳定的真实 preprocess”基础上继续推进
