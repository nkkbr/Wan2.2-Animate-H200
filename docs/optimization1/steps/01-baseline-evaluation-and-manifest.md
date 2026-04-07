# Step 01: 基线、评测与运行清单

## 1. 目标

在修改任何业务逻辑之前，先建立一套能稳定比较不同版本结果的评测与产物管理体系。

这一步的目标不是提升画质，而是解决下面这个根问题：

如果没有统一的评测集、命名规范、运行清单和中间产物保存规范，那么后续所有“优化”都无法被可靠验证。


## 2. 为什么先做这一步

当前 repo 最大的问题不是“没有优化方向”，而是：

- 结果比较过于依赖主观观感
- 同一个视频在不同参数下的实验痕迹不统一
- 没有标准化记录 preprocess / generate 参数
- 很难把某次结果回溯到对应 commit、参数和素材版本

如果这一步不先做，后续会出现三个常见问题：

1. 某次结果看起来更好，但说不清是 preprocess 变了、generate 变了，还是保存格式变了。
2. 某个改动在 A 素材上更好，在 B 素材上更差，却没有统一评测集来揭示这种差异。
3. 长期迭代后无法回答“哪一版是真正的回归”。


## 3. 本步骤的范围

本步骤只做基础设施，不改变核心算法。

要落地的内容：

- 固定评测集定义
- 实验目录结构
- 运行清单 manifest 格式
- 调试产物的命名和位置规范
- 最低限度的客观评测指标定义
- 后续步骤共享的验收模板

本步骤不做：

- mask 质量优化
- seam blending
- background inpainting
- soft mask
- latent memory


## 4. 设计原则

### 4.1 先固定素材，再谈优化

评测集必须覆盖 replacement 真正困难的场景，而不是只选一两个“最容易成功”的视频。

### 4.2 每次运行都必须可追溯

必须能从最终产物反查：

- 输入素材
- 预处理参数
- 生成参数
- 代码版本
- 输出格式

### 4.3 主观与客观评测都要有

replacement 的很多问题仍然需要人工判断，但至少应有客观信号帮助排查：

- seam 差异
- background 区域波动
- face bbox 抖动
- mask 面积波动


## 5. 实施内容

### 5.1 建立固定评测集

建议建立一个小而覆盖面足够的 benchmark set，至少包含：

- 正脸慢动作
- 快速肢体动作
- 头发和衣摆丰富
- 明显遮挡
- 镜头平移或运动明显
- 强光照变化
- 背景复杂

每个 case 应包含：

- 原始驱动视频
- 目标参考全身像
- case id
- 场景标签
- 评测备注

建议增加一个 `benchmark_manifest.json` 或 `benchmark_manifest.yaml`，为后续脚本提供统一入口。

### 5.2 统一实验目录结构

建议所有实验采用统一目录结构，例如：

```text
runs/
  <run_name>/
    manifest.json
    preprocess/
    generate/
    outputs/
    debug/
    metrics/
```

其中：

- `manifest.json`
  保存本次运行的完整配置和环境信息
- `preprocess/`
  保存 preprocess 产物或其软链接
- `generate/`
  保存 clip 级中间产物
- `outputs/`
  保存最终视频和可视化结果
- `debug/`
  保存 overlay、曲线图、关键帧比较
- `metrics/`
  保存自动统计结果

### 5.3 设计统一 manifest

建议定义一个统一的 `manifest.json`，至少包含：

- `run_name`
- `timestamp`
- `git_commit`
- `task`
- `hardware`
- `checkpoint_dir`
- `input_video`
- `input_reference`
- `preprocess_args`
- `generate_args`
- `output_format`
- `output_paths`

其中 `hardware` 应至少记录：

- GPU 型号
- 显存
- CUDA 版本
- PyTorch 版本

### 5.4 定义最低限度的客观指标

第一版不要求很复杂，但至少应定义下面这些统计：

- seam score
  对每个 clip 拼接点前后若干帧做差异统计
- background fluctuation
  在 mask 外区域做帧间波动统计
- mask area curve
  统计 mask 面积在时间上的变化
- face bbox stability
  统计中心点和面积变化率
- runtime / vram
  统计每次运行的时间和显存峰值

### 5.5 定义人工评分模板

建议每个 case 对下面维度做 1-5 分人工评分：

- 身份一致性
- 表情跟随度
- 动作自然度
- 背景稳定性
- 人物边缘质量
- 段间接缝可见度

人工评分模板可以先简单，不必一开始就做成复杂系统。


## 6. 需要改动的代码与文件

这一步建议新增或修改的范围：

- `docs/optimization1/benchmark/` 或类似目录
  保存 benchmark 规范
- `scripts/eval/`
  保存简单统计脚本
- `generate.py`
  增加 `run_name`、`save_debug_dir`、`save_manifest` 等参数
- preprocess 入口
  支持将参数和产物写入 manifest

如果暂时不想改 Python 代码，第一版也可以先用脚本包装，但长期看更建议把 manifest 写入纳入主流程。


## 7. 如何做

建议按下面顺序实施。

1. 先整理 benchmark set，并为每个素材定义 case id。
2. 设计 `manifest.json` 字段。
3. 统一目录结构。
4. 在 preprocess / generate 入口增加 manifest 写入。
5. 增加最简单的统计脚本与人工评分表。

顺序不能反过来。  
如果先写复杂统计脚本但没有固定 benchmark 和 manifest，后面仍然会乱。


## 8. 如何检验

### 8.1 功能检验

对同一个 case 跑一次完整流程，检查是否能得到：

- 结构正确的实验目录
- 完整的 manifest
- 最终输出
- 至少一份自动统计结果

### 8.2 回溯检验

随机拿一条旧结果，确认是否可以仅通过该次输出目录反推出：

- 使用了什么参数
- 使用了哪版代码
- 使用了哪套素材

### 8.3 对照检验

用同一 case 和同一 seed，分别跑两组不同参数，确认：

- 目录结构一致
- manifest 可比较
- 指标结果可并排查看


## 9. 验收标准

只有满足下面条件，才算 Step 01 完成：

- 至少有一套固定 benchmark set 被正式定义
- preprocess 和 generate 的参数都能被完整记录
- 任意一次实验结果都可回溯到输入与配置
- 有最小化的自动统计结果输出
- 有统一的人工评分模板


## 10. 风险与回退

### 风险

- 过早设计过于复杂的评测系统，导致推进缓慢
- 字段设计不稳定，后续频繁改 manifest

### 回退策略

- 第一版只保留真正必要的字段
- 复杂统计可以后置，但目录结构和 manifest 必须先稳定


## 11. 本步骤完成后的收益

这一步完成后，后续每一个优化步骤都将变得更便宜：

- 可以可靠比较前后版本
- 可以快速定位回归
- 可以让多轮实验沉淀为可复用证据，而不是零散截图和主观印象
