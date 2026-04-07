# Step 02: H200 多阶段多分辨率 Preprocess 主链

## 1. 步骤目标

本步骤的目标是把当前 `preprocess_data` 从“单条主链 + 单分辨率分析”升级为：

- 全局层
- 人物 ROI 层
- 脸部 ROI 层

三层协同的多阶段 preprocess 系统，并让 H200 主要用于更高分析分辨率和局部重跑。


## 2. 为什么要做这一步

### 2.1 当前系统还没有真正进入高精度 preprocess 范式

虽然当前代码已经支持：

- H200 preprocess runtime profiles
- analysis/export 区分
- 稳定的 SAM2 smoke

但整体上仍然更像“增强版单阶段 preprocess”，还不是高精度视觉分析系统。

### 2.2 单层分析会直接限制轮廓、动作、脸的上限

一旦全局分辨率不够或局部 ROI 没有单独重跑，最容易受损的是：

- 发丝和手部边界
- 快速动作下的 limb 轨迹
- 脸部细节和头姿

### 2.3 H200 的最大价值恰好在这里

下一阶段最值得用 H200 的地方，不是简单增大导出分辨率，而是：

- 把全局分析做得更准
- 把人物 ROI 做得更精
- 把脸 ROI 做得更稳


## 3. 本步骤的交付结果

完成后应至少交付：

1. 三层 preprocess 数据流
2. analysis resolution 策略升级
3. ROI proposal / ROI rerun 机制
4. per-stage runtime 与 memory stats
5. H200 extreme preset


## 4. 设计原则

### 4.1 全局层负责稳，ROI 层负责精

- 全局层：
  - 跑整帧
  - 做 tracking、粗分割、粗 pose、背景可见性
- 人物 ROI 层：
  - 提升轮廓、身体局部结构
- 脸部 ROI 层：
  - 提升 face landmarks、pose、alpha、parsing

### 4.2 analysis resolution 与 export resolution 完全分离

要求：

- export 分辨率只服务于生成输入格式
- analysis 分辨率只服务于识别精度

不能再让 export 分辨率反向限制 analysis 精度。

### 4.3 必须保留可回退的单阶段路径

第一版多阶段实现必须能回退到当前 `optimization2` 稳定主链，以便排查问题。


## 5. 代码改动范围

重点模块：

- `wan/modules/animate/preprocess/preprocess_data.py`
- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/modules/animate/preprocess/sam_runtime.py`
- 新增 ROI scheduler / multistage preprocess helper
- metadata schema 扩展
- benchmark / runtime summary 脚本


## 6. 实施方案

### 6.1 子步骤 A：定义三层 preprocess 结构

建议结构：

- `global_analysis`
- `person_roi_analysis`
- `face_roi_analysis`

每层都要输出：

- 输入分辨率
- ROI 来源
- 耗时
- 峰值显存
- 置信度摘要

### 6.2 子步骤 B：定义 H200 extreme analysis 配置

建议默认探索值：

- global short side: `1280` 或 `1536`
- person ROI long side: `1536` 到 `2048`
- face ROI: `1024`

第一版不要求所有视频都用最激进配置，但必须支持这条路径。

### 6.3 子步骤 C：增加 ROI 提案与重跑策略

触发条件可以包括：

- 关键点低置信
- 脸部大角度
- 手部高速运动
- mask 边界过细或不稳定

### 6.4 子步骤 D：把 runtime stats 和 metadata 扩展到多阶段

要求：

- 每层独立记录耗时和分辨率
- 每个 ROI 重跑都留 provenance


## 7. 目标指标

相对 `optimization2` 基线，本步骤结束时至少应达到：

- preprocess 稳定性：真实 10 秒 smoke `3/3` 通过
- face bbox jitter：降低 `25%`
- pose jitter：降低 `20%`
- ROI 覆盖率：关键帧手部/脸部在高难样本中 `>= 95%`

同时，单 candidate preprocess 总时间不应超过当前结构化 video baseline 的 `3x`。


## 8. 迭代规则

### Round 1

- 打通三层数据流
- metadata 与 runtime stats 正常输出
- smoke 跑通

### Round 2

- 开启高分辨率 ROI 路径
- 做首轮 jitter / stability 指标对照

### Round 3

- 调整 ROI 触发与重跑规则
- 收敛 H200 extreme preset

若 Round 3 后仍无法显著降低 face/pose jitter，则必须冻结 findings，并回看 ROI 提案机制。


## 9. 验证方案

必须执行：

- `optimization3` suite 中的 multistage preprocess smoke
- 高难 clip 上的 ROI coverage 统计
- `face bbox jitter` / `pose jitter` 指标对照
- H200 runtime/memory profiling


## 10. 风险与回退

### 风险

- 分辨率和 ROI 过高导致时延暴涨
- ROI 提案错误导致局部分析漂移

### 回退

- 保留 `optimization2` 单阶段主链作为 fallback
- 支持分层关闭：
  - `--disable_person_roi_refine`
  - `--disable_face_roi_refine`

