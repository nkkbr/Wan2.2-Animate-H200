# Step 01: 高精度 Benchmark 与质量 Gate

## 1. 步骤目标

本步骤的目标是建立一套真正服务于“极致识别精度”的 benchmark 与质量 gate，让后续每一步的改动都能被客观验证，而不是只靠肉眼感受。

这是本轮所有步骤的基础。  
如果这一步不做，后面即使改了大量代码，也无法可靠判断：

- 边界是否真的更准
- 脸部几何是否真的更稳
- 动作轨迹是否真的更可信
- H200 的额外预算是否真的转化成了精度收益


## 2. 为什么必须先做这一步

### 2.1 当前 benchmark 还不够支撑“极致精度”

当前仓库已经有：

- synthetic checks
- contract checks
- smoke
- runtime stats
- 一轮系统 validation suite

这些足够证明“系统能跑、增强生效、方向正确”，但不足以证明：

- 轮廓 alpha 是否显著更准
- 脸部 landmark 是否显著更准
- pose 在高运动和遮挡下是否显著更稳

### 2.2 后续步骤都需要强依赖 benchmark

本轮 Step 02 到 Step 08 都会引入更复杂的识别和融合逻辑。  
如果没有统一 benchmark，会出现两个问题：

1. 无法知道哪条路线真的更优
2. 会出现“功能越来越多，但真实精度没有稳定提升”的风险


## 3. 本步骤的交付结果

完成后应至少交付：

1. 一套 `optimization3` 专用 benchmark 目录结构
2. 一组代表性 clip 的 case 清单
3. 一批关键帧或片段级标注
4. 一套面向轮廓、脸、动作、背景的客观指标脚本
5. 一套统一的 step gate 规则
6. 一份正式 benchmark 说明文档


## 4. Benchmark 设计

### 4.1 数据集分层

建议把 benchmark 分成三层。

#### A. 快速 smoke 集

用途：

- 每一步开发中快速回归
- 检查功能、契约、基本质量没有退化

建议：

- 2 到 3 个 10 秒以内视频
- 覆盖：
  - 正常单人直立
  - 快速动作
  - 明显边界挑战，例如头发、袖口、手部

#### B. 高精度标注集

用途：

- 做真正的质量门槛评估

建议：

- 5 到 10 个 representative clips
- 每个 clip 至少抽 5 到 10 帧做关键帧标注

标注内容应包括：

- 人物轮廓 / trimap / alpha
- 脸部 landmarks
- 全身关键点
- 遮挡区域
- 边界不确定区域

#### C. 长时序 stress 集

用途：

- 检查时序稳定性和长时段 drift

建议：

- 2 到 3 段 10 秒到 30 秒视频
- 重点覆盖：
  - 转身
  - 手部快速运动
  - 部分遮挡
  - 背景变化


## 5. 指标体系

### 5.1 轮廓指标

必须新增：

- hard mask IoU
- boundary F-score
- trimap error
- soft alpha MAE
- halo ratio
- band gradient strength
- boundary uncertainty calibration

其中 `uncertainty calibration` 建议定义为：

- 误差最高的像素是否集中在高 uncertainty 区域

### 5.2 脸指标

建议至少包括：

- landmark NME
- head pose jitter
- face bbox jitter
- face boundary consistency
- face alpha error

### 5.3 动作指标

建议至少包括：

- PCK 或等价关键点准确率
- body jitter
- hand jitter
- velocity smoothness
- occluded-limb continuity

### 5.4 背景指标

保留并扩展：

- temporal fluctuation
- band-adjacent background stability
- visible support ratio
- unresolved ratio

### 5.5 系统指标

必须记录：

- preprocess total runtime
- per-stage runtime
- peak memory
- generate runtime
- candidate count
- selected candidate score


## 6. 代码改动范围

本步骤预计会新增或修改：

- `docs/optimization3/benchmark/`
- `scripts/eval/`
- `wan/utils/experiment.py`
- `wan/utils/animate_contract.py`
- 可能新增 benchmark manifest / label schema / score summary 工具


## 7. 具体实施方案

### 7.1 子步骤 A：建立 benchmark 目录与 manifest

目标：

- 用统一 manifest 管理 clip、标注、指标和 gate

建议：

- `docs/optimization3/benchmark/README.md`
- `docs/optimization3/benchmark/benchmark_manifest.example.json`
- `scripts/eval/run_optimization3_validation_suite.py`
- `scripts/eval/summarize_optimization3_validation.py`

### 7.2 子步骤 B：定义标注 schema

建议 schema 包含：

- `hard_foreground`
- `soft_alpha`
- `boundary_band`
- `face_landmarks`
- `body_keypoints`
- `occlusion_regions`
- `uncertainty_regions`

### 7.3 子步骤 C：实现指标脚本

至少实现：

- `compute_boundary_precision_metrics.py`
- `compute_face_precision_metrics.py`
- `compute_pose_precision_metrics.py`
- `compute_background_precision_metrics.py`

### 7.4 子步骤 D：定义统一 gate

每一步测评结束后必须生成：

- `summary.json`
- `summary.md`
- `gate_result.json`

并明确：

- 哪些指标达标
- 哪些未达标
- 是否允许进入下一步


## 8. 迭代规则

### Round 1

- 打通 benchmark manifest、指标脚本、suite
- 至少覆盖 smoke 集和一部分标注集

### Round 2

- 补全标注集关键帧
- 跑通所有指标输出
- 确保 gate 能自动判断 pass/fail

### Round 3

- 修正统计口径
- 处理异常 case
- 冻结 benchmark v1

如果 Round 3 后仍然不能稳定输出完整 benchmark 结果，则本步骤失败，后续步骤不得继续。


## 9. 验证与门槛

本步骤本身的“精度提升”门槛不是画质提升，而是 benchmark 能否可靠运行。

通过标准：

1. smoke 集、标注集、stress 集都已登记进 manifest
2. 至少 4 类指标脚本能稳定输出结果
3. `optimization3` suite 能自动生成 gate 结果
4. 对现有 `optimization2` 基线能跑出完整 baseline 分数


## 10. 风险与回退

### 风险

- 标注工作量大
- 指标定义容易漂移
- face / pose / alpha 指标口径不统一

### 回退策略

- 若全量标注集来不及，先做“小而精”的关键帧标注子集
- 先固化评估脚本，再扩展数据量

