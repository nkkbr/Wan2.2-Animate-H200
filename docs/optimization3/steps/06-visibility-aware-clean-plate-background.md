# Step 06: 可见性驱动的 Clean Plate Background 2.0

## 1. 步骤目标

本步骤的目标是把当前 `clean_plate_video` 从“时序更稳的背景近似”升级成“带可见性、支持度、不确定性语义”的背景系统。

交付形态至少应包含：

- `clean_plate`
- `visible_support_map`
- `unresolved_region`
- `background_confidence`
- `background_source_provenance`


## 2. 为什么要做这一步

### 2.1 当前 clean_plate_video 已经有效，但仍有明显上限

`optimization2` 的验证已经证明：

- `clean_plate_video` 明显优于 `clean_plate_image`

但当前系统仍然存在一个关键问题：

- 长期被人物遮挡的区域其实并没有真正恢复，只是被低质量估计或 fallback 补上

### 2.2 背景不确定区域会直接污染边界质量

如果 generate 把低置信背景误认为高置信背景，就会在人物边界处出现：

- 脏边
- 色边
- 轮廓贴边错误
- 时序闪烁

### 2.3 需要显式建模“可恢复”和“不可恢复”

下一阶段背景系统必须承认：

- 有些背景能恢复
- 有些只能估计
- 有些根本不可恢复


## 3. 本步骤的交付结果

1. 背景可见性/支持度 schema
2. clean plate 2.0 构建流程
3. unresolved region 明确输出
4. generate 可消费的 background confidence


## 4. 设计原则

### 4.1 conditioning background 与 evaluation background 分离

- `conditioning background`：
  - 提供生成约束
- `evaluation background`：
  - 用于 debug 和质量评估

### 4.2 不可恢复区域不能伪装成高置信 clean plate

这一步最重要的原则是诚实。  
宁可显式标记 `unresolved`，也不要把低质量补出来的背景当成高置信先验。

### 4.3 visible support 是核心中间变量

所有背景决策都应尽量围绕：

- 这个像素在时序上被看见过多少次
- 看见时的质量如何


## 5. 代码改动范围

- `wan/modules/animate/preprocess/background_clean_plate.py`
- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/utils/animate_contract.py`
- generate 背景消费逻辑
- 新背景指标脚本


## 6. 实施方案

### 6.1 子步骤 A：扩展背景 artifact schema

建议新增：

- `src_visible_support.npz`
- `src_unresolved_region.npz`
- `src_background_confidence.npz`

### 6.2 子步骤 B：构建 visible support map

需要统计：

- 某像素在整个视频中被人物遮挡之外的观察次数
- 观察质量
- 时间跨度

### 6.3 子步骤 C：clean plate 2.0 生成

建议分三类区域处理：

- 高 support 区域：直接视频融合恢复
- 中 support 区域：带权估计
- 低 support / unresolved 区域：显式标注，不伪装为高置信

### 6.4 子步骤 D：背景 confidence 接入 generate

要求 generate 能区分：

- 高置信背景保持
- 低置信背景保守使用


## 7. 目标指标

相对当前 `clean_plate_video` 基线，至少达到：

- `temporal_fluctuation_mean <= 0.9`
- `band_adjacent_background_stability <= 0.27`
- `unresolved_ratio_mean` 相对当前下降 `20%`
- `visible_support` 与背景误差显著相关


## 8. 迭代规则

### Round 1

- 打通 support / unresolved / confidence artifact
- clean plate 2.0 smoke 跑通

### Round 2

- 跑 image/video/video2.0 AB
- 优化 support accumulation 与 fallback

### Round 3

- 聚焦长期遮挡区域和边界附近脏边
- 冻结 background 2.0

若 Round 3 后背景时序稳定性没有显著优于当前基线，则不得继续把其作为默认生成背景。


## 9. 验证方案

必须执行：

- `image` vs `video` vs `video2.0` AB
- band-adjacent stability 统计
- unresolved 区域可视化 review


## 10. 风险与回退

### 风险

- background confidence 设计不合理会误导 generate
- unresolved 区域过大影响生成自由度

### 回退

- 保留现有 `clean_plate_video`
- 支持 `background_mode=clean_plate_video_v1|v2`

