# Step 08: H200 High-Value Compute Allocation

## 1. 目的

本步骤的目标是正式定义 H200 141GB 在 `optimization4` 中的最优预算分配方式。


## 2. 为什么要做

如果没有清晰预算策略，H200 的资源很容易再次被：

- 全图 generate 步数
- 重复无效 AB
- 低价值全局 refine

吞掉。  
而当前真正影响边缘锐度的高价值区域其实是：

- alpha / matting
- boundary ROI refine
- generate-side edge candidate search


## 3. 交付目标

本步骤结束时，应具备：

1. H200 高价值模式定义
2. profiling / accounting
3. 推荐生产 preset
4. 自动化预算分配原则


## 4. 推荐模式

建议至少定义三类模式：

- `edge_preprocess_extreme`
- `edge_generate_search`
- `edge_roi_refine`

每种模式都要明确：

- 目标
- 分辨率
- 候选数
- 最大预算
- 默认启停模块


## 5. 实施方案

### 5.1 建 runtime profiling

不仅记录总时间，还要记录：

- alpha/matting 时间
- ROI refine 时间
- candidate search 时间
- base generate 时间

### 5.2 建预算策略

推荐优先级：

1. alpha / matting
2. boundary ROI refine
3. generate-side candidate search
4. 再考虑整体采样步数

### 5.3 生产 preset

至少输出：

- `hq_h200_edge`
- `hq_h200_edge_search`
- `hq_h200_edge_roi`


## 6. 强指标要求

本步骤的“成功”不只看质量，也看预算是否花得值：

- 额外 runtime 中，至少 **60% 以上** 花在高价值边缘模块
- 真实 quality gain 与 runtime gain 成正相关
- 不再出现“算力主要花在低价值全图路径”这种情况


## 7. 实现-测评-再改闭环

最多 3 轮。

### Round 1

- 建 profiling
- 看当前钱花在哪

### Round 2

- 调预算顺序
- 固化高价值 preset

### Round 3

- 以真实 quality gain 对照 runtime gain
- 冻结生产 preset


## 8. 风险

### 8.1 预算设计过于复杂

应对：

- 只保留 3 个核心 H200 模式

### 8.2 追求极致质量导致产能不可用

应对：

- 每种模式都定义明确预算上限


## 9. 完成标准

只有当：

- H200 的额外算力被明确投向高价值边缘模块
- 质量收益可证明
- 生产 preset 可稳定复现

本步骤才算真正完成。
