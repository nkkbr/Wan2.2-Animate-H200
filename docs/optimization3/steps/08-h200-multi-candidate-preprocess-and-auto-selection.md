# Step 08: H200 多候选 Preprocess 与自动选优

## 1. 步骤目标

本步骤的目标是把 H200 从“高精度单候选 preprocess 运行平台”升级为“多候选高精度 preprocess 选优平台”。

最终目标是：

- 同一段视频可以生成多个高精度 preprocess candidate
- 自动评分
- 自动选出最优 bundle 再送入 generate


## 2. 为什么要做这一步

### 2.1 极致精度往往不是单条固定配置可以覆盖的

同一段视频里：

- 边界最优配置
- face 最优配置
- pose 最优配置

不一定完全一致。  
如果只允许单候选 preprocess，系统会被迫在不同目标之间折中。

### 2.2 H200 的真正优势在于可以把“试错”系统化

相比只跑一套流程，H200 更值得用于：

- 同时生成 2 到 4 套候选
- 用客观指标选最优

### 2.3 自动选优能把高精度能力变成稳定生产力

否则，即使有更高精度能力，也只能依赖人工挑选，难以形成稳定生产流程。


## 3. 本步骤的交付结果

1. 多候选 preprocess orchestration
2. candidate-level scoring
3. auto-selection policy
4. H200 candidate scheduling
5. end-to-end benchmark 与 gate


## 4. 设计原则

### 4.1 候选数要有限但有效

不追求暴力穷举。  
建议第一版每段视频只生成 `2` 到 `4` 个高价值 candidate。

### 4.2 评分必须以客观指标为主

自动选优不能只靠启发式排序。  
应优先使用：

- boundary metrics
- face metrics
- pose metrics
- background stability
- uncertainty summary

### 4.3 选优应可解释

最终选中的 candidate，必须能回答：

- 为什么选它
- 它在哪些指标上优于其他 candidate


## 5. 代码改动范围

- 新增 candidate orchestrator
- `wan/utils/experiment.py`
- `scripts/eval/run_optimization3_candidate_search.py`
- `scripts/eval/select_best_preprocess_candidate.py`
- metadata / manifest 扩展


## 6. 实施方案

### 6.1 子步骤 A：定义 candidate 维度

建议第一版候选只在高价值轴上变化：

- boundary fusion config
- face rerun aggressiveness
- motion refine aggressiveness
- background mode

### 6.2 子步骤 B：统一 candidate score

建议总分由多项指标加权：

- boundary precision
- face precision
- pose stability
- background stability
- uncertainty penalty
- runtime penalty

### 6.3 子步骤 C：H200 candidate scheduling

支持：

- 串行
- 小批并行
- 失败 candidate 自动跳过

### 6.4 子步骤 D：与 generate 串联

最终只把 `best_candidate` 送入 generate 主链。


## 7. 目标指标

相对单候选默认 preprocess，至少达到：

- 自动选中的 candidate 在综合分上优于默认候选的比例 `>= 70%`
- 高难 clip 中优于默认候选的比例 `>= 80%`
- candidate search 本身稳定 `3/3` 通过


## 8. 迭代规则

### Round 1

- 打通 candidate orchestration
- 至少支持 2 候选

### Round 2

- 加入综合评分与自动选优
- 跑 benchmark 集验证

### Round 3

- 优化 scoring 权重
- 冻结 candidate v1

若 Round 3 后自动选优无法显著优于默认单候选，则本步骤失败，继续保留单候选主链。


## 9. 验证方案

必须执行：

- candidate search benchmark
- selected vs default AB
- 高难 clip 优胜率统计
- runtime / memory profiling


## 10. 风险与回退

### 风险

- candidate search 时间过长
- 评分体系与主观质量不一致

### 回退

- 保留单候选稳定主链
- auto-selection 默认关闭，仅作为高精度模式启用

