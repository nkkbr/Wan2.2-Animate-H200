# Step 04 Findings: Semantic Boundary Experts Training Path

## 1. 结论

Step 04 已完成。

这一步的结论是：

- 语义专家训练路径在工程上成立。
- `hair / face / hand` 三类 expert 都能完成训练、bundle 推理和 reviewed benchmark 评估。
- 但三轮都没有通过 promotion gate。
- 因此，这条路线当前只能保留为研究支线，不能升入主线。

最终冻结结果：

- `best-of-three = Round 1`

冻结原因：

- 三轮的 `semantic_pass_count` 都是 `0`
- Round 1 的侵入性最小
- Round 1 的 generate smoke 退化最轻
- Round 2 / Round 3 虽然局部训练指标更强，但 end-to-end 收益没有放大，反而把 seam / background 带坏得更明显


## 2. 本步新增内容

核心文件：

- `wan/utils/semantic_edge_experts.py`
- `scripts/eval/check_semantic_edge_experts.py`
- `scripts/eval/train_semantic_edge_experts.py`
- `scripts/eval/infer_semantic_edge_experts.py`
- `scripts/eval/evaluate_semantic_edge_experts.py`

实现内容：

- shared trunk + semantic-specific heads 的专家模型
- 基于 `face_alpha`、`face_bbox_curve.json`、`src_hand_tracks.json` 和人物边界活动区的 ROI routing
- `hair / face / hand / cloth` 语义 expert 输出写回统一 preprocess bundle
- reviewed benchmark 和 generate smoke 的统一 gate


## 3. 三轮设置

### Round 1

- enabled tags: `hair`
- train dir: `runs/optimization8_step04_round1/train`
- preprocess candidate: `runs/optimization8_step04_round1/preprocess_semantic_experts`

### Round 2

- enabled tags: `hair, face`
- train dir: `runs/optimization8_step04_round2/train`
- preprocess candidate: `runs/optimization8_step04_round2/preprocess_semantic_experts`

### Round 3

- enabled tags: `hair, face, hand`
- train dir: `runs/optimization8_step04_round3/train`
- preprocess candidate: `runs/optimization8_step04_round3/preprocess_semantic_experts`


## 4. 训练阶段观测

训练层面，语义专家不是“完全学不到东西”。

最佳验证指标分别是：

- Round 1
  - `semantic_boundary_f1.hair = 0.2327`
- Round 2
  - `semantic_boundary_f1.face = 0.1028`
  - `semantic_boundary_f1.hair = 0.2417`
- Round 3
  - `semantic_boundary_f1.face = 0.1050`
  - `semantic_boundary_f1.hair = 0.2292`
  - `semantic_boundary_f1.hand = 0.2333`

这说明：

- patch-level 语义监督是能被模型吸收的
- 失败不在“训练完全没学到”
- 失败发生在“局部学习结果无法稳定转化成 full-bundle reviewed edge gain”


## 5. Reviewed Benchmark 结果

基线采用：

- `runs/optimization8_step03_round1/candidate_metrics_v3.json`

### Round 1

- `holdout_boundary_f1_gain_pct = +0.0091%`
- `holdout_alpha_mae_reduction_pct = -0.2143%`
- `holdout_trimap_error_reduction_pct = -0.2550%`
- `semantic_gains_pct.hair = 0.0`
- `semantic_pass_count = 0`

### Round 2

- `holdout_boundary_f1_gain_pct = +0.0238%`
- `holdout_alpha_mae_reduction_pct = -0.2029%`
- `holdout_trimap_error_reduction_pct = -0.1677%`
- `semantic_gains_pct.hair = 0.0`
- `semantic_gains_pct.face = 0.0`
- `semantic_pass_count = 0`

### Round 3

- `holdout_boundary_f1_gain_pct = +0.1182%`
- `holdout_alpha_mae_reduction_pct = +0.9085%`
- `holdout_trimap_error_reduction_pct = +0.9563%`
- `semantic_gains_pct.hair = 0.0`
- `semantic_gains_pct.face = 0.0`
- `semantic_gains_pct.hand = 0.0`
- `semantic_pass_count = 0`

解释：

- Round 3 是 reviewed 指标里最好的，但仍远低于 gate
- Step 04 期望至少两个 semantic 类别显著提升
- 实际上三轮都是 `0 / 2` 或 `0 / 3`


## 6. Generate Smoke 结果

基线 smoke：

- `runs/optimization8_step03_round1_smoke`

### Round 1 smoke

- `runs/optimization8_step04_round1_smoke`
- `seam_degradation_pct = -7.6067%`
- `background_fluctuation_improvement_pct = -10.1252%`

### Round 2 smoke

- `runs/optimization8_step04_round2_smoke`
- `seam_degradation_pct = -56.9007%`
- `background_fluctuation_improvement_pct = -23.0784%`

### Round 3 smoke

- `runs/optimization8_step04_round3_smoke`
- `seam_degradation_pct = -232.7917%`
- `background_fluctuation_improvement_pct = -62.7647%`

这里的符号约定是：

- 正值表示相对基线改善
- 负值表示相对基线退化

所以 smoke 结果说明：

- 语义专家并不是“没有接到主链”
- 它们确实会影响真实生成结果
- 但当前影响方向是负面的，且随着路由覆盖变强，退化更明显


## 7. 为什么失败

当前最合理的失败解释有三条：

### 7.1 ROI patch 学到的语义边界修正，没能在 full-bundle 上形成足够强的全局收益

训练 summary 在变，但 reviewed semantic 指标保持不动，说明：

- local expert 的 patch 内学习结果
- 没有被足够稳定、足够大幅度地回灌到完整 bundle

### 7.2 当前 routing 是启发式的，覆盖区域虽然增加了，但与 reviewed holdout 的关键难点边界并不充分对齐

Round 1 到 Round 3 的 route counts 是：

- Round 1: `35`
- Round 2: `55`
- Round 3: `134`

但随着 route 覆盖增加，reviewed semantic 指标没有起来，generate smoke 反而更差。

这更像是：

- route 确实在工作
- 但工作的位置或 blending 方式不对

### 7.3 当前集成方式仍然太像“局部修补”

它还没有成为真正的强语义主路径，而更像：

- 在 trainable alpha 之后
- 再按语义补一层 residual patch

这类结构的上限很可能仍然不够高。


## 8. 最终判断

Step 04 的最终判断是：

- 工程路径成立
- 研究价值成立
- 主线推广失败

不应做的事：

- 不要把 Step 04 的任一轮升为默认
- 不要继续在这条 shared-trunk semantic expert 路线上做更多小调参

应保留的部分：

- ROI routing 代码
- semantic expert 训练和推理脚手架
- Step 04 的 reviewed + smoke gate

这些都可以作为后续更强语义专家路线的基础设施。


## 9. 下一步建议

Step 04 结束后，最合理的下一步是继续 `optimization8 / Step 05`：

- `05-compositing-aware-losses-and-evaluation-stack.md`

理由：

- 当前 trainable alpha baseline 和 semantic expert 都显示出同一个问题：
  - patch-level 能学
  - full-bundle / smoke 上不成立
- 这通常意味着 loss 和评估目标不够贴近真实 composite 结果

因此后面更值得打的，不是继续堆更多专家，而是：

- 把 compositing-aware loss 和评估先做扎实

