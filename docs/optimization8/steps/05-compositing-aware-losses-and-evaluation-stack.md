# Step 05: Compositing-Aware Losses And Evaluation Stack

## 1. 为什么要做

以前训练路线反复失败，一个重要原因是：

- loss 看起来合理
- 但并没有真正对最终 composite 质量负责

如果 loss 只盯局部像素，就会继续出现：

- val loss 下降
- 最终视频边缘没变好


## 2. 目标

建立 compositing-aware 的 loss 和 evaluation stack，使训练目标真正对应：

- 最终边缘质量
- final composite 质量
- semantic boundary 质量


## 3. 要改哪些模块

建议至少新增：

- `wan/utils/edge_losses.py`
- `scripts/eval/check_compositing_aware_losses.py`
- `scripts/eval/evaluate_compositing_aware_training.py`
- 扩展训练脚本以支持 loss registry


## 4. 具体如何做

### 4.1 loss 组成

建议从少量高价值 loss 开始：

- alpha MAE / SAD
- trimap-focused loss
- boundary-weighted loss
- compositing reconstruction loss
- gradient preservation loss
- contrast preservation loss

### 4.2 evaluation stack

训练评估不能只看 loss，还要同时输出：

- reviewed benchmark metrics
- semantic boundary metrics
- real-video smoke metrics

### 4.3 ablation

不要同时上太多 loss。必须做 ablation，明确：

- 哪个 loss 真有帮助
- 哪个 loss 只是让训练更复杂


## 5. 如何检验

### 5.1 loss correctness

- 数值稳定
- 不爆梯度
- 对 hard samples 有正确响应

### 5.2 quality gain

相对 Step 03 或 Step 04 baseline：

- reviewed benchmark 明显提升
- real-video smoke 至少不退化


## 6. 强指标要求

建议 gate：

- 至少一种 compositing-aware loss 组合能稳定优于纯像素型 baseline
- `boundary_f1 / alpha_mae / trimap_error` 至少两项显著改善


## 7. 实现-测评-再改闭环

### Round 1

- 接最小 loss registry
- 做单一 loss ablation

### Round 2

- 做两组高价值组合 loss
- 跑 reviewed + smoke AB

### Round 3

- 冻结 best-of-three loss stack


## 8. 成功标准

- loss stack 不再只是训练内部指标，而是能带来可感知质量增益
- 为 Step 06 的综合 promotion gate 提供可信依据
