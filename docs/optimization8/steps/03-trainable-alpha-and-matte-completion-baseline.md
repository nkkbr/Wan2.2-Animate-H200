# Step 03: Trainable Alpha And Matte Completion Baseline

## 1. 为什么要做

在第二档路线里，首先值得验证的不是所有花哨专家，而是：

> 在更强数据和更清晰任务定义下，最基础的 trainable alpha / matte completion 路线，是否终于能客观打赢当前非训练基线。

如果这一步都不能成立，那么后面的 semantic experts 和复杂 loss 大概率也很难成立。


## 2. 目标

建立一个受控的 trainable baseline，用于学习：

- alpha refinement
- matte completion

并验证它是否在 reviewed benchmark 与真实 smoke 上优于当前非训练基线。


## 3. 要改哪些模块

建议新增或扩展：

- `wan/utils/trainable_alpha_model.py`
- `scripts/eval/train_trainable_alpha_model.py`
- `scripts/eval/infer_trainable_alpha_model.py`
- `scripts/eval/evaluate_trainable_alpha_model.py`
- `scripts/eval/check_trainable_alpha_model.py`


## 4. 具体如何做

### 4.1 模型范围

先做小而清晰的 baseline，不要一开始就上复杂专家：

- patch-level alpha refiner
- matte completion baseline

### 4.2 输入输出

输入应至少包含：

- foreground patch
- background patch
- soft alpha
- trimap unknown
- boundary ROI mask

输出：

- refined alpha
- completed matte

### 4.3 训练预算

先用可控预算：

- 小模型
- 小 batch
- 少量 epoch
- 明确 early stop


## 5. 如何检验

### 5.1 reviewed benchmark

必须对比：

- `boundary_f1`
- `alpha_mae`
- `trimap_error`
- `alpha_sad`

### 5.2 real-video smoke

至少跑：

- preprocess / infer 路径
- 最小 generate smoke
- 看 composite 后的 `gradient / contrast / halo`


## 6. 强指标要求

建议 gate：

- `boundary_f1` 相对非训练基线提升 `>= 8%`
- `alpha_mae` 下降 `>= 10%`
- `trimap_error` 下降 `>= 10%`

至少满足其中两条，才算 baseline training route 成功。


## 7. 实现-测评-再改闭环

### Round 1

- 跑通最小训练 -> 推理 -> 评测

### Round 2

- 调整模型、采样或训练预算
- 跑 reviewed + smoke AB

### Round 3

- 冻结 best-of-three
- 明确 trainable baseline 是否值得继续放大


## 8. 成功标准

- trainable alpha baseline 正式成形
- 至少在 reviewed benchmark 上显著优于非训练基线
- 并且真实 smoke 不出现明显 regression
