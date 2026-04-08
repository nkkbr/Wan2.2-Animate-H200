# Step 08 Findings: H200 High-Value Compute Allocation and Orchestration

## 1. 结论

Step 08 已执行完成，并按文档要求做了 3 轮“实现 -> 测评 -> 再改”。

最终结论：

- H200 的 `default / high / extreme` 三档质量路径已经正式建立
- tier benchmark、runtime 统计、pairwise edge 对比都已跑通
- 但 **没有任何一轮通过强 gate**
- best-of-three 冻结为 **Round 3**
- `quality_tiers.step08.json` 已写出，但这些 tiers **不能直接升成生产默认**


## 2. 本步骤交付

### 2.1 新增脚本

- [check_h200_orchestration.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_h200_orchestration.py)
- [run_optimization5_h200_tier_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization5_h200_tier_benchmark.py)
- [evaluate_optimization5_h200_tier_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization5_h200_tier_benchmark.py)

### 2.2 新增配置

- [quality_tiers.step08.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/benchmark/quality_tiers.step08.json)

### 2.3 运行路线

Step 08 最终采用的真实 benchmark 路线是：

- preprocess bundle 固定为当前 reviewed GT 最强 baseline：
  - `runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess`
- generate 保持：
  - `replacement_conditioning_mode=legacy`
  - `boundary_refine_mode=none`
- 只测试 H200 上不同 `sample_steps` 的边缘收益 / 代价

这样做的原因是：

- Step 07 之前的 edge heuristic / trainable 路线都没有证明自己值得进生产
- 所以 Step 08 不再去优化未证明有效的模块
- 只测试“多给 H200 采样预算”是否能直接换来边缘收益


## 3. 三轮结果

### 3.1 Round 1

目录：

- [optimization5_step08_round1_tiers](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step08_round1_tiers)

tier：

- `default = 4`
- `high = 8`
- `extreme = 12`

结果：

- `high vs default`
  - `gradient = +44.05%`
  - `contrast = -8.53%`
  - `halo = +0.0067%`
  - `seam = +7.64%`
  - `runtime = +34.42%`
- `extreme vs high`
  - `gradient = +1.37%`
  - `contrast = -3.08%`
  - `halo = +0.49%`
  - `seam = +6.37%`
  - `runtime = +22.70%`

判断：

- 步数增加可以明显抬 `gradient`
- 但 `contrast` 和 `seam` 代价太大


### 3.2 Round 2

目录：

- [optimization5_step08_round2_tiers](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step08_round2_tiers)

tier：

- `default = 4`
- `high = 6`
- `extreme = 8`

结果：

- `high vs default`
  - `gradient = +33.38%`
  - `contrast = -7.08%`
  - `halo = -0.34%`
  - `seam = +10.24%`
  - `runtime = +15.67%`
- `extreme vs high`
  - `gradient = +8.00%`
  - `contrast = -1.56%`
  - `halo = +0.35%`
  - `seam = -2.37%`
  - `runtime = +5.07%`

判断：

- `extreme vs high` 已经把 seam 控下来了
- 但 `high vs default` 仍然明显不够稳


### 3.3 Round 3

目录：

- [optimization5_step08_round3_tiers](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step08_round3_tiers)

tier：

- `default = 4`
- `high = 5`
- `extreme = 6`

结果：

- `high vs default`
  - `gradient = +18.28%`
  - `contrast = -4.55%`
  - `halo = -0.18%`
  - `seam = +5.60%`
  - `runtime = +12.52%`
- `extreme vs high`
  - `gradient = +12.76%`
  - `contrast = -2.65%`
  - `halo = -0.16%`
  - `seam = +4.40%`
  - `runtime = -1.18%`

判断：

- 这是三轮里最保守、最接近“生产 tier”形态的一轮
- 但 `contrast` 仍为负，`seam` 仍超过 `3%` 门槛


## 4. best-of-three 冻结

冻结结果：

- **Round 3**

原因：

- 三轮中副作用最低
- `runtime` 代价也最可控
- 虽然仍未通过 gate，但它最接近“如果必须给出 tier 配置，应该冻结成什么”的答案


## 5. 为什么仍然失败

这一步失败，不是 orchestration 框架坏了，而是说明：

1. 当前最强 baseline preprocess bundle 已经很强
2. 单纯增加 generate `sample_steps`
3. 可以提升 `gradient`
4. 但无法同时保住：
   - `contrast`
   - `halo`
   - `seam`

所以 Step 08 的真正价值是：

- 它证明了“多给 H200 采样预算”并不能自动换来更好的边缘质量
- 当前边缘问题的关键，仍然不是 compute budget 不够，而是 edge reconstruction 路线本身还不够强


## 6. 工程建议

当前可保留的结果是：

- `quality_tiers.step08.json` 作为正式 benchmark / 参考 tier 配置保留
- 但不应直接把 `high` 或 `extreme` 设成生产默认

更准确的生产建议是：

- `default = 4 steps` 继续作为当前主路径
- `high = 5 steps`
- `extreme = 6 steps`

这三档可以保留给后续更强 edge route 的 orchestrator 使用；  
但在现阶段，它们还不是“更锐边缘”的生产答案。


## 7. 最终结论

Step 08 完成，但不通过强 gate。

- orchestration 框架：完成
- H200 quality tiers：完成
- 生产默认提升：未达成

这一步最重要的产出不是新默认配置，而是一个更明确的结论：

> 当前 edge 路线的主要瓶颈不是算力分配，而是边缘生成/重建能力本身。
