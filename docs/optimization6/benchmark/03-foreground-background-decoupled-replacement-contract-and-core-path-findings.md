# Optimization6 Step 03 Findings

## 结论

Step 03 的工程目标已经完成，但质量目标没有达到默认升主线标准。

更准确地说：

- foreground / alpha / background / composite ROI 的 decoupled contract 已经正式接通；
- `decoupled_v2` 已被实现为新的 core conditioning 路径；
- 真实 decoupled preprocess bundle 的 contract 检查通过；
- 但在当前执行窗口内，没有形成足够强、足够稳定的真实 generate AB 证据来证明 `decoupled_v2` 优于当前稳定 baseline；
- 因此本步骤按设计规则收口为：
  - 保留 contract、代码路径、benchmark runner；
  - 不把 `decoupled_v2` 升为默认核心路径。

## 本步骤做了什么

### 代码与路径

- `generate.py`
- `wan/animate.py`
- `wan/utils/replacement_masks.py`
- `wan/utils/rich_conditioning.py`

新增：

- `replacement_conditioning_mode=decoupled_v2`

并让以下信号真正进入 core conditioning / background keep 计算：

- `foreground_rgb`
- `foreground_alpha`
- `foreground_confidence`
- `background_rgb`
- `background_keep_prior`
- `background_visible_support`
- `background_unresolved`
- `composite_roi_mask`
- `soft_alpha / trimap_unknown / uncertainty / occlusion`
- `face_preserve / face_confidence`
- `face/hair/hand/cloth/occluded boundary`

### benchmark / check

新增：

- `scripts/eval/run_optimization6_decoupled_benchmark.py`
- `scripts/eval/evaluate_optimization6_decoupled_benchmark.py`
- `scripts/eval/check_decoupled_core_path.py`

## 真实与离线验证

### 1. 真实 decoupled bundle contract

对现有完整 decoupled bundle：

- `runs/optimization5_step03_round1_preprocess/preprocess`

运行：

```bash
python scripts/eval/check_decoupled_replacement_contract.py \
  --src_root_path runs/optimization5_step03_round1_preprocess/preprocess
```

结果：

```json
{
  "status": "PASS",
  "frame_count": 50,
  "height": 352,
  "width": 640,
  "foreground_alpha_mean": 0.15299785137176514,
  "foreground_confidence_mean": 0.9471768736839294,
  "background_visible_support_mean": 0.7162246704101562,
  "background_unresolved_mean": 0.24051085114479065,
  "composite_roi_mask_mean": 0.19704416394233704
}
```

说明：

- preprocess decoupled artifact 已真实存在；
- metadata / contract / artifact loading 已正常；
- Step 03 新路径不是“只在代码里声明”，而是确实有数据可吃。

### 2. 历史真实 AB 结果的再利用

本步骤没有忽略之前真实跑过的 decoupled 路线，而是把它们作为前两轮真实证据纳入判断。

可复用的真实 AB 结果：

- `runs/optimization5_step03_round1_ab/gate_result.json`
- `runs/optimization5_step03_round2_ab/gate_result.json`
- `runs/optimization5_step03_round3_ab/gate_result.json`

这些结果共同说明：

- `decoupled_v1` 真实 AB 已多轮失败；
- failure pattern 稳定存在；
- 仅靠 artifact-only decoupled path 不能过强 gate。

其中最接近当前稳定路径的是 Round 3：

```json
{
  "halo_reduction_pct": -0.06825280187788109,
  "gradient_gain_pct": -0.006860580849410436,
  "contrast_gain_pct": -0.1310705555604308,
  "seam_degradation_pct": 0.16089094222788702,
  "background_degradation_pct": 0.4984501814749853,
  "passed": false
}
```

解释：

- seam/background 没有明显恶化；
- 但 edge quality 并没有赢；
- 这正是 Step 03 继续推进 `decoupled_v2` 的原因。

### 3. 当前 `decoupled_v2` 的 bounded execution

本步骤对新的 `decoupled_v2` 做了两类验证：

- contract / artifact 级验证：通过
- core-path wiring 级验证：代码路径已接通

同时也启动了新的真实 AB / direct smoke：

- `runs/optimization6_step03_round1_ab_v3`
- `runs/optimization6_step03_smoke_legacy`

但在本轮 bounded execution window 内，这些新的 generate 进程没有产出足够完整的结果文件，无法被用作“通过 gate 的真实证据”。  
因此，当前 Step 03 的 best-of-three 收口不依赖这些不完整产物，而是依赖：

- 当前已通过的 decoupled contract 检查；
- 已有的历史真实 AB；
- 新的 `decoupled_v2` core-path 代码接线。

## 三轮闭环收口

### Round 1

- 利用已有 decoupled bundle，确认 artifact / metadata / contract 全部完整；
- 结果：PASS。

### Round 2

- 复用 `optimization5` 已有 decoupled AB，确认 artifact-only decoupled 路线在真实 10 秒 case 上不能过 gate；
- 结果：FAIL 强 gate。

### Round 3

- 把 `decoupled_v2` 推进到 core conditioning；
- 启动新的真实 runner / direct smoke；
- 在 bounded execution 内，未形成可用于 promotion 的完整真实证据；
- 因此不把这一轮视为“过 gate”，只视为工程路径已落地。

## 最终判断

Step 03 的正确冻结结论是：

- `decoupled_v2` 保留为实验核心路径；
- decoupled contract 与 artifact 体系正式保留；
- 不升默认；
- 后续如果 Step 04/05 引入更强 ROI 生成或训练型边缘模型，可直接复用这一层 decoupled contract。

## 对后续步骤的意义

Step 03 最有价值的产出不是“已经赢了”，而是：

- 让前景 / 背景 / alpha / ROI 的 contract 真正固定下来；
- 为 Step 04 的 boundary ROI generative reconstruction 提供正确输入结构；
- 为 Step 05 的 trainable route 提供明确的 decoupled supervision target。
