# Step 11: Latent Temporal Memory 探索

## 1. 目标

探索替代当前 `decode -> re-encode` temporal handoff 的更稳定机制，降低长视频生成中的时序损失与误差累计。

这一步是明确的高风险探索项，不应在基础步骤未完成前提前推进。


## 2. 为什么这是高风险但值得探索的方向

当前 clip 之间的 handoff 方式是：

- 上一段生成出像素帧
- 取最后若干帧
- 下一段再把这些像素帧重新 encode 成 VAE latent

这会带来：

- 细节损失
- 表观抖动
- 颜色漂移
- 长链条中的误差累积

理论上，更稳定的 memory 方案可能明显改善长视频 quality。  
但它也更容易碰到预训练分布和 VAE 语义的一致性问题，因此风险很高。


## 3. 本步骤的范围

本步骤聚焦探索，不承诺直接进入生产默认路径。

探索方向包括：

- latent-to-latent handoff
- hybrid handoff
- 显式 temporal memory cache

本步骤不做：

- 直接修改主干模型结构
- 引入需要重新训练的新条件分支


## 4. 当前问题

### 4.1 handoff 发生在像素空间

这让上一段的结果在传递前多经历一次：

- decode
- encode

### 4.2 VAE 并不保证严格无损

所以即使上一段结果视觉上不错，下一段拿到的 temporal guidance 也已经不是原始 latent。

### 4.3 误差会沿 clip 链持续传播

当前长视频会形成：

- 第 N 段轻微偏移
- 第 N+1 段继续继承并放大


## 5. 设计原则

### 5.1 先做原型验证，不直接替换主路径

这一项必须保留 baseline 对照，不应直接覆盖现有逻辑。

### 5.2 尽量不改变预训练模型输入维度

优先尝试：

- 只改变 handoff 组织方式

而不是：

- 新增输入通道
- 改主干结构

### 5.3 将探索拆成渐进版本

建议分为：

- Prototype A: 更好的像素 handoff
- Prototype B: latent handoff
- Prototype C: hybrid handoff


## 6. 具体实施方案

### 6.1 Prototype A: 改进像素 handoff

在不改变整体语义的前提下，先尝试：

- 更长 overlap
- 更稳的 seam blending
- 更高保真的 handoff 帧

虽然这仍是像素 handoff，但可作为 latent 路线的强 baseline。

### 6.2 Prototype B: latent handoff

思路是：

- 直接保留上一段末尾的 latent 表示
- 在下一段构造 `y_reft` 时复用 latent，而不是重新 decode/encode

但要重点验证：

- 时间槽位对齐是否正确
- VAE causal 结构是否允许简单拼接
- 结果是否真的比像素 handoff 更稳

### 6.3 Prototype C: hybrid handoff

思路是：

- 背景相关仍走像素或 bg 条件
- 人物 temporal memory 走 latent

这是更复杂但可能更稳的路线。


## 7. 需要改动的文件

主要涉及：

- `wan/animate.py`

必要时辅助：

- debug 工具
- clip 级中间 latent 导出


## 8. 如何实施

建议顺序如下：

1. 先做 Prototype A 的强 baseline。
2. 再做最小化的 latent handoff 原型。
3. 与 baseline 严格对照。
4. 只有当 latent handoff 明显更稳时，才继续做 hybrid。

不要直接一上来就做复杂的 memory 系统。


## 9. 如何检验

### 9.1 continuity 检验

重点看：

- clip 边界接缝
- 长视频后半段稳定性
- 颜色和纹理漂移

### 9.2 误差累计检验

在长 benchmark 上比较：

- 第 10 个 clip
- 第 20 个 clip
- 第 30 个 clip

看误差是否更可控。

### 9.3 数值与显存检验

记录：

- 峰值显存
- 单 clip 时间
- handoff 相关额外内存占用


## 10. 验收标准

只有满足下面条件，才建议继续推进：

- 相比现有 handoff，continuity 明显更稳
- 不引入严重的新伪影
- 显存与复杂度仍在 H200 可接受范围内

如果做不到，应保留为研究分支，不进入主路径。


## 11. 风险与回退

### 风险

- latent 语义与原 handoff 方式不兼容
- 时间对齐出错，造成更难发现的伪影
- 工程复杂度快速上升

### 回退策略

- 始终保留像素 handoff baseline
- 所有 prototype 独立开关控制
- 只在 benchmark 上通过后再考虑推广


## 12. 本步骤完成后的收益

如果成功，这一步有机会显著改善长视频 replacement 的上限。  
但在工程优先级上，它仍应放在 continuity、soft mask、背景质量这些低风险高收益项之后。
