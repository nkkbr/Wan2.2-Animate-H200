# Step 10: Guidance 解耦与参数搜索

## 1. 目标

把当前 animate replacement 中耦合在一起的 guidance 逻辑拆开，并建立一套系统的参数搜索方法。

这一步的目标是：

- 让 face expression control 可以单独调
- 让 text influence 不再意外牵连 face control
- 为 solver / steps / shift / overlap 提供统一实验框架


## 2. 为什么现在做

当前 `guide_scale` 的实际行为不是普通意义上的“整体 CFG 强度”，而更接近：

- 文本 negative prompt
- face condition 置空

这两者的混合权重。

问题在于 replacement 场景中：

- 文本不是主要控制源
- face motion 才更值得精细调节

如果不先解耦，就很难回答：

- 表情不跟，是因为 face 太弱，还是 text 太弱？
- 提高 `guide_scale` 带来的副作用究竟来自哪一支？


## 3. 本步骤的范围

本步骤包括：

- 将 face guidance 与 text guidance 解耦
- 明确 solver / steps / shift / guide 的实验矩阵
- 沉淀 H200 高质量参数 preset

本步骤不做：

- latent memory
- 新模型训练


## 4. 当前问题

### 4.1 guidance 语义不清

当前一个参数同时影响：

- face condition 的 conditional / unconditional 差值
- text context 的正负条件

### 4.2 参数搜索缺少结构

当前参数很容易被“凭感觉”调：

- `sample_steps`
- `sample_shift`
- `sample_solver`
- `sample_guide_scale`

如果没有统一实验矩阵，很难收敛。


## 5. 设计原则

### 5.1 replacement 场景优先服务 face control

prompt 在 replacement 中是弱条件。  
因此解耦后，face guidance 应比 text guidance 更重要。

### 5.2 先做最小改动的解耦

第一版不必引入复杂多分支 CFG，只要先做到：

- face 可单独关/开
- text 可单独关/开

### 5.3 参数搜索必须依托 Step 01 的评测体系

否则会重新回到主观调参状态。


## 6. 具体实施方案

### 6.1 拆分 guidance 参数

建议新增：

- `--face_guide_scale`
- `--text_guide_scale`

保留旧 `sample_guide_scale` 作为兼容入口，但高质量模式应逐步迁移到新参数。

### 6.2 设计三种 unconditional 变体

建议至少支持三种实验模式：

1. 只 null face，保留 text
2. 只改 text，保留 face
3. 同时对两者做 unconditional

这有助于判断当前素材最敏感的控制来源。

### 6.3 建立参数搜索顺序

建议固定下面的搜索顺序：

1. `sample_solver`
2. `sample_steps`
3. `sample_shift`
4. `face_guide_scale`
5. `text_guide_scale`
6. `refert_num`

不要一开始同时搜索所有维度。

### 6.4 建立 H200 高质量参数基线

建议先固定一组 H200 baseline，例如：

- `sample_solver=dpm++`
- `sample_steps=60`
- `sample_shift=5.0`
- `face_guide_scale=1.0`
- `text_guide_scale=1.0`
- `refert_num=5`

然后围绕单个轴做局部搜索。


## 7. 需要改动的文件

建议涉及：

- `generate.py`
- `wan/animate.py`

必要时新增：

- 参数搜索脚本
- 实验结果汇总脚本


## 8. 如何实施

建议顺序如下：

1. 先拆分 face / text guidance 参数。
2. 再增加三种 unconditional 组合模式。
3. 固定 baseline。
4. 基于 benchmark 做参数网格。
5. 汇总并沉淀高质量 preset。


## 9. 如何检验

### 9.1 face 控制检验

重点看：

- 嘴型跟随
- 表情稳定性
- 侧脸或遮挡下的表现

### 9.2 text 影响检验

重点看：

- prompt 是否对角色身份和动作造成不必要扰动
- 提高 text scale 是否真的带来正收益

### 9.3 参数矩阵检验

对 benchmark 统计：

- 身份一致性
- 表情跟随度
- continuity
- 背景稳定性
- 运行时间


## 10. 验收标准

只有满足下面条件，才算本步骤完成：

- face guidance 与 text guidance 可独立调节
- benchmark 上能明确看出各参数轴的收益方向
- 沉淀出一组可靠的 H200 baseline


## 11. 风险与回退

### 风险

- 新参数过多，反而让使用更复杂
- 指标波动较大，难以下结论

### 回退策略

- 对普通模式保留旧接口
- 高质量模式下才使用解耦参数
- 搜索时坚持一次只动一类变量


## 12. 本步骤完成后的收益

这一步完成后，generate 端的调参会从“经验主义”进入“有结构的实验”。  
对长期维护来说，这比单次找到一个好参数更有价值。
