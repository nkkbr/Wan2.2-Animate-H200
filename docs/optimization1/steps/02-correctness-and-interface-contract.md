# Step 02: Correctness 与接口契约修复

## 1. 目标

修复当前 preprocess / generate 之间最基础、最危险的契约问题，确保后续优化建立在一致、可验证的前提上。

这一步的核心不是“提质”，而是“去歧义”。


## 2. 为什么现在做

当前代码里至少存在下面几类契约问题：

- `refert_num` 的 argparse 默认值与实际断言约束不一致
- preprocess 与 generate 对 `src_mask` 的语义理解分散在不同位置
- 颜色空间可能存在隐式翻转风险
- preprocess 输出目录缺少正式 metadata，generate 端只能靠文件名和默认约定读取

如果这些问题不先修掉，后续所有优化都可能建立在错误假设上。


## 3. 本步骤的范围

本步骤聚焦以下内容：

- 统一 preprocess 和 generate 的 I/O 契约
- 修复明显错误的默认值和参数约束
- 明确颜色空间与 mask 语义
- 为后续步骤建立 metadata/manifest 对接格式

本步骤不做：

- lossless 存储
- mask 质量提升
- seam blending
- background inpainting


## 4. 问题列表

### 4.1 `refert_num` 默认值冲突

当前 CLI 默认值是 `77`，但 `wan/animate.py` 中强制要求：

- `refert_num == 1 or 5`

这会造成：

- 用户不传该参数时流程直接失败
- 文档、命令样例和代码默认不一致

### 4.2 `src_mask` 的语义没有被显式记录

当前事实是：

- preprocess 输出的 `src_mask` 表示人物区域
- generate 读取后会做 `1 - mask`
- 然后把它当成“背景保留区域”使用

但这个语义目前没有通过 metadata 对外明确暴露。

这很危险，因为：

- 后续如果换成 soft mask，很容易把语义弄反
- debug 时容易误以为 `src_mask` 直接就是生成控制 mask

### 4.3 颜色空间契约不够明确

replacement 链中同时使用了：

- `decord`
- `cv2`
- preprocess 内部图片路径
- generate 内部的 `cv2.imread(...)[..., ::-1]`

如果不显式定义“全链路统一使用 RGB 还是 BGR”，就容易出现局部反转而不自知。

### 4.4 preprocess 缺少正式 metadata

generate 端当前默认假设：

- 帧数一致
- 尺寸一致
- 文件格式固定
- `src_pose/src_face/src_bg/src_mask/src_ref` 都存在

这种方式对后续演进极不友好，尤其是：

- 一旦支持 `png_seq` 或 `npz`
- 一旦支持 soft mask
- 一旦支持 clean plate 的不同版本

generate 端必须有 metadata 才能稳妥读取。


## 5. 设计原则

### 5.1 显式优于隐式

所有关键语义都必须显式写入 metadata，而不是只靠代码默认逻辑。

### 5.2 兼容当前流程，但不要继续放大历史包袱

这一步可以先兼容现有 `mp4 + png` 结构，但新接口必须为后续步骤预留扩展空间。

### 5.3 所有校验尽量前置

如果输入不满足条件，应在 preprocess 或 generate 开始时尽早报错，而不是到采样阶段才以奇怪方式失败。


## 6. 具体实施方案

### 6.1 修复 `refert_num` 的接口定义

建议：

- 把 CLI 默认值改为 `5`
- 在 `generate.py` 和 `wan/animate.py` 中统一校验逻辑
- 在帮助信息中明确说明为什么推荐 `5`

如果后续要放开任意 overlap，再在 Step 09 中扩展，不要在这一步一次性做太多。

### 6.2 定义 preprocess metadata

建议 preprocess 输出新增一个 `metadata.json`，至少包含：

- `version`
- `frame_count`
- `fps`
- `height`
- `width`
- `channels`
- `mask_semantics`
- `color_space`
- `storage_format`
- `replace_flag`
- `src_files`

其中建议明确写：

- `mask_semantics = "person_foreground"`
- `color_space = "rgb"`

### 6.3 generate 端改为“读 metadata 再读文件”

建议 generate 端：

- 优先读取 `metadata.json`
- 根据 `storage_format` 选择 reader
- 根据 `mask_semantics` 决定后续转换逻辑
- 校验所有输入长度与尺寸

如果 metadata 不存在，可以保留兼容旧目录的 fallback，但应打印明确 warning。

### 6.4 统一颜色空间契约

建议正式规定：

- preprocess 内部的中间 `numpy` 数组统一使用 RGB
- 磁盘图像读取后立即转换到 RGB
- 进入 detector / pose / generate 前，不再做隐式通道切换

同时增加针对关键路径的单元测试：

- `decord -> preprocess ndarray`
- `cv2.imread -> reference image`
- `generate prepare_source`

### 6.5 明确 `src_mask` 的读取和使用语义

建议在代码和文档里统一写清：

- preprocess 输出的是“人物区域 mask”
- generate 内部会派生出“背景保留 mask”

并建议把这两个概念命名区分开，例如：

- `person_mask`
- `background_keep_mask`

这样后续 soft mask 改造时，逻辑会更清楚。


## 7. 需要改动的文件

建议涉及：

- `generate.py`
- `wan/animate.py`
- `wan/modules/animate/preprocess/preprocess_data.py`
- `wan/modules/animate/preprocess/process_pipepline.py`

必要时新增：

- metadata schema 文档
- 小型测试脚本


## 8. 如何实施

建议顺序如下：

1. 先定义 metadata schema。
2. preprocess 写 metadata。
3. generate 读取 metadata。
4. 修复 `refert_num` 默认值和校验。
5. 统一变量命名与日志输出。
6. 增加颜色空间和输入校验测试。

这个顺序不能反过来。  
如果先改 generate 的逻辑，但 metadata 还没有，接口仍然会继续漂。


## 9. 如何检验

### 9.1 契约一致性检验

对一组 preprocess 输出，确认 generate 端能够：

- 正确读取 metadata
- 正确识别尺寸和帧数
- 对缺失文件给出清晰错误信息

### 9.2 参数一致性检验

不传 `--refert_num` 直接运行 animate，确认：

- 不再因为默认值冲突而报错
- 默认行为和文档一致

### 9.3 颜色空间检验

使用带明显红蓝差异的测试图，确认：

- preprocess 输出颜色没有翻转
- generate 读回后颜色保持一致

### 9.4 mask 语义检验

在一个简单样本上检查：

- `src_mask` 代表人物区域
- generate 内部生成的 `background_keep_mask` 符合预期


## 10. 验收标准

只有满足下面条件，才算本步骤完成：

- `refert_num` 默认值与实际行为一致
- preprocess 输出有正式 metadata
- generate 能基于 metadata 做输入校验
- 颜色空间契约被明确并有测试
- `src_mask` 语义在代码、日志和文档中一致


## 11. 风险与回退

### 风险

- 为兼容旧目录结构写了过多分支，导致新逻辑又重新变复杂
- metadata 字段过少，后续仍然不足

### 回退策略

- 保留旧格式 fallback，但新格式必须是主路径
- metadata 第一版先小而全，不追求一次覆盖所有未来场景


## 12. 本步骤完成后的收益

这一步完成后，后续每个高阶优化都会更容易做：

- lossless I/O 可以通过 `storage_format` 扩展
- soft mask 可以通过 `mask_semantics` 和 metadata 平滑接入
- clean plate 与普通 bg 可以通过 metadata 区分
- generate 端不再依赖脆弱的隐式约定
