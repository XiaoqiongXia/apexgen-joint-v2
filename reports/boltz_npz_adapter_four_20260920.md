# Boltz NPZ → Joint-v2：四例适配与审计

日期：2026-09-20。已实现从本地 Boltz 已处理结构 NPZ 到 Joint-v2 complex
record、LMDB、batch 的完整路径，并在当前 `SimplexCodesignModel` 上完成 CPU
前向与反向检查。没有执行优化器更新、正式训练或生成质量评估。

## 四个真实结构

数据目录：`artifacts/datasets/boltz_npz_four_20260920`。
链名是 **Boltz assembly 中的完整名称**，不是原始 author chain 的假定映射。

| PDB | 受体链 | 肽链 | 肽序列 | 口袋残基数 | 肽长度 | 逐原子核对数 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 5HGA | A1, B1 | C1 | RFPLTFGW | 168 | 8 | 1,465 |
| 5K9S | A1 | B1 | TNKKMRRNRFK | 154 | 11 | 1,288 |
| 4EP3 | A1 | B1 | KARVLAEAM | 134 | 9 | 1,081 |
| 7YXN | A1 | C1 | RPAILYALLSS | 79 | 11 | 718 |

总计核对 **574 个残基、4,552 个观察到的原子**。四例中的受体与肽合计覆盖
全部 **20 种标准氨基酸**。5HGA 的最终口袋确实同时保留 A1、B1 两条受体链。
4EP3 保留区域中有 28 个未观察到的标准原子槽，均保持 mask=false，未补坐标。

四例分别有 1、33、5、7 个所选受体链残基因缺少 N/CA/C 而被排除；这些计数
针对所选完整受体链，并非全部位于最终口袋内。原始聚合物索引仍被保留，避免
裁剪或删除后跨缺失区域计算主链扭转角。四例最终均没有肽周围 5 Å 内未纳入条件
的其他观察到的原子。

## 映射验证

- AA 映射按残基名称完成，同时交叉检查 Boltz `res_type`。例如 ALA 2→0、
  GLY 9→7、TRP 19→17。实现不依赖简单的减 2 算术。
- 原子名按 Boltz 的四整数编码解码，再放入模型的 receptor atom38 或
  residue-specific atom14 槽。没有沿用 Boltz 原子数组的排列顺序。
- 保存每个输出残基的全局 residue row、chain row、asym/entity/sym ID，
  以及每个观察到的输出原子的全局 atom row。
- 逐原子反查原始名称、所属残基、观察 mask、坐标；加回口袋平移原点后，
  最大逐坐标分量误差为 **9.5367431640625 × 10⁻⁷ Å**，来自 float32 存储。
- 不使用 `conformer` 作为结构标签。不恢复或伪造 NPZ 已丢失的 author numbering。
- 保存 `mapping.json`，包含全体 AA/atom 映射及模型与原子常数哈希。

映射哈希：
`e99e8c8ddc866e45ad29b3e44b3e68f2ae601e4a965aed10b3ae34de043e612f`。

可核对文件：`mapping.json`、`build.json`、`final_adapter_audit.json`、
`adapter_provenance.json`。最后一个文件记录适配代码及本地 Boltz 字段定义源码哈希。
最终实现重新适配这四例后，关键坐标、AA、mask、链连接数组与已保存记录逐项一致。

## Batch 与模型验证

从 `JointV2Dataset(..., split="smoke")` 重新读取四例，经原有 collator 得到
布局 **[4, 176]**：每个样本把自身口袋与肽串到统一节点轴，再进行 batch padding。

检查结果：

- native AA 和坐标放入独立 targets；静态 condition 中 peptide AA=20（UNKNOWN），
  不含真实肽原子。改变标签的 AA 和坐标后，全部静态 condition 张量保持相同。
- 源链身份、原聚合物索引与有效 C–N link mask 被保留；跨链和跨断点不连成肽键。
- `SimplexCodesignModel` 使用已有 tiny 配置，在 CPU 上完成混合长度四例前向、
  simplex 条件路径、loss 和 backward。所有 loss 与梯度均为有限值。
- 默认输出头为零初始化，首次 backward 有 4 个参数张量的梯度非零，编码器梯度
  为零，这是输出头初始化的结果。另做一次临时非零输出头探针，得到 **76 个**
  非零梯度参数张量，其中 **45 个属于编码器**，确认特征到 loss 的路径可反传。
- 受体平移与旋转保持固定；未执行 optimizer step，未写 checkpoint。

完整模型结果见 `model_smoke.json`。初始化 loss 不代表训练性能或 binder 设计能力。
native 接口用于确定口袋，因此这是给定靶点位点的训练数据；推理时仍需要位点条件。

## 测试

新增 `tests/joint_v2/test_boltz_npz.py` 的 **41 个测试**，涵盖所有标准 AA、
故意倒序的原子、缺失侧链、错误编号、错误元素、重复原子、非法 span、链 mask、
断链、多模型歧义、坐标来源冲突、交联、OXT、LMDB 往返与 padding。

与已有数据集/预处理回归测试一起运行：**123 passed**。新增三个 Python 文件的
Ruff 检查通过。

```bash
source scripts/project_tmp_env.sh
python -m pytest \
  tests/joint_v2/test_boltz_npz.py \
  tests/joint_v2/test_unified_complex_dataset.py \
  tests/joint_v2/test_dataset_view.py \
  tests/joint_v2/test_pdb_preprocessing.py \
  tests/joint_v2/test_preprocessing_stereochemistry_topology.py \
  -q --basetemp="$TMPDIR/pytest"
```

## 使用入口

- 适配 API：`src/apexgen/joint_v2/data/boltz_npz.py`。
- 建库与可选模型检查：`scripts/data/prepare_joint_v2_boltz_npz.py`。
- 字段说明、限制及 Python 使用示例：`docs/joint_v2_boltz_npz_adapter.md`。

使用已复制到数据目录的四个 NPZ 重建一个新的检查面板：

```bash
source scripts/project_tmp_env.sh
python scripts/data/prepare_joint_v2_boltz_npz.py \
  --index artifacts/datasets/boltz_npz_four_20260920/input_index.json \
  --output artifacts/datasets/boltz_npz_four_recheck \
  --model-smoke
```

输出目录必须不存在。当前适配范围为标准蛋白质的线性肽；非标准残基、肽主链缺失、
断链、无法表示的显式交联与多模型输入会拒绝处理。MSA 没有加入当前模型输入。
四例不是同源去重后的训练/验证划分。

## 本地原始归档完整性

来源为本地
`/data2/xiaoqiong/python_project/BasinDiff/dataset/Structure/rcsb_processed_targets.tar`，
文件大小 14,053,543,936 字节。扫描后部时遇到
`tarfile.ReadError: unexpected end of data`，所以**不能宣称整个训练归档已完整下载**。
本次四例均来自可完整读取的 NPZ，所有数组已逐项加载并完成独立身份/坐标审计；
其复制件保存在数据目录 `sources/`。没有重新下载或修改原始归档。

候选筛选中，4FCM、8FZ2、8ANB、1SE0 的所选肽有缺失主链而被拒绝；6YW4 因
包含当前不支持的残基 `48V` 被拒绝，没有用缩短肽链或强制修改残基名称来通过检查。
