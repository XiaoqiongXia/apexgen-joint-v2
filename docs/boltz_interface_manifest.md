# Boltz 蛋白链对界面元信息清单

一行对应一个 NPZ 结构中的一个**无方向蛋白链对**。A/B 是清单的两侧，按源链表
行号排序（`chain_a_row < chain_b_row`），不是必须名为 A、B 的原始链。
训练时可以把同一行解释成 A→B 或 B→A，不必存两份几乎重复的元信息。

本清单不限制链长，不截取待生成链，不要求标准 AA，也不应用训练适配器的几何筛选。
源 `mask=False` 的链也保留在清单中，便于按不同任务自行筛选。

## 本次生成结果（2026-09-20）

输出目录：`artifacts/datasets/boltz_interface_inventory_20260920/`。
读取本地索引覆盖的 44,990 个完整 NPZ，处理失败 0 个；其中 25,057 个结构
产生 625,317 个不同的蛋白链对，全部通过 ≤5 Å 的实际重原子接触复核。
618,568 对的两侧源链 mask 均为真，其余 6,749 对保留源标志供后续筛选。
这些是界面候选数量，不是完成训练质量筛选后的样本数量。

13 项自动测试通过；四个真实结构的六对链用独立全距离矩阵复核，结果一致。
另对全部记录验证了链对唯一性、界面索引列表长度、残基计数上下界、接触距离，
以及 CSV 与 Parquet 的关键标量字段一致性，结果见 `verification.json`。

## 文件

- `interfaces.csv`：全部链对的标量元信息，便于浏览、筛选。
- `interfaces.parquet`：同样的记录，另外保留两侧界面残基索引列表。
- `preview.csv` / `preview.json`：前 20 条记录，方便快速打开。
- `structures.jsonl`：每个输入结构的处理状态及链对数量，包括没有蛋白界面的结构。
- `errors.jsonl`：无法可靠读取或计算的源结构及原因；不会悄悄丢弃。
- `metadata.json`：接触定义、统计、源归档和清单哈希、输出文件哈希、代码哈希。
- `examples.csv`：四个已复核真实结构的六对链，便于快速查看具体例子。
- `four_structure_audit.json`：四个真实结构的独立距离矩阵复核结果。
- `verification.json`：全量清单一致性与完整性检查结果。

## 字段

| 字段 | 含义 |
| --- | --- |
| `pair_id` | 结构内链对标识，例如 `5hga:chain0-chain2`；在这一份源归档中唯一 |
| `source_archive` | 原始 tar 的绝对路径 |
| `source_member` | NPZ 在 tar 中的完整成员路径 |
| `source_npz_filename` | 原始 NPZ 文件名，例如 `5hga.npz` |
| `source_structure_id` | 从原始文件名取得的结构 ID；本批 RCSB 数据对应 PDB ID |
| `source_offset` / `source_size` | NPZ 数据起点的字节偏移与长度，可直接从 tar 读取 |
| `source_npz_sha256` | 该 NPZ 完整字节内容的哈希 |
| `coordinate_source` | 固定为 `atoms.coords` |
| `ensemble_model_count` | 原文件坐标模型数；界面统计使用 `atoms.coords`，不对多个模型取平均 |
| `contact_cutoff_angstrom` | 本次计算使用的重原子接触阈值，默认 5 Å |
| `both_source_masks_true` | 两条链是否都通过源 Boltz 的链 mask |
| `contact_verified` | 按本次阈值与实际坐标重算后是否仍有接触 |
| `minimum_contact_distance_angstrom` | 已确认接触的最近重原子距离；无接触时为空 |

以下字段对 A、B 两侧分别存在，前缀为 `chain_a_` / `chain_b_`：

| 后缀 | 含义 |
| --- | --- |
| `row` | 源 `chains` 表的行号，从 0 开始 |
| `id` | NPZ 中保存的完整链名，如 `A1`、`B2` |
| `asym_id` / `entity_id` / `sym_id` | 源链实例、分子实体与对称副本编号，保留不同编号空间 |
| `source_mask` | 原始 Boltz 链有效标志 |
| `length` | 整条链的残基条目数量，包含未观察到的残基 |
| `residue_table_start` | 该链在全局 `residues` 表中的起始行号 |
| `source_present_residue_count` | 源 `residues.is_present=True` 的数量 |
| `observed_residue_count` | 至少有一个已观察重原子的残基数量，不代表主链完整 |
| `observed_heavy_atom_count` | 已观察重原子数量 |
| `nonstandard_residue_count` | 源 `residues.is_standard=False` 的数量；不是完整的模型 AA 兼容性检查 |
| `interface_residue_count` | 参与这一对链之间接触的不同残基数量 |
| `interface_residue_indices` | 界面残基的源 `residues.res_idx` 值，通常是从 0 开始的链内聚合物位置 |
| `interface_residue_rows` | 同一批界面残基在原始 `residues` 表中的全局行号 |

最后两个列表字段保存在 Parquet / JSON 中。源 NPZ 没有可靠保留原始 author
残基编号或原始 author 链映射，因此清单不伪造这些信息，也不会剥掉链名中的副本后缀。

## 界面残基定义

对 NPZ `interfaces` 中列出的蛋白–蛋白链对，重新计算：一个残基至少有一个
`is_present=True` 且 `element>1` 的原子，与对方链的此类原子距离 **≤5 Å**，
则把该残基计入本侧界面。使用所有观察到的重原子，不只使用 CA 或 CB。

同一残基有多个原子接触时，只计一次。因此 A 侧和 B 侧的界面残基数通常不同。
这里统计的不是接触原子对数、残基对数，也不是埋藏表面积（BSA）。

缺失原子的占位坐标和氢原子不参与计算；非有限的已观察重原子坐标触发显式错误。
源 interface 中重复或反向重复的同一链对合并为一行。当前默认沿用源 Boltz 的
5 Å 阈值；若自定义更大阈值，仍只重算源 interface 已列出的链对，不重新发现新链对。

源 `interfaces.chain_1/chain_2` 按源解析器的实现是链表行号。它们不能通用地视为
entity ID、author chain ID，或假设总与 `asym_id` 相同。

## 读取

```python
import pyarrow.parquet as pq

table = pq.read_table(
    "artifacts/datasets/boltz_interface_inventory_20260920/interfaces.parquet",
    columns=[
        "source_structure_id", "chain_a_id", "chain_b_id",
        "chain_a_length", "chain_b_length",
        "chain_a_interface_residue_count", "chain_b_interface_residue_count",
        "chain_a_interface_residue_indices", "chain_b_interface_residue_indices",
    ],
    filters=[("source_structure_id", "=", "5hga")],
)
print(table.to_pylist())
```

首次建库命令（输出目录须不存在）：

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/data/build_boltz_interface_manifest.py \
  --archive /data2/xiaoqiong/python_project/BasinDiff/dataset/Structure/rcsb_processed_targets.tar \
  --members artifacts/tmp/boltz_inventory_20260920/rcsb_processed_targets_members.csv \
  --output artifacts/datasets/boltz_interface_inventory_20260920 \
  --workers 8 --cutoff 5.0
```

这是一份训练样本构造前的元信息清单。几何接触不直接代表生物学有效结合，也不
代表完整通过模型的化学/几何适配；后续可依任务从清单筛选。当前只处理本地完整
成员索引覆盖的 RCSB 结构。OpenFold 单链集先前盘点没有蛋白链间界面。
