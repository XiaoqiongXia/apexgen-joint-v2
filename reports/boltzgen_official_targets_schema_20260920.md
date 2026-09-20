# BoltzGen 官方 targets.zip 实际格式检查（2026-09-20）

## 数据来源与检查状态

来源：[boltzgen/boltzgen1_train](https://huggingface.co/datasets/boltzgen/boltzgen1_train/tree/main)，
下载文件为 `targets.zip`。本报告检查的是该压缩包中的真实数据，不能仅用当前上游 Python
类定义代替发布数据的实际字段。

**全量下载、SHA256 校验及解压已完成。** 整包 SHA256 与官方 LFS 记录完全一致；
461,134 个文件全部解压并通过外层 ZIP 成员 CRC 校验。
230,566 个 NPZ 的全部数组头已检查，只有一种字段/dtype/维数布局，无 object/pickle dtype。
全量格式普查不等于所有数组值及几何质量均已验证；详细值检查范围仍为下文的 12 例。

官方文件大小为 75,009,352,113 字节；LFS SHA256 为
`b632b09f180216d6bc2769bad93e81c68561dbb6ddfbacd269ae57722809da16`。
ZIP 目录含 230,566 个结构 NPZ、230,566 个配套记录 JSON，以及 `config.yaml`、
`manifest.json`。解压后文件总字节数为 78,428,895,734。

项目内目的目录：`artifacts/datasets/boltzgen1_train_official_20260920/`。
压缩包内部顶层是 `targets/`，解压目的目录设为 `training_data/` 后得到：

```text
training_data/targets/
├── config.yaml
├── manifest.json
├── records/<structure_id>.json
└── structures/<structure_id>.npz
```

## 真实 NPZ 数组

已直接读取并检查 12 个官方 NPZ 的全部数组值：
`5pc9, 7ysh, 3far, 2j3w, 3a2n, 4lgt, 1uo9, 8k94, 4ep3, 5hga, 7yxn, 5k9s`。
全量 230,566 个 NPZ 的字段普查确认：全部具有下面的字段和 dtype。
不同文件的原子、残基、链、连接及坐标模型数量可以不同。

| 数组 | 实际字段 | 含义及读取注意事项 |
|---|---|---|
| `atoms` | `name, coords, is_present, bfactor, plddt` | `name` 是 `<U4` 字符串；`coords` 为 float32 三维坐标；没有 `element` 列。缺失原子的坐标不能用于接触判定，必须读取 `is_present`。 |
| `bonds` | `chain_1, chain_2, res_1, res_2, atom_1, atom_2, type` | 显式连接端点；样例中的 chain/res/atom 分别指对应全局表的行号。不能把该表当成已经列出每条蛋白链全部隐式主链肽键。 |
| `residues` | `name, res_type, res_idx, atom_idx, atom_num, atom_center, atom_disto, is_standard, is_present` | `res_idx` 是链内位置；`atom_idx` 是全局原子表起始偏移。残基名称和 token 应交叉核对。 |
| `chains` | `name, mol_type, entity_id, sym_id, asym_id, atom_idx, atom_num, res_idx, res_num` | 此处 `res_idx` 是全局残基表起始偏移，与 `residues.res_idx` 的含义不同。链表行号、`asym_id`、链名须分别保留。 |
| `interfaces` | `chain_1, chain_2` | 接触链对列表，不是逐残基接触矩阵，也不指定 context/target 方向。局部接触残基需要从坐标重算。 |
| `mask` | 每条链一个 bool | 源处理流程的链有效性标记，不等于本模型全部质量检查已通过。 |
| `coords` | `coords` | 坐标集合，float32 三维坐标。 |
| `ensemble` | `atom_coord_idx, atom_num` | 各坐标模型在 `coords` 中的区间；不能无条件假定所有文件只有一个模型。 |

全量文件没有独立的 `connections` 数组，也没有 `chains.cyclic_period` 字段。
因此不能将当前上游代码新增的字段当成此次发布文件必然包含的字段，字段缺失也不能
单独证明一个目标在化学上是线性的。

## 真实值、编号和四例交叉检查

- 12 个样例的链→残基→原子区间、原子名唯一性、观察到的坐标有限性和坐标模型区间均通过。
- 检查了 9,146 个标准蛋白残基：20 种标准 AA 的 Boltz token 对应 2..21。
  这不能替代向本模型 AA 编号的显式映射。
- 312 条显式 bond 的两端均满足 chain/residue/atom 归属检查；这些样例未出现此前旧数据中
  见到的 chain/atom 字段布局互换现象。该结论不扩展为全量连接值已检查。
- 四例 `5hga, 5k9s, 4ep3, 7yxn` 与先前旧 Boltz 文件相比，原子数量、解码后的原子名、
  presence mask、观察到的坐标、共同残基字段和共同链字段全部一致。
- 四例配套 JSON 中的链编号、链名、长度和 valid 标记与 NPZ 对上。

四例按有效蛋白重原子距离 ≤5 Å 重新计算的界面如下。以下 A/B 仅是链对展示顺序。

| 结构 | 链 A / 链 B | 全链残基数 A / B | 接触残基数 A / B |
|---|---|---:|---:|
| 5hga | A1 / B1 | 275 / 100 | 38 / 30 |
| 5hga | A1 / C1 | 275 / 8 | 38 / 8 |
| 5k9s | A1 / B1 | 458 / 11 | 30 / 10 |
| 5k9s | A1 / C1 | 458 / 11 | 20 / 8 |
| 4ep3 | A1 / B1 | 203 / 9 | 36 / 9 |
| 7yxn | A1 / C1 | 250 / 11 | 16 / 10 |

原子和 AA 都应按名称映射到模型固定槽位，并交叉核对源 token；不能直接把源原子行号
当成 atom14/atom38 槽位。生成片段仍应按用户规则：链对双向展开，每个长度至少 4 的
最大连续接触段分别生成样本，绝不将分隔的片段拼成一个 target；另检查观察到的主链连续性。

## 配套记录与数据数量的含义

四例 `records/*.json` 包含实验方法、分辨率和日期等结构信息，以及链级
`chain_id, chain_name, num_residues, mol_type, cluster_id, msa_id, template_ids, valid`
和 interface 记录。`msa_id` 是引用，实际 MSA 内容不在此次 targets.zip 中；本次没有下载
独立的 msa.zip。聚类信息可以参与之后的分组划分，但不能仅凭存在 cluster_id 就认定无泄漏。

`config.yaml` 记录来源为 RCSB，涉及长度、未知残基、连续 CA 距离及链碰撞等源级筛选。
这些源级筛选不能代替本模型对目标连续片段、AA/atom mapping 和几何质量的检查。

全量记录的实验方法统计：X 射线衍射 191,006 个，电子显微镜 24,541 个，溶液 NMR
14,206 个，其余 813 个为其他方法或混合方法。完整的 29 种方法标签及计数见
`inspection/record_metadata_census.json`。
其中 7 个历史条目的方法为 `solution nmr,theoretical model`：
`1oln, 2bvk, 1e08, 1vyc, 1ur6, 1dwl, 1gx7`。因此该集合是 PDB 来源、以实验结构为主，
但不能把每条记录或每个坐标都说成纯实验测量；这 7 个条目可按后续训练政策单独筛选。

额外核对了 [RCSB 5HGA 官方条目](https://www.rcsb.org/structure/5HGA)：其 X 射线实验方法、
2.20 Å 分辨率、275/100/8 三种蛋白链长度与下载记录相符。
实验复合物中的接触可以用于构造界面生成监督，但从较长蛋白中截出的局部片段不因此
自动具有独立结合实验的验证。

**230,566 是结构文件数，不是可训练界面或片段数。** 样例中就有不含蛋白–蛋白界面的结构。
训练样本数需要在全量数据上筛选蛋白链对、计算接触、双向分段并执行本模型质控后统计。

另外已完整读取 230,566 个配套记录 JSON：文件名 ID、内部 ID 与 NPZ 文件清单一一对应；
各记录中的链数、interface 数及引用的链 ID 均通过一致性检查。
JSON 顶层字段和链记录字段各只有一种布局。元数据统计如下：

| 指标 | 数量 |
|---|---:|
| 全部分子类型的链 | 3,130,376 |
| 蛋白链（`mol_type=0`） | 1,291,686 |
| 元数据列出的蛋白–蛋白链对（每结构内无向去重） | 3,128,129 |
| 其中两条链及 interface 的源 valid 均为 true | 3,088,613 |
| 至少含一个上述源有效蛋白链对的结构 | 128,463 |

这些链对尚未重新计算距离、提取连续片段或经过本模型质控，也没有跨结构序列去重。
因此 **3,088,613 是源元数据中的有效候选链对数，不是最终训练样本数**。
完整结果见数据目录的 `inspection/record_metadata_census.json`。

## 当前适配器的直接读取结果

用四个官方真实 NPZ 调用当前 `adapt_boltz_npz`，四个均被拒绝：

```text
ValueError: unsupported Boltz atoms schema;
requires {'name': 'iu', 'element': 'iu', 'coords': 'f', 'is_present': 'b'}
```

当前适配器支持的是旧版整数原子名及 element/connections 布局，不能直接读取本次官方格式。
需要按真实 schema 增加规范化读取，并处理 bonds 端点及名称映射后再验证模型输入。
本次仅下载、检查和记录证据，没有修改模型或生产转换代码。

详细检查产物已保存到数据目录的 `inspection/`：
`remote_sample_schemas.json`、`real_sample_value_audit.json`、`current_adapter_probe.json`、
`zip_members.csv`、`record_metadata_census.json`。检查脚本保存在
`artifacts/tmp/boltzgen_targets_inspection_20260920/`；全量完成状态以数据目录的 `download_verified.json` 和
`extraction_and_schema_census.json` 为准。
