# Boltz 双向多片段界面训练数据

输入是现有 `interfaces.parquet`。一行是一个无方向接触链对，A/B 按源链表行号排序，
不是受体/binder 标签。分别构造 A→B 和 B→A；`peptide` 是模型内的生成侧角色名，
不表示原始链必须名为 P，也不表示该链本身是一条天然短肽。

## 选择规则

1. 从清单定位原始 NPZ，并核对字节长度、SHA256、链表行、链名、链长和残基索引。
2. A→B 时，把 B 侧界面残基按源 `residues.res_idx` 分成连续段；相邻位置必须差 1。
3. 每个满足长度下限的连续段都保留，各自生成一个独立样本；顺序按原链位置排列。
   B→A 在 A 侧独立枚举。保留的是完整连续接触段，不额外枚举该段内部的重叠子窗口。
4. 每个样本只包含对应一段的实验序列和坐标。**不拼接多个段，也不补入间隔残基把多个段连起来。**
5. 检查该段所有残基都有 N/CA/C，序列位置连续，C–N 距离和连接角度合格。
   不合格则逐段记录拒绝原因，不删除残基或补坐标。同一方向其他合格片段照常保留。
6. 当前不设置长度上限。按用户指定的“大于 3”默认采用 **至少 4 个残基**：
   `--min-fragment-length 4`。短段记为 `skipped_short`，不通过扩展或拼接凑长度。
   该参数是包含端点的下限；模型允许的最低值为 3，可以显式改成 3，但默认不含 3。

例：`10,11,12,13,30,31,32,33,34,50,51,52` → 两个独立样本 `[10,14)` 和
`[30,35)`；最后三个残基组成的段低于默认下限，记为跳过。另一条链分别为这两个样本提供 context。
这里使用 NPZ 的零基聚合物位置；不是 PDB author 残基编号。

## Context 和监督

另一条链提供 context。对**已经选定的生成片段**重新计算另一条链的 ≤5 Å 重原子
接触核心，再保留核心及其 CB 距离 11 Å 内的环境（Gly 使用 CA）。不加入生成侧原链
的其余残基；附近未表示的源原子会记录到 `nearby_excluded_atom_rows`。

两侧坐标减去同一个条件核心 CA 质心，保留真实相对位置。生成侧序列、坐标只进入监督
和带噪训练状态，不进入静态 condition；条件侧保持固定。选点使用实验界面，因此推理
时需要指定目标位点。当前数据用于学习连续界面片段，不等价于已经验证其脱离原蛋白
后仍能独立折叠或结合。

沿用现有 AA 名称/token 双重校验、按原子名称映射 atom14、缺失侧链 mask，以及逐原子
回查原始 NPZ 的审计。原链位置保存在 `peptide_residue_keys`；batch 中生成侧局部位置
为 0..L−1，这只在已确认单段连续后使用。原生片段两端与原链相连的标准肽键允许裁断，
并记录显式边界键；不添加端基、不补 OXT、不改变原坐标。非标准共价交联仍拒绝。

## 构建与读取

对于已经解压的官方 BoltzGen 数据，可以直接从指定的 NPZ 建立清单并转换。
这个入口强制提供结构 ID，不会默认遍历全量；输出目录必须是新目录：

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/data/prepare_joint_v2_boltzgen.py \
  --structures-dir artifacts/datasets/boltzgen1_train_official_20260920/training_data/targets/structures \
  --structure-ids 5hga 5k9s 4ep3 7yxn \
  --output artifacts/datasets/boltzgen_native_four_20260920 \
  --model-smoke
```

外层输出有 `interfaces.parquet`、`interfaces.csv`、`source_audits.json` 和 `summary.json`；
另外自动生成 `sample_inventory.csv` 和 `sample_inventory.md`，每行对应一个被接受的模型样本。
清单列出源结构 ID、源文件及 SHA256、样本 ID、方向、条件/目标链 ID、原链长度、
实际口袋及目标片段长度、目标序列和源链索引范围。口袋的不连续索引逐段列出，
不会用最小/最大位置假装成连续区间。零基范围采用 `[start, stop)`；CSV 同时提供
一基闭区间。两者都是 NPZ 源链位置，不是 PDB 作者残基编号。
CSV 还保留 `record_index`、`shard_id`、`tensor_key` 和 `dataset_source_npz`，可直接定位
实际模型记录。新构建在每条 LMDB 事务提交成功后立即追加 CSV 并 flush，不再为生成
清单重新读取 LMDB，也不收集全部样本清单行。写入前校验界面清单、训练 manifest 与记录的
样本身份、长度、链和序列。入口的 interfaces CSV/Parquet 按源结构逐批写入，随后执行
样本转换；这仍是两阶段流程。去重集合、当前结构和当前批次仍需要内存，未宣称全流程恒定内存。
可输入模型的数据在其 **`dataset/` 子目录**。独立 NPZ 的清单中，`source_archive`
就是该 NPZ 文件路径，`source_offset=0`；不要把 ZIP 的压缩成员 header offset 当成原始 NPZ 偏移。

构建期间可查看 `dataset.inprogress/sample_inventory.csv`、`directions.jsonl` 和
`build_status.json`。每个 LMDB 样本对应一条 CSV 行；默认每 256 个样本换一个 shard。
Parquet manifest 仍按扫描批次写入。成功完成后将 `dataset.inprogress/` 重命名为
`dataset/`，入口再复制 CSV 到外层，并从 CSV 流式生成 Markdown，不读取原子张量。

Python 异常或 Ctrl-C 会保留 `dataset.inprogress/` 和失败/中断状态，不自动删除已写结果。
同名中间目录存在时拒绝覆盖。LMDB 和 CSV 不属于同一事务，进程可能在两次写入之间中断；
此时 CSV 可能落后于 LMDB，Parquet manifest 也可能只包含之前的完整批次。中间结果不能
直接当作完整训练集，也不能简单改名；需检查/恢复，目前未实现自动断点续跑。
CSV 的 flush 用于及时写出用户态缓冲，并非每行 fsync 的断电持久性保证。

原有已索引 tar/Parquet 路径仍可使用：

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/data/prepare_joint_v2_boltz_chain_pairs.py \
  --inventory artifacts/datasets/boltz_interface_inventory_20260920/interfaces.parquet \
  --output artifacts/datasets/boltz_all_interface_fragments_four_20260920 \
  --structure-ids 5hga 5k9s 4ep3 7yxn --min-fragment-length 4 --model-smoke
```

输出目录必须不存在。源 NPZ 按内容哈希保存，每个文件只复制一次。`--structure-ids`
限定输入范围；省略时处理整份清单，两个方向分别质检。默认输出 split 是 `smoke`。
可用 `--split-map splits.json` 显式传入 `{结构ID: "train"/"validation"/...}`；选中的
每个结构都必须有分组。同一结构的所有链对、两个方向及所有片段共用一个 split，避免样本间泄漏。
这个工具不自动做同源序列聚类或去重；正式划分需另外准备。

- `manifest.parquet`：被接受的片段样本，包含源链、方向、`fragment_index`、生成片段起止位置。
  `pair_id` 仍标识原始无方向链对；`sample_id` = `pair_id:direction:fragment起点-终点`，
  因而同一 pair/direction 的多个片段不会互相覆盖。
- `shards/*.lmdb`：可被现有 `JointV2Dataset` 直接读取的 record，默认每 256 条一个 shard。
- `directions.jsonl`：每个方向下**每个连续段**的结果，包括短段、覆盖比例和拒绝原因。
- 若源 NPZ 列有链对，但重算后的双方接触残基均为空，该方向记为 `skipped_no_contact`，
  不调用连续分段，也不阻断其他有效链对；`build.json` 同名计数表示这类方向的数量。
- `mapping.json`：AA 与原子槽位映射契约。
- `build.json` / `metadata.json`：计数、参数、输入和输出身份及代码哈希。
- `model_smoke.json`：可选 CPU 验证汇总，每 8 条样本一批；详细前向/反向、条件独立性、
  条件结构固定和映射复核见 `model_smoke_batch_*.json`。

```python
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.data.batch import collate_joint_v2_records

dataset = JointV2Dataset(
    "artifacts/datasets/boltzgen_native_four_20260920/dataset", split="smoke"
)
try:
    batch = collate_joint_v2_records([dataset[0], dataset[1]])
    batch.condition.validate_model_input()
finally:
    dataset.close()
```

单样本覆盖率 = 该连续段的界面残基数 / 该方向生成侧原始全部界面残基数。一个方向可有
多个独立样本，各自 context 按自己的目标段重新计算；它们的覆盖率可相加统计方向总覆盖率。
条件侧可能包含多个非连续结构片段；
它们保留原链位置和断点 mask，不施加虚假的肽键连接。

四例数据和 smoke 均为开发验证，不是正式训练集划分；不启动优化器或 GPU 训练。
当前转换器使用 v2 映射契约，旧 v1 Boltz 转换产物需重新构建；Dataset 和 collate
都会拒绝不一致的映射，不能手动改哈希绕过。
