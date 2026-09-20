# 官方 BoltzGen NPZ 如何进入当前模型

## 预处理入口

`scripts/data/prepare_joint_v2_boltzgen.py` 的 `prepare_selected()` 是已解压官方 NPZ 的入口。
必须给出 `--structure-ids`；此次只读取 `5hga, 5k9s, 4ep3, 7yxn`。

```bash
source scripts/project_tmp_env.sh
python scripts/data/prepare_joint_v2_boltzgen.py \
  --structures-dir artifacts/datasets/boltzgen1_train_official_20260920/training_data/targets/structures \
  --structure-ids 5hga 5k9s 4ep3 7yxn \
  --output artifacts/datasets/my_new_boltzgen_panel \
  --model-smoke
```

输出路径需不存在。已经完成的四例位于 `artifacts/datasets/boltzgen_native_four_20260920/`。

## 数据经过的各层

| 步骤 | 实现 | 输入与输出 |
|---|---|---|
| 读取结构并识别接触 | `data/boltz_interfaces.py:describe_interfaces()` | 从 NPZ 的 chains/interfaces 找蛋白链对，按有效重原子距离 ≤5 Å 重算双方接触残基。输出 interfaces.parquet / CSV；原链对不带训练方向。 |
| 双向分段 | `data/boltz_chain_pairs.py:fragment_selections()` 与 `adapt_interface_direction()` | A→B、B→A 分别枚举目标链的最大连续接触段。每个长度 ≥4 的段是一个独立样本，不拼接。 |
| 映射与几何检查 | `data/boltz_npz.py:adapt_boltz_npz()` | 校验 AA 名/token、按名称映射原子、保留源行号；检查目标主链连续性和支持的显式连接。每个目标片段单独重算对侧 5 Å 核心及 11 Å CB context。 |
| 存储 | `scripts/data/prepare_joint_v2_boltz_chain_pairs.py:build_dataset()` | 写 dataset/manifest.parquet、shards/*.lmdb、mapping.json；记录每段的接受/拒绝结果、源 NPZ 副本与哈希。 |
| 读取与组成 batch | `JointV2Dataset` → `collate_joint_v2_records()` | 检查映射契约，读取 LMDB record，将 context 与 target 分开、补齐变长尺寸、生成各类 mask。 |
| 加噪、模型与损失 | `sampling/simplex_runtime.py:simplex_fm_losses()` | 采样时间和噪声，构造目标侧带噪状态；SimplexCodesignModel 接收带噪状态、时间及静态条件，再计算三项监督损失。 |

表中 `data/` 和 `sampling/` 均在 `src/apexgen/joint_v2/` 下。

record 的 `pocket_*` 保存 context 序列、atom38 坐标/mask、位置/旋转及残基身份。
`joint_v2_target` 保存 target 序列、atom14 实验坐标/mask、残基位置/旋转等监督。
`boltz_adapter` 和 `interface_pair` 保存原子行号、链角色、片段区间及映射版本。

`batch.condition` 中，context 的序列和坐标可见；target 的原生 AA 为 UNKNOWN，原生
原子坐标不可见。`batch.targets` 单独持有 target 监督。两侧使用同一个平移原点。
裁剪位点本身来自原生界面，因此这是给定位点的条件生成训练。

AA 例如 Boltz ALA=2 → 模型 ALA=0；原子按名称映射到残基对应的 atom14 或 context
atom38。源原子行号只用于追溯，不能直接充当模型原子槽位编号。

## 此次实际训练入口

`scripts/experiments/run_joint_v2_boltzgen_overfit.py` 先用 JointV2Dataset 读取 19 条记录，
再 collate 一次形成 full batch（19 个样本）。每个优化步使用相同样本集合，但重新采样
时间和噪声。该小实验预加载全部记录；普通 DataLoader + 同一 collate 的读取路径也已验证。

```python
dataset = JointV2Dataset(dataset_root, split="smoke")
try:
    records = [dataset[i] for i in range(len(dataset))]
finally:
    dataset.close()
batch = collate_joint_v2_records(records).to(device)

optimizer.zero_grad(set_to_none=True)
losses = simplex_fm_losses(model, batch, generator, precision="float32")
losses["total"].mean().backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
optimizer.step()
```

在 simplex_fm_losses 内部，先采样基础状态及 t∼Uniform[0,1)，再通过
`simplex_training_path` 以 native target 为监督端点构造带噪平移、旋转和序列概率状态。
模型实际调用相当于：`model(noisy_state, time, TaskObservation("J", batch.condition))`。
带噪训练状态依赖 native 标签是训练路径的一部分；标签不直接进入静态 condition。

本次实际 loss 是 CA 平移终点 MSE + 旋转切空间误差 + 序列 CE，等权求和。
虽然记录保留 atom14 实验原子及 mask，此次 simplex 过拟合没有直接训练全部侧链原子，
也没有添加 FAPE、肽键、键角或 clash 辅助 loss。自由采样仅通过残基 R/t 放置 N/CA/C。

已完成 CPU tiny 配置 1,000 步；结果见 `reports/boltzgen19_overfit_results_20260920.md`。
序列和位置明显拟合，但自由生成的骨架几何仍未通过，不能称为结构已充分过拟合。

## 清单和模型数据目录

- 官方原始数据：`artifacts/datasets/boltzgen1_train_official_20260920/`。
- 本次六个链对：`artifacts/datasets/boltzgen_native_four_20260920/interfaces.parquet` / CSV。
- 本次十九个片段：`artifacts/datasets/boltzgen_native_four_20260920/dataset/manifest.parquet`。
- 相同训练清单的可读副本：`artifacts/datasets/boltzgen_native_four_20260920/training_manifest.csv`。
- 当前 BoltzGen 入口自动输出 `sample_inventory.csv` / `sample_inventory.md`：包含数据来源、
  样本/结构 ID、条件/目标链、原链与实际输入长度、源链索引范围和序列；CSV 还包含 LMDB
  shard/key。范围是 NPZ 零基半开区间，另外提供一基闭区间，不等于 PDB 作者编号。
- 传给 JointV2Dataset 的目录：`artifacts/datasets/boltzgen_native_four_20260920/dataset/`。

新构建边转换边保存：每个样本通过检查后提交 LMDB，再追加并 flush
`dataset.inprogress/sample_inventory.csv`；全部成功后发布为 `dataset/`。
入口同时在外层提供同一 CSV 的副本和由 CSV 生成的 Markdown。异常或 Ctrl-C 保留中间目录；
半成品不作为完整数据集发布，当前没有自动续跑功能。

全量提取目录需要显式使用 `--all-structures`（与 `--structure-ids` 互斥）：

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u scripts/data/prepare_joint_v2_boltzgen.py \
  --structures-dir artifacts/datasets/boltzgen1_train_official_20260920/training_data/targets/structures \
  --all-structures --output artifacts/datasets/boltzgen_full_20260920 \
  --min-free-gib 20
```

输出目录必须不存在。先完成全量 interface 清单，再转换模型样本；外层 `progress.json`
记录阶段和源结构进度，转换阶段查看 `dataset.inprogress/build_status.json` 及实时样本 CSV。
默认磁盘保留 20 GiB：源结构扫描时和模型转换的每个批次开始时检查，不保证单批次内不越过阈值。
此命令只做预处理，默认 split 仍为 `smoke`，不启动训练。

正式训练还需要独立制定分组划分及完整质控策略。旧 v1 Boltz 转换产物须重新构建，
当前运行时会拒绝旧映射版本。
