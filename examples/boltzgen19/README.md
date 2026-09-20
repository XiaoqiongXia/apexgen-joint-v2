# BoltzGen 19 个连续界面片段

这份 example 是实际过拟合测试使用的预处理数据的副本：4 个来源结构（5HGA、5K9S、4EP3、7YXN），
6 个无方向蛋白链对，双向筛选后得到 19 个连续片段。所有样本 split=`smoke`。
每个 target 是同一来源链中长度至少 4 的一个最大连续接触段，不跨缺口拼接。

- `dataset/manifest.parquet`：训练索引；`training_manifest.csv`：可读清单。
- `dataset/shards/shard-00000.lmdb/data.mdb`：19 条 record 及 native 监督。
- `dataset/mapping.json`：AA20、target atom14、context atom38 的映射契约。
- `dataset/sources/*.npz`：4 个官方来源文件，文件名为内容 SHA256。
- `dataset/metadata.json`：manifest 与 LMDB 内容哈希；`SHA256SUMS`：example 文件哈希。
- `interfaces.csv`：6 个链对的接触元数据。

来源为 [BoltzGen 官方训练数据](https://huggingface.co/datasets/boltzgen/boltzgen1_train)，
其 `targets.zip` 中的 targets/structures。原下载归档 SHA256：
`b632b09f180216d6bc2769bad93e81c68561dbb6ddfbacd269ae57722809da16`。
原 NPZ 被保留，模型张量经过显式映射与几何检查；没有把缺失原子补成实验观察值。
数据许可与上游声明见随附 `UPSTREAM_DATASET_CARD.md`；模型几何常量的 OpenFold 许可保留在源码中。

LMDB 和清单中的原绝对路径作为历史来源信息保留。`JointV2Dataset` 优先使用本地
`dataset/sources/<sha256>.npz`，所以训练和审计不需要原服务器目录。
不要直接把 CSV 中原服务器路径当成本机路径。

```bash
cd examples/boltzgen19
sha256sum -c SHA256SUMS
```

训练示例见仓库 README / `docs/portable_simplex_training.md`。
本数据没有独立验证/测试划分，仅用于小样本过拟合和接口验证。
