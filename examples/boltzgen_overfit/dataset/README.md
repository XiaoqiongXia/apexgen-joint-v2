# BoltzGen overfitting dataset

This directory contains three LMDB shards (24,576 records in total) with
precomputed pocket/peptide static features, CSV and Parquet inventories, and
source NPZ files. Each shard contains 8,192 records; the split is `smoke`.

Load with `JointV2Dataset(path, split="smoke")` and use
`prepare_training_collator(dataset)` for verified lightweight batch assembly.
See the parent README for download, verification, and training commands.
