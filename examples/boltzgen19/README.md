# BoltzGen: 19 continuous interface fragments

This example copies the processed data used in the overfitting experiment: four source
structures (5HGA, 5K9S, 4EP3, 7YXN), six undirected protein chain pairs, and 19 fragments
after bidirectional selection. Every sample has split=`smoke`. Each target is one maximal
continuous contact segment of at least four residues from a single source chain;
segments separated by gaps are never concatenated.

- `dataset/manifest.parquet`: training index; `training_manifest.csv`: readable inventory.
- `dataset/shards/shard-00000.lmdb/data.mdb`: 19 records and native supervision.
- `dataset/mapping.json`: AA20, target atom14, and context atom38 mapping contract.
- `dataset/sources/*.npz`: four official source files named by content SHA256.
- `dataset/metadata.json`: manifest/LMDB hashes; `SHA256SUMS`: example file checksums.
- `interfaces.csv`: contact metadata for the six chain pairs.

The source is the [official BoltzGen training dataset](https://huggingface.co/datasets/boltzgen/boltzgen1_train),
under targets/structures in `targets.zip`. The original archive SHA256 is:
`b632b09f180216d6bc2769bad93e81c68561dbb6ddfbacd269ae57722809da16`.
Original NPZ files are preserved. Model tensors undergo explicit mapping and geometry
checks; missing atoms are not filled in and presented as experimental observations.
The upstream licensing declaration is included in `UPSTREAM_DATASET_CARD.md`.
The OpenFold license for model geometry constants is retained in the source tree.

Original absolute paths in the LMDB and inventories remain as historical provenance.
`JointV2Dataset` prefers local `dataset/sources/<sha256>.npz` files, so training and audits
do not require the original server directories. Do not interpret historical CSV paths
as paths on your new server.

```bash
cd examples/boltzgen19
sha256sum -c SHA256SUMS
```

See the repository README or `docs/portable_simplex_training.md` for training commands.
This dataset has no independent validation/test split and is intended for small-sample
overfitting and interface validation.
