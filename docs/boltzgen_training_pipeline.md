# From official BoltzGen NPZ files to model training

## Preprocessing entry point

`prepare_selected()` in `scripts/data/prepare_joint_v2_boltzgen.py` reads extracted official
NPZ files. Select structures explicitly with `--structure-ids`; this four-case example uses
`5hga, 5k9s, 4ep3, 7yxn`. Full extraction requires the separate `--all-structures` option.

```bash
source scripts/project_tmp_env.sh
python scripts/data/prepare_joint_v2_boltzgen.py \
  --structures-dir artifacts/datasets/boltzgen1_train_official_20260920/training_data/targets/structures \
  --structure-ids 5hga 5k9s 4ep3 7yxn \
  --output artifacts/datasets/my_new_boltzgen_panel \
  --model-smoke
```

The output must not exist. The original completed panel is at
`artifacts/datasets/boltzgen_native_four_20260920/`.

## Processing stages

| Stage | Implementation | Input/output |
|---|---|---|
| Structure and contacts | `data/boltz_interfaces.py:describe_interfaces()` | Find protein pairs from chains/interfaces and recompute observed heavy-atom contacts at ≤5 Å. Write interfaces Parquet/CSV; pairs have no training direction. |
| Bidirectional segmentation | `data/boltz_chain_pairs.py:fragment_selections()`, `adapt_interface_direction()` | Enumerate maximal consecutive target-contact runs for A→B and B→A. Each run of length ≥4 becomes one sample; no concatenation. |
| Mapping and geometry | `data/boltz_npz.py:adapt_boltz_npz()` | Cross-check AA names/tokens, map atoms by name, preserve source rows, check target backbone continuity and supported explicit connections. Recompute the partner's 5 Å core and 11 Å CB context for each fragment. |
| Storage | `scripts/data/prepare_joint_v2_boltz_chain_pairs.py:build_dataset()` | Write dataset/manifest.parquet, shards/*.lmdb, mapping.json, segment acceptance/rejection records, source NPZ copies, and hashes. |
| Loading and collation | `JointV2Dataset` → `collate_joint_v2_records()` | Validate mapping contracts, load LMDB records, separate context from targets, pad variable lengths, and construct masks. |
| Noising, model, loss | `sampling/simplex_runtime.py:simplex_fm_losses()` | Draw time/noise and construct noisy target states. SimplexCodesignModel receives state, time, and static context; compute three supervised losses. |

Here, `data/` and `sampling/` are under `src/apexgen/joint_v2/`.

`pocket_*` fields store context sequence, atom38 coordinates/masks, translation/rotation,
and residue identities. `joint_v2_target` stores target sequence, experimental atom14
coordinates/masks, frames, and other supervision. `boltz_adapter` and `interface_pair`
preserve source atom rows, chain roles, fragment ranges, and mapping versions.

In `batch.condition`, context sequence and coordinates are visible; native target AA is
UNKNOWN and native target atom coordinates are hidden. `batch.targets` holds supervision
separately. Both sides use one translation origin. Because cropping uses the native
interface, this is conditional generation at a specified site.

For example, Boltz ALA=2 maps to model ALA=0. Atom names map to residue-specific atom14
or context atom38 slots. Source atom rows provide provenance, not model slot numbers.

## Historical overfitting entry point

The original `scripts/experiments/run_joint_v2_boltzgen_overfit.py` loaded all 19 records
and collated them into one full batch. Every step reused the sample set but drew fresh
time/noise. A standard DataLoader using the same collator was also verified.
That historical script is part of the development repository; the standalone GitHub
release uses `apexgen-simplex train`, documented in `portable_simplex_training.md`.

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

`simplex_fm_losses` draws base states and t∼Uniform[0,1), then uses
`simplex_training_path` with native supervised endpoints to construct noisy translation,
rotation, and sequence-probability states. The model call is equivalent to
`model(noisy_state, time, TaskObservation("J", batch.condition))`.
Dependence of noisy training states on native labels is part of the training path;
labels do not directly enter the static condition.

The losses are CA endpoint translation MSE, rotation tangent error, and sequence CE,
with equal default weights. Experimental atom14/masks are retained, but this Simplex
experiment does not directly train complete side chains or add FAPE, peptide-bond,
bond-angle, or clash auxiliary losses. Free sampling places N/CA/C using per-residue R/t.

The historical CPU tiny experiment completed 1,000 steps; see
`reports/boltzgen19_overfit_results_20260920.md`. Sequence and position fitting improved,
but free backbone generation did not pass the geometry criteria. A later fixed-budget
continuation is reported in `reports/boltzgen19_continue5k_results_20260920.md`.

## Inventories and dataset directories

- Official source data: `artifacts/datasets/boltzgen1_train_official_20260920/`.
- Six pairs: `artifacts/datasets/boltzgen_native_four_20260920/interfaces.parquet` / CSV.
- Nineteen fragments: `artifacts/datasets/boltzgen_native_four_20260920/dataset/manifest.parquet`.
- Readable training inventory: `artifacts/datasets/boltzgen_native_four_20260920/training_manifest.csv`.
- New builds automatically write `sample_inventory.csv` / `.md` with provenance, sample/
  structure IDs, condition/target chains, original and actual lengths, source ranges,
  and sequence. CSV also stores LMDB shard/key. Zero-based half-open and one-based inclusive
  ranges both refer to NPZ chain positions, not PDB author numbering.
- Pass `artifacts/datasets/boltzgen_native_four_20260920/dataset/` to JointV2Dataset.

New builds save records as they are converted: commit each accepted sample to LMDB,
append/flush `dataset.inprogress/sample_inventory.csv`, and publish `dataset/` only after
successful completion. The entry point copies the CSV to the outer directory and renders
Markdown from it. Exceptions or Ctrl-C preserve intermediate output. Incomplete data is
not published as a complete dataset, and automatic continuation is not implemented.

To process every extracted structure, explicitly use `--all-structures`, mutually exclusive
with `--structure-ids`:

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u scripts/data/prepare_joint_v2_boltzgen.py \
  --structures-dir artifacts/datasets/boltzgen1_train_official_20260920/training_data/targets/structures \
  --all-structures --output artifacts/datasets/boltzgen_full_20260920 \
  --min-free-gib 20
```

The output must not exist. The complete interface inventory is built first, followed by
model-sample conversion. Outer `progress.json` tracks source/phase progress; during
conversion, inspect `dataset.inprogress/build_status.json` and the live sample CSV.
The default 20 GiB reserve is checked during source scanning and at each conversion-batch
start; a batch can still cross the threshold. This command only preprocesses data,
retains the default `smoke` split, and does not start training.

Formal training still requires independently defined group splits and complete QC policies.
Old v1 Boltz conversion products must be rebuilt; current runtime mapping checks reject them.
