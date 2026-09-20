# Boltz NPZ to Joint-v2: four-case adaptation and audit

Date: 2026-09-20. The local processed Boltz NPZ → Joint-v2 complex record → LMDB → batch
path was implemented and checked with CPU forward/backward through SimplexCodesignModel.
This audit did not run optimizer updates, formal training, or generation-quality evaluation.

## Four real structures

Data: `artifacts/datasets/boltz_npz_four_20260920`.
Chain names are the **complete Boltz assembly names**, not inferred author-chain mappings.

| PDB | Receptor chains | Peptide chain | Peptide sequence | Pocket residues | Peptide length | Audited atoms |
|---|---|---|---|---:|---:|---:|
| 5HGA | A1, B1 | C1 | RFPLTFGW | 168 | 8 | 1,465 |
| 5K9S | A1 | B1 | TNKKMRRNRFK | 154 | 11 | 1,288 |
| 4EP3 | A1 | B1 | KARVLAEAM | 134 | 9 | 1,081 |
| 7YXN | A1 | C1 | RPAILYALLSS | 79 | 11 | 718 |

A total of **574 residues and 4,552 observed atoms** were checked. Receptor plus peptide
residues cover all **20 standard amino acids**. The final 5HGA pocket retains both A1 and
B1 receptor chains. The retained 4EP3 region has 28 unobserved standard-atom slots,
kept at mask=false without coordinate filling.

Respectively 1, 33, 5, and 7 residues in the selected full receptor chains were excluded
for missing N/CA/C. These counts are not restricted to the final pockets. Original polymer
indices are retained to prevent torsion calculation across missing regions after cropping
or removal. None of the four final cases had other observed atoms within 5 Å of the
peptide that were omitted from context.

## Mapping verification

- Map AA by residue name and cross-check Boltz `res_type`: e.g. ALA 2→0, GLY 9→7,
  TRP 19→17. The implementation does not rely on subtracting two.
- Decode four-integer Boltz atom names and place them in receptor atom38 or residue-specific
  atom14 slots, independently of source atom-array order.
- Preserve global residue/chain rows, asym/entity/sym IDs, and the global source row of
  each observed output atom.
- Audit source name, residue ownership, presence mask, and coordinates. After restoring
  the pocket origin, maximum coordinate-component error was
  **9.5367431640625 × 10⁻⁷ Å**, attributable to float32 storage.
- Do not use `conformer` as a structure label or invent author numbering lost by NPZ.
- Save `mapping.json` with complete AA/atom mappings and model/atom-constant hashes.

Historical mapping hash:
`e99e8c8ddc866e45ad29b3e44b3e68f2ae601e4a965aed10b3ae34de043e612f`.

Evidence files: `mapping.json`, `build.json`, `final_adapter_audit.json`, and
`adapter_provenance.json`. The last records adapter and local Boltz field-definition source
hashes. Readapting the four cases with the final implementation reproduced saved coordinates,
AA, masks, and chain-link arrays exactly. This historical v1 mapping was subsequently
superseded by the native BoltzGen v2 adapter.

## Batch and model checks

Reload through `JointV2Dataset(..., split="smoke")` and collate to layout **[4, 176]**:
concatenate each sample's own pocket and peptide along the node axis, then pad the batch.

- Native AA/coordinates are separate targets. Static peptide AA=20 (UNKNOWN), without
  native peptide atoms. Changing target AA/coordinates leaves every static condition
  tensor unchanged.
- Source chain identity, polymer indices, and valid C–N link masks survive; no peptide
  bonds are imposed across chains or breaks.
- The existing tiny SimplexCodesignModel completes variable-length forward, conditional
  Simplex path, loss, and backward on CPU with finite losses/gradients.
- Default output heads are zero-initialized. The first backward has four parameter tensors
  with nonzero gradients and zero encoder gradients. A temporary nonzero-head probe
  produces **76** nonzero-gradient parameter tensors, including **45 encoder tensors**,
  confirming that feature-to-loss backpropagation is available.
- Receptor translation/rotation stay fixed. No optimizer step or checkpoint was produced.

See `model_smoke.json`. Initial loss is not a measure of trained performance or binder-design
ability. Native interfaces determine pockets, so this is site-conditioned training data;
inference still needs site conditions.

## Tests

The 41 new tests in `tests/joint_v2/test_boltz_npz.py` cover every standard AA, intentionally
reversed atom order, missing side chains, incorrect indices/elements, duplicate atoms,
invalid spans, chain masks, breaks, multimodel ambiguity, conflicting coordinate sources,
crosslinks, OXT, LMDB round trips, and padding.

Combined with existing dataset/preprocessing regressions: **123 passed**. Ruff checks
passed for the three new Python files.

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

This historical command refers to the full development repository's regression suite.

## Entry points

- Adapter API: `src/apexgen/joint_v2/data/boltz_npz.py`.
- Dataset builder/optional model smoke: `scripts/data/prepare_joint_v2_boltz_npz.py`.
- Fields, limitations, and Python examples: `docs/joint_v2_boltz_npz_adapter.md`.

Rebuild a new panel using the copied source NPZ files:

```bash
source scripts/project_tmp_env.sh
python scripts/data/prepare_joint_v2_boltz_npz.py \
  --index artifacts/datasets/boltz_npz_four_20260920/input_index.json \
  --output artifacts/datasets/boltz_npz_four_recheck \
  --model-smoke
```

The output must be new. The supported scope is linear peptides of standard protein
residues. Nonstandard residues, missing peptide backbone, chain breaks, unsupported
explicit crosslinks, and multimodel input are rejected. MSA is not a model input.
These four cases are not a homology-deduplicated training/validation split.

## Local archive integrity

Source: `/data2/xiaoqiong/python_project/BasinDiff/dataset/Structure/rcsb_processed_targets.tar`,
14,053,543,936 bytes. Scanning its tail raised
`tarfile.ReadError: unexpected end of data`; therefore the complete training archive
cannot be claimed to have downloaded successfully. These four NPZ members were complete,
all arrays loaded, and source/coordinate identities independently audited. Copies are in
`sources/`. The original archive was neither redownloaded nor modified.

During candidate selection, 4FCM, 8FZ2, 8ANB, and 1SE0 were rejected for missing selected
peptide backbone. 6YW4 was rejected for unsupported residue `48V`. Peptides were not
shortened and residue names were not forced to standard names to pass checks.
