# Official BoltzGen NPZ to Joint-v2: four-case adapter validation (2026-09-20)

## Outcome

Official `5hga, 5k9s, 4ep3, 7yxn` files from
`boltzgen/boltzgen1_train/targets.zip` passed the complete path:
NPZ → interface inventory → bidirectional continuous segments → unified LMDB records →
JointV2Dataset/DataLoader → collate → SimplexCodesignModel forward/backward.
Only the four selected files were processed. No full-dataset conversion, optimizer
updates, or formal training occurred in this validation.

| Item | Result |
|---|---:|
| Source structures | 4 |
| Protein chain pairs | 6 |
| Expanded directions | 12 |
| Maximal continuous contact runs | 132 |
| Runs shorter than 4, skipped | 113 |
| Conversion attempts | 19 |
| Accepted / rejected | 19 / 0 |
| CPU forward/backward | 19 / 19 passed, in batches of 8, 8, 3 |
| Maximum coordinate round-trip error | 9.5367431640625e-7 Å |
| Independent DataLoader read | 19 records, 5 batches |
| Automated tests | 146 passed |

Selection matches the previous four-case policy: each maximal continuous contact run
of at least four residues is independent. No separated runs are joined and no intervening
noncontact residues are inserted. Every target retains observed N/CA/C and must satisfy
source-position and C–N geometry continuity. Context comes from one opposite chain,
with a recomputed 5 Å core and 11 Å CB environment for that segment.

## Code changes

- `data/boltz_schema.py` handles both string atom names and legacy four-integer encoding.
  Interface statistics support protein atom tables without an element column and exclude
  hydrogens and missing atoms from heavy-atom contact calculations.
- `data/boltz_npz.py` accepts official string names and extended bonds without a separate
  connections table. For selected standard proteins, verify residue names, Boltz tokens,
  and legal atom-name templates before identifying elements. If an element column exists,
  enforce its cross-check.
- Interpret bond chain endpoints as **chains-table rows**, checking ownership of global
  residue/atom rows independently. Do not confuse chain rows with asym_id. Standard
  adjacent C–N bonds at explicit fragment boundaries may be cut; other target crosslinks
  are rejected.
- If cyclic_period is supplied and explicitly marks the target chain cyclic, reject it.
  That field is absent from these four official files.
- Update the mapping contract to `apexgen.boltz_structure_adapter.v2`. Dataset and collate
  enforce the Boltz mapping hash and target Joint-v2 contract hash. Rebuild old v1 records;
  replacing version strings/hashes is not sufficient. Non-Boltz sources do not acquire
  these source-specific restrictions.
- Add `scripts/data/prepare_joint_v2_boltzgen.py` for direct conversion from extracted NPZ.
  The four-case validation selected explicit structure IDs. Standalone NPZ provenance uses
  its own path and offset=0, never a ZIP header offset.

AA mapping remains name-based, e.g. Boltz ALA=2 → model ALA=0. Atom names select target
atom14/context atom38 slots. Source array order and global rows are not model slot numbers.

## Verification details

The test total comprises 78 legacy Boltz/pair/segmentation regressions, 35 official-schema
and runtime-contract checks, and 33 shared Dataset/native-supervision/model regressions.
Coverage includes all 20 standard AAs, shuffled atom order, missing side chains, invalid
tokens/names, row-versus-asym bond ownership, boundary bonds, unsupported crosslinks,
cyclic declarations, mapping-version mismatch, and independent multiple segments.

All 19 real samples also underwent:

1. Per-atom source audits of residue/atom identities, source indices, masks, and coordinates.
2. Offline comparison against historical legacy-format records: identical sample-ID sets,
   ranges, AA, atom14/atom38 coordinates/masks, target frames, and context connection masks.
   Historical v1 records were not fed to the current model.
3. Direct distance checks using retained context coordinates: every target residue still
   contacts the context at ≤5 Å.
4. DataLoader reads using the actual variable-length collator and model-input contract.
5. Finite loss/nonzero finite gradients in the current tiny SimplexCodesignModel, fixed
   context coordinates, and unchanged static conditions after modifying native target
   sequence/coordinates.
6. With zero-initialized heads, first-step nonzero gradients concentrate at outputs. A
   separate temporary head-perturbation probe produced nonzero gradients in 45 encoder
   parameter tensors. That temporary model/checkpoint was not saved.

## Outputs and reproduction

Directory: `artifacts/datasets/boltzgen_native_four_20260920/`.

- `interfaces.parquet` / `interfaces.csv`: six original pairs and contacts.
- `summary.json`: processing/model-check summary.
- `tensor_equivalence_and_contacts.json`: 19-sample tensor and retained-contact comparison.
- `validation.json`: check counts and relevant source hashes.
- **`dataset/`**: actual JointV2Dataset input, containing manifest, shards, source copies,
  mapping, segment outcomes, and model-check reports.

Executed command; use a new output path when rerunning:

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/data/prepare_joint_v2_boltzgen.py \
  --structures-dir artifacts/datasets/boltzgen1_train_official_20260920/training_data/targets/structures \
  --structure-ids 5hga 5k9s 4ep3 7yxn \
  --output artifacts/datasets/boltzgen_native_four_20260920 \
  --model-smoke
```

See `docs/boltz_interface_fragment_training.md` for loading. Split is `smoke`.
Logs: `artifacts/tmp/boltzgen_native_four_20260920.log`, `boltzgen_adapter_tests.log`,
`boltzgen_additional_tests.log`, and `boltzgen_shared_tests.log`.

## Scope

This work fixes official-format compatibility, bond indexing, and runtime mapping checks.
Nonstandard residues, multiple coordinate models, missing target backbone, and unsupported
crosslinks remain rejected. Malformed swapped connection fields in historical files are
not guessed or repaired. Full-structure severe-clash and unannotated-covalent screening
are not uniformly integrated into the NPZ route. Format compatibility does not remove
those limits. Formal homology-aware splitting/deduplication and generation quality were
not established by this four-case validation.
