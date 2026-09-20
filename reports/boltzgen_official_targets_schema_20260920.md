# Official BoltzGen targets.zip schema inspection (2026-09-20)

## Source and inspection status

Source: [boltzgen/boltzgen1_train](https://huggingface.co/datasets/boltzgen/boltzgen1_train/tree/main),
file `targets.zip`. This report inspects released data bytes; current upstream Python
class definitions are not a substitute for the actual release schema.

**Full download, SHA256 verification, and extraction completed.** The archive matches
the official LFS hash. All 461,134 files extracted and passed outer ZIP member CRC checks.
Array headers in all 230,566 NPZ files showed one field/dtype/rank layout, with no
object/pickle dtype. This header census is not a complete value or geometry audit;
detailed value checks covered the 12 cases below.

Archive size: 75,009,352,113 bytes. LFS SHA256:
`b632b09f180216d6bc2769bad93e81c68561dbb6ddfbacd269ae57722809da16`.
The ZIP contains 230,566 structure NPZs, 230,566 matching record JSONs, `config.yaml`,
and `manifest.json`. Extracted files total 78,428,895,734 bytes.

Project destination: `artifacts/datasets/boltzgen1_train_official_20260920/`.
The archive's top directory is `targets/`; extracting into `training_data/` produces:

```text
training_data/targets/
├── config.yaml
├── manifest.json
├── records/<structure_id>.json
└── structures/<structure_id>.npz
```

## Actual NPZ arrays

All array values were read directly for 12 official files:
`5pc9, 7ysh, 3far, 2j3w, 3a2n, 4lgt, 1uo9, 8k94, 4ep3, 5hga, 7yxn, 5k9s`.
The complete 230,566-file schema census found the same fields/dtypes below. Atom, residue,
chain, connection, and coordinate-model counts may differ across files.

| Array | Fields | Meaning and reading notes |
|---|---|---|
| `atoms` | `name, coords, is_present, bfactor, plddt` | `name` is a `<U4` string; coords are float32 3D coordinates; no element column. Exclude missing-atom placeholders using is_present. |
| `bonds` | `chain_1, chain_2, res_1, res_2, atom_1, atom_2, type` | Explicit endpoints; sampled chain/res/atom indices refer to global table rows. This table does not enumerate every implicit backbone peptide bond. |
| `residues` | `name, res_type, res_idx, atom_idx, atom_num, atom_center, atom_disto, is_standard, is_present` | res_idx is a chain-local position; atom_idx starts the global atom span. Cross-check residue names and tokens. |
| `chains` | `name, mol_type, entity_id, sym_id, asym_id, atom_idx, atom_num, res_idx, res_num` | Here res_idx starts the global residue span, unlike residues.res_idx. Preserve chain row, asym_id, and chain name separately. |
| `interfaces` | `chain_1, chain_2` | Contact-pair list, not a residue contact matrix or a context/target direction. Recompute local contacts from coordinates. |
| `mask` | One bool per chain | Source chain-validity flag; not proof that all model-specific QC passes. |
| `coords` | `coords` | Coordinate collection of float32 3D positions. |
| `ensemble` | `atom_coord_idx, atom_num` | Each coordinate model's span in coords; do not assume one model for every file. |

No file has a separate `connections` array or `chains.cyclic_period` field. Newly added
upstream fields cannot be assumed to exist in this release; absence of a field alone
does not establish chemically linear target topology.

## Values, indices, and four-case cross-check

- All 12 cases passed chain→residue→atom span, atom-name uniqueness, observed-coordinate
  finiteness, and coordinate-model span checks.
- Checked 9,146 standard protein residues: the 20 standard AAs use Boltz tokens 2..21.
  Explicit mapping into model AA indices is still required.
- Both endpoints of 312 explicit bonds passed chain/residue/atom ownership checks.
  These cases did not exhibit the swapped chain/atom field layout found in some older data.
  This is not a claim that all bond values in the full release were audited.
- For `5hga, 5k9s, 4ep3, 7yxn`, atom counts, decoded names, presence masks, observed
  coordinates, and shared residue/chain fields match the previous legacy Boltz files.
- Chain IDs, names, lengths, and valid flags in matching JSON records agree with NPZ.

Recomputed observed protein heavy-atom contacts at ≤5 Å are shown below. A/B is only
the display order of each pair.

| Structure | Chain A / B | Full-chain residues A / B | Contact residues A / B |
|---|---|---:|---:|
| 5hga | A1 / B1 | 275 / 100 | 38 / 30 |
| 5hga | A1 / C1 | 275 / 8 | 38 / 8 |
| 5k9s | A1 / B1 | 458 / 11 | 30 / 10 |
| 5k9s | A1 / C1 | 458 / 11 | 20 / 8 |
| 4ep3 | A1 / B1 | 203 / 9 | 36 / 9 |
| 7yxn | A1 / C1 | 250 / 11 | 16 / 10 |

Map AA/atoms by name with source-token cross-checks; source atom rows are not atom14/38
slots. Expand both directions and retain each maximal consecutive contact run of at
least four residues as an independent sample. Never stitch separated runs, and check
observed-backbone continuity independently.

## Record metadata and interpretation of counts

The four `records/*.json` files include experimental method, resolution, dates, interface
records, and chain fields `chain_id, chain_name, num_residues, mol_type, cluster_id,
msa_id, template_ids, valid`. `msa_id` is a reference; MSA contents are not in targets.zip.
The separate msa.zip was not downloaded in this inspection. Cluster IDs can inform
future group splits, but their presence alone does not establish absence of leakage.

`config.yaml` identifies RCSB provenance and source filters for length, unknown residues,
consecutive CA distances, and chain clashes. Source filters do not replace model-specific
checks for target continuity, AA/atom mapping, and geometry.

Full-record method counts: X-ray diffraction 191,006; electron microscopy 24,541; solution
NMR 14,206; other or mixed methods 813. All 29 method labels/counts are in
`inspection/record_metadata_census.json`. Seven historical entries have method
`solution nmr,theoretical model`: `1oln, 2bvk, 1e08, 1vyc, 1ur6, 1dwl, 1gx7`.
Thus this is a PDB-derived, predominantly experimental collection, but not every record
or coordinate can be described as a purely experimental measurement. These seven
entries can be handled separately by a subsequent training policy.

The [official RCSB 5HGA entry](https://www.rcsb.org/structure/5HGA) was also checked:
X-ray method, 2.20 Å resolution, and chain lengths 275/100/8 match the downloaded records.
Experimental contacts can supply interface-generation supervision, but a fragment cut
from a larger protein is not automatically an independently validated binder.

**230,566 counts structure files, not usable training interfaces or fragments.** Some
sample files have no protein–protein interface. Final training counts require pair
selection, contact recomputation, bidirectional segmentation, and model-specific QC.

All 230,566 matching JSONs were read. Filename IDs, internal IDs, and NPZ inventory IDs
match one-to-one. Chain counts, interface counts, and referenced chain IDs passed checks.
There is one top-level JSON layout and one chain-record layout.

| Metadata statistic | Count |
|---|---:|
| Chains of all molecular types | 3,130,376 |
| Protein chains (`mol_type=0`) | 1,291,686 |
| Listed protein–protein pairs, undirected deduplication per structure | 3,128,129 |
| Pairs with both chains and interface source-valid | 3,088,613 |
| Structures with at least one such source-valid protein pair | 128,463 |

Distances, consecutive segments, and model QC have not been checked for all these pairs;
cross-structure sequence deduplication has not been performed. **3,088,613 is a source-valid
candidate-pair count, not the final training-sample count.** Full results are in
`inspection/record_metadata_census.json`.

## Adapter behavior at the time of inspection

The then-current `adapt_boltz_npz` rejected all four official files:

```text
ValueError: unsupported Boltz atoms schema;
requires {'name': 'iu', 'element': 'iu', 'coords': 'f', 'is_present': 'b'}
```

That adapter supported legacy integer atom names and element/connections fields.
Native-schema normalization, bond endpoints, and name mapping had to be implemented
before model-input validation. This inspection only downloaded and recorded evidence;
it did not modify the model or production converter. The subsequent fix is documented
in `boltzgen_native_adapter_four_20260920.md`.

Inspection artifacts: `remote_sample_schemas.json`, `real_sample_value_audit.json`,
`current_adapter_probe.json`, `zip_members.csv`, and `record_metadata_census.json`, under
the dataset's `inspection/` directory. Inspection scripts are in
`artifacts/tmp/boltzgen_targets_inspection_20260920/`. Complete-download/extraction status
is recorded in `download_verified.json` and `extraction_and_schema_census.json`.
