# Boltz processed NPZ → Joint-v2

`src/apexgen/joint_v2/data/boltz_npz.py` converts a Boltz structured-array
**structure** NPZ into the existing Joint-v2 complex-record contract. It does not
depend on the Boltz Python package or model, and does not read MSA features.
The adapter supports a canonical protein peptide, one coordinate
model and one or more explicitly selected protein receptor chains.

## Identity and coordinate contract

| Source | Joint-v2 handling |
| --- | --- |
| `residues.name` and `res_type` | Cross-check against the pinned Boltz vocabulary, then map by residue name into `AA3_TO_INDEX`. For example, Boltz ALA=2 → model ALA=0; TRP=19 → TRP=17. |
| `atoms.name` | Accept BoltzGen Unicode names or decode Boltz's four zero-padded integers with `chr(value + 32)`. Map by atom name, never by source position. |
| Protein atom slots | Receptor uses the model's 38-name vocabulary; peptide uses the residue-specific `ATOM14_NAMES`. Empty atom14 slots stay masked. |
| `atoms.element` | When present, cross-check the atomic number. Published BoltzGen arrays omit it: infer C/N/O/S only after validating the canonical residue and its residue-specific atom-name template. No general ligand element inference. |
| `atoms.coords` | Observed coordinates in Å. Subtract the core CA centroid from both receptor and peptide. No idealization, fitting or sidechain reconstruction. |
| `atoms.conformer` | Not used: these are chemical reference coordinates, not observed labels. |
| `atoms.is_present` | Controls atom observation masks. Missing atoms produce zero coordinates, a false mask and source index −1. |
| `residues.is_present` | Check consistency with observed atoms. A peptide with missing N/CA/C is rejected; receptor residues lacking N/CA/C are omitted and recorded. |
| `mask` | Chain validity mask, not a per-residue or per-atom mask. |
| Chain `name`, `asym_id`, `entity_id`, `sym_id` | Preserve assembly-copy identity. Equal entity IDs do not merge chains. Chain selection uses exact names such as `A1`, not assumed PDB author chain `A`. |
| `chains.res_idx` | Global offset into the residue table. |
| `residues.res_idx` | Zero-based position within the polymer; retained through cropping. It is not the original author residue number. |
| `atom_idx`, `atom_num` | Checked spans into the global atom table. Each output atom retains its original row index. |
| `coords` + `ensemble`, if present | Require one complete model and agreement with observed `atoms.coords`. Ambiguous multi-model input is rejected. |
| `bonds`, `connections` | BoltzGen uses extended `bonds` with chain/res/atom endpoints; legacy Boltz has separate `connections`. Chain endpoints are chain TABLE ROWS, not `asym_id`. Check all endpoint memberships; unsupported target crosslinks are rejected. |
| `chains.cyclic_period`, if present | Reject a target source chain declared cyclic under the current linear-target policy. |

`mapping_manifest()` exports all 20 amino-acid mappings, all atom14 slot names,
the receptor atom vocabulary, model-contract hash, atom14-constants hash and a
mapping hash. `audit_boltz_record()` independently reopens the source and checks
every represented residue, chain and atom, including coordinate reconstruction.

Adapter schema is now `apexgen.boltz_structure_adapter.v2`. Both `JointV2Dataset`
and `collate_joint_v2_records` reject Boltz records with a stale adapter/mapping
hash or mismatched target Joint-v2 contract. Rebuild previously converted v1
Boltz records; do not relabel their hashes manually. Non-Boltz record paths are
unaffected by this source-specific validation.

Atom names such as ASP OD1/OD2 are preserved exactly. Symmetry-equivalent atom
renaming is not performed by this adapter. Peptide OXT is recorded as excluded
because the model's atom14 representation has no OXT slot; receptor OXT has a
dedicated slot.

## Crop, topology and supervision

The crop follows the existing preprocessing policy: core receptor residues have
an observed heavy atom within 5 Å of the native peptide; context residues have
CB within 11 Å of a core CB. Gly uses CA, and a virtual CB is used only for crop
selection when observed CB is missing. Virtual CB never becomes a label.

Pocket chain IDs and original polymer positions survive cropping. Torsions are
masked across chain boundaries, omitted residues and implausible C–N links.
The peptide must be complete and continuous; it is not shortened to hide missing
labels. Existing broad bond, angle and stereochemistry checks are reused.

`collate_joint_v2_records()` creates separate `condition` and `targets`:

- Known receptor sequence and coordinates enter the condition.
- The peptide sequence is UNKNOWN=20 in the static condition; its native atom
  coordinates are absent from that condition.
- Native peptide AA IDs, observed atom14, frames and torsions are supervision.
- The flow training state may depend on native labels, as required by the
  supervised conditional path.

The crop itself is selected using the native interface. This is a **site-conditioned
training example**; inference needs a supplied target site or a separate site
selection procedure. Label independence after cropping does not imply that site
selection is independent of the native complex.

## Direct use

Run from the repository root:

```python
from apexgen.joint_v2.data.boltz_npz import adapt_boltz_npz, audit_boltz_record
from apexgen.joint_v2.data.batch import collate_joint_v2_records

record = adapt_boltz_npz(
    "/absolute/path/5hga.npz",
    sample_id="boltz_5hga",
    source_pdb_id="5hga",
    receptor_chain_ids=("A1", "B1"),
    peptide_chain_id="C1",
    split="smoke",
)
audit = audit_boltz_record(record)
batch = collate_joint_v2_records([record])
```

To build a small reusable dataset, provide a JSON list:

```json
[
  {
    "path": "/absolute/path/5hga.npz",
    "sample_id": "boltz_5hga",
    "source_pdb_id": "5hga",
    "receptor_chain_ids": ["A1", "B1"],
    "peptide_chain_id": "C1",
    "split": "smoke"
  }
]
```

```bash
source scripts/project_tmp_env.sh
python scripts/data/prepare_joint_v2_boltz_npz.py \
  --index artifacts/tmp/my_boltz_index.json \
  --output artifacts/datasets/my_boltz_panel \
  --model-smoke
```

The builder copies source NPZs, audits mappings, writes embedded-label LMDB
records, a Parquet manifest, `mapping.json`, `metadata.json` and `build.json`.
It refuses to replace an existing output directory. The optional model smoke
uses CPU and the existing tiny `SimplexCodesignModel` configuration, without
optimizer steps or checkpoints. It checks finite losses and backward gradients,
fixed receptor coordinates, and static-condition independence from labels. A
separate temporary output-head perturbation tests gradients through the encoder;
the default zero-initialized output heads otherwise block upstream gradients on
the first backward pass.

```python
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.data.batch import collate_joint_v2_records

dataset = JointV2Dataset("artifacts/datasets/my_boltz_panel", split="smoke")
try:
    batch = collate_joint_v2_records([dataset[i] for i in range(len(dataset))])
finally:
    dataset.close()
```

## Scope and limitations

- This supports the integer-encoded Boltz schema and the actual string-name
  schema in `boltzgen/boltzgen1_train/targets.zip`, not arbitrary future schemas
  or tokenized model batches. The native schema has no independent `connections`.
- Noncanonical selected residues, masked selected chains, missing peptide
  backbone, peptide breaks and unsupported explicit crosslinks fail closed.
  Close terminal C–N geometry is also screened for possible cyclization.
- No general ligand/nucleic-acid conditioning or arbitrary covalent graph is
  represented. Nearby excluded observed atoms are reported; their absence from
  the model condition must be considered when choosing training examples.
- The processed format cannot recover original author numbering, altloc choices
  or occupancy filtering. Synthetic author numbering is not written into records.
  Upstream omission of covalent annotations cannot be fully repaired here.
- Historical legacy files with swapped connection field layouts still fail
  endpoint checks; the adapter does not guess a repair. Comprehensive full-model
  steric and unannotated covalent-link screening remains separate from this
  compatibility fix; native-format support is not a claim that all PDB entries
  are ready for training without further quality filtering.
- Chain roles are supplied by the caller. A short interacting chain is not, by
  itself, an experimentally validated binder-design example.
- The four-example panel is a data/model compatibility check. It does not supply
  a homology-separated train/validation split or establish generation quality.

See `reports/boltz_npz_adapter_four_20260920.md` for the first real-data audit.
See `reports/boltzgen_native_adapter_four_20260920.md` for the official BoltzGen
files and the complete current model-input verification.
