"""Export codesign backbones with the full observed protein/environment."""

import json
from pathlib import Path

import torch

from apexgen.shared.io.joint_structure import write_joint_structure
from apexgen.joint_v2.evaluation.full_structure import full_structure_metrics, resolve_source
from apexgen.joint_v2.runtime.lineage import sha256_file


def write_codesign_structure(
    record, backbone, aatype, output_stem, *, provenance=None, raw_structure_root=None
):
    """Write PDB and mmCIF; explicitly declare that binder sidechains are absent."""
    if len(backbone) < 3 or backbone.shape != (len(aatype), 3, 3):
        raise ValueError("codesign export requires peptide length >= 3 and N/CA/C coordinates")
    atoms = backbone.new_zeros(len(backbone), 14, 3)
    atoms[:, :3] = backbone
    mask = torch.zeros(len(backbone), 14, dtype=torch.bool, device=backbone.device)
    mask[:, :3] = True
    stem = Path(output_stem)
    paths = [stem.with_suffix(suffix) for suffix in (".pdb", ".mmcif")]
    sidecar = stem.with_suffix(".json")
    if any(p.exists() for p in [*paths, sidecar]):
        raise FileExistsError(f"codesign structure output already exists: {stem}")
    source = resolve_source(record, raw_structure_root)
    metrics = full_structure_metrics(record, atoms, mask, raw_structure_root=raw_structure_root)
    for path in paths:
        write_joint_structure(
            source,
            atoms,
            mask,
            aatype,
            path,
            site_origin=torch.as_tensor(record["site_origin"]),
            replace_chain_id=record["peptide_chain_id"],
            receptor_chain_ids=(record["receptor_chain_id"],),
            model_index=record.get("selected_model_index", 0),
            preserve_environment=True,
        )
    if sha256_file(source) != record["raw_file_sha256"]:
        raise ValueError("source structure changed while exporting codesign result")
    payload = dict(
        schema="apexgen.joint_v2.codesign_full_complex.v1",
        sample_id=record["sample_id"],
        coordinate_frame="global_source_structure_frame",
        selected_model_index=record.get("selected_model_index", 0),
        environment="full_selected_source_model_except_replaced_peptide_chain",
        binder_representation="N_CA_C_only",
        sidechains_evaluated=False,
        all_atom_validity_evaluated=False,
        binding_success_evaluated=False,
        raw_file_sha256=record["raw_file_sha256"],
        generated_aatype=aatype.detach().cpu().tolist(),
        metrics=metrics,
        structures={p.name: sha256_file(p) for p in paths},
        provenance=provenance or {},
    )
    sidecar.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return dict(
        paths=[str(p) for p in paths],
        sidecar=str(sidecar),
        metrics=metrics,
        structure_sha256=payload["structures"],
        sidecar_sha256=sha256_file(sidecar),
    )
