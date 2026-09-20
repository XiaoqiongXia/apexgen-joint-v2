"""Writers for generated atom14 peptide structures."""

from __future__ import annotations

import os
from pathlib import Path

import gemmi
import torch
from torch import Tensor

from apexgen.shared.geometry.joint_residue_constants import AA3_ORDER, ATOM14_NAMES


def _free_chain(model: gemmi.Model) -> str:
    used = {c.name for c in model}
    for name in "XYZABCDEFGHIJKLMNOPQRSTUVWabcdefghijklmnopqrstuvwxyz0123456789":
        if name not in used:
            return name
    raise ValueError("no free chain identifier for generated peptide")


def write_joint_structure(
    receptor_path: str | Path,
    atom14: Tensor,
    atom14_mask: Tensor,
    aatype: Tensor,
    output_path: str | Path,
    *,
    site_origin: Tensor | None = None,
    chain_id: str | None = None,
    replace_chain_id: str | None = None,
    receptor_chain_ids: tuple[str, ...] | None = None,
    model_index: int = 0,
    preserve_environment: bool = True,
) -> str:
    """Replace the peptide in the selected full source model, in global coordinates.

    By default retain all other chains, residues, ligands and waters as supplied.
    receptor_chain_ids validates the intended receptor; it only filters chains
    when preserve_environment is explicitly disabled.
    """
    receptor_path, output_path = Path(receptor_path), Path(output_path)
    if receptor_path.resolve() == output_path.resolve():
        raise ValueError("generated output must not overwrite receptor")
    xyz = atom14.detach().cpu()
    mask = atom14_mask.detach().cpu()
    aa = aatype.detach().cpu()
    if xyz.ndim == 4 and xyz.shape[0] == 1:
        xyz, mask, aa = xyz[0], mask[0], aa[0]
    if xyz.ndim != 3 or xyz.shape[-2:] != (14, 3):
        raise ValueError("atom14 must be [L,14,3]")
    if mask.shape != xyz.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("atom14_mask must be bool [L,14]")
    if aa.shape != xyz.shape[:1] or aa.dtype != torch.long or bool(((aa < 0) | (aa >= 20)).any()):
        raise ValueError("aatype must be int64 [L] in [0,20)")
    if not bool(torch.isfinite(xyz[mask]).all()):
        raise ValueError("generated coordinates must be finite")
    if site_origin is not None:
        xyz = xyz + site_origin.detach().cpu().reshape(1, 1, 3)
    structure = gemmi.read_structure(str(receptor_path), merge_chain_parts=False)
    if type(model_index) is not int or not 0 <= model_index < len(structure):
        raise ValueError("selected source model_index is out of range")
    for index in reversed(range(len(structure))):
        if index != model_index:
            del structure[index]
    structure.setup_entities()
    model = structure[0]
    if receptor_chain_ids is not None:
        if not receptor_chain_ids or len(set(receptor_chain_ids)) != len(receptor_chain_ids):
            raise ValueError("receptor_chain_ids must be non-empty and unique")
        if replace_chain_id in receptor_chain_ids:
            raise ValueError("native peptide chain cannot also be a receptor chain")
        available = {c.name for c in model}
        missing = set(receptor_chain_ids) - available
        if missing:
            raise ValueError(f"receptor chains are absent: {sorted(missing)}")
        keep = set(receptor_chain_ids)
        if replace_chain_id is not None:
            keep.add(replace_chain_id)
        if not preserve_environment:
            for existing_name in [existing.name for existing in model]:
                if existing_name not in keep:
                    model.remove_chain(existing_name)
    if replace_chain_id is not None:
        if replace_chain_id not in {c.name for c in model}:
            raise ValueError(f"native peptide chain {replace_chain_id!r} is absent")
        model.remove_chain(replace_chain_id)
        # Native peptide topology must not survive replacement with a new sequence.
        for index in reversed(range(len(structure.connections))):
            connection = structure.connections[index]
            if replace_chain_id in (connection.partner1.chain_name, connection.partner2.chain_name):
                del structure.connections[index]
    chain_id = chain_id or replace_chain_id or _free_chain(model)
    if chain_id in {c.name for c in model}:
        raise ValueError(f"chain conflict: {chain_id}")
    chain = gemmi.Chain(chain_id)
    used_subchains = {residue.subchain for existing in model for residue in existing}
    subchain = chain_id + "_generated"
    while subchain in used_subchains:
        subchain += "_"
    entity_names = {entity.name for entity in structure.entities}
    entity_name = "generated_binder"
    while entity_name in entity_names:
        entity_name += "_"
    entity = gemmi.Entity(entity_name)
    entity.entity_type = gemmi.EntityType.Polymer
    entity.polymer_type = gemmi.PolymerType.PeptideL
    entity.subchains = [subchain]
    entity.full_sequence = [AA3_ORDER[int(value)] for value in aa]
    structure.entities.append(entity)
    for i in range(xyz.shape[0]):
        if not bool(mask[i].any()):
            continue
        residue = gemmi.Residue()
        residue.name = AA3_ORDER[int(aa[i])]
        residue.seqid = gemmi.SeqId(i + 1, " ")
        residue.entity_type = gemmi.EntityType.Polymer
        residue.entity_id = entity_name
        residue.subchain = subchain
        residue.label_seq = i + 1
        for j, name in enumerate(ATOM14_NAMES[int(aa[i])]):
            if not name or not bool(mask[i, j]):
                continue
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element("S" if name.startswith("S") else name[0])
            atom.occ = 1.0
            atom.b_iso = 0.0
            atom.pos = gemmi.Position(*[float(v) for v in xyz[i, j]])
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    # Keep declared receptor sequences, but discard orphan native-binder entity
    # metadata instead of leaking its old sequence into the generated mmCIF.
    active_subchains = {r.subchain for existing in model for r in existing}
    active_entities = {r.entity_id for existing in model for r in existing}
    for index in reversed(range(len(structure.entities))):
        current = structure.entities[index]
        current.subchains = [name for name in current.subchains if name in active_subchains]
        if not current.subchains and current.name not in active_entities:
            del structure.entities[index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Generation is distributed across ranks on a shared filesystem.  Publish
    # only complete structures so rank 0 cannot hash/evaluate a partially
    # written file from another rank after the collective completes.
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        if output_path.suffix.lower() in {".cif", ".mmcif"}:
            structure.make_mmcif_document().write_file(str(temporary))
        elif output_path.suffix.lower() == ".pdb":
            structure.write_pdb(str(temporary))
        else:
            raise ValueError("output must end in .pdb, .cif, or .mmcif")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return chain_id
