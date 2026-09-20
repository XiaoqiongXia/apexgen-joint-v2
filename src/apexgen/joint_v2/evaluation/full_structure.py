"""Generated-atom checks against the full selected source model, without targets."""

from pathlib import Path

import gemmi
import torch

from apexgen.joint_v2.runtime.lineage import sha256_file


def resolve_source(record, raw_structure_root=None):
    path = Path(record.get("raw_path", ""))
    if not path.is_file() and raw_structure_root is not None:
        path = Path(raw_structure_root) / path.name
    if not path.is_file() or sha256_file(path) != record.get("raw_file_sha256"):
        raise ValueError("full structure source missing or source digest differs")
    return path


def full_structure_metrics(record, atom14, atom14_mask, *, raw_structure_root=None):
    """Check generated observed atoms in the global frame; solvent is separate.

    No native binder coordinates are inspected. Other protein chains, ligands,
    ions and nucleic acids are retained. This is a severe-clash screen, not a
    binding-energy or all-atom validity certificate.
    """
    path = resolve_source(record, raw_structure_root)
    structure = gemmi.read_structure(str(path), merge_chain_parts=False)
    index = record.get("selected_model_index", 0)
    if type(index) is not int or not 0 <= index < len(structure):
        raise ValueError("selected source model_index is out of range")
    model = structure[index]
    if record["peptide_chain_id"] not in {chain.name for chain in model}:
        raise ValueError("source peptide chain is absent from selected model")
    if record["receptor_chain_id"] not in {chain.name for chain in model}:
        raise ValueError("source receptor chain is absent from selected model")
    atoms, waters = [], []
    for chain in model:
        if chain.name == record["peptide_chain_id"]:
            continue
        for residue in chain:
            observed = [a for a in residue if a.occ > 0 and not a.element.is_hydrogen]
            alternates = {a.altloc for a in observed if a.altloc not in {"\x00", " "}}
            if alternates:
                shared = [a for a in observed if a.altloc in {"\x00", " "}]

                def score(alt):
                    selected = [a for a in observed if a.altloc == alt]
                    names = {a.name for a in shared + selected}
                    return (
                        -len(names & {"N", "CA", "C"}),
                        -sum(a.occ for a in selected),
                        alt != "A",
                        alt,
                    )

                alt = min(alternates, key=score)
                observed = [a for a in observed if a.altloc in {"\x00", " ", alt}]
            target = waters if residue.is_water() else atoms
            target.extend([[a.pos.x, a.pos.y, a.pos.z] for a in observed])
    xyz = atom14.detach().float().cpu()
    mask = atom14_mask.detach().bool().cpu()
    xyz = xyz + torch.as_tensor(record["site_origin"], dtype=torch.float32).reshape(1, 1, 3)
    generated = xyz[mask]
    if not len(generated) or not torch.isfinite(generated).all():
        raise ValueError("generated atoms must be nonempty and finite")
    if not atoms:
        raise ValueError("no observed nonwater environment atoms")

    def nearest_to(points):
        nearest = torch.full((len(generated),), torch.inf)
        # Bound memory for long receptors or assemblies.
        points = torch.tensor(points, dtype=torch.float32)
        if not torch.isfinite(points).all():
            raise ValueError("nonfinite full environment coordinates")
        for chunk in points.split(2048):
            nearest = torch.minimum(
                nearest,
                torch.cdist(generated, chunk, compute_mode="donot_use_mm_for_euclid_dist").amin(-1),
            )
        return nearest

    nearest = nearest_to(atoms)
    metrics = dict(
        evaluated=1.0,
        nonwater_atom_count=float(len(atoms)),
        generated_observed_atom_count=float(len(generated)),
        minimum_nonwater_distance_angstrom=float(nearest.min()),
        observed_atom_nonwater_clash_fraction=float((nearest < 2.0).float().mean()),
        observed_atom_nonwater_clash_free=float(not (nearest < 2.0).any()),
        water_atom_count=float(len(waters)),
    )
    if waters:
        metrics["observed_atom_water_overlap_fraction"] = float(
            (nearest_to(waters) < 2.0).float().mean()
        )
    return {"full_structure_quality." + key: value for key, value in metrics.items()}
