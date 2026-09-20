"""Exploratory generation diagnostics; backbone proxies are not binding predictions."""

import math

import torch

from apexgen.joint_v2.sampling.base import sample_base_state
from apexgen.joint_v2.diagnostics.geometry_path_diagnostics import geometry_anatomy
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone
from apexgen.joint_v2.contracts.state import JointFlowState


def stable_batch_base(batch, singles, *, seed, bank, index, seed_fn):
    """Sample each unpadded example before padding, independent of evaluation grouping."""
    device = batch.condition.residue_mask.device
    x = batch.condition.pocket_translation.clone()
    rot = batch.condition.pocket_rotation.clone()
    z = torch.zeros(*batch.condition.layout, 20, device=device)
    for b, sid in enumerate(batch.sample_ids):
        original = singles[sid]
        g = torch.Generator(device=device).manual_seed(seed_fn(seed, sid, index, bank))
        base = sample_base_state(original.condition, generator=g)
        n = base.layout[1]
        x[b, :n] = base.translation[0]
        rot[b, :n] = base.rotation[0]
        z[b, :n] = base.sequence_logits[0]
    return JointFlowState(x, rot, z)


def rank_prior(batches, maximum_length=24):
    counts = torch.ones(maximum_length, 20)
    for batch in batches:
        for b in range(len(batch.sample_ids)):
            aa = batch.targets.endpoint_aatype[b, batch.condition.peptide_mask[b]].cpu()
            counts[torch.arange(len(aa)), aa] += 1
    return counts / counts.sum(-1, keepdim=True)


def prior_endpoint(state, time, mask, prior):
    z = torch.zeros_like(state.sequence_logits)
    a = math.log(381.0)
    for b in range(len(mask)):
        n = int(mask[b].sum())
        t = time[b]
        probabilities = (
            prior[:n].to(z.device).log()
            + a * t / (1 - t).square() * state.sequence_logits[b, mask[b]]
        ).softmax(-1)
        z[b, mask[b]] = a * (probabilities - 1 / 20)
    return z


def edit_similarity(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return 1 - previous[-1] / max(len(a), len(b), 1)


def generation_rows(state, batch, training_sequences):
    """N/CA/C-only chain and clash proxies; native-dependent metrics kept separate."""
    rows = geometry_anatomy(state, batch)
    backbone = reconstruct_backbone(state)
    for b, row in enumerate(rows):
        p = batch.condition.peptide_mask[b]
        k = batch.condition.pocket_mask[b]
        xyz = backbone[b, p]
        ca = state.translation[b, p]
        aa = state.sequence_logits[b, p].argmax(-1).tolist()
        target = batch.targets.endpoint_aatype[b, p]
        atoms = batch.condition.pocket_atom_xyz[b, k][batch.condition.pocket_atom_mask[b, k]]
        cross = torch.cdist(xyz.flatten(0, 1), atoms)
        ca_dist = torch.cdist(ca, atoms).min(-1).values
        cn = (xyz[:-1, 2] - xyz[1:, 0]).norm(dim=-1)
        cn_mae = float((cn - 1.329).abs().mean())
        clash = float((cross.min(-1).values < 2.0).float().mean())
        contact = float((ca_dist < 5.0).float().mean())
        n = len(ca)
        res = torch.arange(n, device=ca.device).repeat_interleave(3)
        nonlocal_mask = (res[:, None] - res[None, :]).abs() > 1
        nonlocal_dist = torch.cdist(xyz.flatten(0, 1), xyz.flatten(0, 1))
        self_clash = bool(((nonlocal_dist < 2.0) & nonlocal_mask).any())
        row.update(
            sequence_accuracy=float(
                (state.sequence_logits[b, p].argmax(-1) == target).float().mean()
            ),
            sequence_logit_rmse=float(
                (
                    state.sequence_logits[b, p]
                    - batch.targets.endpoint_state(batch.condition).sequence_logits[b, p]
                )
                .square()
                .mean()
                .sqrt()
            ),
            cn_absolute_mae_angstrom=cn_mae,
            ca_spacing_mae_angstrom=float(((ca[1:] - ca[:-1]).norm(dim=-1) - 3.8).abs().mean()),
            peptide_atom_clash_fraction_2A=clash,
            contact_residue_fraction_5A=contact,
            mean_CA_to_pocket_heavy_distance=float(ca_dist.mean()),
            nonlocal_backbone_clash=self_clash,
            backbone_valid_proxy=(
                cn_mae <= 0.15 and clash <= 0.05 and contact >= 0.5 and not self_clash
            ),
            nearest_train_edit_similarity=max(edit_similarity(aa, x) for x in training_sequences),
            exact_training_sequence_match=aa in training_sequences,
            generated_aatype=aa,
        )
    return rows


def make_schedule(size, batch_size, steps, seed):
    if size % batch_size:
        raise ValueError("equal-exposure sampler needs divisible training size")
    g = torch.Generator().manual_seed(seed + 2)
    rows = []
    while len(rows) < steps:
        rows.extend(torch.randperm(size, generator=g).reshape(-1, batch_size).tolist())
    return rows[:steps]
