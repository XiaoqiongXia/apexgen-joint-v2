"""Read-only probes of explicit encoder geometry and direct IPA pocket frames.

Feature erasure is an OOD sensitivity test, not an unconditional model. Atom/angle
availability, residue identity, role, chain and position metadata are retained.
IPA perturbations affect only its pocket rigids; endpoint clamps stay native.
"""

from contextlib import contextmanager

import torch

from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone
from apexgen.joint_v2.model.structure_module import Rigid


@contextmanager
def encoder_geometry_erasure(encoder):
    def pair_hook(module, args):
        x = args[0].clone()
        if x.shape[-1] != 64:
            raise ValueError("geometry probe requires the full 64-channel pair contract")
        x[..., :28] = 0  # CA RBF, relative rotation and translation
        x[..., 35:52] = 0  # CB distance RBF including the beyond-range bucket
        x[..., 57:63] = 0  # Rosetta angle sin/cos; retain availability masks
        return (x,)

    def atom_hook(module, args):
        x = args[0].clone()
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, 4)
        x[..., :3] = 0
        return (x.reshape(shape),)

    def angle_hook(module, args):
        x = args[0].clone()
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, 3)
        x[..., :2] = 0
        return (x.reshape(shape),)

    hooks = []
    try:
        for mod, fn in [
            (encoder.pair_projection[0], pair_hook),
            (encoder.atom_projection, atom_hook),
            (encoder.backbone_angle_projection[0], angle_hook),
            (encoder.sidechain_angle_projection[0], angle_hook),
        ]:
            hooks.append(mod.register_forward_pre_hook(fn))
        yield
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def ipa_pocket_perturbation(ipa, pocket_mask, *, kind, translation_scale=10.0):
    if kind not in {"shift6", "rotate90"}:
        raise ValueError(kind)

    def hook(module, args):
        single, pair, rigid, mask = args
        with torch.autocast(device_type=rigid.translation.device.type, enabled=False):
            return transform(single, pair, rigid, mask)

    def transform(single, pair, rigid, mask):
        xyz, rot = rigid.translation.clone(), rigid.rotation.clone()
        if kind == "shift6":
            xyz[pocket_mask] += xyz.new_tensor([6.0 / translation_scale, 0.0, 0.0])
        else:
            q = xyz.new_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            for b in range(len(xyz)):
                p = pocket_mask[b]
                center = xyz[b, p].mean(0)
                xyz[b, p] = (xyz[b, p] - center) @ q.T + center
                rot[b, p] = q @ rot[b, p]
        return single, pair, Rigid(rot, xyz), mask

    handle = ipa.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def aligned_rmsd(x, y):
    """Ordered Kabsch RMSD; forbid reflections, no residue reassignment."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _, vh = torch.linalg.svd(xc.T @ yc)
    correction = torch.eye(3, device=x.device, dtype=x.dtype)
    correction[-1, -1] = torch.linalg.det(u @ vh)
    fit = xc @ (u @ correction @ vh)
    return (fit - yc).square().sum(-1).mean().sqrt()


def geometry_anatomy(state, batch):
    target = batch.targets.endpoint_state(batch.condition)
    xyz, true_xyz = reconstruct_backbone(state), reconstruct_backbone(target)
    rows = []
    for b, sid in enumerate(batch.sample_ids):
        p, k = batch.condition.peptide_mask[b], batch.condition.pocket_mask[b]
        x, y = xyz[b, p], true_xyz[b, p]
        atom_mask = batch.targets.backbone_atom_mask[b, p]
        ca, true_ca = state.translation[b, p], target.translation[b, p]
        dist = torch.cdist(ca, batch.condition.pocket_translation[b, k])
        true_dist = torch.cdist(true_ca, batch.condition.pocket_translation[b, k])
        upper = torch.triu(torch.ones(len(ca), len(ca), device=ca.device, dtype=torch.bool), 1)
        derror = (torch.cdist(ca, ca) - torch.cdist(true_ca, true_ca))[upper]
        relative = state.rotation[b, p].transpose(-1, -2) @ target.rotation[b, p]
        angle = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        cn = (x[:-1, 2] - x[1:, 0]).norm(dim=-1)
        true_cn = (y[:-1, 2] - y[1:, 0]).norm(dim=-1)
        rows.append(
            dict(
                sample_id=sid,
                backbone_rmsd=float((x[atom_mask] - y[atom_mask]).square().sum(-1).mean().sqrt()),
                aligned_backbone_rmsd=float(aligned_rmsd(x[atom_mask], y[atom_mask])),
                centroid_error=float((ca.mean(0) - true_ca.mean(0)).norm()),
                internal_CA_distance_rmse=float(derror.square().mean().sqrt()),
                CN_MAE=float((cn - true_cn).abs().mean()),
                CA_RMSD=float((ca - true_ca).square().sum(-1).mean().sqrt()),
                rotation_degrees=float(torch.rad2deg(angle).mean()),
                per_position_CA_error=(ca - true_ca).norm(dim=-1).tolist(),
                per_position_rotation_degrees=torch.rad2deg(angle).tolist(),
                nearest_pocket_distance=dist.min(-1).values.tolist(),
                nearest_pocket_index=dist.argmin(-1).tolist(),
                target_nearest_pocket_index=true_dist.argmin(-1).tolist(),
                per_position_contacts_8A=(dist < 8).sum(-1).tolist(),
                predicted_CA=ca.tolist(),
            )
        )
    return rows
