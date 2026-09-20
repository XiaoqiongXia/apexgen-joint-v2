"""Exploratory offline refold reward: no gradient through the external predictor.

AF3 scores reweight joint flow matching of generated pairs. Geometry constraints
remain uniformly weighted, so a poor teacher cannot switch off chain repair.
"""
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from apexgen.shared.geometry.joint_residue_constants import (
    AA3_TO_INDEX, BETWEEN_RES_BOND_LENGTH_C_N,
    BETWEEN_RES_COS_ANGLES_CA_C_N, BETWEEN_RES_COS_ANGLES_C_N_CA,
)
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone


def af3_reward_weights(cases, rows, *, temperature=0.5, uniform_mix=0.25):
    """Target-normalized bounded weights, using every AF3 replicate, never Boltz.

    Positive weights express relative preference, NOT biological success.
    Each target has mean weight one; the uniform mixture limits concentration.
    """
    if temperature <= 0 or not 0 <= uniform_mix <= 1:
        raise ValueError('Invalid reward policy')
    grouped = defaultdict(list)
    identities = set()
    for case in cases:
        if case['kind'] != 'generated':
            continue
        if case['name'] in identities:
            raise ValueError('Duplicate candidate')
        identities.add(case['name'])
        matches = [r for r in rows if r['model'] == 'af3' and r['name'] == case['name']]
        expected = {(seed, sample) for seed in [20260908, 20260909] for sample in range(2)}
        if len(matches) != 4 or {(r['seed'], r['sample']) for r in matches} != expected:
            raise ValueError('Incomplete/duplicate AF3 replicates')
        costs = [r['peptide_aligned_CA_RMSD'] / 4
                 + r['receptor_aligned_peptide_CA_RMSD'] / 10
                 + 1 - r['peptide_pocket_contact_fraction']
                 + min(1, 5 * r['peptide_heavy_clash_fraction_2A']) for r in matches]
        if not np.isfinite(costs).all():
            raise ValueError('Nonfinite AF3 reward')
        grouped[case['sample_id']].append(dict(name=case['name'], sample_id=case['sample_id'],
            base_index=case['base_index'], cost=float(np.mean(costs)), replicate_costs=costs,
            absolute_successes=sum(bool(r['geometry_and_site_flag']) for r in matches)))
    result, summaries = [], []
    for sid, group in sorted(grouped.items()):
        costs = np.array([r['cost'] for r in group])
        p = np.exp(-(costs - costs.min()) / temperature)
        p /= p.sum()
        w = uniform_mix + (1 - uniform_mix) * len(group) * p
        for r, weight in zip(group, w):
            result.append(dict(r, weight=float(weight)))
        summaries.append(dict(sample_id=sid, candidates=len(group), mean_weight=float(w.mean()),
            min_weight=float(w.min()), max_weight=float(w.max()),
            effective_sample_size=float(w.sum() ** 2 / (w @ w)),
            absolute_successes=sum(r['absolute_successes'] for r in group)))
    if not result:
        raise ValueError('No generated candidates')
    return result, summaries


def generated_record(record, candidate):
    """Replace ALL peptide supervision with the generated pair (N/CA/C only).

    Legacy collation calls its atom field 'experimental_atom14'; here that field
    explicitly contains generated coordinates, never experimental sidechains.
    """
    bb = np.asarray(candidate['generated_backbone'], dtype=np.float32)
    aa = np.asarray(candidate['generated_aatype'], dtype=np.int64)
    n = len(aa)
    if bb.shape != (n, 3, 3) or n != record['peptide_length']:
        raise ValueError('Generated pair length mismatch')
    x = bb[:, 2] - bb[:, 1]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y = bb[:, 0] - bb[:, 1]
    y -= (x * y).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    rotation = np.stack([x, y, np.cross(x, y)], axis=-1)
    if not np.isfinite(rotation).all() or not np.isfinite(bb).all():
        raise ValueError('Degenerate generated frames')
    atom = np.zeros((n, 14, 3), dtype=np.float32)
    atom[:, :3] = bb
    mask = np.zeros((n, 14), dtype=bool)
    mask[:, :3] = True
    target = dict(aatype=aa, translation=bb[:, 1], rotation=rotation,
                  backbone_torsion=np.zeros((n, 3), dtype=np.float32),
                  experimental_atom14=atom, experimental_atom14_mask=mask)
    return {**record, 'joint_v2_target': target,
            'supervision_source': 'generated_pair_not_native_or_AF_coordinates'}


def geometry_loss(prediction, condition, generated_aatype):
    """Differentiable endpoint chain/steric/site losses; no native coordinates.

    CN references use the generated supervision sequence, so geometry alone does
    not create a spurious incentive to change Pro solely to alter bond length.
    """
    condition.validate_model_input()
    backbone = reconstruct_backbone(prediction)
    values = []
    for i, peptide in enumerate(condition.peptide_mask):
        bb = backbone[i, peptide]
        aa = generated_aatype[i, peptide]
        atoms = bb.reshape(-1, 3)
        refs = torch.as_tensor(BETWEEN_RES_BOND_LENGTH_C_N, device=bb.device)
        cn = (bb[:-1, 2] - bb[1:, 0]).norm(dim=-1)
        bond = (cn - refs[(aa[1:] == AA3_TO_INDEX['PRO']).long()]).square().mean()
        angle = bb.sum() * 0
        for a, b, c, ref in [
            (bb[:-1, 1], bb[:-1, 2], bb[1:, 0], BETWEEN_RES_COS_ANGLES_CA_C_N[0]),
            (bb[:-1, 2], bb[1:, 0], bb[1:, 1], BETWEEN_RES_COS_ANGLES_C_N_CA[0]),
        ]:
            cosine = (F.normalize(a - b, dim=-1) * F.normalize(c - b, dim=-1)).sum(-1)
            angle = angle + (cosine - float(ref)).square().mean()
        owners = torch.arange(len(bb), device=bb.device).repeat_interleave(3)
        nonlocal_mask = (owners[:, None] - owners[None, :]).abs() > 1
        distances = torch.cdist(atoms, atoms, compute_mode='donot_use_mm_for_euclid_dist')
        nearest = distances.masked_fill(~nonlocal_mask, float('inf')).min(-1).values
        self_clash = F.relu(2.2 - nearest).square().mean()
        context = condition.pocket_mask[i]
        xyz = condition.pocket_atom_xyz[i, context]
        mask = condition.pocket_atom_mask[i, context]
        d = torch.cdist(atoms, xyz[mask], compute_mode='donot_use_mm_for_euclid_dist')
        clash = F.relu(2.2 - d.min(-1).values).square().mean()
        core_mask = condition.pocket_core_mask[i, context, None].expand_as(mask)[mask]
        if not bool(core_mask.any()):
            raise ValueError('This bounded experiment requires an observed pocket core')
        core_distance = d[:, core_mask].reshape(len(bb), 3, -1).flatten(1).min(-1).values
        site = F.relu(core_distance - 5).square().mean() / 25
        values.append(torch.stack([bond, angle, self_clash, clash, site]))
    components = torch.stack(values)
    return dict(total=components[:, :4].sum(-1) + 0.1 * components[:, 4],
                bond=components[:, 0], angle=components[:, 1],
                self_clash=components[:, 2], context_clash=components[:, 3], site=components[:, 4])
