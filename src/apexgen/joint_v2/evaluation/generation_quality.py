"""Generation-only runtime evaluation: accepts generated atoms and condition, never targets."""

from collections import defaultdict
from itertools import combinations
import math

import numpy as np
import torch

from apexgen.shared.geometry.joint_residue_constants import (
    AA3_TO_INDEX, BETWEEN_RES_BOND_LENGTH_C_N,
    BETWEEN_RES_COS_ANGLES_CA_C_N, BETWEEN_RES_COS_ANGLES_C_N_CA,
    ATOM14_MASK, ATOM14_DISTANCE_LOWER_BOUND,
)
from apexgen.joint_v2.evaluation.generation_evaluation import ca_pair_distances


PREFIX = 'generation_quality.'
SCHEMA = 'apexgen.joint_v2.generation_quality.v2'
# Descriptive proxies, not calibrated binding or all-atom validity criteria.
POLICY = dict(cn_mae_max_angstrom=.15, backbone_context_clash_fraction_max=.05,
              clash_distance_angstrom=2., contact_distance_angstrom=5.,
              core_contact_fraction_min=.5, core_contact_residues_min=2,
              ca_spacing_reference_angstrom=3.8)


def metric_groups(metrics):
    """Preserve legacy mixed diagnostics but exclude them from generation scoring."""
    return {
        'generation_quality': {k[len(PREFIX):]: v for k, v in metrics.items() if k.startswith(PREFIX)},
        'full_structure_quality': {k.removeprefix('full_structure_quality.'): v for k, v in metrics.items() if k.startswith('full_structure_quality.')},
        'reconstruction_diagnostics': {k: v for k, v in metrics.items() if not k.startswith((PREFIX, 'full_structure_quality.'))},
    }


def _angle(a, b, c):
    first, second = a - b, c - b
    return torch.acos((torch.nn.functional.normalize(first, dim=-1)
                       * torch.nn.functional.normalize(second, dim=-1)).sum(-1).clamp(-1, 1))


@torch.no_grad()
def generation_metrics(backbone, atom14, atom14_mask, aatype, condition, row):
    """Evaluate unpadded generated atoms. All masks come from generation/condition."""
    bb = backbone.float()
    heavy = atom14.float()
    length = len(bb)
    if length < 1 or bb.shape != (length, 3, 3) or heavy.shape != (length, 14, 3):
        raise ValueError('Invalid generated atom layout')
    if atom14_mask.shape != (length, 14) or not bool(atom14_mask[:, :3].all()):
        raise ValueError('Generated backbone atoms must be present')
    if not bool(torch.isfinite(bb).all() and torch.isfinite(heavy[atom14_mask]).all()):
        raise ValueError('Nonfinite generated atoms')
    context = condition.pocket_mask[row]
    context_xyz = condition.pocket_atom_xyz[row, context].float()
    context_mask = condition.pocket_atom_mask[row, context]
    context_core = condition.pocket_core_mask[row, context]
    owners = torch.arange(len(context_xyz), device=bb.device)[:, None].expand_as(context_mask)[context_mask]
    context_atoms = context_xyz[context_mask]
    if not context_atoms.numel():
        raise ValueError('No receptor context atoms for generation evaluation')
    cross_bb = torch.cdist(bb.reshape(-1, 3), context_atoms)
    clash = float((cross_bb.min(-1).values < POLICY['clash_distance_angstrom']).float().mean())
    residue = torch.arange(length, device=bb.device).repeat_interleave(3)
    nonlocal_mask = (residue[:, None] - residue[None, :]).abs() > 1
    distances = torch.cdist(bb.reshape(-1, 3), bb.reshape(-1, 3))
    self_clash = bool(((distances < POLICY['clash_distance_angstrom']) & nonlocal_mask).any())
    result = dict(backbone_context_clash_atom_fraction=clash,
                  backbone_nonlocal_self_clash=float(self_clash),
                  geometry_proxy_available=float(length >= 2),
                  core_residue_count=float(context_core.sum()))
    if length >= 2:
        cn = (bb[:-1, 2] - bb[1:, 0]).norm(dim=-1)
        references = torch.as_tensor(BETWEEN_RES_BOND_LENGTH_C_N, device=bb.device)
        expected = references[(aatype[1:] == AA3_TO_INDEX['PRO']).long()]
        cn_mae = float((cn - expected).abs().mean())
        result.update(cn_bond_mae_to_ideal_angstrom=cn_mae,
                      cn_bond_max_angstrom=float(cn.max()),
                      cn_bond_over_2A_fraction=float((cn > 2).float().mean()),
                      ca_spacing_mae_to_3_8A=float(((bb[1:, 1] - bb[:-1, 1]).norm(dim=-1) - 3.8).abs().mean()))
        for name, angles, cos_ref in [
            ('ca_c_n', _angle(bb[:-1, 1], bb[:-1, 2], bb[1:, 0]), BETWEEN_RES_COS_ANGLES_CA_C_N),
            ('c_n_ca', _angle(bb[:-1, 2], bb[1:, 0], bb[1:, 1]), BETWEEN_RES_COS_ANGLES_C_N_CA),
        ]:
            result[f'{name}_angle_mae_to_ideal_degrees'] = float(
                (angles - math.acos(float(cos_ref[0]))).abs().mean() * 180 / math.pi)
        result['geometry_proxy'] = float(cn_mae <= POLICY['cn_mae_max_angstrom']
                                         and clash <= POLICY['backbone_context_clash_fraction_max']
                                         and not self_clash)
    cross_heavy = torch.cdist(heavy.reshape(-1, 3), context_atoms)
    cross_heavy[~atom14_mask.reshape(-1)] = float('inf')
    observed_clash = (cross_heavy.min(-1).values[atom14_mask.reshape(-1)] < POLICY['clash_distance_angstrom'])
    expected_mask = torch.as_tensor(ATOM14_MASK, device=heavy.device, dtype=torch.bool)[aatype]
    complete = bool((atom14_mask == expected_mask).all())
    owner = torch.arange(length, device=heavy.device)[:, None].expand_as(atom14_mask)[atom14_mask]
    slot = torch.arange(14, device=heavy.device)[None].expand_as(atom14_mask)[atom14_mask]
    atoms = heavy[atom14_mask]
    distances = torch.cdist(atoms, atoms)
    inter = torch.triu(owner[:, None] != owner[None, :], diagonal=1)
    peptide_bond = (owner[:, None] + 1 == owner[None, :]) & (slot[:, None] == 2) & (slot[None, :] == 0)
    inter_clash = ((distances < POLICY['clash_distance_angstrom']) & inter & ~peptide_bond).any()
    lower = torch.as_tensor(ATOM14_DISTANCE_LOWER_BOUND, device=heavy.device, dtype=torch.float32)[aatype]
    intra_valid = atom14_mask[:, :, None] & atom14_mask[:, None, :] & torch.triu(
        torch.ones(14, 14, device=heavy.device, dtype=torch.bool), diagonal=1)
    intra_violation = ((torch.cdist(heavy, heavy) < lower - 1e-5) & intra_valid).any()
    observed_free = not bool(observed_clash.any() or inter_clash or intra_violation)
    result.update(all_atom_geometry_available=float(complete),
                  generated_observed_atom_context_clash_fraction=float(observed_clash.float().mean()),
                  generated_observed_atom_clash_or_violation_free=float(observed_free))
    if 'geometry_proxy' in result:
        result['geometry_proxy'] *= float(observed_free)
        if complete:
            result['all_atom_geometry_proxy'] = result['geometry_proxy']
    contacts = cross_heavy.reshape(length, 14, -1).min(1).values < POLICY['contact_distance_angstrom']
    result['context_heavy_contact_fraction'] = float(contacts.any(-1).float().mean())
    core_atoms = context_core[owners]
    core_observed = context_core & context_mask.any(-1)
    # Missing core coordinates cannot silently become a successful or failed site prediction.
    available = bool(context_core.any() and (core_observed == context_core).all())
    result['core_site_proxy_available'] = float(available)
    if available:
        core_contacts = contacts[:, core_atoms]
        contacted = int(owners[core_atoms][core_contacts.any(0)].unique().numel())
        fraction = float(core_contacts.any(-1).float().mean())
        result.update(core_heavy_contact_fraction=fraction, contacted_core_residues=float(contacted),
                      core_residue_coverage=contacted / int(context_core.sum()),
                      core_site_proxy=float(fraction >= POLICY['core_contact_fraction_min']
                                            and contacted >= POLICY['core_contact_residues_min']))
        if 'geometry_proxy' in result:
            result['geometry_and_core_site_proxy'] = result['geometry_proxy'] * result['core_site_proxy']
    return {PREFIX + k: v for k, v in result.items()}


def summarize_generation_candidates(candidates):
    """Equal target weighting; pair diversity only within the same target/length."""
    if not candidates:
        raise ValueError('Empty generation panel')
    by_target = defaultdict(list)
    identities = set()
    for candidate in candidates:
        identity = candidate.sample_id, candidate.base_index
        if identity in identities:
            raise ValueError('Duplicate target/base in generation panel')
        identities.add(identity)
        by_target[candidate.sample_id].append(candidate)
    per_target = []
    for sid, group in sorted(by_target.items()):
        quality = [metric_groups(c.metrics)['generation_quality'] for c in group]
        if any(not q for q in quality):
            raise ValueError('Generation quality missing from candidate')
        record = dict(sample_id=sid, candidate_count=len(group),
                      metric_means={}, metric_evaluated_candidates={},
                      proxy_counts={},
                      cofold_evaluated_candidates=0, cofold_status='not_evaluated')
        for key in sorted(set().union(*(q.keys() for q in quality))):
            values = [q[key] for q in quality if key in q]
            record['metric_means'][key] = float(np.mean(values))
            record['metric_evaluated_candidates'][key] = len(values)
        for key in ['geometry_proxy', 'core_site_proxy', 'geometry_and_core_site_proxy']:
            values = [q[key] for q in quality if key in q]
            passed = sum(v == 1 for v in values)
            record['proxy_counts'][key] = dict(passed=passed, evaluated=len(values),
                                               not_evaluated=len(group) - len(values))
            record[key + '_any'] = True if passed else (False if len(values) == len(group) else None)
        record['diversity'] = {}
        for label, selected in [
            ('all', list(range(len(group)))),
            ('geometry', [i for i, q in enumerate(quality) if q.get('geometry_proxy') == 1]),
            ('geometry_and_core_site', [i for i, q in enumerate(quality) if q.get('geometry_and_core_site_proxy') == 1]),
        ]:
            pairs = []
            for i, j in combinations(selected, 2):
                a, b = group[i], group[j]
                if a.backbone.shape != b.backbone.shape:
                    raise ValueError('Within-target diversity requires fixed length')
                # CA Kabsch requires >=3 points; short peptides retain sequence diversity only.
                common, internal = (ca_pair_distances(a.backbone[:, 1], b.backbone[:, 1])
                                    if len(a.backbone) >= 3 else (None, None))
                seq = float((a.aatype != b.aatype).float().mean())
                pairs.append((common, internal, seq))
            metrics = {}
            for j, name in enumerate(['ca_common_frame_rmsd', 'ca_internal_shape_rmsd', 'sequence_hamming_fraction']):
                values = [p[j] for p in pairs if p[j] is not None]
                metrics[name] = float(np.mean(values)) if values else None
            record['diversity'][label] = dict(candidate_count=len(selected), pair_count=len(pairs), **metrics)
        per_target.append(record)
    names = sorted(set().union(*(r['metric_means'].keys() for r in per_target)))
    aggregate = {}
    for name in names:
        values = [r['metric_means'][name] for r in per_target if name in r['metric_means']]
        aggregate[name] = dict(target_mean=float(np.mean(values)), evaluated_targets=len(values))
    coverage = {}
    for key in ['geometry_proxy', 'core_site_proxy', 'geometry_and_core_site_proxy']:
        values = [r[key + '_any'] for r in per_target]
        passed = sum(v is True for v in values)
        unresolved = sum(v is None for v in values)
        coverage[key] = dict(targets_with_at_least_one=passed, unresolved_targets=unresolved,
                             total_targets=len(values), observed_lower_fraction=passed / len(values),
                             possible_upper_fraction=(passed + unresolved) / len(values))
    diversity = {}
    for label in ['all', 'geometry', 'geometry_and_core_site']:
        diversity[label] = {}
        for metric in ['ca_common_frame_rmsd', 'ca_internal_shape_rmsd', 'sequence_hamming_fraction']:
            values = [r['diversity'][label][metric] for r in per_target if r['diversity'][label][metric] is not None]
            diversity[label][metric] = dict(target_mean=float(np.mean(values)) if values else None,
                                            evaluated_targets=len(values), total_targets=len(per_target))
    return dict(schema=SCHEMA, policy=POLICY, native_used_for_generation_quality=False,
                reference='generated_atoms_and_condition_only', candidate_count=len(candidates),
                target_count=len(by_target), target_means=aggregate, per_target=per_target,
                target_coverage=coverage, diversity_target_means=diversity,
                cofold=dict(status='not_evaluated', evaluated_candidates=0),
                full_structure_quality=dict(
                    evaluated_candidates=sum('full_structure_quality.evaluated' in c.metrics for c in candidates),
                    observed_atom_clash_free_candidates=sum(c.metrics.get('full_structure_quality.observed_atom_nonwater_clash_free') == 1 for c in candidates),
                    geometry_core_and_full_environment_passed_candidates=sum(
                        c.metrics.get(PREFIX+'geometry_and_core_site_proxy') == 1
                        and c.metrics.get('full_structure_quality.observed_atom_nonwater_clash_free') == 1
                        for c in candidates),
                    all_atom_validity_certified=False),
                selection=dict(native_recovery_gate=False, scalar_ranking_score=None,
                               proxy_calibration='descriptive_only'))
