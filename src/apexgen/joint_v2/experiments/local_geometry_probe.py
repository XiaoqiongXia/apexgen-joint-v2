"""Frozen-model local response probes; native frames are diagnostic inputs only."""
import math

import torch

from apexgen.joint_v2.geometry.rotations import so3_exp, so3_log
from apexgen.joint_v2.contracts.state import JointFlowState


def perturb_native(state, peptide, family, amplitude, seed, sign=1):
    """Single complex; translation amplitudes are RMS angstrom, rotations degrees.

    Global rotations move the entire peptide about its CA centroid, with pocket
    fixed. Internal translations have zero centroid and zero infinitesimal rigid
    rotation component. Local-frame rotations hold all CA positions fixed.
    Internal probes need not preserve covalent geometry or training-path density.
    """
    if state.layout[0] != 1 or sign not in (-1, 1):
        raise ValueError('Expected one complex and sign +/-1')
    if family not in ('translation', 'rigid_rotation', 'internal_ca', 'local_rotation'):
        raise ValueError(family)
    p = peptide[0]
    n = int(p.sum())
    gen = torch.Generator(device=state.translation.device).manual_seed(seed)
    pos = state.translation[0, p]
    centered = pos - pos.mean(0)
    rotation = state.rotation[0, p]
    axis = torch.randn(3, generator=gen, device=pos.device)
    axis = axis / axis.norm()
    new_pos, new_rotation = pos.clone(), rotation.clone()
    if family == 'translation':
        new_pos = pos + sign * amplitude * axis
    elif family == 'rigid_rotation':
        q = so3_exp(axis * (sign * amplitude * math.pi / 180))
        new_pos = centered @ q.T + pos.mean(0)
        new_rotation = q @ rotation
    elif family == 'internal_ca':
        delta = torch.randn(n, 3, generator=gen, device=pos.device)
        delta = delta - delta.mean(0)
        axes = torch.eye(3, device=pos.device)
        # SVD projection removes all representable rigid rotational directions.
        basis = torch.stack([torch.cross(a.expand_as(centered), centered, dim=-1).flatten() for a in axes], -1)
        u, s, _ = torch.linalg.svd(basis.double(), full_matrices=False)
        u = u[:, s > s.max() * 1e-8]
        v = delta.double().flatten()
        delta = (v - u @ (u.T @ v)).reshape(n, 3).float()
        delta = delta / delta.square().sum(-1).mean().sqrt().clamp_min(1e-8)
        new_pos = pos + sign * amplitude * delta
    else:
        axes = torch.randn(n, 3, generator=gen, device=pos.device)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        new_rotation = rotation @ so3_exp(axes * (sign * amplitude * math.pi / 180))
    x, r = state.translation.clone(), state.rotation.clone()
    x[0, p], r[0, p] = new_pos, new_rotation
    return JointFlowState(x, r, state.sequence_logits.clone())


def error_vectors(state, target):
    return (state.translation - target.translation,
            so3_log(target.rotation.transpose(-1, -2) @ state.rotation))


def vector_stats(proposed, desired, mask):
    """Norm and directional metrics; undefined zero-denominator values stay NaN."""
    a, b = proposed * mask[..., None], desired * mask[..., None]
    aa, bb = a.square().sum((-1, -2)), b.square().sum((-1, -2))
    dot = (a * b).sum((-1, -2))
    nan = torch.full_like(dot, float('nan'))
    return {
        'cosine': torch.where((aa * bb) > 1e-16, dot / (aa * bb).sqrt().clamp_min(1e-16), nan),
        'projection': torch.where(bb > 1e-12, dot / bb.clamp_min(1e-12), nan),
        'norm_ratio': torch.where(bb > 1e-12, (aa / bb.clamp_min(1e-12)).sqrt(), nan),
    }


def response_metrics(state, prediction, target, clean_prediction, mask):
    """Absolute error, correction direction, and bias-subtracted endpoint response.

    Rotation response uses native-relative log coordinates, whereas correction
    direction uses the input frame's tangent coordinates. These are finite
    perturbation measurements, not an exact Jacobian or posterior oracle.
    """
    inp = error_vectors(state, target)
    out = error_vectors(prediction, target)
    clean = error_vectors(clean_prediction, target)
    count = mask.sum(-1)
    result = {}
    for name, vi, vo, vc, factor in zip(('ca', 'rotation'), inp, out, clean, (1., 180 / math.pi)):
        for label, v in [('input', vi), ('output', vo), ('clean_bias', vc), ('response', vo-vc)]:
            sq = v.square().sum(-1)
            result[f'{name}_{label}_rms'] = ((sq * mask).sum(-1) / count).sqrt() * factor
        for k, v in vector_stats(vo-vc, vi, mask).items():
            result[f'{name}_response_{k}'] = v
    updates = (prediction.translation-state.translation,
               so3_log(state.rotation.transpose(-1, -2) @ prediction.rotation))
    desired = (target.translation-state.translation,
               so3_log(state.rotation.transpose(-1, -2) @ target.rotation))
    for name, a, b in zip(('ca', 'rotation'), updates, desired):
        for k, v in vector_stats(a, b, mask).items():
            result[f'{name}_correction_{k}'] = v
    result['ca_improves'] = result['ca_output_rms'] < result['ca_input_rms'] - 1e-5
    result['rotation_improves'] = result['rotation_output_rms'] < result['rotation_input_rms'] - 1e-4
    return result
