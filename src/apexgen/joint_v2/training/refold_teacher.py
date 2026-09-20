"""Screened offline structure teachers and actual-rollout suffix regularization."""
from contextlib import nullcontext

import numpy as np
import torch

from apexgen.joint_v2.sampling.flow import endpoint_step
from apexgen.joint_v2.training.refold_reward import geometry_loss
from apexgen.shared.training.precision import network_autocast


def teacher_to_condition(backbone, receptor_rotation, receptor_translation,
                         condition_to_raw_rotation, condition_to_raw_translation):
    """Receptor-align the prediction, then invert the receptor-only frame mapping."""
    raw = np.asarray(backbone) @ receptor_rotation + receptor_translation
    return (raw - condition_to_raw_translation) @ np.asarray(condition_to_raw_rotation).T


def select_supported_teacher(rows, *, maximum_rmsd=2.0):
    """Select an actual medoid, never average incompatible protein structures.

    Input rows have already passed confidence, geometry and site checks. Require
    support from both prediction seeds within a common-receptor-frame radius.
    """
    choices = []
    for row in rows:
        ca = np.asarray(row['generated_backbone'])[:, 1]
        distances = [float(np.sqrt(np.mean(np.sum(
            (ca - np.asarray(other['generated_backbone'])[:, 1]) ** 2, axis=-1)))) for other in rows]
        cluster = [other for other, distance in zip(rows, distances) if distance <= maximum_rmsd]
        if len({other['seed'] for other in cluster}) < 2:
            continue
        score = (-len(cluster), float(np.mean([d for d in distances if d <= maximum_rmsd])),
                 -row['plddt'], row['seed'], row['sample'])
        choices.append((score, row, cluster))
    if not choices:
        return None
    _, chosen, cluster = min(choices, key=lambda item: item[0])
    return dict(chosen, support=[dict(seed=r['seed'], sample=r['sample']) for r in cluster])


def rollout_with_suffix_grad(model, base, observation, *, suffix_steps=4):
    """Run the current model for all 20 solver steps; detach only the prefix.

    This is truncated backpropagation through actual sampling, not a native or
    teacher interpolation path. Encoder gradients are rebuilt for the suffix.
    """
    if not 1 <= suffix_steps <= 20:
        raise ValueError('suffix_steps must be within 1..20')
    state = observation.clamp(base)
    device = state.translation.device
    boundary = 20 - suffix_steps
    encoding = None
    prefix_detached = None
    for index in range(20):
        grad = index >= boundary
        with nullcontext() if grad else torch.no_grad():
            if index == 0 or index == boundary:
                if index == boundary:
                    state = state.detach()
                    prefix_detached = not state.translation.requires_grad and not state.sequence_logits.requires_grad
                with network_autocast(device, 'bfloat16'):
                    encoding = model.encode_complex(observation)
            time = torch.full((state.layout[0],), index / 20, device=device)
            with network_autocast(device, 'bfloat16'):
                pred = model.decode(state, time, observation, encoding, return_intermediates=False)
            state = observation.clamp(endpoint_step(state, pred, condition=observation.pocket,
                time=time, next_time=torch.full_like(time, (index + 1) / 20),
                sequence_time_power=getattr(model,'sequence_time_power',1.0)))
    return state, dict(solver_steps=20, gradient_steps=suffix_steps,
                       prefix_detached=prefix_detached,
                       final_translation_requires_grad=state.translation.requires_grad,
                       final_sequence_requires_grad=state.sequence_logits.requires_grad)


def rollout_constraints(state, condition):
    """Native-free final-sample chain/steric loss and strengthened core attraction."""
    # Treat Pro identity as a detached chemical lookup, not an incentive to mutate.
    aa = state.sequence_logits.detach().argmax(-1)
    parts = geometry_loss(state, condition, aa)
    geometric = parts['bond'] + parts['angle'] + parts['self_clash'] + parts['context_clash']
    # Site has its own weight in the runner; the old 0.1 factor is not inherited.
    return dict(parts, geometry=geometric)
