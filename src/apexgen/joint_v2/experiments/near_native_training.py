"""Independent distribution and time-normalized supervision continuation controls."""
import hashlib

import torch

from apexgen.joint_v2.experiments.fixed_sequence_scale import FixedSequenceScaleModel
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.model.task_factorization import task_losses, task_objective_weights, task_path

CONTROLS = ('baseline', 'near_native', 'relative_update')
CONTRACT = dict(
    schema='apexgen.joint_v2.near_native_training.v1',
    architecture='Unchanged FixedSequenceScaleModel G_s, original accumulating frames and 1-t gate',
    parent='Original n128 baseline raw step32000; restore optimizer, EMA and all CPU/CUDA/base RNG states',
    near_native='t>=0.75: deterministic fair hash bit selects native frames or native-centered translation noise with original path rotations; all other inputs and all losses unchanged',
    relative_update='Original path and t sampler; multiply final translation and geodesic rotation squared losses by (1-T)/(1-t)^2, T=0.95; other losses unchanged',
    evaluation='Original path, clean-input preservation, paired perturbation response and complete native-sequence-fixed rollout; no native frames in encoder',
    compatibility='Exploratory training-control identity; not formal or generically checkpoint-compatible',
)
CONTRACT_SHA256 = canonical_sha256(CONTRACT)


class NearNativeTrainingModel(FixedSequenceScaleModel):
    def __init__(self, config, control, code_seed=20260906):
        if control not in CONTROLS: raise ValueError(control)
        super().__init__(config, code_seed)
        self.control = control


def build_fit_model(config, task, protocol):
    if task != 'G_s': raise ValueError('Requires native-sequence-fixed geometry task')
    return NearNativeTrainingModel(config, protocol['control'], protocol['position_code_seed'])


def augmentation_choices(seed, step, sample_ids, device):
    # Counter-based independent stream; never consumes the paired base/t RNG.
    bits = [hashlib.sha256(f'near_native_v1:{seed}:{step}:{sid}'.encode()).digest()[0] & 1 for sid in sample_ids]
    return torch.tensor(bits, device=device, dtype=torch.bool)


def training_state(base, batch, observation, time, control, *, seed, step):
    state = task_path(base, batch, observation, time)
    family = torch.zeros_like(time, dtype=torch.long)
    if control != 'near_native':
        if control not in CONTROLS: raise ValueError(control)
        return state, family
    high = time >= .75
    centered = augmentation_choices(seed, step, batch.sample_ids, time.device)
    target = observation.clamp(batch.targets.endpoint_state(batch.condition))
    p = batch.condition.peptide_mask
    noisy_x = target.translation + (1-time)[:,None,None] * base.translation
    proposal_x = torch.where(centered[:,None,None], noisy_x, target.translation)
    proposal_r = torch.where(centered[:,None,None,None], state.rotation, target.rotation)
    update = high[:,None] & p
    out = JointFlowState(torch.where(update[...,None], proposal_x, state.translation),
                         torch.where(update[...,None,None], proposal_r, state.rotation), state.sequence_logits)
    family = torch.where(high, torch.where(centered, 2, 1), family)
    return out, family


def relative_update_weight(time, maximum=.95):
    """E[w]=1 for t~Uniform[0,T); finite on the predeclared evaluation grid.

    This changes time weighting, not the truth target or model parameterization.
    Translation endpoint error / (1-t) is exactly residual-update error.
    Rotation uses the existing intrinsic endpoint geodesic error with the same
    scale, not a claimed equality to finite input-tangent vector differences.
    """
    if not 0 < maximum < 1: raise ValueError('maximum must be between zero and one')
    if bool(((time < 0) | (time > maximum + 1e-6)).any()): raise ValueError('Time outside declared support')
    return (1-maximum) / (1-time.float()).square()


def control_weights(task, control):
    if control not in CONTROLS: raise ValueError(control)
    return task_objective_weights(task)


def control_losses(prediction, batch, task, control, time):
    values = task_losses(prediction, batch, task)
    if control not in CONTROLS: raise ValueError(control)
    values['unweighted_total'] = values['total']
    values['unweighted_final_translation'] = values['final_translation']
    values['unweighted_final_rotation'] = values['final_rotation']
    weight = relative_update_weight(time) if control == 'relative_update' else torch.ones_like(time, dtype=torch.float32)
    values['relative_update_weight'] = weight
    if control == 'relative_update':
        values['final_translation'] = values['final_translation'] * weight
        values['final_rotation'] = values['final_rotation'] * weight
        values['total'] = sum(values[k] * w for k,w in task_objective_weights(task).items())
    return values
