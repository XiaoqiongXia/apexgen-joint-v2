"""Exploratory early categorical supervision and covalent endpoint output layer.

Existing checkpoint tensors initialize the underlying model only. The chain
output contract is new and must be recorded with the exploratory checkpoint.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F
from apexgen.joint_v2.geometry.chain_projection import internal_from_backbone, build_chain, initial_pose
from apexgen.joint_v2.sampling.flow import endpoint_step
from apexgen.joint_v2.model.variants.real_sequence_transfer import RealSequenceTransferModel, sequence_output
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.shared.training.precision import network_autocast


def decode_with_scores(model, state, time, observation, encoding):
    """Expose posterior scores or direct endpoint readout logits, matching forward."""
    if not isinstance(model, RealSequenceTransferModel):
        raise TypeError('Expected the exploratory real sequence transfer model')
    pred = super(RealSequenceTransferModel, model).decode(
        state, time, observation, encoding, return_intermediates=False)
    current = observation.clamp(state)
    with torch.autocast(device_type=state.translation.device.type, enabled=False):
        sequence, scores = sequence_output(pred.sequence_logits, current.sequence_logits, time,
            mode=model.sequence_endpoint_mode, power=model.sequence_time_power)
        sequence = torch.where(observation.pocket.peptide_mask[..., None], sequence, 0.)
    return replace(pred, sequence_logits=sequence), scores


def early_categorical_loss(scores, labels, mask):
    """Stable CE on pre-softmax scores, with labels detached and padding masked."""
    if not scores or labels.shape != mask.shape:
        raise ValueError('Expected nonempty scores and matched labels/mask')
    weight = mask.float()
    losses = [((F.cross_entropy(s.float().transpose(1, 2), labels.detach().clamp(0, 19),
               reduction='none') * weight).sum(-1) / weight.sum(-1)) for s in scores]
    return torch.stack(losses).mean(0)


def covalent_output(state, condition):
    """Extract torsions, build exact bonds/angles/planes, fit to own generated atoms.

    The cis/trans choice before generated Pro is detached; non-Pro is trans.
    This output layer guarantees local covalent geometry, not clash freedom.
    No teacher or native coordinates are accepted. SVD is differentiated.
    """
    raw = reconstruct_backbone(state)
    translation, rotation = state.translation.clone(), state.rotation.clone()
    for i, mask in enumerate(condition.peptide_mask):
        bb = raw[i, mask][None]
        aa = state.sequence_logits[i, mask].detach().argmax(-1)[None]
        valid = torch.ones_like(aa, dtype=torch.bool)
        torsions, omega = internal_from_backbone(bb, aa)
        chain = build_chain(torsions, aa, omega, valid)
        r, t = initial_pose(chain, bb, valid)
        bb = (chain @ r[:, None] + t[:, None, None])[0]
        x = F.normalize(bb[:, 2] - bb[:, 1], dim=-1)
        y = bb[:, 0] - bb[:, 1]
        y = F.normalize(y - (x * y).sum(-1, keepdim=True) * x, dim=-1)
        rotation[i, mask] = torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)
        translation[i, mask] = bb[:, 1]
    return JointFlowState(translation, rotation, state.sequence_logits)


def rollout(model, base, observation, *, chain=False, collect_scores=True):
    """Full 20-step true rollout; output layer is applied once after integration."""
    state = observation.clamp(base)
    with network_autocast(state.translation.device, 'bfloat16'):
        encoding = model.encode_complex(observation)
    early_scores = []
    for step in range(20):
        time = torch.full((state.layout[0],), step / 20, device=state.translation.device)
        with network_autocast(state.translation.device, 'bfloat16'):
            pred, scores = decode_with_scores(model, state, time, observation, encoding)
        if collect_scores and step <= 6:
            early_scores.append(scores)
        state = observation.clamp(endpoint_step(state, pred, condition=observation.pocket,
            time=time, next_time=torch.full_like(time, (step + 1) / 20),
            sequence_time_power=model.sequence_time_power))
    raw = state
    return (covalent_output(raw, observation.pocket) if chain else raw), raw, early_scores
