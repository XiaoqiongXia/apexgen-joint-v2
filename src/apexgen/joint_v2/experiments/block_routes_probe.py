"""Read-only decomposition and small interventions inside one shared block.

Affine decomposition freezes LayerNorm statistics at the observed activation.
It is an exact accounting identity (up to FP32 replay error), not a causal share.
Interventions instead recompute all downstream LayerNorm statistics.
"""
import torch
import torch.nn.functional as F
from apexgen.joint_v2.model.variants.dynamic_pair import pair_features

ROUTES = ('single_skip', 'ipa_message', 'norm1_bias', 'transition_message', 'norm2_head_bias')
CASES = ('baseline', 'skip_initial', 'ipa_single_initial', 'ipa_q_initial', 'transition_zero') + tuple(
    f'{route}_{sign}' for route in ('skip', 'ipa', 'transition', 'q') for sign in ('minus', 'plus'))


def block(sm, single, q, rigid, pair, condition, time, initial=None, case='baseline'):
    if sm.training or sm.sequence_attention_topology != 'global':
        raise ValueError('Probe requires eval mode and global IPA')
    if case not in CASES:
        raise ValueError(case)
    peptide = condition.peptide_mask[..., None]
    skip, ipa_single, ipa_q = single, single, q
    if case == 'skip_initial': skip = initial[0]
    if case == 'ipa_single_initial': ipa_single = initial[0]
    if case == 'ipa_q_initial': ipa_q = initial[1]
    factor = .95 if case.endswith('_minus') else 1.05
    if case.startswith('skip_') and case != 'skip_initial': skip = skip * factor
    if case.startswith('q_'): ipa_q = ipa_q * factor
    gamma, beta = sm.time_film(sm.time_conditioner(time)).chunk(2, dim=-1)
    if hasattr(sm, 'dynamic_pair'):
        geometry = pair_features(rigid, condition.residue_mask, condition.peptide_mask, sm.translation_scale)
        pair = pair + sm.dynamic_pair(geometry).to(pair.dtype)
    conditioned = (ipa_single + ipa_q) * (1. + peptide * gamma[:, None]) + peptide * beta[:, None]
    message = sm.ipa(conditioned, pair, rigid, condition.residue_mask)
    if case in ('ipa_minus', 'ipa_plus'): message = message * factor
    z = skip + message
    normed = sm.layer_norm_ipa(sm.ipa_dropout(z))
    current = normed
    for layer in sm.transition.layers:
        current = layer(current)
    transition = current - normed
    if case == 'transition_zero': current = normed
    if case in ('transition_minus', 'transition_plus'): current = normed + transition * factor
    final = sm.transition.layer_norm(sm.transition.dropout(current))
    update = (sm.backbone_update(final) * (1. - time.float())[:, None, None]).float()
    output = rigid.masked(rigid.compose_q_update_vec(update), condition.peptide_mask)
    return final, update, output, dict(skip=skip, message=message, z=z, normed=normed,
                                      transition=transition, current=current, final=final)


def decompose(sm, activation, time):
    """Gated six-vector components; rotation entries are quaternion vectors, not degrees."""
    a = {k: v.double() for k, v in activation.items()}
    center = lambda x: x - x.mean(-1, keepdim=True)
    ln1, ln2 = sm.layer_norm_ipa, sm.transition.layer_norm
    gain1 = ln1.weight.double() / (a['z'].var(-1, unbiased=False, keepdim=True) + ln1.eps).sqrt()
    gain2 = ln2.weight.double() / (a['current'].var(-1, unbiased=False, keepdim=True) + ln2.eps).sqrt()
    head = sm.backbone_update.linear
    linear = lambda x: F.linear(x, head.weight.double(), None)
    gate = (1. - time.float()).double()[:, None, None]
    vectors = {
        'single_skip': linear(gain2 * center(gain1 * center(a['skip']))),
        'ipa_message': linear(gain2 * center(gain1 * center(a['message']))),
        'norm1_bias': linear(gain2 * center(ln1.bias.double())).expand(*a['skip'].shape[:-1], 6),
        'transition_message': linear(gain2 * center(a['transition'])),
        'norm2_head_bias': (linear(ln2.bias.double()) + head.bias.double()).expand(*a['skip'].shape[:-1], 6),
    }
    return {k: v * gate for k, v in vectors.items()}


def channel_vectors(update, rigid, mask, scale):
    u = update[:, mask].double()
    ca = torch.einsum('bnij,bnj->bni', rigid.rotation[:, mask].double(), u[..., 3:]) * scale
    return {'ca': ca, 'quaternion_vector': u[..., :3]}


def accounting(parts, total, initial_parts=None, initial_total=None):
    """Per-component norms and signed projections, including changes from stage 1."""
    energy = lambda x: x.square().sum(-1).mean()
    dot = lambda x, y: (x * y).sum(-1).mean()
    rows = []
    for route, value in parts.items():
        proj = dot(value, total)
        row = dict(route=route, component_rms=float(energy(value).sqrt()),
                   total_mse=float(energy(total)), projection=float(proj))
        if initial_parts is not None:
            dv, dt = value - initial_parts[route], total - initial_total
            row.update(change_component_rms=float(energy(dv).sqrt()),
                       total_change_mse=float(energy(dt)), change_projection=float(dot(dv, dt)))
        rows.append(row)
    return rows
