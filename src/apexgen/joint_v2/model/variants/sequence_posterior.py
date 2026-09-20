"""Exact synthetic sequence posterior and an isolated Joint-v2 sequence decoder.

This is an exploratory contract extension, not a compatible formal checkpoint.
The fixed geometry and frozen encoding contain no sampled sequence labels.
"""

import math

import torch
from torch import nn

from apexgen.joint_v2.contracts.contract import UnifiedComplexCondition, JOINT_V2_CONTRACT_SHA256
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.model.network import JointV2Model
from apexgen.joint_v2.contracts.state import JointFlowState, center_sequence_logits

SCALE = math.log(381.0)
CONTRACT = dict(
    schema="apexgen.joint_v2.synthetic_sequence_posterior.v1",
    parent_contract=JOINT_V2_CONTRACT_SHA256,
    state="20D centered logits, linear path, centered unit Gaussian base",
    prior="uniform initial; iid uniform OR .15 uniform + .55 self + .30 next cyclic",
    observation="one constant pocket; label-independent straight frames; frozen encoder",
    input="LN20-linear plus zero-initialized raw linear branch; disabled in normalized arm",
    decoder="existing JointV2StructureModule task S, shared IPA and sequence modules",
    target="exact posterior mean or sampled endpoint; MSE only, original (1-t) gate",
    compatibility="experimental only; reject different synthetic contract/config/input/target",
)
CONTRACT_SHA256 = canonical_sha256(CONTRACT)
ARCHITECTURE = dict(
    single_dim=64,
    pair_dim=32,
    encoder_attention_heads=4,
    encoder_blocks=1,
    decoder_blocks=4,
    dropout=0.0,
    c_ipa=8,
    ipa_heads=4,
    ipa_query_key_points=2,
    ipa_value_points=4,
    transition_layers=1,
    angle_hidden_dim=32,
    angle_blocks=1,
    sequence_head_blocks=2,
    time_embedding_dim=64,
    time_embedding_frequencies=16,
)


def transition_matrix(prior, *, device="cpu", dtype=torch.float32):
    if prior not in {"iid", "markov"}:
        raise ValueError("unknown prior")
    uniform = torch.full((20, 20), 1 / 20, device=device, dtype=dtype)
    if prior == "iid":
        return uniform
    eye = torch.eye(20, device=device, dtype=dtype)
    return 0.15 * uniform + 0.55 * eye + 0.30 * eye.roll(1, dims=1)


def sample_labels(uniforms, transition):
    """Use explicit uniforms so the two input arms consume identical random streams."""
    labels = [(uniforms[:, 0] * 20).long().clamp_max(19)]
    cdf = transition.cumsum(-1)
    for i in range(1, uniforms.shape[1]):
        labels.append((uniforms[:, i, None] > cdf[labels[-1]]).sum(-1).clamp_max(19))
    return torch.stack(labels, 1)


def endpoints(labels):
    return SCALE * (torch.nn.functional.one_hot(labels, 20).float() - 1 / 20)


def markov_marginals(log_emission, log_transition, log_initial=None):
    """Exact forward/backward marginals in log space; supports test state counts."""
    if log_emission.ndim != 3 or log_emission.shape[1] < 1:
        raise ValueError("emissions must have shape [batch, positive length, states]")
    k = log_emission.shape[-1]
    if log_transition.shape != (k, k):
        raise ValueError("transition shape mismatch")
    if log_initial is None:
        log_initial = log_emission.new_full((k,), -math.log(k))
    alpha = [torch.log_softmax(log_initial + log_emission[:, 0], -1)]
    for i in range(1, log_emission.shape[1]):
        a = torch.logsumexp(alpha[-1][:, :, None] + log_transition, dim=1)
        alpha.append(torch.log_softmax(a + log_emission[:, i], -1))
    beta = [torch.zeros_like(alpha[0])]
    for i in range(log_emission.shape[1] - 2, -1, -1):
        b = torch.logsumexp(
            log_transition + (log_emission[:, i + 1] + beta[-1])[:, None, :], dim=-1
        )
        beta.append(torch.log_softmax(b, -1))
    return torch.stack(
        [torch.softmax(a + b, -1) for a, b in zip(alpha, reversed(beta), strict=True)], 1
    )


def posterior(z, time, prior):
    if time.shape != (z.shape[0],) or bool(((time < 0) | (time >= 1)).any()):
        raise ValueError("posterior requires batch times in [0,1)")
    coefficient = time / (1 - time).square()
    emission = SCALE * coefficient[:, None, None] * z
    if prior == "iid":
        q = emission.softmax(-1)
    elif prior == "markov":
        q = markov_marginals(
            emission, transition_matrix(prior, device=z.device, dtype=z.dtype).log()
        )
    else:
        raise ValueError("unknown prior")
    return q, SCALE * (q - 1 / 20)


def draw(batch, length, prior, generator, *, time_max=0.95):
    device = generator.device
    # Random draws have exactly the same order for both priors and input arms.
    u = torch.rand(batch, length, device=device, generator=generator)
    labels = sample_labels(u, transition_matrix(prior, device=device))
    base = center_sequence_logits(
        torch.randn(batch, length, 20, device=device, generator=generator)
    )
    t = torch.rand(batch, device=device, generator=generator) * time_max
    target = endpoints(labels)
    return labels, base, target, t


def fixed_condition(batch, length, device):
    shape = (batch, length + 1)
    residue = torch.ones(shape, dtype=torch.bool, device=device)
    pocket = torch.zeros_like(residue)
    pocket[:, 0] = True
    peptide = ~pocket
    xyz = torch.zeros(*shape, 3, device=device)
    rotation = torch.eye(3, device=device).expand(*shape, 3, 3).clone()
    aa = torch.full(shape, 20, device=device, dtype=torch.long)
    aa[:, 0] = 0
    atoms = torch.zeros(*shape, 38, 3, device=device)
    atoms[:, 0, :3] = torch.tensor(
        [[-1.2, 1.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], device=device
    )
    atom_mask = torch.zeros(*shape, 38, device=device, dtype=torch.bool)
    atom_mask[:, 0, :3] = True
    condition = UnifiedComplexCondition(
        residue,
        pocket,
        peptide,
        xyz,
        rotation,
        aa,
        atoms,
        atom_mask,
        pocket.clone(),
        torch.arange(length + 1, device=device).expand(batch, -1),
        peptide.long(),
        torch.zeros(*shape, 3, 2, device=device),
        torch.zeros(*shape, 3, dtype=torch.bool, device=device),
        torch.zeros(*shape, 4, 2, device=device),
        torch.zeros(*shape, 4, dtype=torch.bool, device=device),
    )
    condition.validate_invariants()
    translation = xyz.clone()
    translation[:, 1:, 0] = torch.arange(length, device=device) * 3.8
    translation[:, 1:, 1] = 8.0
    return condition, translation, rotation


class SequenceInput(nn.Module):
    def __init__(self, norm, projection, raw):
        super().__init__()
        self.norm, self.normalized = norm, projection
        self.raw = nn.Linear(20, projection.out_features, bias=False)
        nn.init.zeros_(self.raw.weight)
        self.use_raw = raw

    def forward(self, z):
        # The zero control branch has identical parameter layout and initialization.
        return self.normalized(self.norm(z)) + self.raw(z) * float(self.use_raw)


class PosteriorSequenceModel(nn.Module):
    def __init__(self, input_mode):
        super().__init__()
        if input_mode not in {"normalized", "raw"}:
            raise ValueError("unknown sequence input")
        network = JointV2Model(**ARCHITECTURE)
        self.encoder, self.decoder = network.encoder, network.decoder
        self.input_mode = input_mode
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        module = self.decoder.structure_module
        for branch in [module.backbone_update, module.angle_resnet]:
            for parameter in branch.parameters():
                parameter.requires_grad_(False)
        module.sequence_projection = SequenceInput(
            module.sequence_norm, module.sequence_projection, input_mode == "raw"
        )
        module.sequence_norm = nn.Identity()
        self._cache = {}

    def forward(self, z, time):
        key = (z.shape[0], z.shape[1], str(z.device))
        if key not in self._cache:
            condition, translation, rotation = fixed_condition(*z.shape[:2], z.device)
            # Always cache FP32 frozen features, independent of caller autocast.
            with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
                encoding = self.encoder(condition)
            self._cache[key] = (condition, translation, rotation, encoding)
        condition, translation, rotation, encoding = self._cache[key]
        full_z = torch.cat((z.new_zeros(z.shape[0], 1, 20), z), 1)
        state = JointFlowState(translation, rotation, full_z)
        pred = self.decoder.structure_module(
            state, time, condition, encoding, task="S", return_intermediates=False
        )
        return pred.sequence_logits[:, 1:]
