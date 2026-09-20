"""Exploratory probability endpoint and bounded velocity-weight controls."""

import math

import torch

from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.model.variants.sequence_posterior import PosteriorSequenceModel, SCALE
from apexgen.joint_v2.model.variants.sequence_head_controls import CONTRACT_SHA256 as PARENT_SHA256
from apexgen.joint_v2.contracts.state import center_sequence_logits

CONTRACT = dict(
    schema="apexgen.joint_v2.iid_calibration.v1",
    parent_contract=PARENT_SHA256,
    model="Joint-v2 raw, direct mean endpoint, frozen constant condition and geometry",
    head="linear h OR SCALE*(softmax(20*h/SCALE)-1/20), FP32; matched zero-init Jacobian",
    loss="oracle endpoint MSE OR analytically mean-normalized capped inverse (1-t)^2 MSE",
    data="iid uniform, centered Gaussian base, same linear path and endpoint code",
    compatibility="exploratory only; manifest must match head, loss, seed and contract",
)
CONTRACT_SHA256 = canonical_sha256(CONTRACT)


def probability_endpoint(head):
    """At h=0 the Jacobian equals the centered linear-head projector."""
    return center_sequence_logits(SCALE * ((20.0 / SCALE * head.float()).softmax(-1) - 0.05))


class CalibrationModel(PosteriorSequenceModel):
    def __init__(self, head_mode):
        if head_mode not in {"linear", "probability"}:
            raise ValueError("unknown calibration head")
        super().__init__("raw")
        self.head_mode = head_mode
        self.decoder.structure_module.sequence_output_parameterization = "direct"

    def forward(self, z, time):
        h = super().forward(z, time)
        if self.head_mode == "linear":
            return h
        with torch.autocast(device_type=z.device.type, enabled=False):
            return probability_endpoint(h)


def weight_normalizer(time_max, cap):
    if not 0 < time_max < 1 or cap < 1:
        raise ValueError("invalid weighting parameters")
    cutoff = 1 - 1 / math.sqrt(cap)
    upper = min(time_max, cutoff)
    return (1 / (1 - upper) - 1 + cap * max(0, time_max - cutoff)) / time_max


def time_weights(time, mode, *, time_max=0.95, cap=25.0):
    if mode == "uniform":
        return torch.ones_like(time)
    if mode != "velocity":
        raise ValueError("unknown loss mode")
    return (1 - time).square().reciprocal().clamp_max(cap) / weight_normalizer(time_max, cap)


def calibration_loss(prediction, target, time, mode, *, time_max=0.95, cap=25.0):
    if mode == "uniform":
        # Preserve the previous baseline's exact floating-point reduction order.
        return (prediction - target).square().mean()
    per_sequence = (prediction - target).square().mean((1, 2))
    return (per_sequence * time_weights(time, mode, time_max=time_max, cap=cap)).mean()
