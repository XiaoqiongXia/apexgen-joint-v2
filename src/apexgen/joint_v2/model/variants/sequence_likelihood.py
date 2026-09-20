"""Exploratory known observation likelihood plus learned sequence messages.

The model never receives the synthetic transition matrix or posterior teacher.
The explicit term assumes the centered unit Gaussian / linear path contract.
"""

import torch

from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.model.variants.sequence_calibration import (
    CONTRACT_SHA256 as PARENT_SHA256,
    probability_endpoint,
)
from apexgen.joint_v2.model.variants.sequence_posterior import PosteriorSequenceModel, SCALE
from apexgen.joint_v2.contracts.state import center_sequence_logits

CONTRACT = dict(
    schema="apexgen.joint_v2.likelihood_correction.v1",
    parent_contract=PARENT_SHA256,
    model="raw direct Joint-v2, frozen constant condition and geometry",
    logits="20*h/SCALE, optionally plus SCALE*t/(1-t)^2*z; FP32",
    endpoint="SCALE*(softmax(logits)-1/20), numerically centered",
    loss="uniform exact Markov posterior mean endpoint MSE",
    initialization="matched tensors, zero head; different initial functions by design",
    compatibility="exploratory only; centered Gaussian linear-path contract required",
)
CONTRACT_SHA256 = canonical_sha256(CONTRACT)


def observation_logits(z, time):
    if z.ndim != 3 or z.shape[-1] != 20 or time.shape != (z.shape[0],):
        raise ValueError("expected z [B,L,20], time [B]")
    if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time >= 1)).any()):
        raise ValueError("observation likelihood requires times in [0,1)")
    with torch.autocast(device_type=z.device.type, enabled=False):
        t = time.float()
        return SCALE * (t / (1 - t).square())[:, None, None] * z.float()


class LikelihoodModel(PosteriorSequenceModel):
    def __init__(self, head_mode):
        if head_mode not in {"learned", "likelihood"}:
            raise ValueError("unknown likelihood head")
        super().__init__("raw")
        self.head_mode = head_mode
        self.decoder.structure_module.sequence_output_parameterization = "direct"

    def forward(self, z, time):
        h = super().forward(z, time)
        with torch.autocast(device_type=z.device.type, enabled=False):
            if self.head_mode == "learned":
                return probability_endpoint(h)
            logits = observation_logits(z, time) + 20.0 / SCALE * h.float()
            return center_sequence_logits(SCALE * (logits.softmax(-1) - 0.05))
