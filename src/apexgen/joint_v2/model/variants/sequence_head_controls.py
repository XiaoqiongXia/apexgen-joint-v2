"""Exploratory iid output controls; no formal checkpoint compatibility."""

import torch
from torch import nn

from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.model.variants.sequence_posterior import (
    ARCHITECTURE as JOINT_ARCHITECTURE,
    CONTRACT_SHA256 as PARENT_SHA256,
    PosteriorSequenceModel,
)
from apexgen.joint_v2.contracts.state import center_sequence_logits

CONTRACT = dict(
    schema="apexgen.joint_v2.iid_head_controls.v1",
    parent_contract=PARENT_SHA256,
    prior="iid uniform, centered Gaussian linear path, exact posterior mean MSE",
    input="raw available in all arms; Joint-v2 retains normalized plus raw input",
    output="residual z+(1-t)*head versus direct head; both centered, FP32 output",
    initialization="matched tensors within backbone, zero final head; initial predictions differ",
    backbone="Joint-v2 fixed-condition task S or positionwise MLP with scalar time",
    compatibility="exploratory contract only; no formal checkpoint compatibility",
)
CONTRACT_SHA256 = canonical_sha256(CONTRACT)
ARCHITECTURES = dict(joint=JOINT_ARCHITECTURE, mlp=dict(width=218, hidden_layers=3, time="scalar"))


class PositionMLP(nn.Module):
    def __init__(self, output_mode):
        super().__init__()
        if output_mode not in {"residual", "direct"}:
            raise ValueError("unknown output mode")
        self.output_mode = output_mode
        self.input_layer = nn.Linear(21, 218)
        self.hidden_1 = nn.Linear(218, 218)
        self.hidden_2 = nn.Linear(218, 218)
        self.output_layer = nn.Linear(218, 20)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, z, time):
        t = time[:, None, None].expand(*z.shape[:2], 1)
        x = torch.nn.functional.silu(self.input_layer(torch.cat((z, t), -1)))
        x = torch.nn.functional.silu(self.hidden_1(x))
        x = torch.nn.functional.silu(self.hidden_2(x))
        with torch.autocast(device_type=z.device.type, enabled=False):
            head = self.output_layer(x.float())
            endpoint = z + (1 - t) * head if self.output_mode == "residual" else head
            return center_sequence_logits(endpoint)


def make_model(backbone, output_mode):
    if output_mode not in {"residual", "direct"}:
        raise ValueError("unknown output mode")
    if backbone == "mlp":
        return PositionMLP(output_mode)
    if backbone == "joint":
        model = PosteriorSequenceModel("raw")
        model.decoder.structure_module.sequence_output_parameterization = output_mode
        return model
    raise ValueError("unknown backbone")
