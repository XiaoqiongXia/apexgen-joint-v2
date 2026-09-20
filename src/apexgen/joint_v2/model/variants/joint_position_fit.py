"""Exploratory J fits with position conditions separate from the generated sequence state."""

import math

import torch
from torch import nn

from apexgen.joint_v2.experiments.condition_controls import seed_for
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.model.structure_module import OpenFoldLinear
from apexgen.joint_v2.model.task_factorization import TaskFactorizationModel

JOINT_POSITION_CONTRACT = dict(
    schema="apexgen.joint_v2.joint_position_fit.v1",
    task="J: generated translation, rotation and sequence logits; no fixed native peptide observation",
    code="20D Gaussian seeded by fixed code seed and integer rank; centered/native L2 matched",
    constant="repeat rank-zero random code at every peptide position",
    random="use the fixed random code at each integer peptide rank",
    route="independent encoder.position_norm/projection added before encoder blocks, peptide mask only",
    sequence="original sequence norm/projection, latent, sequence head and flow remain active",
    initialization="append independent OpenFoldLinear after original J modules; code generation uses local RNG",
    evaluation="one fixed evaluation seed for both training seeds; fresh training base/t each optimizer step",
)
JOINT_POSITION_CONTRACT_SHA256 = canonical_sha256(JOINT_POSITION_CONTRACT)


def position_codes(mask, mode, code_seed):
    if mode not in {"constant", "random"}:
        raise ValueError("joint position mode must be constant or random")
    z = torch.zeros(*mask.shape, 20, device=mask.device)
    count = int(mask.sum(-1).max())
    codes = []
    for rank in range(count if mode == "random" else 1):
        g = torch.Generator().manual_seed(seed_for(code_seed, f"position:{rank}"))
        code = torch.randn(20, generator=g)
        code -= code.mean()
        code *= math.log(381.0) * math.sqrt(19 / 20) / code.norm()
        codes.append(code)
    table = torch.stack(codes).to(mask.device)
    for b in range(len(mask)):
        idx = mask[b].nonzero().flatten()
        z[b, idx] = table[: len(idx)] if mode == "random" else table[0]
    return z


class JointPositionFitModel(TaskFactorizationModel):
    def __init__(self, config, *, position_mode, code_seed):
        if position_mode not in {"constant", "random"}:
            raise ValueError("unsupported position mode")
        super().__init__(config, "J")
        self.position_mode = position_mode
        self.code_seed = int(code_seed)
        architecture = config["architecture"]
        width = architecture.get("encoder_single_dim", architecture["single_dim"])
        self.encoder.position_norm = nn.LayerNorm(20)
        self.encoder.position_projection = OpenFoldLinear(20, width)

    def encode_complex(self, observation):
        if observation.task != "J":
            raise ValueError("joint position fit requires J observations")
        z = position_codes(observation.pocket.peptide_mask, self.position_mode, self.code_seed)
        injected = self.encoder.position_projection(self.encoder.position_norm(z))
        return self.encoder(observation.pocket, peptide_single=injected)


def build_fit_model(config, task, protocol):
    mode = protocol.get("joint_position_mode", "none")
    if mode != "none":
        if (
            task != "J"
            or protocol.get("condition_mode", "native") != "native"
            or protocol.get("condition_route", "decoder") != "decoder"
        ):
            raise ValueError(
                "joint position fits require original J flow/observations and decoder sequence route"
            )
        return JointPositionFitModel(
            config, position_mode=mode, code_seed=protocol["position_code_seed"]
        )
    return TaskFactorizationModel(
        config, task, condition_route=protocol.get("condition_route", "decoder")
    )
