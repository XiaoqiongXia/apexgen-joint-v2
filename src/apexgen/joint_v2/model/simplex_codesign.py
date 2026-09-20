"""Joint geometry plus a simplex-conditioned categorical denoiser."""

from copy import deepcopy

from torch import nn

from apexgen.joint_v2.contracts.simplex_state import SimplexJointState
from apexgen.joint_v2.model.variants.real_sequence_transfer import build_real_transfer


class SimplexCodesignModel(nn.Module):
    """Output sequence_logits are categorical scores for CE, not simplex state."""

    def __init__(self, config, *, position_code_seed=20260919):
        super().__init__()
        config = deepcopy(config)
        config["architecture"]["structure_module"]["stop_rotation_gradient_between_blocks"] = False
        self.network = build_real_transfer(
            config,
            "J",
            dict(
                sequence_topology="global",
                joint_position_mode="random",
                position_code_seed=position_code_seed,
                sequence_endpoint_mode="direct",
            ),
        )

    def encode_complex(self, observation):
        return self.network.encode_complex(observation)

    def decode(self, state, time, observation, encoding):
        if not isinstance(state, SimplexJointState):
            raise TypeError("Simplex model requires SimplexJointState")
        return self.network.decode(
            state.network_state(), time, observation, encoding, return_intermediates=False
        )

    def forward(self, state, time, observation):
        return self.decode(state, time, observation, self.encode_complex(observation))
