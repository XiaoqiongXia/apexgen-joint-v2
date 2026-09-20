"""Joint sequence-structure endpoint decoder."""

from __future__ import annotations

from torch import Tensor, nn

from apexgen.joint_v2.contracts.contract import (
    JointEndpointPrediction,
    UnifiedComplexCondition,
    UnifiedComplexEncoding,
)
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.model.structure_module import JointV2StructureModule


class JointEndpointDecoder(nn.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        blocks: int = 8,
        c_ipa: int = 16,
        ipa_heads: int = 12,
        query_key_points: int = 4,
        value_points: int = 8,
        dropout: float = 0.1,
        transition_layers: int = 1,
        angle_hidden_dim: int = 128,
        angle_blocks: int = 2,
        sequence_head_blocks: int = 2,
        translation_scale: float = 10.0,
        time_embedding_dim: int = 64,
        time_embedding_frequencies: int = 16,
        epsilon: float = 1e-8,
        inf: float = 1e5,
        stop_rotation_gradient: bool = True,
        geometry_update_time_gate: bool = True,
    ) -> None:
        super().__init__()
        self.structure_module = JointV2StructureModule(
            c_s=single_dim,
            c_z=pair_dim,
            c_ipa=c_ipa,
            c_angle=angle_hidden_dim,
            no_heads_ipa=ipa_heads,
            no_qk_points=query_key_points,
            no_v_points=value_points,
            dropout=dropout,
            blocks=blocks,
            transition_layers=transition_layers,
            angle_blocks=angle_blocks,
            sequence_head_blocks=sequence_head_blocks,
            translation_scale=translation_scale,
            time_embedding_dim=time_embedding_dim,
            time_frequencies=time_embedding_frequencies,
            eps=epsilon,
            inf=inf,
            stop_rotation_gradient=stop_rotation_gradient,
            geometry_update_time_gate=geometry_update_time_gate,
        )

    def forward(
        self,
        state: JointFlowState,
        time: Tensor,
        condition: UnifiedComplexCondition,
        encoding: UnifiedComplexEncoding,
        *,
        return_intermediates: bool = True,
    ) -> JointEndpointPrediction:
        return self.structure_module(
            state,
            time,
            condition,
            encoding,
            return_intermediates=return_intermediates,
        )
