"""Joint-v2 unified-complex encoder and sequence-structure decoder."""

from __future__ import annotations

from torch import Tensor, nn

from apexgen.joint_v2.contracts.contract import (
    JointEndpointPrediction,
    UnifiedComplexCondition,
    UnifiedComplexEncoding,
)
from apexgen.joint_v2.model.decoder import JointEndpointDecoder
from apexgen.joint_v2.model.encoder import UnifiedComplexEncoder
from apexgen.joint_v2.contracts.state import JointFlowState


class JointV2Model(nn.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        encoder_attention_heads: int,
        encoder_blocks: int,
        decoder_blocks: int,
        encoder_single_dim: int | None = None,
        encoder_pair_dim: int | None = None,
        encoder_feature_mode: str = "full",
        dropout: float = 0.1,
        c_ipa: int = 16,
        ipa_heads: int = 12,
        ipa_query_key_points: int = 4,
        ipa_value_points: int = 8,
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
        self.encoder = UnifiedComplexEncoder(
            single_dim=single_dim,
            pair_dim=pair_dim,
            encoder_single_dim=encoder_single_dim,
            encoder_pair_dim=encoder_pair_dim,
            feature_mode=encoder_feature_mode,
            blocks=encoder_blocks,
            heads=encoder_attention_heads,
            dropout=dropout,
        )
        self.decoder = JointEndpointDecoder(
            single_dim=single_dim,
            pair_dim=pair_dim,
            blocks=decoder_blocks,
            c_ipa=c_ipa,
            ipa_heads=ipa_heads,
            query_key_points=ipa_query_key_points,
            value_points=ipa_value_points,
            dropout=dropout,
            transition_layers=transition_layers,
            angle_hidden_dim=angle_hidden_dim,
            angle_blocks=angle_blocks,
            sequence_head_blocks=sequence_head_blocks,
            translation_scale=translation_scale,
            time_embedding_dim=time_embedding_dim,
            time_embedding_frequencies=time_embedding_frequencies,
            epsilon=epsilon,
            inf=inf,
            stop_rotation_gradient=stop_rotation_gradient,
            geometry_update_time_gate=geometry_update_time_gate,
        )

    def encode_complex(self, condition: UnifiedComplexCondition) -> UnifiedComplexEncoding:
        return self.encoder(condition)

    def decode(
        self,
        state: JointFlowState,
        time: Tensor,
        condition: UnifiedComplexCondition,
        encoding: UnifiedComplexEncoding,
        *,
        return_intermediates: bool = True,
    ) -> JointEndpointPrediction:
        return self.decoder(
            state,
            time,
            condition,
            encoding,
            return_intermediates=return_intermediates,
        )

    def forward(
        self,
        state: JointFlowState,
        time: Tensor,
        condition: UnifiedComplexCondition,
        *,
        return_intermediates: bool = True,
    ) -> JointEndpointPrediction:
        return self.decode(
            state,
            time,
            condition,
            self.encode_complex(condition),
            return_intermediates=return_intermediates,
        )


def build_joint_v2_model(config: dict) -> JointV2Model:
    architecture = config["architecture"]
    structure = architecture["structure_module"]
    time = architecture["time_conditioning"]
    angle = architecture["angle_head"]
    return JointV2Model(
        single_dim=architecture["single_dim"],
        pair_dim=architecture["pair_dim"],
        encoder_single_dim=architecture["encoder_single_dim"],
        encoder_pair_dim=architecture["encoder_pair_dim"],
        encoder_feature_mode=architecture["encoder_feature_mode"],
        encoder_attention_heads=architecture["encoder_attention_heads"],
        encoder_blocks=architecture["encoder_blocks"],
        decoder_blocks=architecture["decoder_blocks"],
        dropout=structure["dropout_rate"],
        c_ipa=structure["c_ipa"],
        ipa_heads=structure["no_heads_ipa"],
        ipa_query_key_points=structure["no_qk_points"],
        ipa_value_points=structure["no_v_points"],
        transition_layers=structure["no_transition_layers"],
        angle_hidden_dim=angle["hidden_dim"],
        angle_blocks=angle["blocks"],
        sequence_head_blocks=architecture["sequence_module"]["final_head_blocks"],
        translation_scale=architecture["translation_scale_factor"],
        time_embedding_dim=time["embedding_dim"],
        time_embedding_frequencies=time["frequencies"],
        epsilon=structure["epsilon"],
        inf=structure["inf"],
        stop_rotation_gradient=structure["stop_rotation_gradient_between_blocks"],
        geometry_update_time_gate=structure.get("geometry_update_time_gate", True),
    )
