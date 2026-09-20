"""Fail-closed contract and tensors for sequence--structure Joint-v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from apexgen.joint_v2.data.native_targets import observed_backbone_frames

from apexgen.joint_v2.contracts.state import (
    AMINO_ACID_TYPES,
    BACKBONE_ANGLE_SLOTS,
    JointFlowState,
    POCKET_ATOM_SLOTS,
    SEQUENCE_EPSILON,
    SIDECHAIN_ANGLE_SLOTS,
    UNKNOWN_AATYPE,
    debug_invariants_enabled,
    sequence_endpoint_logits,
)


JOINT_V2_CONTRACT = {
    "schema_version": "apexgen.joint_v2.sequence_structure_endpoint.contract.v5",
    "architecture": "joint_v2_unified_complex_sequence_structure_endpoint_refinement",
    "dynamic_state": {
        "fields": ["translation", "rotation", "sequence_logits"],
        "precision": "float32",
        "sequence_types": AMINO_ACID_TYPES,
        "sequence_coordinate": "centered_logits",
        "public_translation_unit": "angstrom",
        "rotation_convention": "local_to_global",
        "generated_role": "peptide",
        "fixed_role": "pocket",
    },
    "static_condition": {
        "layout": "unified_pocket_peptide_residue_axis",
        "peptide_aatype": "unknown_placeholder",
        "native_peptide_geometry": "forbidden",
        "pocket_atom_slots": POCKET_ATOM_SLOTS,
        "encoder_reusable_across_rollout": True,
    },
    "supervision": {
        "translation": "observed_peptide_CA_in_pocket_coordinate_system",
        "rotation": "observed_N_CA_C_frame_local_to_global",
        "sequence": "same_observed_residue_aatype",
        "backbone_atoms": "unmodified_observed_N_CA_C",
        "backbone_torsions": "extracted_from_observed_N_CA_C",
        "missing_or_degenerate_backbone": "reject_no_idealized_fallback",
        "legacy_whole_chain_idealized_labels": "never_used",
        "prediction_atom_placement": "fixed_local_template_not_a_label_transform",
    },
    "encoder": {
        "internal_single_pair_widths": "positive_integer_configured_per_run",
        "decoder_single_pair_widths": "positive_integer_configured_per_run",
        "output_projection": "identity_or_linear_from_encoder_to_decoder_width",
        "feature_ablation": (
            "full_or_cumulative_information_only_ablation_with_fixed_tensor_layout"
        ),
        "feature_modes": [
            "full",
            "no_metadata_torsions",
            "backbone_atoms",
            "compact_geometry",
        ],
    },
    "decoder": {
        "parameterization": "endpoint_refinement",
        "structure_module": "openfold_monomer_ipa_be2ec184",
        "rigid_update": "compose_q_update_vec",
        "blocks": "positive_integer_configured_per_run",
        "parameter_sharing": True,
        "pocket_frame_update": "forbidden",
        "time_conditioning": "fourier_peptide_film_every_block",
        "block_update_time_gate": "one_minus_t_frame_every_block_and_final_sequence",
        "sequence_projection": "initial_zt_shared_layernorm20_linear_to_latent",
        "sequence_latent": "shared_zero_final_geometry_conditioned_residual_before_every_ipa",
        "sequence_update": "final_only_resnet_zero_final_linear_one_minus_t_centered_residual",
        "sequence_stop_gradient": False,
        "angle_head": "shared_final_only_phi_psi_omega_chi1_chi2_chi3_chi4_auxiliary",
        "angle_head_slots": ["phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4"],
        "angle_head_projection": "shared_trunk_separate_3_plus_4_linear",
        "sidechain_projection_initialization": "appended_after_ec6a13a_modules",
    },
    "flow": {
        "base_translation": "peptide_independent_gaussian_sigma_5_angstrom",
        "base_rotation": "peptide_independent_haar_so3",
        "base_sequence": "peptide_centered_standard_gaussian_logits",
        "sequence_epsilon": SEQUENCE_EPSILON,
        "pocket_base": "copy_static_condition",
        "training_time": "uniform_half_open_0_1",
        "offpath": "disabled",
        "sampler": "20_step_endpoint_fractional_geodesic",
        "sequence_sampler": "20_step_endpoint_fractional_centered_logit",
        "sequence_decoding": "argmax",
        "query_grid": "0.00_0.05_through_0.95",
    },
    "loss": {
        "reduction": "sample_first_batch_second",
        "trajectory": "unclamped_origin_point_peptide_and_bidirectional_cross_fape",
        "fape_length_scale_angstrom": 10.0,
        "final_terms": [
            "translation",
            "rotation_geodesic",
            "backbone_n_ca_c",
            "backbone_angle_sincos",
            "sidechain_angle_symmetry_aware_sincos",
            "final_sequence_centered_logit_mse",
            "final_sequence_soft_target_cross_entropy",
        ],
        "sequence_weights": {
            "sigma_logit": 1.0,
            "logit": 1.0,
            "soft_ce": 1.0,
            "total": 1.0,
            "blocks": "final_only",
        },
        "sidechain_symmetry": "chi_pi_periodic_minimum_equivalent_target",
        "training_chain_and_clash": False,
    },
    "precision": {
        "network": "bfloat16_or_float32",
        "geometry_sequence_state_ipa_points_so3_fape_and_losses": "float32",
    },
    "checkpoint": {
        "schema": "apexgen.joint_v2.sequence_structure_endpoint.checkpoint.v1",
        "frame_endpoint_checkpoint": "reject",
        "legacy_velocity_torsion_checkpoint": "reject",
    },
}

JOINT_V2_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(JOINT_V2_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def require_joint_v2_contract(payload: Mapping[str, Any], *, source: str = "artifact") -> None:
    """Reject artifacts that do not carry the exact endpoint contract digest."""

    observed = payload.get("joint_v2_contract_sha256")
    if observed != JOINT_V2_CONTRACT_SHA256:
        raise ValueError(
            f"{source} Joint-v2 sequence-structure contract mismatch: expected "
            f"{JOINT_V2_CONTRACT_SHA256}, got {observed!r}"
        )


def _same_device(values: tuple[Tensor, ...], name: str) -> torch.device:
    devices = {value.device for value in values}
    if len(devices) != 1:
        raise ValueError(f"{name} tensors must be on one device")
    return values[0].device


@dataclass(frozen=True)
class UnifiedComplexCondition:
    """Static encoder condition with native geometry only for pocket residues."""

    residue_mask: Tensor
    pocket_mask: Tensor
    peptide_mask: Tensor
    pocket_translation: Tensor
    pocket_rotation: Tensor
    aatype: Tensor
    pocket_atom_xyz: Tensor
    pocket_atom_mask: Tensor
    pocket_core_mask: Tensor
    sequence_index: Tensor
    chain_index: Tensor
    pocket_backbone_angles_sin_cos: Tensor
    pocket_backbone_angle_mask: Tensor
    pocket_sidechain_angles_sin_cos: Tensor
    pocket_sidechain_angle_mask: Tensor

    def __post_init__(self) -> None:
        if self.residue_mask.ndim != 2 or self.residue_mask.dtype != torch.bool:
            raise TypeError("residue_mask must be bool [batch, graph]")
        layout = self.residue_mask.shape
        bool_layout = {
            "pocket_mask": self.pocket_mask,
            "peptide_mask": self.peptide_mask,
            "pocket_core_mask": self.pocket_core_mask,
        }
        for name, value in bool_layout.items():
            if value.shape != layout or value.dtype != torch.bool:
                raise TypeError(f"{name} must be bool [batch, graph]")
        if self.pocket_translation.shape != layout + (3,):
            raise ValueError("pocket_translation must be [batch, graph, 3]")
        if self.pocket_rotation.shape != layout + (3, 3):
            raise ValueError("pocket_rotation must be [batch, graph, 3, 3]")
        if self.pocket_atom_xyz.shape != layout + (POCKET_ATOM_SLOTS, 3):
            raise ValueError("pocket_atom_xyz must be [batch, graph, 38, 3]")
        if (
            self.pocket_atom_mask.shape != layout + (POCKET_ATOM_SLOTS,)
            or self.pocket_atom_mask.dtype != torch.bool
        ):
            raise TypeError("pocket_atom_mask must be bool [batch, graph, 38]")
        for name, value in (
            ("aatype", self.aatype),
            ("sequence_index", self.sequence_index),
            ("chain_index", self.chain_index),
        ):
            if value.shape != layout or value.dtype != torch.long:
                raise TypeError(f"{name} must be int64 [batch, graph]")
        angle_specs = (
            (
                "pocket_backbone_angles_sin_cos",
                self.pocket_backbone_angles_sin_cos,
                layout + (BACKBONE_ANGLE_SLOTS, 2),
                None,
            ),
            (
                "pocket_backbone_angle_mask",
                self.pocket_backbone_angle_mask,
                layout + (BACKBONE_ANGLE_SLOTS,),
                torch.bool,
            ),
            (
                "pocket_sidechain_angles_sin_cos",
                self.pocket_sidechain_angles_sin_cos,
                layout + (4, 2),
                None,
            ),
            (
                "pocket_sidechain_angle_mask",
                self.pocket_sidechain_angle_mask,
                layout + (4,),
                torch.bool,
            ),
        )
        for name, value, shape, dtype in angle_specs:
            if value.shape != shape or (dtype is not None and value.dtype != dtype):
                raise TypeError(f"{name} has an invalid shape or dtype")
        geometry = (
            self.pocket_translation,
            self.pocket_rotation,
            self.pocket_atom_xyz,
            self.pocket_backbone_angles_sin_cos,
            self.pocket_sidechain_angles_sin_cos,
        )
        if any(value.dtype != torch.float32 for value in geometry):
            raise TypeError("condition geometry must be float32")
        _same_device(tuple(self.__dict__.values()), "condition")
        if debug_invariants_enabled():
            self.validate_invariants()

    @property
    def layout(self) -> torch.Size:
        return self.residue_mask.shape

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "UnifiedComplexCondition":
        return UnifiedComplexCondition(
            **{
                name: value.to(device=device, non_blocking=non_blocking)
                for name, value in self.__dict__.items()
            }
        )

    def validate_invariants(self) -> None:
        """Perform value checks at data/debug boundaries, never in rollout loops."""

        peptide_or_padding = ~self.pocket_mask
        identity = torch.eye(3, device=self.pocket_rotation.device).expand_as(self.pocket_rotation)
        violations = (
            (self.pocket_mask & self.peptide_mask).any()
            | (self.residue_mask != (self.pocket_mask | self.peptide_mask)).any()
            | (self.pocket_mask.sum(-1) == 0).any()
            | (self.aatype[peptide_or_padding] != UNKNOWN_AATYPE).any()
            | (self.pocket_translation[peptide_or_padding] != 0).any()
            | (self.pocket_rotation[peptide_or_padding] != identity[peptide_or_padding]).any()
            | (self.pocket_atom_xyz[peptide_or_padding] != 0).any()
            | (self.pocket_atom_mask & ~self.pocket_mask[..., None]).any()
            | (self.pocket_core_mask & ~self.pocket_mask).any()
            | (self.pocket_backbone_angles_sin_cos[peptide_or_padding] != 0).any()
            | (self.pocket_backbone_angle_mask & ~self.pocket_mask[..., None]).any()
            | (self.pocket_sidechain_angles_sin_cos[peptide_or_padding] != 0).any()
            | (self.pocket_sidechain_angle_mask & ~self.pocket_mask[..., None]).any()
            | (self.aatype[self.pocket_mask] < 0).any()
            | (self.aatype[self.pocket_mask] >= UNKNOWN_AATYPE).any()
            | ~torch.isfinite(self.pocket_atom_xyz).all()
            | ~torch.isfinite(self.pocket_backbone_angles_sin_cos).all()
            | ~torch.isfinite(self.pocket_sidechain_angles_sin_cos).all()
        )
        if bool(violations.detach().cpu()):
            raise ValueError("UnifiedComplexCondition invariant failed")
        self.fixed_state().validate(self.residue_mask)

    def validate_model_input(self) -> None:
        """Training and generation require at least three observed peptide slots.

        Storage/inspection may retain shorter records, but they must never enter
        a model or training objective. Check independently of debug invariants.
        """
        if bool((self.peptide_mask.sum(-1) < 3).any()):
            raise ValueError("model input requires peptide length >= 3 (greater than 2)")

    def fixed_state(self) -> JointFlowState:
        return JointFlowState(
            self.pocket_translation,
            self.pocket_rotation,
            torch.zeros(
                *self.layout,
                AMINO_ACID_TYPES,
                dtype=torch.float32,
                device=self.residue_mask.device,
            ),
        )


@dataclass(frozen=True)
class PeptideNativeTargets:
    """Supervision-only native peptide sequence endpoint and geometry."""

    endpoint_aatype: Tensor
    endpoint_translation: Tensor
    endpoint_rotation: Tensor
    atom14_xyz: Tensor
    atom14_mask: Tensor
    backbone_xyz: Tensor
    backbone_atom_mask: Tensor
    backbone_angles_sin_cos: Tensor
    backbone_angle_mask: Tensor
    sidechain_angles_sin_cos: Tensor
    sidechain_angle_mask: Tensor

    def __post_init__(self) -> None:
        if self.endpoint_translation.ndim != 3 or self.endpoint_translation.shape[-1] != 3:
            raise ValueError("endpoint_translation must be [batch, graph, 3]")
        layout = self.endpoint_translation.shape[:-1]
        if self.endpoint_aatype.shape != layout or self.endpoint_aatype.dtype != torch.long:
            raise TypeError("endpoint_aatype must be int64 [batch, graph]")
        if self.endpoint_rotation.shape != layout + (3, 3):
            raise ValueError("endpoint_rotation must be [batch, graph, 3, 3]")
        if self.atom14_xyz.shape != layout + (14, 3):
            raise ValueError("atom14_xyz must be [batch, graph, 14, 3]")
        if self.atom14_mask.shape != layout + (14,) or self.atom14_mask.dtype != torch.bool:
            raise TypeError("atom14_mask must be bool [batch, graph, 14]")
        if self.backbone_xyz.shape != layout + (3, 3):
            raise ValueError("backbone_xyz must be [batch, graph, N_CA_C, xyz]")
        if (
            self.backbone_atom_mask.shape != layout + (3,)
            or self.backbone_atom_mask.dtype != torch.bool
        ):
            raise TypeError("backbone_atom_mask must be bool [batch, graph, 3]")
        if self.backbone_angles_sin_cos.shape != layout + (BACKBONE_ANGLE_SLOTS, 2):
            raise ValueError("backbone_angles_sin_cos must be [batch, graph, 3, 2]")
        if (
            self.backbone_angle_mask.shape != layout + (BACKBONE_ANGLE_SLOTS,)
            or self.backbone_angle_mask.dtype != torch.bool
        ):
            raise TypeError("backbone_angle_mask must be bool [batch, graph, 3]")
        if self.sidechain_angles_sin_cos.shape != layout + (SIDECHAIN_ANGLE_SLOTS, 2):
            raise ValueError("sidechain_angles_sin_cos must be [batch, graph, 4, 2]")
        if (
            self.sidechain_angle_mask.shape != layout + (SIDECHAIN_ANGLE_SLOTS,)
            or self.sidechain_angle_mask.dtype != torch.bool
        ):
            raise TypeError("sidechain_angle_mask must be bool [batch, graph, 4]")
        geometry = (
            self.endpoint_translation,
            self.endpoint_rotation,
            self.atom14_xyz,
            self.backbone_xyz,
            self.backbone_angles_sin_cos,
            self.sidechain_angles_sin_cos,
        )
        if any(value.dtype != torch.float32 for value in geometry):
            raise TypeError("native target geometry must be float32")
        _same_device(tuple(self.__dict__.values()), "targets")

    @property
    def layout(self) -> torch.Size:
        return self.endpoint_translation.shape[:-1]

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PeptideNativeTargets":
        return PeptideNativeTargets(
            **{
                name: value.to(device=device, non_blocking=non_blocking)
                for name, value in self.__dict__.items()
            }
        )

    def endpoint_state(self, condition: UnifiedComplexCondition) -> JointFlowState:
        if self.layout != condition.layout:
            raise ValueError("target and condition layouts differ")
        peptide = condition.peptide_mask
        return JointFlowState(
            translation=torch.where(
                peptide[..., None], self.endpoint_translation, condition.pocket_translation
            ),
            rotation=torch.where(
                peptide[..., None, None], self.endpoint_rotation, condition.pocket_rotation
            ),
            sequence_logits=sequence_endpoint_logits(
                self.endpoint_aatype,
                peptide,
                epsilon=SEQUENCE_EPSILON,
            ),
        )

    def validate_for(self, condition: UnifiedComplexCondition) -> None:
        """Validate supervision at the data boundary without exposing it to the model."""

        if self.layout != condition.layout:
            raise ValueError("target and condition layouts differ")
        outside_peptide = ~condition.peptide_mask
        identity = torch.eye(3, device=self.endpoint_rotation.device).expand_as(
            self.endpoint_rotation
        )
        violations = (
            (self.endpoint_aatype[outside_peptide] != UNKNOWN_AATYPE).any()
            | (self.endpoint_aatype[condition.peptide_mask] < 0).any()
            | (self.endpoint_aatype[condition.peptide_mask] >= AMINO_ACID_TYPES).any()
            | (self.endpoint_translation[outside_peptide] != 0).any()
            | (self.endpoint_rotation[outside_peptide] != identity[outside_peptide]).any()
            | (self.atom14_xyz[outside_peptide] != 0).any()
            | (self.atom14_mask & ~condition.peptide_mask[..., None]).any()
            | (self.backbone_xyz[outside_peptide] != 0).any()
            | (self.backbone_atom_mask & ~condition.peptide_mask[..., None]).any()
            | (self.backbone_angles_sin_cos[outside_peptide] != 0).any()
            | (self.backbone_angle_mask & ~condition.peptide_mask[..., None]).any()
            | (self.sidechain_angles_sin_cos[outside_peptide] != 0).any()
            | (self.sidechain_angle_mask & ~condition.peptide_mask[..., None]).any()
            | ~torch.isfinite(self.atom14_xyz).all()
            | ~torch.isfinite(self.backbone_xyz).all()
            | ~torch.isfinite(self.backbone_angles_sin_cos).all()
            | ~torch.isfinite(self.sidechain_angles_sin_cos).all()
        )
        if bool(violations.detach().cpu()):
            raise ValueError("PeptideNativeTargets invariant failed")
        peptide = condition.peptide_mask
        observed = self.atom14_xyz[peptide][:, :3]
        observed_mask = self.atom14_mask[peptide][:, :3]
        if not torch.equal(self.backbone_atom_mask[peptide], observed_mask) or not torch.equal(
            self.backbone_xyz[peptide], observed
        ):
            raise ValueError("native backbone labels must preserve observed atom coordinates")
        frames = observed_backbone_frames(observed, observed_mask)
        translation_matches = torch.allclose(
            self.endpoint_translation[peptide], frames.translation, atol=1e-5, rtol=0
        )
        rotation_matches = torch.allclose(
            self.endpoint_rotation[peptide], frames.rotation, atol=1e-5, rtol=0
        )
        if not translation_matches or not rotation_matches:
            raise ValueError("native endpoint frames must come from observed N/CA/C")
        self.endpoint_state(condition).validate(condition.residue_mask)


@dataclass(frozen=True)
class UnifiedComplexEncoding:
    """Cached static encoder output; it contains no native peptide geometry."""

    single: Tensor
    pair: Tensor

    def __post_init__(self) -> None:
        if self.single.ndim != 3:
            raise ValueError("encoding.single must be [batch, graph, channel]")
        if self.pair.ndim != 4 or self.pair.shape[:3] != (
            self.single.shape[0],
            self.single.shape[1],
            self.single.shape[1],
        ):
            raise ValueError("encoding.pair must be [batch, graph, graph, channel]")
        if self.single.device != self.pair.device:
            raise ValueError("encoding tensors must be on one device")


@dataclass(frozen=True)
class JointEndpointPrediction:
    """Final joint endpoint plus optional per-block trajectory and angle head."""

    translation: Tensor
    rotation: Tensor
    sequence_logits: Tensor
    block_translation: Tensor | None
    block_rotation: Tensor | None
    block_sequence_logits: Tensor | None
    unnormalized_backbone_angles: Tensor
    backbone_angles_sin_cos: Tensor
    unnormalized_sidechain_angles: Tensor
    sidechain_angles_sin_cos: Tensor

    def __post_init__(self) -> None:
        if self.translation.ndim != 3 or self.translation.shape[-1] != 3:
            raise ValueError("prediction.translation must be [batch, graph, 3]")
        layout = self.translation.shape[:-1]
        if self.rotation.shape != layout + (3, 3):
            raise ValueError("prediction.rotation must be [batch, graph, 3, 3]")
        if self.sequence_logits.shape != layout + (AMINO_ACID_TYPES,):
            raise ValueError("prediction.sequence_logits must be [batch, graph, 20]")
        angle_shape = layout + (BACKBONE_ANGLE_SLOTS, 2)
        if (
            self.unnormalized_backbone_angles.shape != angle_shape
            or self.backbone_angles_sin_cos.shape != angle_shape
        ):
            raise ValueError("prediction angles must be [batch, graph, 3, 2]")
        sidechain_shape = layout + (SIDECHAIN_ANGLE_SLOTS, 2)
        if (
            self.unnormalized_sidechain_angles.shape != sidechain_shape
            or self.sidechain_angles_sin_cos.shape != sidechain_shape
        ):
            raise ValueError("prediction sidechain angles must be [batch, graph, 4, 2]")
        intermediates = (
            self.block_translation,
            self.block_rotation,
            self.block_sequence_logits,
        )
        if (self.block_translation is None) != (self.block_rotation is None):
            raise ValueError("block frame trajectory tensors must be present together")
        if self.block_translation is not None:
            if (
                self.block_translation.ndim != 4
                or self.block_translation.shape[:1] + self.block_translation.shape[2:]
                != self.translation.shape
            ):
                raise ValueError("block_translation must be [batch, blocks, graph, 3]")
            expected_rotation = self.block_translation.shape[:-1] + (3, 3)
            if self.block_rotation is None or self.block_rotation.shape != expected_rotation:
                raise ValueError("block_rotation must be [batch, blocks, graph, 3, 3]")
        if self.block_sequence_logits is not None:
            if self.block_translation is None:
                raise ValueError("block sequence logits require a block frame trajectory")
            expected_sequence = self.block_translation.shape[:-1] + (AMINO_ACID_TYPES,)
            if self.block_sequence_logits.shape != expected_sequence:
                raise ValueError(
                    "block_sequence_logits must be [batch, blocks, graph, 20]"
                )
        geometry = (
            self.translation,
            self.rotation,
            self.sequence_logits,
            self.unnormalized_backbone_angles,
            self.backbone_angles_sin_cos,
            self.unnormalized_sidechain_angles,
            self.sidechain_angles_sin_cos,
        )
        if any(value.dtype != torch.float32 for value in geometry):
            raise TypeError("prediction state and angle tensors must be float32")
        if any(
            value is not None and value.dtype != torch.float32
            for value in intermediates
        ):
            raise TypeError("prediction block trajectory tensors must be float32")
        if debug_invariants_enabled():
            self.validate()

    def validate(self) -> None:
        """Run explicit value checks outside normal compiled forward paths."""

        intermediates = (
            self.block_translation,
            self.block_rotation,
            self.block_sequence_logits,
        )
        finite_values = (
            self.translation,
            self.rotation,
            self.sequence_logits,
            self.unnormalized_backbone_angles,
            self.backbone_angles_sin_cos,
            self.unnormalized_sidechain_angles,
            self.sidechain_angles_sin_cos,
        ) + tuple(value for value in intermediates if value is not None)
        if any(not bool(torch.isfinite(value).all()) for value in finite_values):
            raise ValueError("prediction tensors must be finite")
        if bool((self.sequence_logits.sum(-1).abs() > 1e-4).any()):
            raise ValueError("prediction sequence logits must be centered")
        if self.block_sequence_logits is not None and bool(
            (self.block_sequence_logits.sum(-1).abs() > 1e-4).any()
        ):
            raise ValueError("prediction block sequence logits must be centered")
