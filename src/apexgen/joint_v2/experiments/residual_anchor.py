# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal OpenFold monomer StructureModule adapted for Joint-v2.

The IPA, transition, update and angle-head formulas are adapted from OpenFold
commit be2ec1841f16c966c65ae0e7599ebbadc725757d. Joint-v2 deliberately omits
the optional CUDA/LMA/offload paths and atom14 coordinate reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apexgen.joint_v2.model.variants.neighbor_attention import peptide_neighbor_attention

from apexgen.joint_v2.contracts.contract import (
    JointEndpointPrediction,
    UnifiedComplexCondition,
    UnifiedComplexEncoding,
)
from apexgen.joint_v2.contracts.state import (
    AMINO_ACID_TYPES,
    BACKBONE_ANGLE_SLOTS,
    JointFlowState,
    SIDECHAIN_ANGLE_SLOTS,
    TORSION_ANGLE_SLOTS,
    center_sequence_logits,
)


from apexgen.joint_v2.model.structure_module import JointV2StructureModule, Rigid
from apexgen.joint_v2.runtime.lineage import canonical_sha256

from apexgen.joint_v2.model.variants.dynamic_pair import DynamicPairStructureModule, pair_features

CONTRACT = dict(schema="apexgen.joint_v2.residual_anchor.v1",
    skip="S_initial + alpha*(S_current-S_initial), full residue single stream; initial is projected encoder single, not detached",
    other_paths="IPA gets current S+Q, current geometry and dynamic pair; Q transition and both norms unchanged",
    identity="alpha=1 uses original skip exactly; first block always uses original single; no added parameters",
    scope="Exploratory fixed-sequence geometry continuation, not formal training")
CONTRACT_SHA256 = canonical_sha256(CONTRACT)

class ResidualAnchorStructureModule(DynamicPairStructureModule):
    def __init__(self, base, alpha):
        nn.Module.__init__(self)
        if not 0. <= alpha <= 1.: raise ValueError(alpha)
        if not hasattr(base, 'dynamic_pair'): raise ValueError('Dynamic-pair parent required')
        for name in ("sequence_output_parameterization", "sequence_attention_topology",
                     "blocks", "translation_scale", "stop_rotation_gradient"):
            setattr(self, name, getattr(base, name))
        for name, child in base.named_children(): self.add_module(name, child)
        self.residual_alpha = float(alpha)
        self.train(base.training)

    def forward(
        self,
        state: JointFlowState,
        time: Tensor,
        condition: UnifiedComplexCondition,
        encoding: UnifiedComplexEncoding,
        *,
        return_intermediates: bool = True,
        task: str = "J",
        trace: list | None = None,
        initialize_sequence_latent: bool = True,
    ) -> JointEndpointPrediction:
        if task not in {"J", "S", "G_s", "G_0"}:
            raise ValueError(f"unknown refinement task {task}")
        if state.layout != condition.layout or encoding.single.shape[:2] != condition.layout:
            raise ValueError("decoder state, condition and encoding layouts differ")
        if time.shape != (condition.layout[0],):
            raise ValueError("time must be [batch]")
        single = self.layer_norm_s(encoding.single)
        pair = self.layer_norm_z(encoding.pair)
        initial_single = single
        single = self.linear_in(single)
        anchor_single = single
        time_embedding = self.time_conditioner(time)
        gamma, beta = self.time_film(time_embedding).chunk(2, dim=-1)
        block_update_gate = (1.0 - time.float())[:, None, None]
        peptide = condition.peptide_mask[..., None]
        fixed_state = JointFlowState(
            translation=torch.where(
                condition.peptide_mask[..., None],
                state.translation,
                condition.pocket_translation,
            ),
            rotation=torch.where(
                condition.peptide_mask[..., None, None],
                state.rotation,
                condition.pocket_rotation,
            ),
            sequence_logits=torch.where(
                peptide,
                state.sequence_logits,
                torch.zeros_like(state.sequence_logits),
            ),
        )
        rigid = Rigid.from_state(fixed_state, self.translation_scale)
        sequence_single = (
            torch.zeros_like(single)
            if task == "G_0" or not initialize_sequence_latent
            else self.sequence_projection(self.sequence_norm(fixed_state.sequence_logits))
        )
        sequence_single = torch.where(peptide, sequence_single, 0.0)
        translations: list[Tensor] = []
        rotations: list[Tensor] = []

        final_state = fixed_state
        attention_mask = (
            peptide_neighbor_attention(condition, self.ipa.no_heads)
            if self.sequence_attention_topology == "neighbors"
            else None
        )
        for block_index in range(self.blocks):
            geometry = pair_features(rigid, condition.residue_mask, condition.peptide_mask, self.translation_scale)
            dynamic_pair = self.dynamic_pair(geometry)
            current_pair = pair + dynamic_pair.to(pair.dtype)
            combined = single + sequence_single
            conditioned = combined * (1.0 + peptide * gamma[:, None]) + peptide * beta[:, None]
            incoming_single = single
            skip_single = (single if self.residual_alpha == 1. or block_index == 0
                           else anchor_single + self.residual_alpha * (single - anchor_single))
            if attention_mask is None:
                ipa_message = self.ipa(conditioned, current_pair, rigid, condition.residue_mask)
            else:
                ipa_message = self.ipa(conditioned, current_pair, rigid, condition.residue_mask,
                                       attention_mask=attention_mask)
            single = skip_single + ipa_message
            single = self.layer_norm_ipa(self.ipa_dropout(single))
            single = self.transition(single)
            update = (
                torch.zeros(*single.shape[:2], 6, device=single.device, dtype=single.dtype)
                if task == "S"
                else self.backbone_update(single) * block_update_gate
            )
            with torch.autocast(device_type=single.device.type, enabled=False):
                update = update.float()
                if task != "S":
                    proposed = rigid.compose_q_update_vec(update)
                    rigid = rigid.masked(proposed, condition.peptide_mask)
                raw_export = rigid.export(self.translation_scale, fixed_state.sequence_logits)
                exported = JointFlowState(
                    translation=torch.where(
                        peptide,
                        raw_export.translation if task != "S" else fixed_state.translation,
                        fixed_state.translation,
                    ),
                    rotation=torch.where(
                        peptide[..., None],
                        raw_export.rotation if task != "S" else fixed_state.rotation,
                        fixed_state.rotation,
                    ),
                    sequence_logits=torch.where(
                        peptide, raw_export.sequence_logits, fixed_state.sequence_logits
                    ),
                )
                final_state = exported
            if return_intermediates:
                translations.append(exported.translation)
                rotations.append(exported.rotation)
            if task != "G_0":
                sequence_single = self.sequence_latent_transition(
                    sequence_single,
                    single,
                    condition.peptide_mask,
                )
            if trace is not None:
                entry = {
                    "iteration": block_index,
                    "incoming_single": incoming_single,
                    "skip_single": skip_single,
                    "ipa_message": ipa_message,
                    "anchor_single": anchor_single,
                    "dynamic_geometry": geometry,
                    "dynamic_pair": dynamic_pair,
                    "current_pair": current_pair,
                    "single": single,
                    "sequence_latent": sequence_single,
                    "update": update,
                    "translation": exported.translation,
                    "rotation": exported.rotation,
                }
                for value in entry.values():
                    if isinstance(value, Tensor) and value.requires_grad:
                        value.retain_grad()
                trace.append(entry)
            if self.stop_rotation_gradient and block_index + 1 < self.blocks:
                rigid = rigid.stop_rotation_gradient()

        if task in {"J", "S"}:
            sequence_single = self.final_sequence_resnet(single, initial_single, sequence_single)
        with torch.autocast(device_type=single.device.type, enabled=False):
            sequence_delta = (
                self.sequence_update(sequence_single.float()).float()
                if task in {"J", "S"}
                else torch.zeros_like(fixed_state.sequence_logits)
            )
            proposed_sequence = (
                center_sequence_logits(
                    fixed_state.sequence_logits + sequence_delta * block_update_gate
                    if self.sequence_output_parameterization == "residual"
                    else sequence_delta
                )
                if task in {"J", "S"}
                else fixed_state.sequence_logits
            )
            sequence = torch.where(peptide, proposed_sequence, 0.0)
            raw_export = rigid.export(self.translation_scale, sequence)
            final_state = JointFlowState(
                translation=torch.where(
                    peptide,
                    raw_export.translation if task != "S" else fixed_state.translation,
                    fixed_state.translation,
                ),
                rotation=torch.where(
                    peptide[..., None],
                    raw_export.rotation if task != "S" else fixed_state.rotation,
                    fixed_state.rotation,
                ),
                sequence_logits=torch.where(
                    peptide, raw_export.sequence_logits, fixed_state.sequence_logits
                ),
            )

        unnormalized_angles, angles = self.angle_resnet(single, initial_single)
        peptide_angles = condition.peptide_mask[..., None, None]
        unnormalized_angles = torch.where(peptide_angles, unnormalized_angles, 0.0)
        angles = torch.where(peptide_angles, angles, 0.0)
        unnormalized_backbone = unnormalized_angles[..., :BACKBONE_ANGLE_SLOTS, :]
        backbone_angles = angles[..., :BACKBONE_ANGLE_SLOTS, :]
        unnormalized_sidechain = unnormalized_angles[..., BACKBONE_ANGLE_SLOTS:, :]
        sidechain_angles = angles[..., BACKBONE_ANGLE_SLOTS:, :]
        return JointEndpointPrediction(
            translation=final_state.translation,
            rotation=final_state.rotation,
            sequence_logits=final_state.sequence_logits,
            block_translation=(torch.stack(translations, dim=1) if translations else None),
            block_rotation=(torch.stack(rotations, dim=1) if rotations else None),
            block_sequence_logits=None,
            unnormalized_backbone_angles=unnormalized_backbone,
            backbone_angles_sin_cos=backbone_angles,
            unnormalized_sidechain_angles=unnormalized_sidechain,
            sidechain_angles_sin_cos=sidechain_angles,
        )
