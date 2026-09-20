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


_TRUNCATED_NORMAL_STD = 0.8796256610342398
_IPA_POINT_WEIGHT_INITIAL = 0.541324854612918


class OpenFoldLinear(nn.Linear):
    """Linear layer with the initializers used by the pinned OpenFold source."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        init: str = "default",
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if init == "final":
            nn.init.zeros_(self.weight)
        elif init == "normal":
            nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        elif init in {"default", "relu"}:
            scale = 1.0 if init == "default" else 2.0
            std = math.sqrt(scale / max(1, in_features)) / _TRUNCATED_NORMAL_STD
            nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        elif init == "glorot":
            nn.init.xavier_uniform_(self.weight)
        else:
            raise ValueError(f"unknown OpenFold linear initializer: {init}")


@dataclass(frozen=True)
class Rigid:
    """Small matrix-backed rigid transform used only inside StructureModule."""

    rotation: Tensor  # float32 [B, N, 3, 3]
    translation: Tensor  # float32 [B, N, 3], internal scale

    @classmethod
    def from_state(cls, state: JointFlowState, translation_scale: float) -> "Rigid":
        return cls(state.rotation.float(), state.translation.float() / translation_scale)

    def apply(self, local_points: Tensor) -> Tensor:
        translation = self.translation.reshape(
            self.translation.shape[:2] + (1,) * (local_points.ndim - 3) + (3,)
        )
        return torch.einsum("bnij,bn...j->bn...i", self.rotation, local_points) + translation

    def invert_apply(self, global_points: Tensor) -> Tensor:
        translation = self.translation.reshape(
            self.translation.shape[:2] + (1,) * (global_points.ndim - 3) + (3,)
        )
        return torch.einsum("bnji,bn...j->bn...i", self.rotation, global_points - translation)

    def compose_q_update_vec(self, update: Tensor) -> "Rigid":
        """Compose OpenFold's local ``(1, qx, qy, qz), translation`` update."""

        update = update.float()
        if update.shape != self.translation.shape[:-1] + (6,):
            raise ValueError("rigid update must be float32 [batch, graph, 6]")
        vector = update[..., :3]
        one = torch.ones_like(vector[..., :1])
        quaternion = torch.cat((one, vector), dim=-1)
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = quaternion.unbind(-1)
        delta_rotation = torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(self.rotation.shape)
        rotation = self.rotation @ delta_rotation
        translation = self.translation + torch.einsum(
            "bnij,bnj->bni", self.rotation, update[..., 3:]
        )
        return Rigid(rotation, translation)

    def masked(self, proposed: "Rigid", update_mask: Tensor) -> "Rigid":
        return Rigid(
            torch.where(update_mask[..., None, None], proposed.rotation, self.rotation),
            torch.where(update_mask[..., None], proposed.translation, self.translation),
        )

    def stop_rotation_gradient(self) -> "Rigid":
        return Rigid(self.rotation.detach(), self.translation)

    def export(self, translation_scale: float, sequence_logits: Tensor) -> JointFlowState:
        return JointFlowState(
            translation=self.translation.float() * translation_scale,
            rotation=self.rotation.float(),
            sequence_logits=sequence_logits.float(),
        )


class PointProjection(nn.Module):
    def __init__(self, c_s: int, points: int, heads: int) -> None:
        super().__init__()
        self.points = points
        self.heads = heads
        self.linear = OpenFoldLinear(c_s, heads * points * 3)

    def forward(self, single: Tensor, rigid: Rigid) -> Tensor:
        local = self.linear(single)
        # OpenFold stores all x, then all y, then all z coordinates.
        local = torch.stack(torch.chunk(local, 3, dim=-1), dim=-1)
        local = local.reshape(*single.shape[:-1], self.heads, self.points, 3).float()
        with torch.autocast(device_type=single.device.type, enabled=False):
            return rigid.apply(local)


class InvariantPointAttention(nn.Module):
    """Pure-PyTorch monomer IPA matching the pinned OpenFold mathematical path."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        *,
        inf: float = 1e5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        hidden = c_hidden * no_heads
        self.linear_q = OpenFoldLinear(c_s, hidden)
        self.linear_kv = OpenFoldLinear(c_s, 2 * hidden)
        self.linear_q_points = PointProjection(c_s, no_qk_points, no_heads)
        self.linear_kv_points = PointProjection(c_s, no_qk_points + no_v_points, no_heads)
        self.linear_b = OpenFoldLinear(c_z, no_heads)
        self.head_weights = nn.Parameter(torch.full((no_heads,), _IPA_POINT_WEIGHT_INITIAL))
        output_dim = no_heads * (c_z + c_hidden + no_v_points * 4)
        self.linear_out = OpenFoldLinear(output_dim, c_s, init="final")

    def forward(
        self, single: Tensor, pair: Tensor, rigid: Rigid, mask: Tensor,
        *, attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, length, _ = single.shape
        if pair.shape != (batch, length, length, self.c_z):
            raise ValueError("IPA pair layout differs from single layout")
        if mask.shape != (batch, length) or mask.dtype != torch.bool:
            raise TypeError("IPA mask must be bool [batch, graph]")
        if attention_mask is not None and (
            attention_mask.shape != (batch, self.no_heads, length, length)
            or attention_mask.dtype != torch.bool
        ):
            raise ValueError("IPA attention mask must be bool [B,H,N,N]")

        q = self.linear_q(single).reshape(batch, length, self.no_heads, self.c_hidden)
        kv = self.linear_kv(single).reshape(batch, length, self.no_heads, 2 * self.c_hidden)
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        q_points = self.linear_q_points(single, rigid)
        kv_points = self.linear_kv_points(single, rigid)
        k_points, v_points = torch.split(kv_points, (self.no_qk_points, self.no_v_points), dim=-2)
        pair_logits = self.linear_b(pair)

        with torch.autocast(device_type=single.device.type, enabled=False):
            scalar_logits = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            scalar_logits = scalar_logits * math.sqrt(1.0 / (3.0 * self.c_hidden))
            pair_logits = pair_logits.float().permute(0, 3, 1, 2)
            pair_logits = pair_logits * math.sqrt(1.0 / 3.0)
            distance_squared = (
                (q_points[:, :, None] - k_points[:, None, :]).square().sum(dim=(-1, -2))
            )
            point_weight = F.softplus(self.head_weights.float()) * math.sqrt(
                1.0 / (3.0 * (self.no_qk_points * 9.0 / 2.0))
            )
            point_logits = -0.5 * distance_squared.permute(0, 3, 1, 2)
            point_logits = point_logits * point_weight[None, :, None, None]
            pair_mask = mask[:, :, None] & mask[:, None, :]
            logits = scalar_logits + pair_logits + point_logits
            logits = logits + self.inf * (pair_mask[:, None].float() - 1.0)
            if attention_mask is not None:
                # Padded queries are discarded below; avoid all -inf softmax.
                allowed = attention_mask | ~mask[:, None, :, None]
                logits = logits.masked_fill(~allowed, -torch.inf)
            attention = torch.softmax(logits, dim=-1)

            scalar_output = torch.einsum("bhij,bjhc->bihc", attention, v.float())
            scalar_output = scalar_output.flatten(-2)
            point_output_global = torch.einsum("bhij,bjhpc->bihpc", attention, v_points.float())
            point_output_local = rigid.invert_apply(point_output_global)
            point_norm = torch.sqrt(point_output_local.square().sum(-1) + self.eps).flatten(-2)
            point_flat = point_output_local.reshape(batch, length, -1, 3)
            point_x, point_y, point_z = point_flat.unbind(-1)
            pair_output = torch.einsum("bhij,bijc->bihc", attention, pair.float()).flatten(-2)
            combined = torch.cat(
                (
                    scalar_output,
                    point_x,
                    point_y,
                    point_z,
                    point_norm,
                    pair_output,
                ),
                dim=-1,
            )
        output = self.linear_out(combined.to(pair.dtype))
        return torch.where(mask[..., None], output, 0.0)


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c_s: int) -> None:
        super().__init__()
        self.linear_1 = OpenFoldLinear(c_s, c_s, init="relu")
        self.linear_2 = OpenFoldLinear(c_s, c_s, init="relu")
        self.linear_3 = OpenFoldLinear(c_s, c_s, init="final")

    def forward(self, single: Tensor) -> Tensor:
        update = self.linear_1(single)
        update = F.relu(update)
        update = self.linear_2(update)
        update = F.relu(update)
        return single + self.linear_3(update)


class StructureModuleTransition(nn.Module):
    def __init__(self, c_s: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(StructureModuleTransitionLayer(c_s) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(c_s)

    def forward(self, single: Tensor) -> Tensor:
        for layer in self.layers:
            single = layer(single)
        return self.layer_norm(self.dropout(single))


class BackboneUpdate(nn.Module):
    def __init__(self, c_s: int) -> None:
        super().__init__()
        self.linear = OpenFoldLinear(c_s, 6, init="final")

    def forward(self, single: Tensor) -> Tensor:
        return self.linear(single)


class AngleResnetBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear_1 = OpenFoldLinear(hidden, hidden, init="relu")
        self.linear_2 = OpenFoldLinear(hidden, hidden, init="final")

    def forward(self, value: Tensor) -> Tensor:
        update = self.linear_1(F.relu(value))
        return value + self.linear_2(F.relu(update))


class AngleResnet(nn.Module):
    def __init__(
        self,
        c_s: int,
        hidden: int,
        blocks: int,
        angles: int = 3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.angles = angles
        self.eps = eps
        self.linear_in = OpenFoldLinear(c_s, hidden)
        self.linear_initial = OpenFoldLinear(c_s, hidden)
        self.layers = nn.ModuleList(AngleResnetBlock(hidden) for _ in range(blocks))
        self.linear_out = OpenFoldLinear(hidden, angles * 2)
        self.sidechain_linear_out: OpenFoldLinear | None = None

    def add_sidechain_angles(self, angles: int) -> None:
        """Append chi outputs without perturbing initialization of existing modules."""

        if angles <= 0 or self.sidechain_linear_out is not None:
            raise ValueError("sidechain angle output must be added exactly once")
        self.sidechain_linear_out = OpenFoldLinear(
            self.linear_out.in_features,
            angles * 2,
        )
        self.angles += angles

    def forward(self, single: Tensor, initial_single: Tensor) -> tuple[Tensor, Tensor]:
        value = self.linear_in(F.relu(single)) + self.linear_initial(F.relu(initial_single))
        for layer in self.layers:
            value = layer(value)
        activated = F.relu(value)
        outputs = [self.linear_out(activated)]
        if self.sidechain_linear_out is not None:
            outputs.append(self.sidechain_linear_out(activated))
        unnormalized = torch.cat(outputs, dim=-1).reshape(*value.shape[:-1], self.angles, 2)
        with torch.autocast(device_type=value.device.type, enabled=False):
            unnormalized = unnormalized.float()
            norm = torch.sqrt(unnormalized.square().sum(-1, keepdim=True).clamp_min(self.eps))
            normalized = unnormalized / norm
        return unnormalized, normalized


class FinalSequenceResnet(nn.Module):
    """Final-only sequence head conditioned on structure and sequence latents."""

    def __init__(self, c_s: int, blocks: int) -> None:
        super().__init__()
        self.linear_in = OpenFoldLinear(c_s, c_s)
        self.linear_initial = OpenFoldLinear(c_s, c_s)
        self.linear_sequence = OpenFoldLinear(c_s, c_s)
        self.layers = nn.ModuleList(AngleResnetBlock(c_s) for _ in range(blocks))

    def forward(
        self,
        single: Tensor,
        initial_single: Tensor,
        sequence_context: Tensor,
    ) -> Tensor:
        value = (
            self.linear_in(F.relu(single))
            + self.linear_initial(F.relu(initial_single))
            + self.linear_sequence(F.relu(sequence_context))
        )
        for layer in self.layers:
            value = layer(value)
        return value


class SequenceLatentTransition(nn.Module):
    """Update the peptide sequence latent from its aligned geometry latent."""

    def __init__(self, c_s: int) -> None:
        super().__init__()
        self.sequence_norm = nn.LayerNorm(c_s)
        self.geometry_norm = nn.LayerNorm(c_s)
        self.linear_1 = OpenFoldLinear(2 * c_s, c_s, init="relu")
        self.linear_2 = OpenFoldLinear(c_s, c_s, init="final")

    def forward(
        self,
        sequence_single: Tensor,
        geometry_single: Tensor,
        peptide_mask: Tensor,
    ) -> Tensor:
        if sequence_single.shape != geometry_single.shape:
            raise ValueError("sequence and geometry latents must have matching shapes")
        if peptide_mask.shape != sequence_single.shape[:-1]:
            raise ValueError("peptide mask must match the latent residue layout")
        message = self.linear_1(
            torch.cat(
                (
                    self.sequence_norm(sequence_single),
                    self.geometry_norm(geometry_single),
                ),
                dim=-1,
            )
        )
        message = self.linear_2(F.relu(message))
        return torch.where(
            peptide_mask[..., None],
            sequence_single + message,
            0.0,
        )


class FourierTimeConditioning(nn.Module):
    def __init__(self, embedding_dim: int = 64, frequencies: int = 16) -> None:
        super().__init__()
        self.frequencies = frequencies
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * frequencies, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        time = time.float()
        frequency = torch.arange(1, self.frequencies + 1, device=time.device).float()
        phase = 2.0 * math.pi * time[:, None] * frequency
        return self.mlp(torch.cat((time[:, None], torch.sin(phase), torch.cos(phase)), -1))


class JointV2StructureModule(nn.Module):
    """Shared-block joint endpoint refinement initialized from input ``x_t``."""

    def __init__(
        self,
        *,
        c_s: int,
        c_z: int,
        c_ipa: int = 16,
        c_angle: int = 128,
        no_heads_ipa: int = 12,
        no_qk_points: int = 4,
        no_v_points: int = 8,
        dropout: float = 0.1,
        blocks: int = 8,
        transition_layers: int = 1,
        angle_blocks: int = 2,
        sequence_head_blocks: int = 2,
        translation_scale: float = 10.0,
        time_embedding_dim: int = 64,
        time_frequencies: int = 16,
        eps: float = 1e-8,
        inf: float = 1e5,
        stop_rotation_gradient: bool = True,
        sequence_output_parameterization: str = "residual",
        sequence_attention_topology: str = "global",
    ) -> None:
        super().__init__()
        if sequence_output_parameterization not in {"residual", "direct"}:
            raise ValueError("unknown sequence output parameterization")
        self.sequence_output_parameterization = sequence_output_parameterization
        if sequence_attention_topology not in {"global", "neighbors"}:
            raise ValueError("unknown sequence attention topology")
        if sequence_attention_topology == "neighbors" and no_heads_ipa < 4:
            raise ValueError("neighbor routing requires at least four IPA heads")
        self.sequence_attention_topology = sequence_attention_topology
        self.blocks = blocks
        self.translation_scale = translation_scale
        self.stop_rotation_gradient = stop_rotation_gradient
        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_in = OpenFoldLinear(c_s, c_s)
        self.sequence_norm = nn.LayerNorm(AMINO_ACID_TYPES)
        self.sequence_projection = OpenFoldLinear(AMINO_ACID_TYPES, c_s)
        self.ipa = InvariantPointAttention(
            c_s, c_z, c_ipa, no_heads_ipa, no_qk_points, no_v_points, inf=inf, eps=eps
        )
        self.ipa_dropout = nn.Dropout(dropout)
        self.layer_norm_ipa = nn.LayerNorm(c_s)
        self.transition = StructureModuleTransition(c_s, transition_layers, dropout)
        self.backbone_update = BackboneUpdate(c_s)
        self.sequence_update = OpenFoldLinear(c_s, AMINO_ACID_TYPES, init="final")
        self.time_conditioner = FourierTimeConditioning(time_embedding_dim, time_frequencies)
        self.time_film = nn.Linear(time_embedding_dim, 2 * c_s)
        nn.init.zeros_(self.time_film.weight)
        nn.init.zeros_(self.time_film.bias)
        self.angle_resnet = AngleResnet(
            c_s,
            c_angle,
            angle_blocks,
            angles=BACKBONE_ANGLE_SLOTS,
            eps=eps,
        )
        self.final_sequence_resnet = FinalSequenceResnet(c_s, sequence_head_blocks)
        self.sequence_latent_transition = SequenceLatentTransition(c_s)
        # Initialize this new projection after every ec6a13a module so all
        # pre-existing parameters retain their exact same-seed initialization.
        self.angle_resnet.add_sidechain_angles(SIDECHAIN_ANGLE_SLOTS)
        if self.angle_resnet.angles != TORSION_ANGLE_SLOTS:
            raise RuntimeError("seven-torsion angle-head slot contract is inconsistent")

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
        condition.validate_model_input()
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
            combined = single + sequence_single
            conditioned = combined * (1.0 + peptide * gamma[:, None]) + peptide * beta[:, None]
            if attention_mask is None:
                single = single + self.ipa(conditioned, pair, rigid, condition.residue_mask)
            else:
                single = single + self.ipa(
                    conditioned, pair, rigid, condition.residue_mask, attention_mask=attention_mask
                )
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
