"""Static unified pocket--peptide encoder for frame-endpoint Joint-v2."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from apexgen.shared.storage.features import distance_rbf, rosetta_orientations, virtual_cb
from apexgen.joint_v2.contracts.contract import UnifiedComplexCondition, UnifiedComplexEncoding
from apexgen.joint_v2.contracts.state import UNKNOWN_AATYPE


ENCODER_FEATURE_MODES = (
    "full",
    "no_metadata_torsions",
    "backbone_atoms",
    "compact_geometry",
)


def _feature_switches(mode: str) -> dict[str, bool]:
    if mode not in ENCODER_FEATURE_MODES:
        raise ValueError(f"unsupported encoder feature mode: {mode}")
    return {
        "pocket_position": mode == "full",
        "pocket_core": mode == "full",
        "torsions": mode == "full",
        "all_atoms": mode in {"full", "no_metadata_torsions"},
        "cb_geometry": mode != "compact_geometry",
        "rosetta_orientation": mode != "compact_geometry",
    }


def _normalized_position(mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Tensor-layout rank only; polymer separation uses condition.sequence_index."""
    rank = mask.long().cumsum(-1) - 1
    denominator = (mask.sum(-1) - 1).clamp_min(1).to(dtype)
    return torch.where(mask, rank.to(dtype) / denominator[:, None], 0.0)


class EncoderBlock(nn.Module):
    """Pair-biased scalar attention followed by single/pair transitions."""

    def __init__(self, c_s: int, c_z: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.single_norm = nn.LayerNorm(c_s)
        self.attention = nn.MultiheadAttention(c_s, heads, dropout=dropout, batch_first=True)
        self.pair_bias = nn.Linear(c_z, heads, bias=False)
        self.pair_context = nn.Sequential(nn.LayerNorm(c_z), nn.Linear(c_z, c_s), nn.SiLU())
        self.single_transition = nn.Sequential(
            nn.LayerNorm(c_s), nn.Linear(c_s, 4 * c_s), nn.SiLU(), nn.Linear(4 * c_s, c_s)
        )
        self.left_pair = nn.Linear(c_s, c_z)
        self.right_pair = nn.Linear(c_s, c_z)
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(c_z), nn.Linear(c_z, 2 * c_z), nn.SiLU(), nn.Linear(2 * c_z, c_z)
        )

    def forward(self, single: Tensor, pair: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        pair_mask = mask[:, :, None] & mask[:, None, :]
        batch, length = mask.shape
        bias = self.pair_bias(pair).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~mask[:, None, None, :], -torch.inf)
        update, _ = self.attention(
            self.single_norm(single),
            self.single_norm(single),
            self.single_norm(single),
            attn_mask=bias.reshape(batch * self.heads, length, length),
            need_weights=False,
        )
        weights = pair_mask[..., None].to(pair.dtype)
        context = (pair * weights).sum(2) / weights.sum(2).clamp_min(1.0)
        single = single + update + self.pair_context(context)
        single = single + self.single_transition(single)
        single = torch.where(mask[..., None], single, 0.0)
        pair = pair + self.left_pair(single)[:, :, None] + self.right_pair(single)[:, None, :]
        pair = pair + self.pair_transition(pair)
        pair = torch.where(pair_mask[..., None], pair, 0.0)
        return single, pair


class UnifiedComplexEncoder(nn.Module):
    """Encode pocket chemistry/geometry and peptide length placeholders once."""

    _ATOM_FEATURES = 38 * 4
    _PAIR_FEATURES = 64

    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        blocks: int,
        heads: int,
        encoder_single_dim: int | None = None,
        encoder_pair_dim: int | None = None,
        feature_mode: str = "full",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_mode not in ENCODER_FEATURE_MODES:
            raise ValueError(
                f"feature_mode must be one of {ENCODER_FEATURE_MODES}, got {feature_mode!r}"
            )
        self.feature_mode = feature_mode
        encoder_single_dim = single_dim if encoder_single_dim is None else encoder_single_dim
        encoder_pair_dim = pair_dim if encoder_pair_dim is None else encoder_pair_dim
        self.role_projection = nn.Sequential(
            nn.Linear(3, encoder_single_dim),
            nn.SiLU(),
            nn.Linear(encoder_single_dim, encoder_single_dim),
        )
        self.aatype_embedding = nn.Embedding(UNKNOWN_AATYPE + 1, encoder_single_dim)
        self.atom_projection = nn.Linear(self._ATOM_FEATURES, encoder_single_dim)
        self.core_embedding = nn.Embedding(2, encoder_single_dim)
        self.backbone_angle_projection = nn.Sequential(
            nn.Linear(9, encoder_single_dim),
            nn.SiLU(),
            nn.Linear(encoder_single_dim, encoder_single_dim),
        )
        self.sidechain_angle_projection = nn.Sequential(
            nn.Linear(12, encoder_single_dim),
            nn.SiLU(),
            nn.Linear(encoder_single_dim, encoder_single_dim),
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(self._PAIR_FEATURES, encoder_pair_dim),
            nn.SiLU(),
            nn.Linear(encoder_pair_dim, encoder_pair_dim),
        )
        self.blocks = nn.ModuleList(
            EncoderBlock(encoder_single_dim, encoder_pair_dim, heads, dropout)
            for _ in range(blocks)
        )
        self.single_norm = nn.LayerNorm(encoder_single_dim)
        self.pair_norm = nn.LayerNorm(encoder_pair_dim)
        self.single_output_projection = (
            nn.Identity()
            if encoder_single_dim == single_dim
            else nn.Linear(encoder_single_dim, single_dim)
        )
        self.pair_output_projection = (
            nn.Identity() if encoder_pair_dim == pair_dim else nn.Linear(encoder_pair_dim, pair_dim)
        )

    def _static_features(self, condition: UnifiedComplexCondition) -> tuple[Tensor, Tensor, Tensor]:
        """Construct all coordinate features in an explicit FP32 geometry island."""

        switches = _feature_switches(self.feature_mode)
        pocket = condition.pocket_mask
        peptide = condition.peptide_mask
        pocket_position = _normalized_position(pocket, torch.float32)
        peptide_position = _normalized_position(peptide, torch.float32)
        encoded_pocket_position = pocket_position if switches["pocket_position"] else 0.0
        single = torch.stack(
            (pocket.float(), peptide.float(), encoded_pocket_position + peptide_position), dim=-1
        )

        pocket_pair = pocket[:, :, None] & pocket[:, None, :]
        geometry_weight = pocket_pair[..., None].float()
        translation = condition.pocket_translation
        rotation = condition.pocket_rotation
        distance = torch.cdist(translation, translation)
        centers = torch.linspace(0.0, 30.0, 16, device=distance.device)
        rbf = torch.exp(-((distance[..., None] - centers) / 2.0).square()) * geometry_weight
        relative_rotation = torch.einsum("bnji,bmjk->bnmik", rotation, rotation).flatten(-2)
        displacement = translation[:, None] - translation[:, :, None]
        relative_translation = torch.einsum("bnji,bnmj->bnmi", rotation, displacement)
        role_index = peptide.long()
        role_pair = torch.nn.functional.one_hot(
            role_index[:, :, None] * 2 + role_index[:, None, :], 4
        ).float()
        pocket_delta = (pocket_position[:, None] - pocket_position[:, :, None])[..., None]
        peptide_delta = (peptide_position[:, None] - peptide_position[:, :, None])[..., None]
        pocket_delta = pocket_delta * pocket_pair[..., None]
        if not switches["pocket_position"]:
            pocket_delta = torch.zeros_like(pocket_delta)
        peptide_delta = peptide_delta * (peptide[:, :, None] & peptide[:, None, :])[..., None]
        same_role = (role_index[:, :, None] == role_index[:, None, :])[..., None].float()
        base_pair = torch.cat(
            (
                rbf,
                relative_rotation * geometry_weight,
                relative_translation * geometry_weight,
                role_pair,
                pocket_delta,
                peptide_delta,
                same_role,
            ),
            dim=-1,
        )

        xyz = condition.pocket_atom_xyz
        atom_mask = condition.pocket_atom_mask
        n = torch.where(atom_mask[..., 0, None], xyz[..., 0, :], 0.0)
        ca = torch.where(atom_mask[..., 1, None], xyz[..., 1, :], 0.0)
        c = torch.where(atom_mask[..., 2, None], xyz[..., 2, :], 0.0)
        backbone_mask = pocket & atom_mask[..., 0] & atom_mask[..., 1] & atom_mask[..., 2]
        cb = torch.where(atom_mask[..., 4, None], xyz[..., 4, :], virtual_cb(n, ca, c))
        cb_pair_mask = backbone_mask[:, :, None] & backbone_mask[:, None, :]
        cb_rbf = distance_rbf(torch.cdist(cb, cb)) * cb_pair_mask[..., None]
        if not switches["cb_geometry"]:
            cb_rbf = torch.zeros_like(cb_rbf)
        same_chain = condition.chain_index[:, :, None] == condition.chain_index[:, None, :]
        separation = condition.sequence_index[:, None] - condition.sequence_index[:, :, None]
        separation = separation.clamp(-32, 32).float() / 32.0
        separation = torch.where(same_chain, separation, 2.0)[..., None] * geometry_weight
        core_pair_index = (
            condition.pocket_core_mask[:, :, None].long() * 2
            + condition.pocket_core_mask[:, None, :].long()
        )
        core_pair = torch.nn.functional.one_hot(core_pair_index, 4).float() * geometry_weight
        if not switches["pocket_core"]:
            core_pair = torch.zeros_like(core_pair)
        orientation = rosetta_orientations(n, ca, cb, backbone_mask)
        orientation_features = (
            torch.cat(
                (
                    orientation.omega_sin_cos,
                    orientation.theta_sin_cos,
                    orientation.phi_sin_cos,
                    orientation.mask[..., None].float(),
                ),
                dim=-1,
            )
            * geometry_weight
        )
        if not switches["rosetta_orientation"]:
            orientation_features = torch.zeros_like(orientation_features)
        pair = torch.cat((base_pair, cb_rbf, separation, core_pair, orientation_features), dim=-1)

        local_xyz = torch.einsum("bnji,bnaj->bnai", rotation, xyz - translation[:, :, None])
        local_xyz = torch.where(atom_mask[..., None], local_xyz, 0.0)
        encoded_atom_mask = atom_mask
        if not switches["all_atoms"]:
            backbone_atom_slots = torch.zeros(
                atom_mask.shape[-1], dtype=torch.bool, device=atom_mask.device
            )
            backbone_atom_slots[:3] = True
            backbone_atom_slots[4] = True
            encoded_atom_mask = atom_mask & backbone_atom_slots
            local_xyz = torch.where(encoded_atom_mask[..., None], local_xyz, 0.0)
        atom = torch.cat((local_xyz, encoded_atom_mask[..., None].float()), dim=-1).flatten(-2)
        return single, pair, atom

    def forward(
        self, condition: UnifiedComplexCondition, *, peptide_single: Tensor | None = None
    ) -> UnifiedComplexEncoding:
        condition.validate_model_input()
        with torch.autocast(device_type=condition.residue_mask.device.type, enabled=False):
            single_features, pair_features, atom_features = self._static_features(condition)
            backbone_angles = torch.cat(
                (
                    condition.pocket_backbone_angles_sin_cos,
                    condition.pocket_backbone_angle_mask[..., None].float(),
                ),
                dim=-1,
            ).flatten(-2)
            sidechain_angles = torch.cat(
                (
                    condition.pocket_sidechain_angles_sin_cos,
                    condition.pocket_sidechain_angle_mask[..., None].float(),
                ),
                dim=-1,
            ).flatten(-2)
            if not _feature_switches(self.feature_mode)["torsions"]:
                backbone_angles = torch.zeros_like(backbone_angles)
                sidechain_angles = torch.zeros_like(sidechain_angles)

        single = self.role_projection(single_features) + self.aatype_embedding(condition.aatype)
        pocket_detail = (
            self.atom_projection(atom_features)
            + self.core_embedding(
                condition.pocket_core_mask.long()
                if _feature_switches(self.feature_mode)["pocket_core"]
                else torch.zeros_like(condition.pocket_core_mask, dtype=torch.long)
            )
            + self.backbone_angle_projection(backbone_angles)
            + self.sidechain_angle_projection(sidechain_angles)
        )
        single = single + torch.where(condition.pocket_mask[..., None], pocket_detail, 0.0)
        if peptide_single is not None:
            if peptide_single.shape != single.shape:
                raise ValueError("peptide injection must match encoder single shape")
            single = single + torch.where(condition.peptide_mask[..., None], peptide_single, 0.0)
        pair = self.pair_projection(pair_features)
        pair_mask = condition.residue_mask[:, :, None] & condition.residue_mask[:, None, :]
        single = torch.where(condition.residue_mask[..., None], single, 0.0)
        pair = torch.where(pair_mask[..., None], pair, 0.0)
        for block in self.blocks:
            single, pair = block(single, pair, condition.residue_mask)
        single = self.single_output_projection(self.single_norm(single))
        pair = self.pair_output_projection(self.pair_norm(pair))
        return UnifiedComplexEncoding(
            single=torch.where(condition.residue_mask[..., None], single, 0.0),
            pair=torch.where(pair_mask[..., None], pair, 0.0),
        )
