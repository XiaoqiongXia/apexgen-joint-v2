from __future__ import annotations

import numpy as np
import pytest
import torch

from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.shared.geometry.sidechain import build_atom14
from apexgen.joint_v2.model.network import JointV2Model
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone
from apexgen.joint_v2.contracts.state import JointFlowState


def make_record(
    sample_id: str = "toy", *, pocket_offset: float = 0.0, peptide_length: int = 4
) -> dict:
    pocket_length = 2
    pocket_xyz = np.zeros((pocket_length, 38, 3), dtype=np.float32)
    pocket_mask = np.zeros((pocket_length, 38), dtype=np.bool_)
    for residue in range(pocket_length):
        center = pocket_offset + 3.0 * residue
        pocket_xyz[residue, 0] = [center - 1.2, 1.0, 0.0]
        pocket_xyz[residue, 1] = [center, 0.0, 0.0]
        pocket_xyz[residue, 2] = [center + 1.5, 0.0, 0.0]
        pocket_mask[residue, :3] = True
    endpoint_translation = torch.zeros(1, peptide_length, 3)
    endpoint_translation[0, :, 0] = torch.arange(peptide_length) * 3.8
    endpoint_rotation = torch.eye(3).expand(1, peptide_length, 3, 3).clone()
    native = reconstruct_backbone(
        JointFlowState(
            endpoint_translation.float(),
            endpoint_rotation.float(),
            torch.zeros(1, peptide_length, 20),
        )
    )[0]
    peptide_aatype = torch.arange(peptide_length, dtype=torch.long)
    backbone_with_oxygen = torch.zeros(peptide_length, 4, 3)
    backbone_with_oxygen[:, :3] = native
    backbone_with_oxygen[:, 3] = native[:, 2] + torch.tensor([0.5, 0.8, 0.0])
    chi = torch.linspace(-1.2, 1.2, peptide_length * 4).reshape(peptide_length, 4)
    experimental_atom14, experimental_atom14_mask = build_atom14(
        backbone_with_oxygen,
        peptide_aatype,
        chi,
    )
    experimental = experimental_atom14.numpy()
    experimental_mask = experimental_atom14_mask.numpy()
    return {
        "sample_id": sample_id,
        "source_pdb_id": "source",
        "split": "validation",
        "peptide_length": peptide_length,
        "pocket_aatype": np.array([0, 1], dtype=np.int64),
        "pocket_atom_xyz": pocket_xyz,
        "pocket_atom_mask": pocket_mask,
        "pocket_residue_rotation": np.broadcast_to(
            np.eye(3, dtype=np.float32), (pocket_length, 3, 3)
        ).copy(),
        "pocket_residue_translation": np.array(
            [[pocket_offset, 0.0, 0.0], [pocket_offset + 3.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "pocket_core_mask": np.array([True, False], dtype=np.bool_),
        "pocket_residue_keys": [
            {"auth_chain_id": "A", "label_seq_id": index + 1} for index in range(pocket_length)
        ],
        "joint_v2_target": {
            "aatype": peptide_aatype.numpy(),
            "translation": endpoint_translation[0].numpy(),
            "rotation": endpoint_rotation[0].numpy(),
            "backbone_torsion": np.zeros((peptide_length, 3), dtype=np.float32),
            "experimental_atom14": experimental,
            "experimental_atom14_mask": experimental_mask,
        },
    }


@pytest.fixture
def toy_batch():
    return collate_joint_v2_records([make_record()])


@pytest.fixture
def record_factory():
    return make_record


@pytest.fixture
def small_model():
    torch.manual_seed(7)
    return JointV2Model(
        single_dim=32,
        pair_dim=16,
        encoder_attention_heads=4,
        encoder_blocks=1,
        decoder_blocks=2,
        dropout=0.0,
        c_ipa=4,
        ipa_heads=4,
        ipa_query_key_points=2,
        ipa_value_points=3,
        transition_layers=1,
        angle_hidden_dim=16,
        angle_blocks=1,
        stop_rotation_gradient=True,
    )
