"""Supervise adjacent-residue geometry against observed atoms, without projection."""

import torch

from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone


def adjacent_peptide_mask(condition):
    return (
        condition.peptide_mask[:, :-1]
        & condition.peptide_mask[:, 1:]
        & (condition.chain_index[:, :-1] == condition.chain_index[:, 1:])
        & (condition.sequence_index[:, 1:] == condition.sequence_index[:, :-1] + 1)
    )


def adjacent_translation_loss(prediction, batch):
    """Observed neighbor CA displacement vector MSE in A^2, without chain projection.

    Unlike distance-only supervision, this retains the direction of each local
    connection in the shared pocket coordinate system. A common translation
    cancels, and a common rigid rotation of prediction and labels preserves loss.
    """
    mask = adjacent_peptide_mask(batch.condition)
    with torch.autocast(device_type=prediction.translation.device.type, enabled=False):
        x = prediction.translation.float()
        y = batch.targets.endpoint_translation.float()
        error = ((x[:, 1:] - x[:, :-1]) - (y[:, 1:] - y[:, :-1])).square().sum(-1)
        return torch.where(mask, error, 0.0).sum(-1) / mask.sum(-1).clamp_min(1)


def native_connection_loss(prediction, batch):
    """Mean squared error (A^2) of four distances spanning each peptide bond.

    C--N controls the bond; CA--N and C--CA constrain its flanking angles;
    CA--CA additionally constrains the relative orientation across the bond.
    All reference distances come from observed PDB backbone coordinates.
    Pocket/padding, chain boundaries and sequence-index gaps are excluded.
    """
    adjacent = adjacent_peptide_mask(batch.condition)
    with torch.autocast(device_type=prediction.translation.device.type, enabled=False):
        predicted = reconstruct_backbone(prediction)
        observed = batch.targets.backbone_xyz.float()

        def distances(x):
            return torch.stack(
                [
                    (x[:, :-1, a] - x[:, 1:, b]).norm(dim=-1)
                    for a, b in ((2, 0), (1, 0), (2, 1), (1, 1))
                ],
                dim=-1,
            )

        error = (distances(predicted) - distances(observed)).square().mean(-1)
        return torch.where(adjacent, error, 0.0).sum(-1) / adjacent.sum(-1).clamp_min(1)
