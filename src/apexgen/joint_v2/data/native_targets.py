"""Residue-frame labels extracted directly from observed peptide backbone atoms."""

import torch
from torch import Tensor

from apexgen.shared.storage.features import residue_frames
from apexgen.shared.geometry.rigid import Rigid


def observed_backbone_frames(backbone: Tensor, atom_mask: Tensor) -> Rigid:
    """Use CA as origin and the observed N/CA/C plane as local-to-global axes.

    A rigid frame does not encode internal bond lengths or angles. Keep the
    original atom coordinates as independent supervision; never reconstruct a
    chain to obtain these frame labels or substitute a template for missing atoms.
    """

    if backbone.shape[-2:] != (3, 3) or atom_mask.shape != backbone.shape[:-1]:
        raise ValueError("observed backbone must end in [N_CA_C, xyz] with matching mask")
    if atom_mask.dtype != torch.bool:
        raise TypeError("observed backbone atom mask must be bool")
    if not bool(atom_mask.all()):
        raise ValueError("observed peptide frames require complete N/CA/C atoms")
    if not bool(torch.isfinite(backbone).all()):
        raise ValueError("observed peptide backbone contains nonfinite coordinates")
    frames, valid = residue_frames(
        backbone[..., 0, :], backbone[..., 1, :], backbone[..., 2, :]
    )
    if not bool(valid.all()):
        raise ValueError("observed peptide backbone has degenerate N/CA/C frames")
    return frames
