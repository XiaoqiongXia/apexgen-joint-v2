"""Native-independent peptide-plane regularization and explicit diagnostics.

Both cis and trans are planar. This term does not establish all-atom validity.
"""
import torch
from torch.nn import functional as F
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone


def plane_components(backbone):
    """Return squared sine, nearest-plane degrees, and nondegenerate-link mask."""
    a, b = backbone[:-1, 1], backbone[:-1, 2]
    c, d = backbone[1:, 0], backbone[1:, 1]
    axis = F.normalize(c - b, dim=-1, eps=1e-8)
    v, w = a - b, d - c
    v = v - (v * axis).sum(-1, keepdim=True) * axis
    w = w - (w * axis).sum(-1, keepdim=True) * axis
    valid = ((c - b).norm(dim=-1) > 1e-6) & (v.norm(dim=-1) > 1e-6) & (w.norm(dim=-1) > 1e-6)
    v, w = F.normalize(v, dim=-1, eps=1e-8), F.normalize(w, dim=-1, eps=1e-8)
    sine = (torch.cross(axis, v, dim=-1) * w).sum(-1)
    cosine = (v * w).sum(-1)
    degrees = torch.rad2deg(torch.atan2(sine.abs(), cosine.abs()))
    return sine.square(), degrees, valid


def plane_loss(state, condition):
    backbone = reconstruct_backbone(state)
    values = []
    for i, mask in enumerate(condition.peptide_mask):
        squared_sine, _, valid = plane_components(backbone[i, mask])
        if not len(squared_sine):
            values.append(backbone[i].sum() * 0)
        else:
            # Degenerate links cannot count as planar; other chain terms supply
            # the bond/angle restoring force at exactly degenerate coordinates.
            values.append(torch.where(valid, squared_sine, torch.ones_like(squared_sine)).mean())
    return torch.stack(values)


def plane_metrics(backbone):
    with torch.no_grad():
        _, degrees, valid = plane_components(torch.as_tensor(backbone, dtype=torch.float64))
        if len(degrees) == 0:
            return dict(plane_mean_degrees=0., plane_max_degrees=0., plane_nondegenerate=True, plane_proxy=True)
        good = bool(valid.all())
        return dict(plane_mean_degrees=float(degrees.mean()), plane_max_degrees=float(degrees.max()),
                    plane_nondegenerate=good,
                    plane_proxy=good and float(degrees.mean()) <= 10. and float(degrees.max()) <= 30.)
