"""Explicit fixed weights and non-mutating shared-trunk gradient diagnostics."""

from dataclasses import asdict, dataclass
import math

import torch


OBJECTIVES = ("final_translation", "rotation_tangent", "sequence_ce")


@dataclass(frozen=True)
class SimplexLossWeights:
    final_translation: float = 1.0
    rotation_tangent: float = 1.0
    sequence_ce: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Simplex loss weights must be finite and positive")

    def as_dict(self):
        return asdict(self)


def shared_parameters(model):
    """Encoder and shared geometric decoder trunk; exclude modality output heads."""
    trunk = ("ipa", "transition", "linear_in", "layer_norm_s", "layer_norm_z", "layer_norm_ipa")
    prefixes = ("network.encoder.",) + tuple(
        f"network.decoder.structure_module.{name}." for name in trunk
    )
    items = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith(prefixes)
    ]
    if not items:
        raise ValueError("No shared simplex trunk parameters found")
    return items


def shared_gradient_statistics(model, losses):
    """Raw objective gradients before clipping/Adam; leaves parameter .grad untouched."""
    items = shared_parameters(model)
    parameters = [p for _, p in items]
    gradients = [
        torch.autograd.grad(losses[name].mean(), parameters, allow_unused=True, retain_graph=i < 2)
        for i, name in enumerate(OBJECTIVES)
    ]
    gram = torch.zeros(3, 3, device=parameters[0].device, dtype=torch.float64)
    for index in range(len(parameters)):
        for i in range(3):
            a = gradients[i][index]
            if a is None:
                continue
            for j in range(i, 3):
                b = gradients[j][index]
                if b is not None:
                    dot = (a.detach().float() * b.detach().float()).sum(dtype=torch.float64)
                    gram[i, j] += dot
                    if i != j:
                        gram[j, i] += dot
    gram = gram.cpu()
    norms = gram.diag().clamp_min(0).sqrt()
    if not bool(torch.isfinite(gram).all()):
        raise FloatingPointError("Nonfinite shared gradients")
    return dict(
        gradient_l2=dict(zip(OBJECTIVES, norms.tolist(), strict=True)),
        gram=gram.tolist(),
        cosine=[
            [
                None
                if norms[i] * norms[j] == 0
                else float((gram[i, j] / (norms[i] * norms[j])).clamp(-1, 1))
                for j in range(3)
            ]
            for i in range(3)
        ],
    )


def calibrate_weights(statistics):
    """Equalize panel RMS gradient norms, with translation weight fixed to one."""
    if not statistics:
        raise ValueError("An independent gradient calibration panel is required")
    norms = {
        name: math.sqrt(sum(row["gradient_l2"][name] ** 2 for row in statistics) / len(statistics))
        for name in OBJECTIVES
    }
    if any(not math.isfinite(v) or v <= 1e-12 for v in norms.values()):
        raise ValueError(
            "Degenerate calibration gradients; do not invert zero initialization gradients"
        )
    return SimplexLossWeights(
        **{name: norms["final_translation"] / value for name, value in norms.items()}
    ), norms
