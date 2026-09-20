"""Dirichlet FM path and simplex-preserving integration (Stark et al., ICML 2024).

Path: Dir(1 + (alpha-1)*one_hot(label)); alpha(t)=1+(alpha_max-1)*t.
Conditional field in alpha time: C(x_i,alpha)*(e_i-x).
C = -d_alpha I_x(alpha,K-1) / ((1-x)*BetaPDF(x;alpha,K-1)).
See https://arxiv.org/abs/2402.05841, equations 14-17.
"""

import math

import numpy as np
from scipy import special
import torch


MAX_SUPPORTED_ALPHA = 32.0
SMALL_X = 1e-5


def validate_alpha_max(alpha_max):
    if not math.isfinite(alpha_max) or not 1 < alpha_max <= MAX_SUPPORTED_ALPHA:
        raise ValueError("alpha_max must be finite and in (1, 32]")
    return float(alpha_max)


def alpha_at(time, alpha_max):
    validate_alpha_max(alpha_max)
    if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
        raise ValueError("Dirichlet time must lie in [0,1]")
    return 1 + (alpha_max - 1) * time


def sample_dirichlet(concentration, *, generator=None):
    """Generator-controlled Gamma sampling, matching torch's Dirichlet construction."""
    if not bool(torch.isfinite(concentration).all()) or bool((concentration <= 0).any()):
        raise ValueError("Dirichlet concentrations must be positive and finite")
    draws = torch._standard_gamma(concentration.float(), generator=generator)
    return draws / draws.sum(-1, keepdim=True)


def conditional_simplex(labels, mask, time, *, alpha_max=8.0, generator=None):
    if labels.shape != mask.shape or time.shape != (len(mask),):
        raise ValueError("Expected labels/mask [B,L], time [B]")
    if bool(((labels[mask] < 0) | (labels[mask] >= 20)).any()):
        raise ValueError("Invalid peptide amino acid")
    safe = torch.where(mask, labels, 0)
    onehot = torch.nn.functional.one_hot(safe, 20).float()
    concentration = 1 + (alpha_at(time, alpha_max) - 1)[:, None, None] * onehot
    p = sample_dirichlet(concentration, generator=generator)
    return torch.where(mask[..., None], p, 0.0)


def coefficient_reference(x, alpha, classes=20):
    """Float64 central alpha derivative; use complementary CDF in the upper tail.

    The endpoint limits are analytic. No numerical division by a vanishing PDF
    at the boundary, and no subtraction of near-one CDFs in the upper tail.
    """
    x, alpha = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
    )
    if (
        classes < 2
        or not np.isfinite(alpha).all()
        or not np.isfinite(x).all()
        or np.any(alpha < 1)
        or np.any((x < 0) | (x > 1))
    ):
        raise ValueError("Invalid Dirichlet coefficient arguments")
    b = classes - 1
    interior = (x > 0) & (x < 1)
    safe = np.where(interior, x, 0.5)
    h = 1e-4
    lower = -(special.betainc(alpha + h, b, safe) - special.betainc(alpha - h, b, safe)) / (2 * h)
    upper = (special.betaincc(alpha + h, b, safe) - special.betaincc(alpha - h, b, safe)) / (2 * h)
    numerator = np.where(safe < alpha / (alpha + b), lower, upper)
    log_denom = (alpha - 1) * np.log(safe) + b * np.log1p(-safe) - special.betaln(alpha, b)
    value = numerator / np.exp(log_denom)
    value = np.where(x == 0, 0, value)
    value = np.where(x == 1, (special.digamma(alpha + b) - special.digamma(alpha)) / b, value)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise FloatingPointError("Invalid Dirichlet coefficient")
    return value


class DirichletField:
    """Bilinear FP32 lookup of a Float64 reference, used only in sampling."""

    def __init__(self, alpha_max=8.0, *, classes=20, alpha_points=513, x_points=4097, device="cpu"):
        self.alpha_max = validate_alpha_max(alpha_max)
        if classes < 2 or alpha_points < 2 or x_points < 5:
            raise ValueError("Invalid coefficient grid")
        self.classes = classes
        self.alpha_points, self.x_points = alpha_points, x_points
        alphas = np.linspace(1, self.alpha_max, alpha_points)[:, None]
        # Resolve the logarithmic slope near zero without constructing tiny PDFs.
        split = x_points // 4 + 1
        xs = np.concatenate(
            (np.geomspace(SMALL_X, 0.01, split), np.linspace(0.01, 1, x_points - split + 1)[1:])
        )[None, :]
        self.x_grid = torch.as_tensor(xs[0], dtype=torch.float32, device=device)
        self.table = torch.as_tensor(
            coefficient_reference(xs, alphas, classes), dtype=torch.float32, device=device
        )

    def coefficient(self, x, alpha):
        if not bool(torch.isfinite(x).all() & torch.isfinite(alpha).all()):
            raise ValueError("Nonfinite simplex field input")
        if bool(((x < 0) | (x > 1)).any()) or bool(
            ((alpha < 1) | (alpha > self.alpha_max + 1e-5)).any()
        ):
            raise ValueError("Simplex field input outside table domain")
        a = ((alpha - 1) / (self.alpha_max - 1) * (self.alpha_points - 1)).clamp(
            0, self.alpha_points - 1
        )
        ai = a.long().clamp_max(self.alpha_points - 2)
        lookup_x = x.clamp_min(SMALL_X)
        xi = (torch.searchsorted(self.x_grid, lookup_x.contiguous(), right=True) - 1).clamp(
            0, self.x_points - 2
        )
        wa = a - ai
        wx = (lookup_x - self.x_grid[xi]) / (self.x_grid[xi + 1] - self.x_grid[xi])
        low = self.table[ai, xi] * (1 - wx) + self.table[ai, xi + 1] * wx
        high = self.table[ai + 1, xi] * (1 - wx) + self.table[ai + 1, xi + 1] * wx
        value = low * (1 - wa) + high * wa
        # Leading small-x Beta-CDF expansion. Relative correction is O(K*x).
        small = (
            x
            / alpha
            * (
                -x.clamp_min(torch.finfo(x.dtype).tiny).log()
                + 1 / alpha
                + torch.digamma(alpha)
                - torch.digamma(alpha + self.classes - 1)
            )
        )
        return torch.where(x < SMALL_X, small, value)

    def weights(self, x, posterior, alpha):
        return posterior * self.coefficient(x, alpha)

    def velocity(self, x, posterior, alpha):
        weights = self.weights(x, posterior, alpha)
        return weights - x * weights.sum(-1, keepdim=True)

    def step(self, x, posterior, alpha, next_alpha, *, max_alpha_step=0.05):
        """Frozen-classifier exponential midpoint, substepping C in alpha.

        Each step is a convex combination: simplex invariance without projection.
        The inner solver is second order for a fixed classifier. The overall
        coupled sampler still freezes the classifier over each model step.
        """
        if not math.isfinite(max_alpha_step) or max_alpha_step <= 0:
            raise ValueError("max_alpha_step must be positive")
        if x.shape != posterior.shape or x.shape[-1] != self.classes:
            raise ValueError("State and posterior must have matching class dimensions")
        if not bool(torch.isfinite(next_alpha).all()) or bool(
            ((next_alpha < 1) | (next_alpha > self.alpha_max + 1e-5)).any()
        ):
            raise ValueError("next_alpha outside table domain")
        if not bool(torch.isfinite(posterior).all()) or bool((posterior < 0).any()):
            raise ValueError("Invalid class posterior")
        for name, p in (("state", x), ("posterior", posterior)):
            if not torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]), atol=2e-5, rtol=0):
                raise ValueError(name + " must sum to one")
        delta = next_alpha - alpha
        if bool((delta < 0).any()):
            raise ValueError("Cannot integrate backwards")
        count = max(1, math.ceil(float(delta.max()) / max_alpha_step))
        step = delta / count

        def advance(value, weights, dt):
            rate = weights.sum(-1, keepdim=True)
            move = -torch.expm1(-dt * rate)
            return (1 - move) * value + move * weights / rate.clamp_min(
                torch.finfo(value.dtype).tiny
            )

        for i in range(count):
            weights = self.weights(x, posterior, alpha + i * step)
            midpoint = advance(x, weights, step / 2)
            middle_weights = self.weights(midpoint, posterior, alpha + (i + 0.5) * step)
            x = advance(x, middle_weights, step)
        return x
