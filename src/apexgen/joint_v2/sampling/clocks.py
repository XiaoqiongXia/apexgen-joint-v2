"""Matched sequence clock for Gaussian interpolation, likelihood and integration."""

import math

from torch import Tensor


def validate_sequence_time_power(power: float) -> float:
    if isinstance(power, bool) or not math.isfinite(power) or power < 1:
        raise ValueError("sequence_time_power must be finite and >= 1")
    return float(power)


def sequence_time(time: Tensor, power: float = 1.0) -> Tensor:
    """s(t)=t**power, with the identity clock preserved exactly for controls."""
    power = validate_sequence_time_power(power)
    return time if power == 1 else time.pow(power)
