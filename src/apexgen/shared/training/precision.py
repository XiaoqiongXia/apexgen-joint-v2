"""Precision policy shared by training, validation, and generation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def network_autocast(device: torch.device, precision: str) -> ContextManager:
    """Use BF16 only for CUDA network operations; geometry owns FP32 islands."""
    if precision == "float32":
        return nullcontext()
    if precision != "bfloat16":
        raise ValueError(f"unsupported network precision: {precision}")
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
