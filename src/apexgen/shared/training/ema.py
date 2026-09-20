"""FP32 exponential moving average with explicit state naming."""

from __future__ import annotations

from collections.abc import Collection
from contextlib import contextmanager

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must lie in (0, 1)")
        self.decay = decay
        self.shadow = {
            name: value.detach().float().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: nn.Module, *, excluded_names: Collection[str] = ()) -> None:
        excluded = set(excluded_names)
        unknown = excluded - self.shadow.keys()
        if unknown:
            raise ValueError(f"EMA exclusions contain unknown names: {sorted(unknown)}")
        groups: dict[tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for name, value in model.state_dict().items():
            if name in self.shadow and name not in excluded:
                shadow = self.shadow[name]
                current = value.detach().to(device=shadow.device, dtype=shadow.dtype)
                shadows, values = groups.setdefault((shadow.device, shadow.dtype), ([], []))
                shadows.append(shadow)
                values.append(current)
        for shadows, values in groups.values():
            torch._foreach_lerp_(shadows, values, 1 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        if state["decay"] != self.decay or set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA state is incompatible")
        self.shadow = {
            name: value.to(device=self.shadow[name].device, dtype=torch.float32).clone()
            for name, value in state["shadow"].items()
        }

    def model_state_dict(self, model: nn.Module) -> dict:
        result = model.state_dict()
        return {name: self.shadow.get(name, value).clone() for name, value in result.items()}

    @contextmanager
    def apply_to(self, model: nn.Module):
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.model_state_dict(model))
        try:
            yield
        finally:
            model.load_state_dict(original)
