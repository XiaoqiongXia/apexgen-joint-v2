"""Batched proper rigid transforms using row-major point tensors."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class Rigid:
    rotation: Tensor
    translation: Tensor

    def __post_init__(self) -> None:
        if self.rotation.shape[-2:] != (3, 3):
            raise ValueError("rotation must end in shape (3, 3)")
        if self.translation.shape[-1] != 3:
            raise ValueError("translation must end in dimension 3")
        if self.rotation.shape[:-2] != self.translation.shape[:-1]:
            raise ValueError("rotation and translation batch shapes must agree")

    def apply(self, points: Tensor) -> Tensor:
        return (self.rotation @ points.unsqueeze(-1)).squeeze(-1) + self.translation

    def compose(self, other: Rigid) -> Rigid:
        rotation = self.rotation @ other.rotation
        translation = self.apply(other.translation)
        return Rigid(rotation, translation)

    def inverse(self) -> Rigid:
        rotation = self.rotation.transpose(-1, -2)
        translation = -(rotation @ self.translation.unsqueeze(-1)).squeeze(-1)
        return Rigid(rotation, translation)
