"""Deterministic distributed sampling with an explicit resumable cursor."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sized
from typing import Any

from torch.utils.data import DistributedSampler
import torch


class ResumableDistributedSampler(DistributedSampler):
    """A ``DistributedSampler`` whose rank-local sample offset is checkpointed."""

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.start_index = 0

    def set_epoch_and_start(self, epoch: int, start_index: int = 0) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a non-negative integer")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or not 0 <= start_index <= self.num_samples
        ):
            raise ValueError("sampler start index lies outside the rank-local epoch")
        super().set_epoch(epoch)
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        indices = list(super().__iter__())
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def state_dict(
        self, *, epoch: int | None = None, start_index: int | None = None
    ) -> dict[str, Any]:
        epoch = self.epoch if epoch is None else epoch
        start_index = self.start_index if start_index is None else start_index
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler checkpoint epoch must be a non-negative integer")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or not 0 <= start_index <= self.num_samples
        ):
            raise ValueError("sampler checkpoint start index is invalid")
        return {
            "schema_version": "apexgen.joint_v2.sampler.v1",
            "epoch": epoch,
            "start_index": start_index,
            "dataset_length": len(self.dataset),
            "num_samples": self.num_samples,
            "total_size": self.total_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema_version": "apexgen.joint_v2.sampler.v1",
            "dataset_length": len(self.dataset),
            "num_samples": self.num_samples,
            "total_size": self.total_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
        }
        observed = {name: state.get(name) for name in expected}
        if observed != expected:
            raise ValueError(
                f"sampler checkpoint differs from runtime: expected={expected}, observed={observed}"
            )
        self.set_epoch_and_start(int(state["epoch"]), int(state["start_index"]))


class ResumableSizeBucketBatchSampler:
    """Deterministic graph-size buckets with equal DDP batches and exact resume."""

    def __init__(
        self,
        costs: list[int],
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int,
        bucket_window_batches: int = 32,
    ) -> None:
        if not costs or any(isinstance(value, bool) or value <= 0 for value in costs):
            raise ValueError("bucket costs must be non-empty positive integers")
        if batch_size <= 0 or num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("bucket sampler topology is invalid")
        if bucket_window_batches <= 0:
            raise ValueError("bucket_window_batches must be positive")
        self.costs = tuple(int(value) for value in costs)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.bucket_window_batches = bucket_window_batches
        self.epoch = 0
        self.start_batch = 0
        self.global_batch_size = batch_size * num_replicas
        self.batches_per_epoch = (len(costs) + self.global_batch_size - 1) // self.global_batch_size
        self.costs_sha256 = hashlib.sha256(
            json.dumps(self.costs, separators=(",", ":")).encode()
        ).hexdigest()

    def set_epoch_and_start(self, epoch: int, start_batch: int = 0) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("bucket sampler epoch must be a non-negative integer")
        if not 0 <= start_batch <= self.batches_per_epoch:
            raise ValueError("bucket sampler cursor lies outside the epoch")
        self.epoch = epoch
        self.start_batch = start_batch

    def _global_batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.costs), generator=generator).tolist()
        window = self.global_batch_size * self.bucket_window_batches
        ordered: list[int] = []
        for offset in range(0, len(order), window):
            ordered.extend(sorted(order[offset : offset + window], key=self.costs.__getitem__))
        required = self.batches_per_epoch * self.global_batch_size
        missing = required - len(ordered)
        repeats = (missing + len(order) - 1) // len(order)
        ordered.extend((order * repeats)[:missing])
        batches = [
            ordered[offset : offset + self.global_batch_size]
            for offset in range(0, required, self.global_batch_size)
        ]
        permutation = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[index] for index in permutation]

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._global_batches()
        for global_indices in batches[self.start_batch :]:
            interleaved = torch.tensor(global_indices).reshape(self.batch_size, self.num_replicas)
            yield interleaved[:, self.rank].tolist()

    def __len__(self) -> int:
        return self.batches_per_epoch - self.start_batch

    def state_dict(
        self, *, epoch: int | None = None, start_batch: int | None = None
    ) -> dict[str, Any]:
        return {
            "schema_version": "apexgen.joint_v2.size_bucket_sampler.v1",
            "epoch": self.epoch if epoch is None else epoch,
            "start_batch": self.start_batch if start_batch is None else start_batch,
            "dataset_length": len(self.costs),
            "costs_sha256": self.costs_sha256,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "seed": self.seed,
            "bucket_window_batches": self.bucket_window_batches,
            "batches_per_epoch": self.batches_per_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict(epoch=0, start_batch=0)
        for name in ("epoch", "start_batch"):
            expected.pop(name)
        observed = {name: state.get(name) for name in expected}
        if observed != expected:
            raise ValueError("bucket sampler checkpoint differs from runtime")
        self.set_epoch_and_start(int(state["epoch"]), int(state["start_batch"]))
