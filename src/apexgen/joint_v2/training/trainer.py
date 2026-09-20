"""Optimizer, EMA and checkpoint runtime for Joint-v2."""

from __future__ import annotations

import math
import os
import random
import re
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from apexgen.joint_v2.sampling.base import sample_base_state
from apexgen.joint_v2.data.batch import JointV2Batch
from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256, require_joint_v2_contract
from apexgen.joint_v2.training.loss import JointEndpointLossWeights
from apexgen.joint_v2.runtime.lineage import sha256_file
from apexgen.joint_v2.training.step import joint_endpoint_training_step
from apexgen.shared.training.ema import ExponentialMovingAverage


def _unwrapped(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def build_optimizer(model: nn.Module, *, learning_rate: float, weight_decay: float) -> AdamW:
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("optimizer hyperparameters are invalid")
    no_decay_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (
            no_decay
            if name.endswith("bias") or id(parameter) in no_decay_ids or "head_weights" in name
            else decay
        )
        target.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: Optimizer,
    *,
    maximum_steps: int,
    warmup_steps: int,
    minimum_learning_rate: float,
) -> LambdaLR:
    if maximum_steps <= 0 or not 0 <= warmup_steps < maximum_steps:
        raise ValueError("scheduler step counts are invalid")
    peak = float(optimizer.param_groups[0]["lr"])
    if not 0 <= minimum_learning_rate <= peak:
        raise ValueError("minimum learning rate must lie in [0, peak]")
    minimum_ratio = minimum_learning_rate / peak

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(maximum_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return minimum_ratio + 0.5 * (1 - minimum_ratio) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def inspect_model_initialization_checkpoint(
    path: str | Path,
    *,
    expected_dataset_identity_sha256: str,
    expected_data_view_identity_sha256: str,
    model_state_view: str,
    allow_cross_data_view: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate a model-only exploratory initialization checkpoint.

    Optimizer, scheduler, EMA runtime, and RNG state are deliberately not inherited.
    Velocity/torsion checkpoints are rejected before any state-dict loading.
    """

    if model_state_view not in {"ema", "raw"}:
        raise ValueError("initialization model state must be 'ema' or 'raw'")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"initialization checkpoint is absent: {source}")
    digest = sha256_file(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if sha256_file(source) != digest:
        raise RuntimeError("initialization checkpoint changed while loading")
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "apexgen.joint_v2.sequence_structure_endpoint.checkpoint.v1"
    ):
        raise ValueError("initialization checkpoint schema mismatch")
    require_joint_v2_contract(payload, source="initialization checkpoint")
    if payload.get("dataset_identity_sha256") != expected_dataset_identity_sha256:
        raise ValueError("initialization checkpoint dataset identity mismatch")
    source_data_view = payload.get("data_view_identity_sha256")
    if not isinstance(source_data_view, str) or len(source_data_view) != 64:
        raise ValueError("initialization checkpoint has no data-view identity")
    crosses_data_view = source_data_view != expected_data_view_identity_sha256
    if crosses_data_view and not allow_cross_data_view:
        raise ValueError("initialization checkpoint data-view identity mismatch")
    source_contract = payload.get("joint_v2_contract_sha256")
    if not isinstance(source_contract, str) or len(source_contract) != 64:
        raise ValueError("initialization checkpoint has no contract digest")
    source_config = payload.get("config_sha256")
    source_run = payload.get("run_id")
    source_step = payload.get("optimizer_step")
    if not isinstance(source_config, str) or len(source_config) != 64:
        raise ValueError("initialization checkpoint has no config digest")
    if not isinstance(source_run, str) or not source_run:
        raise ValueError("initialization checkpoint has no run identity")
    if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step <= 0:
        raise ValueError("initialization checkpoint has an invalid optimizer step")
    state_name = "ema_state" if model_state_view == "ema" else "model_state"
    state = payload.get(state_name)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"initialization checkpoint has no usable {state_name}")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"initialization checkpoint {state_name} is not a tensor state dict")
    _require_finite(state, state_name)
    metadata = {
        "mode": "model_weights_only_optimizer_scheduler_ema_rng_reset",
        "source_path": str(source),
        "source_sha256": digest,
        "source_contract_sha256": source_contract,
        "source_config_sha256": source_config,
        "source_run_id": source_run,
        "source_optimizer_step": source_step,
        "source_data_view_identity_sha256": source_data_view,
        "target_data_view_identity_sha256": expected_data_view_identity_sha256,
        "cross_data_view_initialization": crosses_data_view,
        "model_state_view": model_state_view,
    }
    return payload, metadata


def initialize_model_from_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    expected_dataset_identity_sha256: str,
    expected_data_view_identity_sha256: str,
    model_state_view: str,
    expected_source_sha256: str | None = None,
    allow_cross_data_view: bool = False,
) -> dict[str, Any]:
    """Strictly initialize model parameters while resetting all training state."""

    payload, metadata = inspect_model_initialization_checkpoint(
        path,
        expected_dataset_identity_sha256=expected_dataset_identity_sha256,
        expected_data_view_identity_sha256=expected_data_view_identity_sha256,
        model_state_view=model_state_view,
        allow_cross_data_view=allow_cross_data_view,
    )
    if expected_source_sha256 is not None and metadata["source_sha256"] != (expected_source_sha256):
        raise ValueError("initialization checkpoint digest differs from run binding")
    state_name = "ema_state" if model_state_view == "ema" else "model_state"
    _unwrapped(model).load_state_dict(payload[state_name], strict=True)
    for name, parameter in _unwrapped(model).state_dict().items():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"initialized model parameter is non-finite: {name}")
    return metadata


@dataclass(frozen=True)
class JointV2StepMetrics:
    optimizer_step: int
    total: float
    trajectory_fape: float
    peptide_fape: float
    cross_fape: float
    final_translation: float
    final_rotation: float
    final_backbone_n_ca_c: float
    backbone_angle: float
    backbone_angle_norm: float
    sidechain_angle: float
    sidechain_angle_norm: float
    sequence_logit: float
    sequence_logit_rmse: float
    sequence_soft_ce: float
    sequence_accuracy: float
    sequence_hard_nll: float
    sequence_perplexity: float
    sequence_entropy: float
    gradient_norm: float
    learning_rate: float
    sampled_time_mean: float
    encoder_frozen: bool


class JointV2Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        ema: ExponentialMovingAverage,
        generator: torch.Generator,
        translation_sigma_angstrom: float,
        gradient_clip: float,
        network_precision: str,
        loss_weights: JointEndpointLossWeights | None = None,
        training_time_max: float = 1.0,
        encoder_joint_warmup_steps: int | None = None,
    ) -> None:
        if gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if not 0.0 < training_time_max <= 1.0:
            raise ValueError("training_time_max must lie in (0, 1]")
        if encoder_joint_warmup_steps is not None and (
            isinstance(encoder_joint_warmup_steps, bool)
            or not isinstance(encoder_joint_warmup_steps, int)
            or encoder_joint_warmup_steps < 0
        ):
            raise ValueError("encoder_joint_warmup_steps must be null or non-negative")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.generator = generator
        self.translation_sigma_angstrom = translation_sigma_angstrom
        self.gradient_clip = gradient_clip
        self.network_precision = network_precision
        self.loss_weights = loss_weights
        self.training_time_max = training_time_max
        self.encoder_joint_warmup_steps = encoder_joint_warmup_steps
        self.optimizer_step = 0
        if encoder_joint_warmup_steps is None:
            self._encoder_parameters: tuple[nn.Parameter, ...] = ()
            self._encoder_ema_names: tuple[str, ...] = ()
        else:
            encoder = getattr(_unwrapped(model), "encoder", None)
            if not isinstance(encoder, nn.Module):
                raise ValueError("encoder freezing requires model.encoder")
            self._encoder_parameters = tuple(encoder.parameters())
            if not self._encoder_parameters:
                raise ValueError("model.encoder has no parameters to freeze")
            self._encoder_ema_names = tuple(
                name for name in self.ema.shadow if name.startswith("encoder.")
            )
            if not self._encoder_ema_names:
                raise ValueError("EMA state has no encoder entries to freeze")

    @property
    def encoder_frozen(self) -> bool:
        return (
            self.encoder_joint_warmup_steps is not None
            and self.optimizer_step >= self.encoder_joint_warmup_steps
        )

    def train_step(self, batch: JointV2Batch) -> JointV2StepMetrics:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        base = sample_base_state(
            batch.condition,
            translation_sigma_angstrom=self.translation_sigma_angstrom,
            generator=self.generator,
        )
        output = joint_endpoint_training_step(
            self.model,
            base,
            batch.condition,
            batch.targets,
            weights=self.loss_weights,
            generator=self.generator,
            network_precision=self.network_precision,
            training_time_max=self.training_time_max,
        )
        output.losses["total"].backward()
        encoder_frozen = self.encoder_frozen
        if encoder_frozen:
            for parameter in self._encoder_parameters:
                parameter.grad = None
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.ema.update(
            _unwrapped(self.model),
            excluded_names=self._encoder_ema_names if encoder_frozen else (),
        )
        self.optimizer_step += 1
        return JointV2StepMetrics(
            optimizer_step=self.optimizer_step,
            total=float(output.losses["total"].detach()),
            trajectory_fape=float(output.losses["trajectory_fape"].detach()),
            peptide_fape=float(output.losses["peptide_fape"].detach()),
            cross_fape=float(output.losses["cross_fape"].detach()),
            final_translation=float(output.losses["final_translation"].detach()),
            final_rotation=float(output.losses["final_rotation"].detach()),
            final_backbone_n_ca_c=float(output.losses["final_backbone_n_ca_c"].detach()),
            backbone_angle=float(output.losses["backbone_angle"].detach()),
            backbone_angle_norm=float(output.losses["backbone_angle_norm"].detach()),
            sidechain_angle=float(output.losses["sidechain_angle"].detach()),
            sidechain_angle_norm=float(output.losses["sidechain_angle_norm"].detach()),
            sequence_logit=float(output.losses["sequence_logit"].detach()),
            sequence_logit_rmse=float(output.losses["sequence_logit_rmse"].detach()),
            sequence_soft_ce=float(output.losses["sequence_soft_ce"].detach()),
            sequence_accuracy=float(output.losses["sequence_accuracy"].detach()),
            sequence_hard_nll=float(output.losses["sequence_hard_nll"].detach()),
            sequence_perplexity=float(output.losses["sequence_perplexity"].detach()),
            sequence_entropy=float(output.losses["sequence_entropy"].detach()),
            gradient_norm=float(gradient_norm),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            sampled_time_mean=float(output.time.mean()),
            encoder_frozen=encoder_frozen,
        )


def globally_reduce_step_metrics(
    metrics: JointV2StepMetrics, *, device: torch.device | str
) -> JointV2StepMetrics:
    """Average batch-dependent training metrics over every DDP rank."""

    if not dist.is_available() or not dist.is_initialized():
        return metrics
    names = (
        "total",
        "trajectory_fape",
        "peptide_fape",
        "cross_fape",
        "final_translation",
        "final_rotation",
        "final_backbone_n_ca_c",
        "backbone_angle",
        "backbone_angle_norm",
        "sidechain_angle",
        "sidechain_angle_norm",
        "sequence_logit",
        "sequence_logit_rmse",
        "sequence_soft_ce",
        "sequence_accuracy",
        "sequence_hard_nll",
        "sequence_perplexity",
        "sequence_entropy",
        "gradient_norm",
        "sampled_time_mean",
    )
    values = torch.tensor(
        [getattr(metrics, name) for name in names],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    reduced = {name: float(value) for name, value in zip(names, values, strict=True)}
    return JointV2StepMetrics(
        optimizer_step=metrics.optimizer_step,
        learning_rate=metrics.learning_rate,
        encoder_frozen=metrics.encoder_frozen,
        **reduced,
    )


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _require_finite(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"checkpoint contains non-finite tensor: {name}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite(child, f"{name}[{index}]")


def validated_checkpoint_model_state(
    payload: dict[str, Any], model: nn.Module, *, model_state_view: str
) -> dict[str, Tensor]:
    """Return a finite, layout-compatible raw or EMA model state without mutation."""

    if model_state_view not in {"raw", "ema"}:
        raise ValueError("checkpoint model state view must be 'raw' or 'ema'")
    state_name = "ema_state" if model_state_view == "ema" else "model_state"
    state = payload.get(state_name)
    expected = _unwrapped(model).state_dict()
    if not isinstance(state, dict) or set(state) != set(expected) or any(
        not isinstance(state[name], torch.Tensor) or state[name].shape != value.shape
        for name, value in expected.items()
    ):
        raise ValueError(f"checkpoint {state_name} layout differs from runtime")
    _require_finite(state, state_name)
    return state


def _validate_rng_state(state: Any) -> None:
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("checkpoint random state has an invalid layout")
    if not isinstance(state["python"], tuple):
        raise ValueError("checkpoint Python random state is invalid")
    if not isinstance(state["numpy"], tuple) or len(state["numpy"]) != 5:
        raise ValueError("checkpoint NumPy random state is invalid")
    if (
        not isinstance(state["torch"], torch.Tensor)
        or state["torch"].dtype != torch.uint8
        or state["torch"].ndim != 1
    ):
        raise ValueError("checkpoint Torch random state is invalid")
    cuda_states = state["cuda"]
    if not isinstance(cuda_states, list) or any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1
        for value in cuda_states
    ):
        raise ValueError("checkpoint CUDA random state is invalid")
    if torch.cuda.is_available() and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("checkpoint CUDA random state count differs from runtime")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator().set_state(state["torch"])
        for index, cuda_state in enumerate(cuda_states):
            torch.Generator(device=f"cuda:{index}").set_state(cuda_state)
    except Exception as error:
        raise ValueError("checkpoint random state cannot be restored") from error


def _distributed_identity() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _local_runtime_state(trainer: JointV2Trainer, extra: dict[str, Any] | None) -> dict[str, Any]:
    rank, world_size = _distributed_identity()
    return {
        "rank": rank,
        "world_size": world_size,
        "generator_state": trainer.generator.get_state(),
        "rng_state": _rng_state(),
        "extra": {} if extra is None else extra,
    }


def _gather_runtime_states(local: dict[str, Any]) -> list[dict[str, Any]] | None:
    rank, world_size = _distributed_identity()
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if rank != 0:
        return None
    if any(value is None for value in gathered):
        raise RuntimeError("DDP checkpoint did not gather every rank runtime state")
    states = [value for value in gathered if value is not None]
    observed_ranks = [int(value["rank"]) for value in states]
    if observed_ranks != list(range(world_size)):
        raise RuntimeError("DDP checkpoint rank runtime states are not ordered and complete")
    return states


def save_checkpoint(
    path: str | Path,
    *,
    trainer: JointV2Trainer,
    run_id: str,
    config_sha256: str,
    dataset_identity_sha256: str,
    data_view_identity_sha256: str,
    extra: dict[str, Any] | None = None,
    rank_runtime_extra: dict[str, Any] | None = None,
) -> None:
    """Collect rank-local state on every rank and atomically write on rank zero.

    In a distributed process group this function is collective: every rank must
    call it at the same optimizer step.
    """

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("checkpoint run_id must be a non-empty string")
    for name, value in (
        ("config", config_sha256),
        ("dataset identity", dataset_identity_sha256),
        ("data-view identity", data_view_identity_sha256),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"checkpoint {name} SHA-256 is invalid")
    if trainer.optimizer_step <= 0:
        raise ValueError("checkpoint requires at least one completed optimizer step")
    if extra is not None and not isinstance(extra, dict):
        raise TypeError("checkpoint extra must be a mapping")
    if rank_runtime_extra is not None and not isinstance(rank_runtime_extra, dict):
        raise TypeError("checkpoint rank runtime extra must be a mapping")
    rank, world_size = _distributed_identity()
    rank_runtime_states = _gather_runtime_states(_local_runtime_state(trainer, rank_runtime_extra))
    status: list[str | None] = [None]
    if rank == 0:
        try:
            if rank_runtime_states is None:
                raise RuntimeError("rank zero did not receive DDP runtime states")
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            model = _unwrapped(trainer.model)
            payload = {
                "schema_version": "apexgen.joint_v2.sequence_structure_endpoint.checkpoint.v1",
                "joint_v2_contract_sha256": JOINT_V2_CONTRACT_SHA256,
                "run_id": run_id,
                "config_sha256": config_sha256,
                "dataset_identity_sha256": dataset_identity_sha256,
                "data_view_identity_sha256": data_view_identity_sha256,
                "model_state": model.state_dict(),
                "ema_runtime_state": trainer.ema.state_dict(),
                "ema_state": trainer.ema.model_state_dict(model),
                "optimizer_state": trainer.optimizer.state_dict(),
                "scheduler_state": (
                    None if trainer.scheduler is None else trainer.scheduler.state_dict()
                ),
                "optimizer_step": trainer.optimizer_step,
                "world_size": world_size,
                "rank_runtime_states": rank_runtime_states,
                "extra": {} if extra is None else extra,
            }
            temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
            try:
                torch.save(payload, temporary)
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"checkpoint save failed on rank zero: {status[0]}")


def load_checkpoint(
    path: str | Path,
    *,
    trainer: JointV2Trainer,
    expected_run_id: str,
    expected_config_sha256: str,
    expected_dataset_identity_sha256: str,
    expected_data_view_identity_sha256: str,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore shared training state and the calling rank's exact random streams."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Joint-v2 checkpoint payload must be a mapping")
    if payload.get("schema_version") != (
        "apexgen.joint_v2.sequence_structure_endpoint.checkpoint.v1"
    ):
        raise ValueError("Joint-v2 checkpoint schema mismatch")
    require_joint_v2_contract(payload, source="checkpoint")
    if payload.get("run_id") != expected_run_id:
        raise ValueError("Joint-v2 checkpoint run identity mismatch")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("Joint-v2 checkpoint config mismatch")
    if payload.get("dataset_identity_sha256") != expected_dataset_identity_sha256:
        raise ValueError("Joint-v2 checkpoint dataset identity mismatch")
    if payload.get("data_view_identity_sha256") != expected_data_view_identity_sha256:
        raise ValueError("Joint-v2 checkpoint data-view identity mismatch")
    rank, world_size = _distributed_identity()
    if payload.get("world_size") != world_size:
        raise ValueError(
            f"checkpoint world size {payload.get('world_size')} differs from runtime {world_size}"
        )
    runtime_states = payload.get("rank_runtime_states")
    if not isinstance(runtime_states, list) or len(runtime_states) != world_size:
        raise ValueError("checkpoint rank runtime states are incomplete")
    for expected_rank, runtime_state in enumerate(runtime_states):
        if not isinstance(runtime_state, dict):
            raise ValueError("checkpoint rank runtime state must be a mapping")
        if runtime_state.get("rank") != expected_rank or runtime_state.get(
            "world_size"
        ) != world_size:
            raise ValueError("checkpoint rank runtime identity differs from runtime")
        generator = runtime_state.get("generator_state")
        if (
            not isinstance(generator, torch.Tensor)
            or generator.dtype != torch.uint8
            or generator.ndim != 1
        ):
            raise ValueError("checkpoint generator state is invalid")
        try:
            torch.Generator(device=trainer.generator.device).set_state(generator)
        except Exception as error:
            raise ValueError("checkpoint generator state cannot be restored") from error
        _validate_rng_state(runtime_state.get("rng_state"))
        if not isinstance(runtime_state.get("extra", {}), dict):
            raise ValueError("checkpoint rank runtime extra must be a mapping")
    local_runtime = runtime_states[rank]
    optimizer_step = payload.get("optimizer_step")
    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step <= 0
    ):
        raise ValueError("checkpoint optimizer step is invalid")
    generator_state = local_runtime["generator_state"]
    model_state = payload.get("model_state")
    ema_state = payload.get("ema_runtime_state")
    optimizer_state = payload.get("optimizer_state")
    model_view_state = payload.get("ema_state")
    if not all(
        isinstance(value, dict)
        for value in (model_state, model_view_state, ema_state, optimizer_state)
    ):
        raise ValueError("checkpoint learned state is incomplete")
    validated_checkpoint_model_state(payload, trainer.model, model_state_view="raw")
    validated_checkpoint_model_state(payload, trainer.model, model_state_view="ema")
    expected_ema = trainer.ema.state_dict()
    shadow = ema_state.get("shadow")
    if (
        ema_state.get("decay") != expected_ema["decay"]
        or not isinstance(shadow, dict)
        or set(shadow) != set(expected_ema["shadow"])
        or any(
            not isinstance(shadow[name], torch.Tensor)
            or shadow[name].shape != expected.shape
            for name, expected in expected_ema["shadow"].items()
        )
    ):
        raise ValueError("checkpoint EMA state differs from runtime")
    if set(optimizer_state) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer state layout is invalid")
    scheduler_state = payload.get("scheduler_state")
    if (trainer.scheduler is None) != (scheduler_state is None):
        raise ValueError("checkpoint scheduler differs from runtime")
    if scheduler_state is not None and not isinstance(scheduler_state, dict):
        raise ValueError("checkpoint scheduler state layout is invalid")

    _require_finite(model_state, "model_state")
    _require_finite(model_view_state, "ema_state")
    _require_finite(ema_state, "ema_runtime_state")
    _require_finite(optimizer_state, "optimizer_state")
    _require_finite(scheduler_state, "scheduler_state")
    model = _unwrapped(trainer.model)
    previous_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    previous_ema = copy.deepcopy(trainer.ema.state_dict())
    previous_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    previous_scheduler = (
        None if trainer.scheduler is None else copy.deepcopy(trainer.scheduler.state_dict())
    )
    previous_step = trainer.optimizer_step
    previous_generator = trainer.generator.get_state().clone()
    previous_rng = _rng_state() if restore_rng else None
    try:
        model.load_state_dict(model_state)
        trainer.ema.load_state_dict(ema_state)
        trainer.optimizer.load_state_dict(optimizer_state)
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(scheduler_state)
        trainer.optimizer_step = optimizer_step
        trainer.generator.set_state(generator_state)
        if restore_rng:
            _restore_rng_state(local_runtime["rng_state"])
    except Exception as error:
        model.load_state_dict(previous_model)
        trainer.ema.load_state_dict(previous_ema)
        trainer.optimizer.load_state_dict(previous_optimizer)
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(previous_scheduler)
        trainer.optimizer_step = previous_step
        trainer.generator.set_state(previous_generator)
        if previous_rng is not None:
            _restore_rng_state(previous_rng)
        raise RuntimeError("checkpoint restoration failed without changing runtime") from error
    payload["loaded_rank_runtime_extra"] = local_runtime.get("extra", {})
    return payload


_PERIODIC_CHECKPOINT = re.compile(r"^checkpoint_(\d{8})\.pt$")


def prune_periodic_checkpoints(directory: str | Path, *, keep_last: int) -> list[Path]:
    """Remove only old, strictly named periodic checkpoints; preserve final artifacts."""

    if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last <= 0:
        raise ValueError("keep_last must be a positive integer")
    root = Path(directory)
    matched: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _PERIODIC_CHECKPOINT.fullmatch(path.name)
        if match is not None and path.is_file():
            matched.append((int(match.group(1)), path))
    matched.sort()
    removed = [path for _, path in matched[:-keep_last]]
    for path in removed:
        path.unlink()
    return removed
