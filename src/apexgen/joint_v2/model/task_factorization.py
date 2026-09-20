"""Task-specific observations, aligned targets, losses and sampling for stage one."""

import torch
from torch import nn

from apexgen.joint_v2.data.batch import JointV2Batch
from apexgen.joint_v2.sampling.flow import conditional_path, endpoint_step
from apexgen.joint_v2.training.loss import JointEndpointLossWeights, joint_endpoint_loss_per_sample
from apexgen.joint_v2.model.network import build_joint_v2_model
from apexgen.joint_v2.contracts.task_contract import TASKS, TaskObservation


def observation_from_batch(batch: JointV2Batch, task: str) -> TaskObservation:
    """Training adapter; exposing labels is allowed only for that task's observed modality."""
    if task == "S":
        return TaskObservation(
            task,
            batch.condition,
            batch.targets.endpoint_translation,
            batch.targets.endpoint_rotation,
        )
    if task == "G_s":
        return TaskObservation(
            task,
            batch.condition,
            sequence_logits=batch.targets.endpoint_state(batch.condition).sequence_logits,
        )
    return TaskObservation(task, batch.condition)


def aligned_batch(batch: JointV2Batch) -> JointV2Batch:
    """Validate the native-frame contract without rewriting observed atom labels.

    Kept for existing callers. Whole-chain idealized targets and template-derived
    coordinate replacements are no longer the supervision used by Joint-v2.
    """
    batch.targets.validate_for(batch.condition)
    return batch


def task_objective_weights(task: str) -> dict[str, float]:
    if task not in TASKS:
        raise ValueError("unknown task")
    weights = {}
    if task != "S":
        weights.update(
            trajectory_fape=1.0,
            final_translation=1.0,
            final_rotation=1.0,
            final_backbone_n_ca_c=1.0,
        )
    if task in {"J", "S"}:
        weights.update(sequence_logit=1.0, sequence_soft_ce=1.0)
    return weights


def task_losses(prediction, batch: JointV2Batch, task: str):
    # Compute the shared metrics, but sum only active objectives; no 0*inactive gradients.
    weights = JointEndpointLossWeights(
        trajectory_fape=float(task != "S"),
        final_translation=float(task != "S"),
        final_rotation=float(task != "S"),
        final_backbone=float(task != "S"),
        backbone_angle=0,
        backbone_angle_norm=0,
        sidechain_angle=0,
        sidechain_angle_norm=0,
        sequence_total=float(task in {"J", "S"}),
    )
    values = joint_endpoint_loss_per_sample(
        prediction, batch.condition, batch.targets, weights=weights
    )
    active = task_objective_weights(task)
    values["total"] = sum(values[name] * weight for name, weight in active.items())
    return values


def task_path(base, batch, observation, time, *, sequence_time_power=1.0):
    return observation.clamp(
        conditional_path(
            base=base,
            targets=batch.targets,
            condition=batch.condition,
            time=time,
            sequence_time_power=sequence_time_power,
        )
    )


class TaskFactorizationModel(nn.Module):
    def __init__(self, config, task: str, *, condition_route: str = "decoder"):
        super().__init__()
        if task not in TASKS:
            raise ValueError("unknown task")
        if condition_route not in {"encoder", "decoder"}:
            raise ValueError("unknown condition route")
        if condition_route == "encoder":
            if task != "G_s":
                raise ValueError("encoder condition route is an exploratory G_s control")
            architecture = config["architecture"]
            if (
                architecture.get("encoder_single_dim", architecture["single_dim"])
                != architecture["single_dim"]
            ):
                raise ValueError(
                    "matched projection requires equal encoder and decoder single widths"
                )
        # Construct all modules in identical order before marking unused parameters inactive.
        network = build_joint_v2_model(config)
        self.encoder, self.decoder = network.encoder, network.decoder
        self.task = task
        self.condition_route = condition_route
        inactive = ["decoder.structure_module.angle_resnet."]
        if task == "S":
            inactive += ["decoder.structure_module.backbone_update."]
        if task in {"G_s", "G_0"}:
            inactive += [
                "decoder.structure_module.sequence_update.",
                "decoder.structure_module.final_sequence_resnet.",
            ]
        if task == "G_0":
            inactive += [
                "decoder.structure_module.sequence_norm.",
                "decoder.structure_module.sequence_projection.",
                "decoder.structure_module.sequence_latent_transition.",
            ]
        for name, parameter in self.named_parameters():
            if any(name.startswith(prefix) for prefix in inactive):
                parameter.requires_grad_(False)

    def encode_complex(self, observation):
        if observation.task != self.task:
            raise ValueError("observation task differs from model")
        if self.condition_route == "encoder":
            module = self.decoder.structure_module
            code = module.sequence_projection(module.sequence_norm(observation.sequence_logits))
            return self.encoder(observation.pocket, peptide_single=code)
        return self.encoder(observation.pocket)

    def decode(self, state, time, observation, encoding, *, trace=None, return_intermediates=True):
        if observation.task != self.task:
            raise ValueError("observation task differs from model")
        return self.decoder.structure_module(
            observation.clamp(state),
            time,
            observation.pocket,
            encoding,
            task=self.task,
            initialize_sequence_latent=self.condition_route == "decoder",
            trace=trace,
            return_intermediates=return_intermediates,
        )

    def forward(self, state, time, observation, *, trace=None):
        return self.decode(state, time, observation, self.encode_complex(observation), trace=trace)


def task_rollout(base, observation, endpoint_fn):
    state = observation.clamp(base)
    for index in range(20):
        time = torch.full((state.layout[0],), index / 20, device=state.translation.device)
        state = observation.clamp(
            endpoint_step(
                state,
                endpoint_fn(state, time),
                condition=observation.pocket,
                time=time,
                next_time=torch.full_like(time, (index + 1) / 20),
            )
        )
    return state
