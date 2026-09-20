"""Portable, single-device Simplex training, native evaluation and generation.

Checkpoints embed configuration and bind data by content, never by server paths.
This is an exploratory runner, not a formal Joint-v2 training lock.
"""

import argparse
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch
import yaml

from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.contracts.task_contract import TaskObservation
from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.evaluation.generation_quality import generation_metrics
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone
from apexgen.joint_v2.geometry.rotations import so3_log
from apexgen.joint_v2.model.simplex_codesign import SimplexCodesignModel
from apexgen.joint_v2.runtime.lineage import joint_v2_dataset_identity, sha256_file
from apexgen.joint_v2.sampling.dirichlet import DirichletField, validate_alpha_max
from apexgen.joint_v2.sampling.simplex_runtime import (
    CONTRACT, CONTRACT_SHA256, sample_simplex, sample_simplex_base,
    simplex_fm_losses, simplex_losses, simplex_training_path,
)
from apexgen.joint_v2.training.simplex_weighting import SimplexLossWeights
from apexgen.shared.training.precision import network_autocast


SCHEMA = "apexgen.joint_v2.portable_simplex.v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    return validate_config(config)


def validate_config(config):
    if set(config) != {"schema", "model", "training", "sampling"} or config["schema"] != SCHEMA:
        raise ValueError("expected portable Simplex configuration")
    train = config["training"]
    expected = {"seed", "batch_size", "steps", "learning_rate", "weight_decay",
                "gradient_clip", "checkpoint_every", "precision", "alpha_max", "weights"}
    if set(train) != expected:
        raise ValueError("unknown or missing training options")
    for key in ("batch_size", "steps", "checkpoint_every"):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(train["seed"]) is not int or train["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("learning_rate", "gradient_clip", "weight_decay"):
        value = train[key]
        if not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(f"invalid {key}")
    if train["precision"] not in {"float32", "bfloat16"}:
        raise ValueError("precision must be float32 or bfloat16")
    validate_alpha_max(train["alpha_max"])
    SimplexLossWeights(**train["weights"])
    if set(config["sampling"]) != {"steps"} or type(config["sampling"]["steps"]) is not int or config["sampling"]["steps"] < 1:
        raise ValueError("sampling.steps must be a positive integer")
    if set(config["model"]) != {"architecture"}:
        raise ValueError("model only accepts architecture; flow and losses belong to the Simplex runtime")
    if config["model"]["architecture"]["structure_module"]["stop_rotation_gradient_between_blocks"]:
        raise ValueError("Simplex requires full rotation gradients")
    if type(config["model"]["architecture"]["structure_module"].get("geometry_update_time_gate", True)) is not bool:
        raise ValueError("geometry_update_time_gate must be a boolean")
    return config


def configure_device(name, threads):
    if threads < 1:
        raise ValueError("cpu threads must be positive")
    torch.set_num_threads(threads)
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("supported devices are cpu and cuda")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def build_model(config, device):
    return SimplexCodesignModel(
        config["model"], position_code_seed=config["training"]["seed"]
    ).to(device)


def read_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != SCHEMA or saved.get("runtime_contract_sha256") != CONTRACT_SHA256:
        raise ValueError("checkpoint schema/runtime mismatch; legacy experiment checkpoints are not portable checkpoints")
    validate_config(saved["config"])
    return saved


def load_checkpoint(path, device):
    saved = read_checkpoint(path)
    model = build_model(saved["config"], device)
    model.load_state_dict(saved["model_state"], strict=True)
    return model, saved


def training_batches(count, batch_size, seed, start, stop):
    """One permutation per epoch; resume directly at the next minibatch."""
    batches = math.ceil(count / batch_size)
    step = start
    while step < stop:
        epoch, offset = divmod(step, batches)
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        for batch_index in range(offset, batches):
            if step >= stop:
                return
            yield step, order[batch_index * batch_size:(batch_index + 1) * batch_size]
            step += 1


def save_checkpoint(path, model, optimizer, generator, config, identity, split, step, device):
    payload = dict(
        schema=SCHEMA, runtime_contract_sha256=CONTRACT_SHA256, config=config,
        dataset_identity=identity, split=split, step=step, device_type=device.type,
        model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        torch_version=str(torch.__version__),
    )
    temporary = path.with_suffix(".pt.partial")
    torch.save(payload, temporary)
    temporary.replace(path)
    write_json(path.parent / "latest.json", {"checkpoint": path.name, "step": step,
                                           "sha256": sha256_file(path)})


def train(args):
    config = load_config(args.config)
    options = config["training"]
    steps = args.steps if args.steps is not None else options["steps"]
    if steps < 1:
        raise ValueError("steps must be positive")
    device = configure_device(args.device, args.cpu_threads)
    torch.manual_seed(options["seed"])
    dataset = JointV2Dataset(args.dataset, split=args.split)
    try:
        if not len(dataset):
            raise ValueError("training split is empty")
        identity = joint_v2_dataset_identity(args.dataset)
        model = build_model(config, device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=options["learning_rate"], weight_decay=options["weight_decay"],
        )
        generator = torch.Generator(device=device).manual_seed(options["seed"] + 1)
        start = 0
        if args.resume:
            saved = read_checkpoint(args.resume)
            if saved.get("distributed_training"):
                raise ValueError("DDP diagnostic checkpoints support evaluation/sampling, not exact-state continuation")
            if saved["config"] != config or saved["dataset_identity"] != identity or saved["split"] != args.split:
                raise ValueError("resume configuration or dataset identity mismatch")
            if saved["device_type"] != device.type or saved["torch_version"] != str(torch.__version__):
                raise ValueError("resume requires the same device type and PyTorch version; evaluation/sampling allow relocation")
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            generator.set_state(saved["generator_state"].cpu())
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), device)
            start = saved["step"]
        if steps <= start:
            raise ValueError("requested total steps must exceed checkpoint step")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run.json", dict(
            schema=SCHEMA, formal=False, config=config, runtime_contract=CONTRACT,
            dataset_identity=identity, split=args.split, samples=len(dataset),
            total_steps=steps, start_step=start, device=str(device),
            python=platform.python_version(), torch=str(torch.__version__),
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            resume_sha256=sha256_file(args.resume) if args.resume else None,
        ))
        weights = SimplexLossWeights(**options["weights"])
        model.train()
        with (output / "training.jsonl").open("w", buffering=1) as log:
            for step, indices in training_batches(len(dataset), options["batch_size"], options["seed"], start, steps):
                batch = collate_joint_v2_records([dataset[i] for i in indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                losses = simplex_fm_losses(model, batch, generator, alpha_max=options["alpha_max"],
                                          precision=options["precision"], weights=weights)
                loss = losses["total"].mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                row = dict(step=step + 1, samples=len(indices), gradient_norm=float(norm),
                           **{key: float(value.detach().mean()) for key, value in losses.items()})
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step == start or (step + 1) % 25 == 0 or step + 1 == steps:
                    print(json.dumps(row), flush=True)
                if (step + 1) % options["checkpoint_every"] == 0 or step + 1 == steps:
                    save_checkpoint(output / f"checkpoint_{step + 1:08d}.pt", model, optimizer,
                                    generator, config, identity, args.split, step + 1, device)
        write_json(output / "completion.json", dict(status="completed", steps=steps, formal=False))
    finally:
        dataset.close()


def prediction_metrics(prediction, batch):
    """One sample; pose errors stay in the fixed context frame (no alignment)."""
    mask = batch.condition.peptide_mask[0]
    backbone = reconstruct_backbone(prediction)[0, mask]
    labels = prediction.sequence_logits[0, mask].argmax(-1)
    atoms = backbone.new_zeros(len(labels), 14, 3)
    atom_mask = torch.zeros(len(labels), 14, dtype=torch.bool, device=backbone.device)
    atoms[:, :3], atom_mask[:, :3] = backbone, True
    quality = generation_metrics(backbone, atoms, atom_mask, labels, batch.condition, 0)
    angle = so3_log(batch.targets.endpoint_rotation[0, mask].transpose(-1, -2) @ prediction.rotation[0, mask])
    return dict(
        ca_rmsd_angstrom=float((prediction.translation[0, mask] - batch.targets.endpoint_translation[0, mask]).square().sum(-1).mean().sqrt()),
        rotation_degrees=float(angle.norm(dim=-1).mean() * 180 / torch.pi),
        sequence_accuracy=float((labels == batch.targets.endpoint_aatype[0, mask]).float().mean()),
        cn_mae_angstrom=float(quality["generation_quality.cn_bond_mae_to_ideal_angstrom"]),
        geometry_pass=bool(quality["generation_quality.geometry_proxy"]),
    )


@torch.no_grad()
def infer(args):
    if args.bases < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("bases and limit must be positive")
    device = configure_device(args.device, args.cpu_threads)
    model, saved = load_checkpoint(args.checkpoint, device)
    model.eval()
    config = saved["config"]
    options = config["training"]
    steps = config["sampling"]["steps"] if args.sampling_steps is None else args.sampling_steps
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    dataset = JointV2Dataset(args.dataset, split=args.split)
    try:
        if not len(dataset):
            raise ValueError("selected split is empty")
        identity = joint_v2_dataset_identity(args.dataset)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        count = min(len(dataset), args.limit or len(dataset))
        write_json(output / "run.json", dict(
            schema=SCHEMA, command=args.command, checkpoint_sha256=sha256_file(args.checkpoint),
            dataset_identity=identity, training_dataset_identity=saved["dataset_identity"],
            same_training_view=identity == saved["dataset_identity"] and args.split == saved["split"],
            split=args.split, samples=count, bases=args.bases, seed=args.seed, sampling_steps=steps,
            selection="first N included manifest rows", formal=False,
        ))
        field = DirichletField(options["alpha_max"], device=device)
        totals, counts = {}, {}

        def record_row(log, row):
            log.write(json.dumps(row, allow_nan=False) + "\n")
            group = row["kind"]
            counts[group] = counts.get(group, 0) + 1
            target = totals.setdefault(group, {})
            for key, value in row["metrics"].items():
                target[key] = target.get(key, 0.0) + float(value)

        with (output / "metrics.jsonl").open("w", buffering=1) as log:
            for index in range(count):
                record = dataset[index]
                batch = collate_joint_v2_records([record]).to(device)
                for base_index in range(args.bases):
                    seed = args.seed + index * args.bases + base_index
                    result = sample_simplex(model, batch.condition,
                        generator=torch.Generator(device=device).manual_seed(seed),
                        steps=steps, alpha_max=options["alpha_max"], field=field, precision=options["precision"])
                    prediction = JointFlowState(result.state.translation, result.state.rotation, result.sequence_logits)
                    if args.command == "sample":
                        mask = batch.condition.peptide_mask[0]
                        filename = f"sample_{index:06d}_base_{base_index:03d}.npz"
                        np.savez_compressed(output / filename,
                            generated_backbone=reconstruct_backbone(prediction)[0, mask].cpu().numpy(),
                            generated_aatype=prediction.sequence_logits[0, mask].argmax(-1).cpu().numpy(),
                            context_atom_xyz=record["pocket_atom_xyz"], context_atom_mask=record["pocket_atom_mask"],
                            context_aatype=record["pocket_aatype"], site_origin=record["site_origin"],
                        )
                        log.write(json.dumps(dict(sample_id=batch.sample_ids[0], seed=seed, arrays=filename)) + "\n")
                        continue
                    record_row(log, dict(sample_id=batch.sample_ids[0], seed=seed, kind="rollout",
                                         metrics=prediction_metrics(prediction, batch)))
                    generator = torch.Generator(device=device).manual_seed(seed + 1000000)
                    base = sample_simplex_base(batch.condition, generator=generator)
                    with network_autocast(device, options["precision"]):
                        encoding = model.encode_complex(TaskObservation("J", batch.condition))
                    for time_value in (0.0, 0.05, 0.25, 0.5, 0.9):
                        time = torch.tensor([time_value], device=device)
                        state = simplex_training_path(base, batch, time, alpha_max=options["alpha_max"], generator=generator)
                        with network_autocast(device, options["precision"]):
                            pred = model.decode(state, time, TaskObservation("J", batch.condition), encoding)
                        losses = simplex_losses(pred, batch, state, weights=SimplexLossWeights(**options["weights"]))
                        metrics = prediction_metrics(pred, batch)
                        metrics.update({key: float(value.mean()) for key, value in losses.items()})
                        record_row(log, dict(sample_id=batch.sample_ids[0], seed=seed, kind=f"denoise_t{time_value:g}", metrics=metrics))
                print(f"{args.command}: {index + 1}/{count}", flush=True)
        write_json(output / "summary.json", dict(
            samples=count, bases=args.bases,
            metrics={group: {key: value / counts[group] for key, value in values.items()}
                     for group, values in totals.items()},
            counts=counts, note="equal sample/base weighting; geometry_pass is a fraction; no structural alignment",
        ))
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="minibatch training or exact-state continuation")
    training.add_argument("--config", required=True, type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--steps", type=int, help="total step budget, including resumed steps")
    for name in ("evaluate", "sample"):
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", required=True, type=Path)
        sub.add_argument("--bases", type=int, default=1)
        sub.add_argument("--limit", type=int)
        sub.add_argument("--seed", type=int, default=20260921)
        sub.add_argument("--sampling-steps", type=int)
    for sub in commands.choices.values():
        sub.add_argument("--dataset", required=True, type=Path)
        sub.add_argument("--split", default="smoke")
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    (train if args.command == "train" else infer)(args)


if __name__ == "__main__":
    main()
