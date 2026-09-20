"""DDP full-batch overfitting diagnostic, launched with torchrun.

Each update consumes the selected split exactly once, without padding samples.
Checkpoints support portable evaluation and sampling; continuation is not supported.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.runtime.lineage import joint_v2_dataset_identity, sha256_file
from apexgen.joint_v2.runtime.portable import (
    SCHEMA, build_model, configure_device, load_config, save_checkpoint, write_json,
)
from apexgen.joint_v2.sampling.simplex_runtime import CONTRACT, simplex_fm_losses
from apexgen.joint_v2.training.simplex_weighting import SimplexLossWeights


def shard_indices(count, world_size, rank):
    if world_size < 2 or count < world_size or not 0 <= rank < world_size:
        raise ValueError("DDP requires at least two ranks and one distinct sample per rank")
    return list(range(rank, count, world_size))


def global_mean_loss(local_losses, total_samples, world_size):
    # DDP averages rank gradients; cancel that average before weighting samples.
    return local_losses.sum() * (world_size / total_samples)


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = configure_device(f"cuda:{local_rank}", args.cpu_threads)
    dist.init_process_group("nccl", device_id=device)
    dataset = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        config = load_config(args.config)
        options = config["training"]
        steps = args.steps if args.steps is not None else options["steps"]
        if steps < 1:
            raise ValueError("steps must be positive")
        dataset = JointV2Dataset(args.dataset, split=args.split)
        count = len(dataset)
        indices = shard_indices(count, world, rank)
        if options["batch_size"] != count:
            raise ValueError("this full-batch diagnostic requires training.batch_size == split size")
        identity = joint_v2_dataset_identity(args.dataset)
        torch.manual_seed(options["seed"])
        model = build_model(config, device)
        ddp = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=options["learning_rate"], weight_decay=options["weight_decay"],
        )
        generator = torch.Generator(device=device).manual_seed(options["seed"] + 1 + rank)
        weights = SimplexLossWeights(**options["weights"])
        batch = collate_joint_v2_records([dataset[i] for i in indices]).to(device)
        assignments = [None] * world
        dist.all_gather_object(assignments, dict(rank=rank, indices=indices,
            sample_ids=list(batch.sample_ids), gpu=torch.cuda.get_device_name(device)))
        output = Path(args.output)
        setup_error = [None]
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=False)
                write_json(output / "run.json", dict(
                    schema=SCHEMA, formal=False, config=config, runtime_contract=CONTRACT,
                    dataset_identity=identity, split=args.split, samples=count,
                    total_steps=steps, start_step=0, world_size=world, backend="nccl",
                    assignments=assignments, device="cuda", precision=options["precision"],
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    python=platform.python_version(), torch=str(torch.__version__),
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    reduction="equal sample weights; full split once per update; no duplicates",
                    continuation_supported=False,
                ))
            except Exception as exc:
                setup_error[0] = repr(exc)
        dist.broadcast_object_list(setup_error, src=0)
        if setup_error[0]:
            raise RuntimeError(setup_error[0])

        def checkpoint(step):
            digests = [None] * world
            dist.all_gather_object(digests, model_digest(model))
            if len(set(digests)) != 1:
                raise RuntimeError("DDP model states diverged across ranks")
            if rank == 0:
                path = output / f"checkpoint_{step:08d}.pt"
                save_checkpoint(path, model, optimizer, generator, config, identity,
                                args.split, step, device)
                payload = torch.load(path, map_location="cpu", weights_only=True)
                payload["distributed_training"] = dict(world_size=world, continuation_supported=False)
                temporary = path.with_suffix(".pt.partial")
                torch.save(payload, temporary)
                temporary.replace(path)
                write_json(output / "latest.json", dict(checkpoint=path.name, step=step,
                                                        sha256=sha256_file(path)))
                write_json(output / f"rank_sync_{step:08d}.json", dict(step=step, model_sha256=digests))
            dist.barrier()

        checkpoint(0)
        ddp.train()
        started = time.monotonic()
        log = (output / "training.jsonl").open("w", buffering=1) if rank == 0 else None
        try:
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)
                losses = simplex_fm_losses(ddp, batch, generator,
                    alpha_max=options["alpha_max"], precision=options["precision"], weights=weights)
                loss = global_mean_loss(losses["total"], count, world)
                finite = torch.isfinite(loss).to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("nonfinite training loss on a rank")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"],
                                                      error_if_nonfinite=True)
                optimizer.step()
                keys = sorted(losses)
                totals = torch.stack([losses[key].detach().sum() for key in keys])
                dist.all_reduce(totals)
                if rank == 0:
                    row = dict(step=step, samples=count, gradient_norm=float(norm),
                        elapsed_seconds=time.monotonic() - started,
                        **dict(zip(keys, (totals / count).cpu().tolist())))
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    if step == 1 or step % 25 == 0 or step == steps:
                        print(json.dumps(row), flush=True)
                if step % options["checkpoint_every"] == 0 or step == steps:
                    checkpoint(step)
        finally:
            if log is not None:
                log.close()
        if rank == 0:
            write_json(output / "completion.json", dict(status="completed", steps=steps,
                world_size=world, samples=count, exposures_per_sample=steps,
                elapsed_seconds=time.monotonic() - started, formal=False,
                note="Execution completed; fitting quality requires separate evaluation."))
    finally:
        if dataset is not None:
            dataset.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cpu-threads", type=int, default=2)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
