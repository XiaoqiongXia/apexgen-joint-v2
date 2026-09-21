"""Resumable DDP minibatch training for portable Simplex datasets."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from apexgen.joint_v2.data.training_collate import prepare_training_collator
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.runtime.distributed import global_mean_loss, model_digest
from apexgen.joint_v2.runtime.lineage import joint_v2_dataset_identity, sha256_file
from apexgen.joint_v2.runtime.portable import (
    SCHEMA, build_model, configure_device, load_config, read_checkpoint, training_batches, write_json,
)
from apexgen.joint_v2.sampling.simplex_runtime import CONTRACT, CONTRACT_SHA256, simplex_fm_losses
from apexgen.joint_v2.training.simplex_weighting import SimplexLossWeights

MODE = 'apexgen.ddp_minibatch.v1'


def validate_batch_layout(count, batch_size, world):
    if world < 2 or min(count, batch_size) < world:
        raise ValueError('each rank must receive at least one distinct sample')
    tail = count % batch_size
    if tail and tail < world:
        raise ValueError('final minibatch is smaller than world size; choose another batch size')


def rank_indices(indices, rank, world):
    if len(indices) < world or not 0 <= rank < world:
        raise ValueError('invalid nonempty distributed batch')
    return indices[rank::world]


def train(args):
    device = configure_device(f"cuda:{int(os.environ['LOCAL_RANK'])}", args.cpu_threads)
    dist.init_process_group('nccl', device_id=device)
    dataset = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        config = load_config(args.config)
        options = config['training']
        steps = args.steps if args.steps is not None else options['steps']
        if steps < 1:
            raise ValueError('steps must be positive')
        dataset = JointV2Dataset(args.dataset, split=args.split)
        count = len(dataset)
        validate_batch_layout(count, options['batch_size'], world)
        identity = joint_v2_dataset_identity(args.dataset)
        training_collator = prepare_training_collator(dataset, verified_identity=identity,
            geometry_checks=getattr(args, 'geometry_checks', 'auto'))
        torch.manual_seed(options['seed'])
        model = build_model(config, device)
        ddp = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=options['learning_rate'], weight_decay=options['weight_decay'])
        generator = torch.Generator(device=device).manual_seed(options['seed'] + 1 + rank)
        weights = SimplexLossWeights(**options['weights'])
        start = 0
        saved = None
        if args.resume:
            saved = read_checkpoint(args.resume)
            protocol = saved.get('distributed_training', {})
            if protocol.get('mode') != MODE or protocol.get('world_size') != world:
                raise ValueError('resume requires a minibatch DDP checkpoint with identical world size')
            if saved['config'] != config or saved['dataset_identity'] != identity or saved['split'] != args.split:
                raise ValueError('resume configuration, dataset or split mismatch')
            if saved['torch_version'] != str(torch.__version__) or saved['device_type'] != device.type:
                raise ValueError('resume requires identical PyTorch version and device type')
            if len(protocol.get('rank_rng_states', [])) != world:
                raise ValueError('checkpoint is missing per-rank RNG states')
            model.load_state_dict(saved['model_state'], strict=True)
            optimizer.load_state_dict(saved['optimizer_state'])
            start = saved['step']
        if steps <= start:
            raise ValueError('total step budget must exceed checkpoint step')
        output = Path(args.output)
        failure = [None]
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=False)
                write_json(output / 'run.json', dict(schema=SCHEMA, mode=MODE, formal=False,
                    config=config, runtime_contract=CONTRACT, dataset_identity=identity,
                    input_validation=training_collator.report,
                    split=args.split, samples=count, total_steps=steps, start_step=start,
                    world_size=world, global_batch_size=options['batch_size'],
                    batches_per_epoch=math.ceil(count/options['batch_size']),
                    backend='nccl', device='cuda', gpu=torch.cuda.get_device_name(device),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    python=platform.python_version(), torch=str(torch.__version__),
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    reduction='sample-weighted global mean; shuffled once per epoch; no duplication or dropped tail',
                    continuation_supported=True,
                    resume_sha256=sha256_file(args.resume) if args.resume else None))
            except Exception as exc:
                failure[0] = repr(exc)
        dist.broadcast_object_list(failure, src=0)
        if failure[0]:
            raise RuntimeError(failure[0])
        if saved is not None:
            state = saved['distributed_training']['rank_rng_states'][rank]
            generator.set_state(state['generator'].cpu())
            torch.set_rng_state(state['torch'].cpu())
            torch.cuda.set_rng_state(state['cuda'].cpu(), device)
            del saved

        def checkpoint(step):
            digests, states = [None]*world, [None]*world
            dist.all_gather_object(digests, model_digest(model))
            if len(set(digests)) != 1:
                raise RuntimeError('DDP model weights diverged')
            rng = dict(generator=generator.get_state(), torch=torch.get_rng_state(),
                       cuda=torch.cuda.get_rng_state(device))
            dist.all_gather_object(states, rng)
            if rank == 0:
                path = output / f'checkpoint_{step:08d}.pt'
                payload = dict(schema=SCHEMA, runtime_contract_sha256=CONTRACT_SHA256,
                    config=config, dataset_identity=identity, split=args.split, step=step,
                    device_type=device.type, torch_version=str(torch.__version__),
                    model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    generator_state=states[0]['generator'], torch_rng_state=states[0]['torch'],
                    cuda_rng_state=states[0]['cuda'], distributed_training=dict(mode=MODE,
                        world_size=world, continuation_supported=True, rank_rng_states=states))
                temporary = path.with_suffix('.pt.partial')
                torch.save(payload, temporary)
                temporary.replace(path)
                latest = output / 'latest.partial.json'
                write_json(latest, dict(checkpoint=path.name, step=step, sha256=sha256_file(path)))
                latest.replace(output / 'latest.json')
                write_json(output / f'rank_sync_{step:08d}.json', dict(step=step, model_sha256=digests))
            dist.barrier()

        checkpoint(start)
        ddp.train()
        started = time.monotonic()
        log = (output/'training.jsonl').open('w', buffering=1) if rank == 0 else None
        try:
            for index, indices in training_batches(count, options['batch_size'], options['seed'], start, steps):
                local = rank_indices(indices, rank, world)
                batch = training_collator([dataset[i] for i in local]).to(device)
                optimizer.zero_grad(set_to_none=True)
                losses = simplex_fm_losses(ddp, batch, generator, alpha_max=options['alpha_max'],
                                          precision=options['precision'], weights=weights)
                loss = global_mean_loss(losses['total'], len(indices), world)
                finite = torch.isfinite(loss).to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('nonfinite loss on a rank')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'],
                                                      error_if_nonfinite=True)
                optimizer.step()
                keys = sorted(losses)
                totals = torch.stack([losses[key].detach().sum() for key in keys])
                dist.all_reduce(totals)
                step = index + 1
                if rank == 0:
                    row = dict(step=step, samples=len(indices),
                        epoch=index//math.ceil(count/options['batch_size']),
                        gradient_norm=float(norm), elapsed_seconds=time.monotonic()-started,
                        **dict(zip(keys,(totals/len(indices)).cpu().tolist())))
                    log.write(json.dumps(row,allow_nan=False)+'\n')
                    if step == start+1 or step%25 == 0 or step == steps:
                        print(json.dumps(row),flush=True)
                if step%options['checkpoint_every'] == 0 or step == steps:
                    checkpoint(step)
        finally:
            if log is not None:
                log.close()
        if rank == 0:
            write_json(output/'completion.json',dict(status='completed',steps=steps,
                start_step=start,world_size=world,samples=count,formal=False,
                elapsed_seconds=time.monotonic()-started))
    finally:
        if dataset is not None:
            dataset.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config','dataset','output']:
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--split',default='smoke')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--geometry-checks', choices=('auto', 'full'), default='auto',
                        help='auto uses verified static features; full enables per-batch geometry checks')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
