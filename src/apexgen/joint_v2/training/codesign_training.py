"""Shared deterministic updates for a separately locked codesign training run."""
import hashlib
import math
import torch
from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.sampling.codesign_runtime import fm_losses, teacher_endpoint_losses
from apexgen.joint_v2.model.task_factorization import aligned_batch


def training_schedule(train_ids, teachers, *, epochs, batch_size, seed):
    if len(set(train_ids))!=len(train_ids) or not train_ids or not teachers:
        raise ValueError('Nonempty unique training IDs and teachers required')
    if epochs<1 or batch_size<1:raise ValueError('Invalid training budget')
    if len({t['teacher_id'] for t in teachers})!=len(teachers):raise ValueError('Duplicate teacher ID')
    by_target={}
    for teacher in teachers:
        if teacher['sample_id'] not in train_ids:raise ValueError('Teacher outside training split')
        by_target.setdefault(teacher['sample_id'],[]).append(teacher['teacher_id'])
    def rank(value):return hashlib.sha256(value.encode()).hexdigest()
    targets=sorted(by_target)
    for sid in targets:by_target[sid].sort()
    schedule=[];visits={sid:0 for sid in targets}
    for epoch in range(epochs):
        ordered=sorted(train_ids,key=lambda sid:rank(f'{seed}:native:{epoch}:{sid}'))
        for offset in range(0,len(ordered),batch_size):
            step=len(schedule)
            cycle=step//len(targets)
            order=sorted(targets,key=lambda sid:rank(f'{seed}:teacher:{cycle}:{sid}'))
            sid=order[step%len(targets)]
            pool=by_target[sid];teacher=pool[visits[sid]%len(pool)];visits[sid]+=1
            stochastic_seed=int(rank(f'{seed}:stochastic:{step}')[:15],16)
            schedule.append(dict(step=step+1,epoch=epoch+1,train_ids=ordered[offset:offset+batch_size],
                                 teacher_id=teacher,stochastic_seed=stochastic_seed))
    return schedule


def codesign_update(model,optimizer,native_records,teacher_record,base,*,device,entry,config):
    torch.manual_seed(entry['stochastic_seed'])
    if device.type=='cuda':torch.cuda.manual_seed_all(entry['stochastic_seed'])
    generator=torch.Generator(device=device).manual_seed(entry['stochastic_seed'])
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr']=config['learning_rate']*min(1.,entry['step']/config['warmup_steps'])
    metrics={}
    model.train()
    for name,records,tmax,weight in [
        ('native_fm',native_records,config['native_time_max'],config['native_weight']),
        ('teacher_fm',[teacher_record],config['teacher_time_max'],config['teacher_fm_weight'])]:
        batch=aligned_batch(collate_joint_v2_records(records).to(device))
        parts=fm_losses(model,batch,generator,time_max=tmax,connection_weight=config.get('connection_weight',0.0),adjacent_translation_weight=config.get('adjacent_translation_weight',0.0))
        loss=weight*parts['total'].mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite '+name)
        loss.backward();metrics[name]=float(loss.detach())
        metrics[name+'_rotation_tangent']=float(parts['rotation_tangent'].detach().mean())
        if 'native_connections' in parts:metrics[name+'_native_connections']=float(parts['native_connections'].detach().mean())
        if 'adjacent_translation' in parts:metrics[name+'_adjacent_translation']=float(parts['adjacent_translation'].detach().mean())
        # Retain endpoint angle error for diagnosis; it is not in the FM total.
        metrics[name+'_rotation_geodesic_diagnostic']=float(parts['final_rotation'].detach().mean())
        del batch,parts,loss
    model.eval()  # Match the deterministic sampler while retaining gradients.
    batch=aligned_batch(collate_joint_v2_records([teacher_record]).to(device))
    _,_,parts=teacher_endpoint_losses(model,batch,base)
    loss=config['teacher_endpoint_weight']*parts['total'].mean()
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite teacher endpoint')
    loss.backward();metrics['teacher_endpoint']=float(loss.detach())
    metrics['early_sequence']=float(parts['joint']['early_sequence'].detach().mean())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
    optimizer.step()
    metrics.update(gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'])
    if not all(math.isfinite(v) for v in metrics.values()):raise FloatingPointError('Nonfinite update metrics')
    return metrics
