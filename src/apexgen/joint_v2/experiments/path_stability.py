"""Matched full-path and endpoint replay objectives; no architecture changes."""
from dataclasses import fields,is_dataclass
import random
import torch
from apexgen.joint_v2.sampling.base import sample_base_state
from apexgen.joint_v2.experiments.clean_support import training_input,corrupt_one,denoise_losses
from apexgen.joint_v2.experiments.clean_proposal import proposal_losses,training_objective
from apexgen.joint_v2.geometry.rotations import so3_log
from apexgen.joint_v2.experiments.geometry_controls import chain_continuity_terms
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone
from apexgen.joint_v2.experiments.local_denoise_task import seed_for,state_slice
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.model.task_factorization import observation_from_batch,task_path
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.shared.training.precision import network_autocast

ARMS=('local','path','endpoint','chain')
CONTRACT=dict(schema='apexgen.joint_v2.path_stability.v1',
    architecture='Unchanged DynamicPairStructureModule, 8 shared iterations, full rotation gradients',
    initialization='Warm from coverage-transfer n128/8192 per seed; exact raw/Adam/EMA/RNG restoration',
    local='Original normalized local endpoint + clean proposal, every arm every step, new per-target visit prefix',
    path='32 balanced times spanning 0..0.95; two independent stable bases per target; final CA MSE/(10 A)^2 + final SO3 angle MSE/(1 rad)^2',
    endpoint='Additional quarter-weighted clean/joint .05/.1 replay and same clean proposal penalty; no additional inference gate',
    chain='Additional .1-weighted final path CN and two bond-angle continuity terms; no projection and no native input',
    controls='local; local+path; local+path+.25*endpoint; local+path+.1*chain. No endpoint+chain combination in this round',
    scope='Exploratory fixed native sequence; repeated development14 and additional7, not blind test or formal training')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def cat_states(states):
    return JointFlowState(**{k:torch.cat([getattr(s,k) for s in states]) for k in ('translation','rotation','sequence_logits')})


def tree_row(value,index):
    if isinstance(value,torch.Tensor):return value[index:index+1]
    if is_dataclass(value):return type(value)(**{f.name:tree_row(getattr(value,f.name),index) for f in fields(value)})
    return value


def micro_category(sid,epoch):
    categories=[('clean',.05),('clean',.1),('joint',.05),('joint',.1)]
    random.Random(seed_for(sid,'path_stability_micro_category',epoch//4)).shuffle(categories)
    return categories[epoch%4]


def path_time(sid,epoch,seed):
    bins=list(range(32));random.Random(seed_for(sid,f'path_stability_times:{seed}',epoch//32)).shuffle(bins)
    return .95*bins[epoch%32]/31


def make_cases(batch,singles,epoch,p):
    obs=observation_from_batch(batch,'G_s');target=obs.clamp(batch.targets.endpoint_state(batch.condition));device=target.translation.device
    local,t,a,meta=training_input(batch,obs,'mixed',epoch,bank=p['noise_bank'])
    clean=torch.tensor([r['family']=='clean' for r in meta],device=device)
    x=batch.condition.pocket_translation.clone();rot=batch.condition.pocket_rotation.clone();z=torch.zeros_like(target.sequence_logits)
    path_meta=[];micro=[];micro_meta=[]
    for i,sid in enumerate(batch.sample_ids):
        base_seed=seed_for(sid,f"path_stability_base:{p['seed']}",epoch*2+i%2)
        g=torch.Generator(device=device).manual_seed(base_seed)
        base=sample_base_state(singles[sid].condition,generator=g);n=base.layout[1]
        x[i,:n]=base.translation[0];rot[i,:n]=base.rotation[0];z[i,:n]=base.sequence_logits[0]
        path_meta.append(dict(sample_id=sid,time=path_time(sid,epoch,p['seed']),base_seed=base_seed,base_index=i%2))
        family,severity=micro_category(sid,epoch);sign=-1 if i%2==0 else 1;noise_seed=seed_for(sid,'path_stability_micro_v1',epoch)
        micro.append(corrupt_one(state_slice(target,i),obs.pocket.peptide_mask[i:i+1],family,severity,noise_seed,sign))
        micro_meta.append(dict(family=family,severity=severity,time=1-.2*severity,seed=noise_seed,sign=sign))
    base=obs.clamp(JointFlowState(x,rot,z));pt=torch.tensor([r['time'] for r in path_meta],device=device)
    path=task_path(base,batch,obs,pt)
    ma=torch.tensor([r['severity'] for r in micro_meta],device=device);mc=torch.tensor([r['family']=='clean' for r in micro_meta],device=device)
    return dict(obs=obs,target=target,local=dict(state=local,time=t,severity=a,clean=clean,meta=meta),
                path=dict(state=path,time=pt,meta=path_meta,base=base),
                endpoint=dict(state=cat_states(micro),time=1-.2*ma,severity=ma,clean=mc,meta=micro_meta))


def path_losses(pred,target,condition):
    with torch.autocast(device_type=pred.translation.device.type,enabled=False):
        mask=condition.peptide_mask;count=mask.sum(-1).clamp_min(1)
        ca=(((pred.translation.float()-target.translation).square().sum(-1))*mask).sum(-1)/count/100.
        rotation=(so3_log(target.rotation.transpose(-1,-2)@pred.rotation.float()).square().sum(-1)*mask).sum(-1)/count
    return dict(translation=ca,rotation=rotation,total=ca+rotation)


def weighted_total(terms,arm):
    if arm not in ARMS:raise ValueError(arm)
    value=terms['local']
    if arm!='local':value=value+terms['path']
    if arm=='endpoint':value=value+.25*terms['endpoint']
    if arm=='chain':value=value+.1*terms['chain']
    return value


def objective(model,cases,p,device):
    """All arms observe the same three forwards; only declared loss terms differ."""
    obs=cases['obs'];target=cases['target'];mask=obs.pocket.peptide_mask
    terms={};details={};traces={};predictions={}
    for name in ['local','path','endpoint']:
        c=cases[name];trace=[]
        with network_autocast(device,p['network_precision']):pred=model(c['state'],c['time'],obs,trace=trace)
        assert torch.equal(pred.sequence_logits,target.sequence_logits)
        assert torch.equal(pred.translation[~mask],target.translation[~mask]) and torch.equal(pred.rotation[~mask],target.rotation[~mask])
        predictions[name]=pred;traces[name]=trace
        if name=='path':
            end=path_losses(pred,target,obs.pocket);terms[name]=end['total'].mean()
            xyz=reconstruct_backbone(JointFlowState(pred.translation,pred.rotation,pred.sequence_logits))
            chain=chain_continuity_terms(xyz,obs.pocket);terms['chain']=chain['chain_continuity'].mean();details['chain']=chain
            details[name]=end
        else:
            end=denoise_losses(pred,target,mask,c['severity'])
            aux=proposal_losses(trace,mask,c['severity'],model.decoder.structure_module.translation_scale)
            terms[name]=training_objective(end,aux,c['clean'],1.)
            details[name]=dict(endpoint=end['total'],proposal=aux['total']*c['clean'])
    return weighted_total(terms,p['arm']),terms,details,traces,predictions
