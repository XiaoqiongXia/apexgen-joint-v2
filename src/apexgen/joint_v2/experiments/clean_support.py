"""Balanced clean/small-noise support; unchanged model with full rotation gradient."""
import random
import torch
from apexgen.joint_v2.experiments.local_denoise_task import seed_for,state_slice,corrupt_one as original_corrupt,denoise_losses as original_losses
from apexgen.joint_v2.experiments.local_geometry_probe import perturb_native
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.runtime.lineage import canonical_sha256

ARMS=('fixed','mixed')
CONTRACT=dict(schema='apexgen.joint_v2.clean_support.v1',
 initialization='Archived random step0, fresh Adam/EMA, restored CPU/CUDA RNG; both arms full rotation gradients from step0',
 task='128 targets, native sequence fixed; paired local geometry recovery, not full-base generation',
 fixed='joint a=.5,t=.9 every visit; exact historical input bank and global target schedule',
 mixed='Per-target shuffled four-visit blocks: joint .5/.25/.1 and clean, each once; clean nominal severity cycles .1/.25/.5 across blocks with target-specific phase',
 loss='Same final translation/a^2 plus rotation/(20deg*a)^2 extended to a>=.05; clean uses nonzero nominal severity/time, never t=1',
 comparison='Equal 8000 steps and125 target visits, not equal exposure within each corruption category; expands support and its implied normalized weighting')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def category(arm,sample_id,epoch):
    if arm not in ARMS or epoch<0:raise ValueError((arm,epoch))
    if arm=='fixed':return 'joint',.5
    block,slot=divmod(epoch,4);order=[0,1,2,3]
    random.Random(seed_for(sample_id,'clean_support_category',block)).shuffle(order)
    index=order[slot]
    if index<3:return 'joint',[.5,.25,.1][index]
    phase=seed_for(sample_id,'clean_support_clean_phase',0)%3
    return 'clean',[.1,.25,.5][(block+phase)%3]


def corrupt_one(target,peptide,family,severity,seed,sign=1):
    if family not in ('clean','joint') or not .05-1e-6<=severity<=1+1e-6:raise ValueError((family,severity))
    if severity>=.25:return original_corrupt(target,peptide,family,severity,seed,sign)
    if family=='clean':return target
    moved=perturb_native(target,peptide,'internal_ca',severity,seed,sign)
    return perturb_native(moved,peptide,'local_rotation',20*severity,(seed+7919)%(2**32),sign)


def denoise_losses(prediction,target,peptide,severity):
    if bool(((severity<.05-1e-6)|(severity>1+1e-6)).any()):raise ValueError('Outside clean-support range')
    reference=severity.clamp_min(.25)
    losses=original_losses(prediction,target,peptide,reference)
    factor=(reference.float()/severity.float()).square()
    return {k:v*factor for k,v in losses.items()}


def training_input(batch,obs,arm,epoch,bank='two_target_fit'):
    ids=batch.sample_ids
    if len(ids)!=4 or ids[0]!=ids[1] or ids[2]!=ids[3]:raise ValueError('Expected two target sign pairs')
    target=obs.clamp(batch.targets.endpoint_state(batch.condition));parts=[];meta=[]
    for i,sid in enumerate(ids):
        family,a=category(arm,sid,epoch);seed=seed_for(sid,bank,epoch);sign=-1 if i%2==0 else 1
        parts.append(corrupt_one(state_slice(target,i),batch.condition.peptide_mask[i:i+1],family,a,seed,sign))
        meta.append(dict(family=family,severity=a,time=1-.2*a,seed=seed,sign=sign))
    state=JointFlowState(**{k:torch.cat([getattr(s,k) for s in parts]) for k in ('translation','rotation','sequence_logits')})
    severity=torch.tensor([r['severity'] for r in meta],device=state.translation.device)
    return state,1-.2*severity,severity,meta
