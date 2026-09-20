"""Low-noise geometry-only task, independent of the full generative flow objective."""
import hashlib
import math
import torch
from apexgen.joint_v2.experiments.fixed_sequence_scale import FixedSequenceScaleModel
from apexgen.joint_v2.geometry.rotations import so3_log
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.experiments.local_geometry_probe import perturb_native
from apexgen.joint_v2.contracts.state import JointFlowState

FAMILIES=('clean','translation','internal_ca','local_rotation','rigid_rotation','joint')
OBJECTIVES=('translation','rotation')
CONTRACT=dict(schema='apexgen.joint_v2.local_denoise_task.v1',
    task='G_s; low-noise local geometry restoration, not full-base peptide generation',
    architecture='Original FixedSequenceScaleModel; unchanged 8-block accumulating IPA and 1-t gate',
    initialization='cold: archived random step0 weights; warm: original step32000 weights; both reset AdamW/EMA, same random step0 CPU/CUDA states and new paired corruption generator',
    corruption='six equally likely families; severity Uniform[.25,1); translation amplitude severity Angstrom; rotation amplitude20*severity degrees; t=1-.2*severity',
    objective='Only final CA squared error/severity^2 plus final intrinsic rotation squared error/(20deg*severity)^2; sample first',
    contract='Exploratory distinct task; no formal checkpoint compatibility claim')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def seed_for(sample_id, bank, index):
    return int.from_bytes(hashlib.sha256(f'local_denoise_v1:{sample_id}:{bank}:{index}'.encode()).digest()[:4],'little')


def state_slice(state,index):
    return JointFlowState(**{k:getattr(state,k)[index:index+1] for k in ('translation','rotation','sequence_logits')})


def corrupt_one(target, peptide, family, severity, seed, sign=1):
    if family not in FAMILIES or not .25-1e-6 <= severity <= 1.+1e-6:
        raise ValueError('Unknown family or severity outside fixed low-noise support')
    if family=='clean':return target
    if family=='joint':
        moved=perturb_native(target,peptide,'internal_ca',severity,seed,sign)
        return perturb_native(moved,peptide,'local_rotation',20*severity,(seed+7919)%(2**32),sign)
    amplitude=20*severity if family in ('local_rotation','rigid_rotation') else severity
    return perturb_native(target,peptide,family,amplitude,seed,sign)


def sample_corruption(batch,observation,generator):
    device=batch.condition.peptide_mask.device;n=len(batch.sample_ids)
    severity=.25+.75*torch.rand(n,device=device,generator=generator)
    family=torch.randint(len(FAMILIES),(n,),device=device,generator=generator)
    seeds=torch.randint(0,2**31-1,(n,),device=device,generator=generator)
    signs=2*torch.randint(2,(n,),device=device,generator=generator)-1
    target=observation.clamp(batch.targets.endpoint_state(batch.condition));states=[]
    for i,(f,s,seed,sign) in enumerate(zip(family.tolist(),severity.tolist(),seeds.tolist(),signs.tolist())):
        states.append(corrupt_one(state_slice(target,i),batch.condition.peptide_mask[i:i+1],FAMILIES[f],s,seed,sign))
    state=JointFlowState(**{k:torch.cat([getattr(v,k) for v in states]) for k in ('translation','rotation','sequence_logits')})
    return state,1-.2*severity,severity,family,seeds,signs


def denoise_losses(prediction,target,peptide,severity):
    with torch.autocast(device_type=prediction.translation.device.type,enabled=False):
        if bool(((severity<.25-1e-6)|(severity>1+1e-6)).any()):raise ValueError('Severity outside task support')
        count=peptide.sum(-1).clamp_min(1)
        dx=(prediction.translation.float()-target.translation).square().sum(-1)
        dr=so3_log(target.rotation.transpose(-1,-2)@prediction.rotation.float()).square().sum(-1)
        translation=(dx*peptide).sum(-1)/count/severity.float().square()
        rotation=(dr*peptide).sum(-1)/count/(math.radians(20)*severity.float()).square()
    return dict(translation=translation,rotation=rotation,total=translation+rotation)


class LocalDenoiseModel(FixedSequenceScaleModel):
    def __init__(self,config,initialization,code_seed=20260906):
        if initialization not in ('cold','warm'):raise ValueError(initialization)
        super().__init__(config,code_seed)
        self.initialization=initialization
