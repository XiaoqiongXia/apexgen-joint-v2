"""Balanced target-pair schedules with matched per-target corruption prefixes."""
import hashlib
from apexgen.joint_v2.diagnostics.generalization_diagnostics import make_schedule
from apexgen.joint_v2.runtime.lineage import canonical_sha256

CONTRACT=dict(schema='apexgen.joint_v2.paired_denoise_scale.v1',task='Unchanged two-target joint local denoising task at a=.5,t=.9, extended to nested32/128 receptors',
    sampler='2 distinct targets/step; +/- corruption per target; seeded balanced epochs; corruption index is per-target visit count',
    initialization='same archived random step0 per seed; fresh Adam/EMA; unchanged8 shared refinements and rotation stop',
    comparisons='equal8000steps; equal125 direction pairs per target at32/2000 versus128/8000; common32,extra96,development14 evaluated separately',
    scope='Exploratory local geometry recovery with native sequence; no formal or full-base generation claim')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def paired_schedule(size,steps,seed):
    if size not in [2,32,128]:raise ValueError(size)
    return [(indices,step//(size//2)) for step,indices in enumerate(make_schedule(size,2,steps,seed))]


def target_digests(state,condition):
    result=[]
    for i in range(state.layout[0]):
        h=hashlib.sha256();mask=condition.residue_mask[i]
        for key in ['translation','rotation','sequence_logits']:
            h.update(getattr(state,key)[i,mask].detach().cpu().numpy().tobytes())
        result.append(h.hexdigest())
    return result
