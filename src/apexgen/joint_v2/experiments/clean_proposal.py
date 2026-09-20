"""Clean-only per-iteration geometry-update energy; original runtime unchanged."""
import math
import torch
from apexgen.joint_v2.runtime.lineage import canonical_sha256

CONTRACT=dict(schema='apexgen.joint_v2.clean_proposal.v1',
 objective='Original normalized final CA+SO3 loss plus lambda * clean_indicator * sum over 8 local update energies, averaged over all batch samples',
 translation='Actual gated local translation * structure translation scale; residue mean squared norm / nominal severity squared',
 rotation='Exact relative quaternion angle 2 atan(norm(qxyz)) from quaternion (1,qxyz); residue mean angle squared / (20deg*nominal severity) squared',
 controls='lambda=0 versus lambda=1; no change to noisy endpoint objective, architecture, recurrence, pocket or conditioning',
 initialization='Paired continuation from mixed-support step8000, restoring model, Adam, EMA, CPU/CUDA RNG; 4096 further updates; no warmup reset',
 schedule='128 targets, two target +/- pairs per batch; 64 balanced visits per target, corruption category epochs128..191; same new-bank noise for both arms',
 scope='Exploratory fixed-native-sequence local geometry continuation; not formal training or full-base generation')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def proposal_losses(trace,peptide,severity,translation_scale):
    """Return per-sample energy without selecting clean rows or averaging samples."""
    if not trace or bool((severity<=0).any()):raise ValueError('Nonempty trace and positive nominal severity required')
    updates=torch.stack([x['update'].float() for x in trace])
    if updates.shape[1:3]!=peptide.shape:raise ValueError('Trace/mask mismatch')
    with torch.autocast(device_type=updates.device.type,enabled=False):
        count=peptide.sum(-1).clamp_min(1)
        dx=updates[...,3:]*translation_scale
        angle=2*torch.atan(torch.linalg.vector_norm(updates[...,:3],dim=-1))
        ca=(dx.square().sum(-1)*peptide).sum(-1)/count/severity.float().square()
        rot=(angle.square()*peptide).sum(-1)/count/(math.radians(20)*severity.float()).square()
    return dict(translation=ca.sum(0),rotation=rot.sum(0),total=(ca+rot).sum(0))


def training_objective(endpoint,proposal,clean,coefficient):
    if coefficient not in (0.,1.):raise ValueError(coefficient)
    # Avoid adding a zero-weight graph to the reference objective.
    if coefficient==0:return endpoint['total'].mean()
    return (endpoint['total']+clean.to(endpoint['total'].dtype)*proposal['total']).mean()
