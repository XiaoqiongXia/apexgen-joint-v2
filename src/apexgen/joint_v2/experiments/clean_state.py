"""Clean intermediate-state supervision; preserve the historical proposal control."""
import math
import torch
from apexgen.joint_v2.geometry.rotations import so3_log
from apexgen.joint_v2.experiments.clean_proposal import proposal_losses,training_objective as original_objective
from apexgen.joint_v2.runtime.lineage import canonical_sha256

CONTRACT=dict(schema='apexgen.joint_v2.clean_state.v1',
 objective='Original final endpoint loss plus clean_indicator * coefficient * auxiliary energy, averaged over all batch samples',
 state_auxiliary='Sum over all 8 exported states: native-referenced CA MSE/a^2 + native-local SO3 angle MSE/(20deg*a)^2; full gradients',
 proposal_control='Exactly the prior clean-proposal energy and coefficient1; exact historical replay checked',
 calibration='One fixed state coefficient sqrt(sum(||g_proposal||^2)/sum(||g_state||^2)), using 2 initial seeds x3 times on a batch of4 fixed training targets; pre-clip BF16-network gradients, no validation',
 schedule='Same mixed8000 model/Adam/EMA/RNG,4096 further steps,category epochs128..191 and clean_proposal_v1 noise bank as historical paired continuation',
 scope='Exploratory native-sequence local geometry recovery, not a formal or full-base generation run')
CONTRACT_SHA256=canonical_sha256(CONTRACT)
CALIBRATION_KEYS=('initial_sources','base_config','data_config','panel_path','position_code_seed','calibration_train_ids','calibration_severities','network_precision')


def state_losses(trace,target,peptide,severity):
    if not trace or bool((severity<=0).any()):raise ValueError('Nonempty trace and positive nominal severity required')
    x=torch.stack([r['translation'].float() for r in trace]);rotation=torch.stack([r['rotation'].float() for r in trace])
    if x.shape[1:3]!=peptide.shape:raise ValueError('Trace/mask mismatch')
    with torch.autocast(device_type=x.device.type,enabled=False):
        count=peptide.sum(-1).clamp_min(1)
        ca=((x-target.translation.float()).square().sum(-1)*peptide).sum(-1)/count/severity.float().square()
        angles=so3_log(target.rotation.float().transpose(-1,-2)@rotation)
        rot=(angles.square().sum(-1)*peptide).sum(-1)/count/(math.radians(20)*severity.float()).square()
    return dict(translation=ca.sum(0),rotation=rot.sum(0),total=(ca+rot).sum(0))


def auxiliary_losses(trace,target,peptide,severity,translation_scale,arm):
    if arm=='proposal':return proposal_losses(trace,peptide,severity,translation_scale)
    if arm=='state':return state_losses(trace,target,peptide,severity)
    raise ValueError(arm)


def training_objective(endpoint,auxiliary,clean,coefficient):
    if not math.isfinite(coefficient) or coefficient<0:raise ValueError(coefficient)
    if coefficient in (0.,1.):return original_objective(endpoint,auxiliary,clean,coefficient)
    return (endpoint['total']+clean.to(endpoint['total'].dtype)*(coefficient*auxiliary['total'])).mean()
