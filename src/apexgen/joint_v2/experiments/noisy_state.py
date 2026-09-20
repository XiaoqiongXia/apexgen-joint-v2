"""Noisy intermediate-state supervision versus equal-scale endpoint weighting."""
from apexgen.joint_v2.experiments.clean_state import state_losses
from apexgen.joint_v2.experiments.clean_proposal import proposal_losses, training_objective as proposal_objective
from apexgen.joint_v2.runtime.lineage import canonical_sha256
import math

CONTRACT = dict(schema='apexgen.joint_v2.noisy_state.v1',
    objective='Keep original final endpoint and clean proposal penalty; add noisy-only auxiliary, averaged over all batch samples',
    intermediate='Sum native-referenced normalized CA+SO3 state errors over iterations1..7; final iteration8 excluded; full recurrent gradients',
    endpoint='Add one extra noisy final endpoint loss, coefficient1; equal initial RMS auxiliary gradient scale control',
    proposal='Zero extra noisy auxiliary: exact historical clean-proposal numerical control in preflight; archived complete control used for full comparison',
    calibration='One fixed coefficient sqrt(sum(||g_noisy_endpoint||^2)/sum(||g_noisy_intermediate||^2)) over2seeds x3severities,4training targets x2 signs,one fixed independent direction; preclip BF16 gradients, no validation',
    scope='128 targets, two seeds, mixed8000 full state restoration,4096paired continuation steps; fixed native sequence/local recovery')
CONTRACT_SHA256 = canonical_sha256(CONTRACT)
CALIBRATION_KEYS = ('initial_sources','base_config','data_config','panel_path','position_code_seed',
                    'calibration_train_ids','calibration_severities','calibration_noise_bank','network_precision')


def intermediate_losses(trace, target, peptide, severity):
    if len(trace) != 8:
        raise ValueError('Expected all eight shared iterations')
    return state_losses(trace[:-1], target, peptide, severity)


def auxiliary_losses(trace, target, peptide, severity, endpoint, arm):
    if arm in ('proposal', 'endpoint'):
        return endpoint
    if arm == 'intermediate':
        return intermediate_losses(trace, target, peptide, severity)
    raise ValueError(arm)


def training_objective(endpoint, proposal, auxiliary, clean, coefficient):
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError(coefficient)
    base = proposal_objective(endpoint, proposal, clean, 1.)
    if coefficient == 0:
        return base
    return base + coefficient * (auxiliary['total'] * (~clean).to(auxiliary['total'].dtype)).mean()
