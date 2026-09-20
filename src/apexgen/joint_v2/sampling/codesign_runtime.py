"""Joint-v2 codesign sampling and loss contract shared by training and evaluation.

Using this runtime does not authorize a formal run. Formal provenance is checked
by the separate run lock. Output frames are the unprojected integration endpoint;
N/CA/C atoms are placed independently in those frames without rebuilding a chain.
"""
from dataclasses import fields
import hashlib
import json
import math
import torch
from apexgen.joint_v2.sampling.base import sample_base_state
from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256, UnifiedComplexCondition
from apexgen.joint_v2.sampling.early_sequence_chain import rollout, early_categorical_loss
from apexgen.joint_v2.training.joint_sample_consistency import joint_sample_consistency
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.training.objective_ablation import objective_losses, objective_weights
from apexgen.joint_v2.geometry.peptide_plane import plane_loss
from apexgen.joint_v2.model.variants.real_sequence_transfer import CONTRACT_SHA256 as MODEL_CONTRACT_SHA256
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone
from apexgen.joint_v2.training.refold_teacher import rollout_constraints
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.contracts.task_contract import TaskObservation
from apexgen.joint_v2.model.task_factorization import task_path
from apexgen.shared.training.precision import network_autocast
from apexgen.joint_v2.training.native_connections import native_connection_loss, adjacent_translation_loss

FM_OBJECTIVE_ARM='both'

CONTRACT=dict(schema='apexgen.joint_v2.codesign_runtime.v8',state_contract=JOINT_V2_CONTRACT_SHA256,
    model_contract=MODEL_CONTRACT_SHA256,task='J',steps=20,
    output='raw integrated residue frames; independent N/CA/C placement x=R*x_local+t; no covalent projection or rigid refit',
    covalent_projection=False,
    sequence_clock='model protocol sequence_time_power: s=t**power; matched path/likelihood/integrator; default 1',
    sequence_endpoint='explicit model protocol likelihood (default) or direct centered endpoint; same target MSE and integrator, direct softmax is readout only',
    native_connections='optional observed C-N, CA-N, C-CA, CA-CA distance MSE; explicit connection_weight, default 0',
    adjacent_translation='optional observed adjacent CA displacement vector MSE in A^2; explicit adjacent_translation_weight, default 0',
    sequence_supervision='pre-softmax categorical scores at t=0,.05,.10,.15,.20,.25,.30',
    teacher_mapping='stored observed-clamped base tensors bound to exact static condition; no seed-only reconstruction',
    fm_objective_arm=FM_OBJECTIVE_ARM,
    minimal_objective='explicit objective_arm=minimal: translation /100, tangent rotation /pi^2, centered sequence endpoint MSE; unit weights; no auxiliary losses',
    rotation_supervision='mean_peptide ||Log(R_s^T R_hat_1)-Log(R_s^T R_1)||^2 / pi^2',
    rotation_time_weight='(1-s)^2 weighted angular velocity MSE; no inverse-time reweighting',
    rotation_branch='shared principal SO(3) log in FP32, symmetric-axis recovery near pi',
    native_fm='both objective arm, random interpolation time below configured native_time_max',
    teacher_fm='same both objective arm, random interpolation time below configured teacher_time_max',
    final_success='own-design refold plus generation geometry/core and effective sequence diversity')
CONTRACT_SHA256=canonical_sha256(CONTRACT)
BASE_SCHEMA='apexgen.joint_v2.stored_base.v1'


def tensors_digest(named):
    digest=hashlib.sha256()
    for name,value in named:
        value=value.detach().cpu().contiguous()
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def condition_digest(condition):
    if not isinstance(condition,UnifiedComplexCondition):raise TypeError('Expected Joint-v2 static condition')
    return tensors_digest((field.name,getattr(condition,field.name)) for field in fields(condition))


def pack_base(base,condition):
    if base.layout!=condition.layout:raise ValueError('Base and condition layouts differ')
    base=TaskObservation('J',condition).clamp(base)
    state={name:getattr(base,name).detach().cpu().clone() for name in ['translation','rotation','sequence_logits']}
    return dict(schema=BASE_SCHEMA,condition_sha256=condition_digest(condition),state=state,
        state_sha256=tensors_digest(state.items()))


def unpack_base(payload,condition):
    if payload.get('schema')!=BASE_SCHEMA or payload.get('condition_sha256')!=condition_digest(condition):
        raise ValueError('Stored noise belongs to a different condition or schema')
    state=payload['state']
    if list(state)!=['translation','rotation','sequence_logits'] or tensors_digest(state.items())!=payload['state_sha256']:
        raise ValueError('Stored noise tensor identity mismatch')
    device=condition.pocket_translation.device
    base=JointFlowState(**{name:value.to(device=device).clone() for name,value in state.items()})
    if base.layout!=condition.layout:raise ValueError('Stored noise layout mismatch')
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):raise ValueError('Nonfinite stored noise')
    clamped=TaskObservation('J',condition).clamp(base)
    if any(not torch.equal(getattr(base,name),getattr(clamped,name)) for name in state):
        raise ValueError('Stored base contains noncanonical observed fields')
    return base


def sample_codesign(model,condition,*,base=None,generator=None,collect_scores=False):
    if base is None:base=sample_base_state(condition,generator=generator)
    if base.layout!=condition.layout:raise ValueError('Base layout does not match condition')
    return rollout(model,base,TaskObservation('J',condition),chain=False,collect_scores=collect_scores)


def fm_losses(model,batch,generator,*,time_max,connection_weight=0.0,adjacent_translation_weight=0.0,objective_arm=FM_OBJECTIVE_ARM):
    if not 0<time_max<1:raise ValueError('FM time limit must lie strictly between 0 and 1')
    if not math.isfinite(connection_weight) or connection_weight<0:raise ValueError('connection_weight must be finite and nonnegative')
    if not math.isfinite(adjacent_translation_weight) or adjacent_translation_weight<0:raise ValueError('adjacent_translation_weight must be finite and nonnegative')
    objective_weights(objective_arm)
    if objective_arm=='minimal' and (connection_weight or adjacent_translation_weight):
        raise ValueError('minimal objective excludes connection and adjacent translation losses')
    observation=TaskObservation('J',batch.condition)
    base=sample_base_state(batch.condition,generator=generator)
    time=torch.rand(base.layout[0],device=base.translation.device,generator=generator)*time_max
    state=task_path(base,batch,observation,time,sequence_time_power=model.sequence_time_power)
    with network_autocast(base.translation.device,'bfloat16'):prediction=model(state,time,observation)
    losses=objective_losses(prediction,batch,state,objective_arm)
    if connection_weight:
        losses['native_connections']=native_connection_loss(prediction,batch)
        losses['total']=losses['total']+connection_weight*losses['native_connections']
    if adjacent_translation_weight:
        losses['adjacent_translation']=adjacent_translation_loss(prediction,batch)
        losses['total']=losses['total']+adjacent_translation_weight*losses['adjacent_translation']
    return losses


def teacher_endpoint_losses(model,batch,base):
    state,raw,scores=sample_codesign(model,batch.condition,base=base,collect_scores=True)
    joint=joint_sample_consistency(reconstruct_backbone(state),state.sequence_logits,
        batch.targets.backbone_xyz,batch.targets.endpoint_aatype,batch.condition.peptide_mask)
    joint['early_sequence']=early_categorical_loss(scores,batch.targets.endpoint_aatype,batch.condition.peptide_mask)
    parts=rollout_constraints(state,batch.condition);parts['plane']=plane_loss(state,batch.condition)
    total=sum(joint[key] for key in ['pose','shape','early_sequence'])+5*sum(parts[key] for key in ['geometry','site','plane'])
    return state,raw,dict(joint=joint,geometry=parts,total=total)
