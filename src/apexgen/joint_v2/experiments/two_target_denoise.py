"""Paired perturbations and reversible decoder-path interventions for diagnostics."""
from contextlib import contextmanager
import torch
from apexgen.joint_v2.experiments.local_denoise_task import corrupt_one,seed_for,state_slice
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.runtime.lineage import canonical_sha256

CONTRACT=dict(schema='apexgen.joint_v2.two_target_denoise.v1',task='Native-sequence fixed; joint internal-CA/local-frame denoising on two frozen training targets',
    training='cold archived step0; fixed vs resampled antithetic noise; a=.5,t=.9; original shared8 decoder, rotation stop enabled; two target pairs per batch',
    scope='New narrow optimization diagnostic, not directly comparable to previous six-family task; frozen1/8 and rotation-stop probes are not retrained variants')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def paired_state(batch,obs,bank,index,severity=.5):
    if len(batch.sample_ids)!=4 or batch.sample_ids[0]!=batch.sample_ids[1] or batch.sample_ids[2]!=batch.sample_ids[3]:
        raise ValueError('Expected [target0,target0,target1,target1]')
    target=obs.clamp(batch.targets.endpoint_state(batch.condition));parts=[];seeds=[]
    for i,sid in enumerate(batch.sample_ids):
        seed=seed_for(sid,bank,index);seeds.append(seed)
        parts.append(corrupt_one(state_slice(target,i),batch.condition.peptide_mask[i:i+1],'joint',severity,seed,-1 if i%2==0 else 1))
    return JointFlowState(**{k:torch.cat([getattr(v,k) for v in parts]) for k in ['translation','rotation','sequence_logits']}),seeds


@contextmanager
def decoder_path(model,blocks,stop_rotation):
    if blocks not in [1,8]:raise ValueError(blocks)
    sm=model.decoder.structure_module;original=(sm.blocks,sm.stop_rotation_gradient)
    sm.blocks=blocks;sm.stop_rotation_gradient=stop_rotation
    try:yield
    finally:sm.blocks,sm.stop_rotation_gradient=original
