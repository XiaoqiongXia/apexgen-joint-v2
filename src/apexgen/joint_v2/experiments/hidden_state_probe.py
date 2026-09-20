"""Single-block, fixed-geometry hidden-state interventions; no training runtime edits."""
from types import SimpleNamespace
import math
import torch
from apexgen.joint_v2.model.structure_module import Rigid
from apexgen.joint_v2.model.variants.dynamic_pair import pair_features
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.geometry.rotations import so3_exp,so3_log
from apexgen.joint_v2.experiments.local_denoise_task import seed_for
from apexgen.joint_v2.runtime.lineage import canonical_sha256

CONTRACT=dict(schema='apexgen.joint_v2.hidden_state_probe.v1',
 donors='Incoming single/sequence latents of clean 8-block trajectory, either naturally accumulating geometry or reset to native before EVERY block',
 queries='One shared block at identical native or controlled perturbed geometry, fixed time, same static pair/pocket/weights; old and dynamic IPA geometry refreshed',
 factorial='Initial (S1,Q1), and (Sk,Qk),(Sk,Q1),(S1,Qk) for k2..8, all residues of the single stream including pocket; sequence latent remains peptide-only',
 perturbations='CA-only centered internal translation and local SO3-only rotations, per-residue RMS normalized; 4 sample-stable directions, +/- and two amplitudes; hidden donors frozen across perturbations',
 outputs='One-block update bias, factorial vector effects and interaction, finite central odd response, actual output error, cross-channel response; no Jacobian spectrum claim',
 scope='Read-only counterfactual dynamics on native-sequence development panel, not retraining, not a sampler or a causal percentage of total generalization failure')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def prepare(sm,state,time,condition,encoding):
    peptide=condition.peptide_mask[...,None]
    fixed=JointFlowState(torch.where(peptide,state.translation,condition.pocket_translation),torch.where(peptide[...,None],state.rotation,condition.pocket_rotation),torch.where(peptide,state.sequence_logits,torch.zeros_like(state.sequence_logits)))
    single=sm.linear_in(sm.layer_norm_s(encoding.single))
    sequence=sm.sequence_projection(sm.sequence_norm(fixed.sequence_logits))
    sequence=torch.where(peptide,sequence,0.)
    return fixed,single,sequence,sm.layer_norm_z(encoding.pair)


def one_block(sm,single,sequence,rigid,pair,condition,time):
    assert sm.sequence_attention_topology=='global'
    peptide=condition.peptide_mask[...,None]
    gamma,beta=sm.time_film(sm.time_conditioner(time)).chunk(2,dim=-1)
    current_pair=pair
    if hasattr(sm,'dynamic_pair'):
        geometry=pair_features(rigid,condition.residue_mask,condition.peptide_mask,sm.translation_scale)
        current_pair=pair+sm.dynamic_pair(geometry).to(pair.dtype)
    conditioned=(single+sequence)*(1.+peptide*gamma[:,None])+peptide*beta[:,None]
    single=single+sm.ipa(conditioned,current_pair,rigid,condition.residue_mask)
    single=sm.layer_norm_ipa(sm.ipa_dropout(single));single=sm.transition(single)
    update=(sm.backbone_update(single)*(1.-time.float())[:,None,None]).float()
    rigid=rigid.masked(rigid.compose_q_update_vec(update),condition.peptide_mask)
    sequence=sm.sequence_latent_transition(sequence,single,condition.peptide_mask)
    return single,sequence,update,rigid


def export(rigid,fixed,condition,scale):
    raw=rigid.export(scale,fixed.sequence_logits);p=condition.peptide_mask[...,None]
    return JointFlowState(torch.where(p,raw.translation,fixed.translation),torch.where(p[...,None],raw.rotation,fixed.rotation),fixed.sequence_logits)


def unroll(sm,state,time,condition,encoding,history='natural'):
    if history not in ['natural','clamped']:raise ValueError(history)
    fixed,single,sequence,pair=prepare(sm,state,time,condition,encoding)
    native=Rigid.from_state(fixed,sm.translation_scale);rigid=native;tr=[];snapshots=[]
    for i in range(sm.blocks):
        if history=='clamped':rigid=native
        snapshots.append((single,sequence))
        single,sequence,update,rigid=one_block(sm,single,sequence,rigid,pair,condition,time)
        final=export(rigid,fixed,condition,sm.translation_scale)
        tr.append(dict(iteration=i+1,translation=final.translation,rotation=final.rotation,update=update,single=single,sequence_latent=sequence))
    return final,tr,snapshots,pair


def combinations(blocks=8):
    return [(1,'initial',0,0)]+[(k+1,case,i,j) for k in range(1,blocks) for case,i,j in [('both',k,k),('single_late',k,0),('sequence_late',0,k)]]


def perturbations(target,mask,sid,directions=4):
    """Return singleton states; independent local generators preserve global RNG."""
    result=[dict(family='clean',amplitude=0.,direction=-1,sign=0,state=target)]
    for family,amplitudes in [('ca',[.02,.1]),('rotation',[.4,2.])]:
        for d in range(directions):
            g=torch.Generator(device=target.translation.device).manual_seed(seed_for(sid,'hidden_state_'+family,d))
            noise=torch.randn(target.translation.shape,device=target.translation.device,generator=g)
            noise=torch.where(mask[...,None],noise,0.)
            count=mask.sum(-1).clamp_min(1)
            if family=='ca':noise=torch.where(mask[...,None],noise-noise.sum(1,keepdim=True)/count[:,None,None],0.)
            norm=(noise.square().sum(-1).sum(-1)/count).sqrt()
            noise=noise/norm[:,None,None]
            for amplitude in amplitudes:
                for sign in [-1,1]:
                    dx=noise*amplitude*sign
                    if family=='ca':state=JointFlowState(target.translation+dx,target.rotation,target.sequence_logits)
                    else:state=JointFlowState(target.translation,target.rotation@so3_exp(dx*math.pi/180),target.sequence_logits)
                    result.append(dict(family=family,amplitude=amplitude,direction=d,sign=sign,state=state))
    return result


def error_vectors(state,target,peptide):
    return (state.translation[:,peptide]-target.translation[:,peptide]).double(), (so3_log(target.rotation[:,peptide].transpose(-1,-2)@state.rotation[:,peptide])*180/math.pi).double()


def factorial(v00,v10,v01,v11):
    """Exact vector identity and projections onto total change, per channel."""
    e=lambda x:x.square().sum(-1).mean()
    dot=lambda x,y:(x*y).sum(-1).mean()
    delta=v11-v00;s=v10-v00;q=v01-v00;inter=v11-v10-v01+v00
    assert torch.allclose(s+q+inter,delta,atol=1e-10,rtol=1e-9)
    values=dict(initial_rms=e(v00).sqrt(),late_rms=e(v11).sqrt(),change_mse=e(delta),single_effect_rms=e(s).sqrt(),sequence_effect_rms=e(q).sqrt(),interaction_rms=e(inter).sqrt(),single_projection=dot(s,delta),sequence_projection=dot(q,delta),interaction_projection=dot(inter,delta))
    assert torch.allclose(values['change_mse'],sum(values[k] for k in ['single_projection','sequence_projection','interaction_projection']),atol=1e-10,rtol=1e-9)
    return {k:float(v) for k,v in values.items()}
