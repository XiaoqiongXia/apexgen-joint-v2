"""Frozen shared-iteration dynamics probes; independent of historical runtime."""
from types import SimpleNamespace
import torch
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.model.structure_module import Rigid
from apexgen.joint_v2.model.variants.joint_position_fit import position_codes
from apexgen.joint_v2.experiments.local_denoise_task import seed_for
from apexgen.joint_v2.geometry.rotations import so3_log
from apexgen.joint_v2.runtime.lineage import canonical_sha256

CASES=('intact','reset_hidden','freeze_ipa_geometry','repeat_first','sequence_shuffle_0','sequence_shuffle_1','rank_shuffle_0','rank_shuffle_1')
CONTRACT=dict(schema='apexgen.joint_v2.recurrent_drift.v1',
 dynamics='2x2: carry/reset pre-iteration single+sequence latents; current/frozen initial IPA rigid; actual rigid composition always accumulates',
 repeat_first='Reset both hidden latents and freeze IPA geometry: identical local proposal each iteration, composed in accumulating frames',
 sequence='Two deterministic composition-preserving shuffles of initial sequence latent input only; output native sequence remains clamped',
 rank='Two deterministic permutations of extra random rank-code injection; original normalized rank and signed pair rank remain',
 scope='Read-only counterfactual dynamics, not retrained variants or a new sampler; all pocket pathways retained')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def permutation(length,sid,kind,index):
    if length<2:return torch.arange(length)
    gen=torch.Generator().manual_seed(seed_for(sid,'recurrent_drift_'+kind,index))
    forbidden=[torch.arange(length)]
    if index==1 and length>2:forbidden.append(permutation(length,sid,kind,0))
    while True:
        order=torch.randperm(length,generator=gen)
        if not any(torch.equal(order,p) for p in forbidden):return order


def shuffle_peptide(values,mask,sid,kind,index):
    result=values.clone()
    for b in range(len(mask)):
        ids=mask[b].nonzero().flatten();order=permutation(len(ids),sid,kind,index).to(ids.device)
        result[b,ids]=values[b,ids[order]]
    return result


def encode_case(model,obs,case,sid):
    if not case.startswith('rank_shuffle'):return model.encode_complex(obs)
    z=position_codes(obs.pocket.peptide_mask,'random',model.code_seed)
    z=shuffle_peptide(z,obs.pocket.peptide_mask,sid,'rank',int(case[-1]))
    injected=model.encoder.position_projection(model.encoder.position_norm(z))
    return model.encoder(obs.pocket,peptide_single=injected)


def unroll(model,state,time,obs,encoding,case,sid):
    if case not in CASES:raise ValueError(case)
    sm=model.decoder.structure_module;condition=obs.pocket;state=obs.clamp(state)
    assert sm.blocks==8 and not sm.stop_rotation_gradient and sm.sequence_attention_topology=='global'
    single=sm.layer_norm_s(encoding.single);pair=sm.layer_norm_z(encoding.pair);single=sm.linear_in(single)
    gamma,beta=sm.time_film(sm.time_conditioner(time)).chunk(2,dim=-1)
    gate=(1.-time.float())[:,None,None];peptide=condition.peptide_mask[...,None]
    fixed=JointFlowState(torch.where(peptide,state.translation,condition.pocket_translation),torch.where(peptide[...,None],state.rotation,condition.pocket_rotation),torch.where(peptide,state.sequence_logits,torch.zeros_like(state.sequence_logits)))
    rigid=Rigid.from_state(fixed,sm.translation_scale);initial_rigid=rigid;initial_single=single
    z=fixed.sequence_logits
    if case.startswith('sequence_shuffle'):z=shuffle_peptide(z,condition.peptide_mask,sid,'sequence',int(case[-1]))
    sequence=sm.sequence_projection(sm.sequence_norm(z));sequence=torch.where(peptide,sequence,0.);initial_sequence=sequence
    trace=[]
    for iteration in range(8):
        if case in ('reset_hidden','repeat_first'):single,sequence=initial_single,initial_sequence
        before_single,before_sequence=single,sequence
        conditioned=(single+sequence)*(1.+peptide*gamma[:,None])+peptide*beta[:,None]
        read_rigid=initial_rigid if case in ('freeze_ipa_geometry','repeat_first') else rigid
        single=single+sm.ipa(conditioned,pair,read_rigid,condition.residue_mask)
        single=sm.layer_norm_ipa(sm.ipa_dropout(single));single=sm.transition(single)
        update=(sm.backbone_update(single)*gate).float()
        with torch.autocast(device_type=single.device.type,enabled=False):
            rigid=rigid.masked(rigid.compose_q_update_vec(update),condition.peptide_mask)
            raw=rigid.export(sm.translation_scale,fixed.sequence_logits)
            final=JointFlowState(torch.where(peptide,raw.translation,fixed.translation),torch.where(peptide[...,None],raw.rotation,fixed.rotation),fixed.sequence_logits)
        sequence=sm.sequence_latent_transition(sequence,single,condition.peptide_mask)
        trace.append(dict(iteration=iteration+1,translation=final.translation,rotation=final.rotation,update=update,
            single=single,sequence_latent=sequence,single_memory=before_single-initial_single,sequence_memory=before_sequence-initial_sequence,
            single_change=single-before_single,sequence_change=sequence-before_sequence))
    return SimpleNamespace(translation=final.translation,rotation=final.rotation,sequence_logits=final.sequence_logits),trace


def alignment_metrics(x,rotation,target_x,target_rotation):
    """Proper Kabsch alignment; row x@Q -> target and column frames Q.T@R."""
    x=x.double();target_x=target_x.double();rotation=rotation.double();target_rotation=target_rotation.double()
    cx=x.mean(-2,keepdim=True);cy=target_x.mean(-2,keepdim=True);a=x-cx;b=target_x-cy
    u,s,vh=torch.linalg.svd(a.transpose(-1,-2)@b)
    d=torch.ones_like(s);d[...,-1]=torch.linalg.det(u@vh).sign();q=(u*d[...,None,:])@vh
    identifiable=(s[...,1]>s[...,0]*1e-8)&(s[...,0]>1e-12)
    def rms(v):return v.square().sum(-1).mean(-1).sqrt()
    shape=rms(a@q-b);centroid=(cx-cy).square().sum(-1).squeeze(-1).sqrt();centered=rms(a-b)
    pose=so3_log(q.float()).square().sum(-1).sqrt()*180/torch.pi
    frame=so3_log((target_rotation.transpose(-1,-2)@q.transpose(-1,-2)[...,None,:,:]@rotation).float()).square().sum(-1).mean(-1).sqrt()*180/torch.pi
    return dict(ca_aligned_rms=shape,ca_centroid_rms=centroid,ca_centered_rms=centered,
        pose_rotation_degrees=torch.where(identifiable,pose,torch.nan),frame_aligned_degrees=torch.where(identifiable,frame,torch.nan),pose_identifiable=identifiable.double())


def accumulation_gram(clean_positions,native):
    """World CA displacements [K,L,3]; exact total displacement energy identity."""
    positions=clean_positions.double();native=native.double()
    previous=torch.cat([native[None],positions[:-1]],0);delta=positions-previous
    gram=torch.einsum('ilc,jlc->ij',delta,delta)/delta.shape[1]
    total=(positions[-1]-native).square().sum(-1).mean();assert torch.allclose(gram.sum(),total,atol=1e-10,rtol=1e-9)
    diagonal=gram.diag();denom=(diagonal[:,None]*diagonal[None,:]).sqrt()
    cosine=torch.where(denom>1e-20,gram/denom,torch.nan)
    coherence=total/(len(positions)*diagonal.sum()).clamp_min(1e-20)
    return dict(gram=gram,cosine=cosine,diagonal_energy=diagonal.sum(),cross_energy=gram.sum()-diagonal.sum(),total_energy=total,coherence=coherence)
