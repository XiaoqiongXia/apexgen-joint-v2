"""Reversible pocket information-path interventions and paired error decomposition."""
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType
import torch
from apexgen.joint_v2.runtime.lineage import canonical_sha256

ROUTES=('intact','encoder_off','decoder_off','both_off')
CONTRACT=dict(schema='apexgen.joint_v2.pocket_bias_routes.v1',
 encoder_off='Block peptide-query/pocket-key scalar attention AND pair-context average in every encoder block; retain pair values and pocket queries',
 decoder_off='Block peptide-query/pocket-key IPA attention at every refinement, including scalar, pair and geometry reads; retain pocket queries',
 unchanged='Native sequence, peptide rank, current peptide frames, learned parameters and checkpoint files are unchanged',
 decomposition='In native-relative CA and SO3-log error coordinates: e_even=(e+ + e-)/2, e_odd=(e+ - e-)/2; even=b_clean+curvature; exact signed-pair MSE=even^2+odd^2',
 scope='Inference path sensitivity on trained models; ablated inputs are distribution shifts, clean bias uses oracle native input, not a deployable correction')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def encoder_forward(block,single,pair,mask,peptide,pocket):
    pair_mask=mask[:,:,None]&mask[:,None,:]
    allowed=~(peptide[:,:,None]&pocket[:,None,:])
    batch,length=mask.shape
    bias=block.pair_bias(pair).permute(0,3,1,2)
    bias=bias.masked_fill(~mask[:,None,None,:],-torch.inf)
    bias=bias.masked_fill(~allowed[:,None],-torch.inf)
    update,_=block.attention(block.single_norm(single),block.single_norm(single),block.single_norm(single),attn_mask=bias.reshape(batch*block.heads,length,length),need_weights=False)
    weights=(pair_mask&allowed)[...,None].to(pair.dtype)
    context=(pair*weights).sum(2)/weights.sum(2).clamp_min(1.)
    single=single+update+block.pair_context(context)
    single=single+block.single_transition(single)
    single=torch.where(mask[...,None],single,0.)
    pair=pair+block.left_pair(single)[:,:,None]+block.right_pair(single)[:,None,:]
    pair=pair+block.pair_transition(pair)
    pair=torch.where(pair_mask[...,None],pair,0.)
    return single,pair


@contextmanager
def route_context(model,condition,route,*,noop=False):
    if route not in ROUTES:raise ValueError(route)
    saved=[];peptide=condition.peptide_mask;pocket=condition.pocket_mask
    if noop:pocket=torch.zeros_like(pocket)
    def patch(module,fn):
        saved.append((module,'forward' in module.__dict__,module.__dict__.get('forward')))
        module.forward=MethodType(fn,module)
    try:
        if route in ('encoder_off','both_off'):
            def enc(self,single,pair,mask):
                return encoder_forward(self,single,pair,mask,peptide.expand(mask.shape[0],-1),pocket.expand(mask.shape[0],-1))
            for block in model.encoder.blocks:patch(block,enc)
        if route in ('decoder_off','both_off'):
            ipa=model.decoder.structure_module.ipa;original=ipa.forward
            def dec(self,single,pair,rigid,mask,*,attention_mask=None):
                allowed=~(peptide[:,:,None]&pocket[:,None,:]);allowed=allowed[:,None].expand(mask.shape[0],self.no_heads,-1,-1)
                if attention_mask is not None:allowed=allowed&attention_mask
                return original(single,pair,rigid,mask,attention_mask=allowed)
            patch(ipa,dec)
        yield
    finally:
        for module,had,value in reversed(saved):
            if had:module.forward=value
            else:delattr(module,'forward')


def changed_pocket(condition):
    """Same layout/peptide; receptor translation plus chemistry sensitivity probe."""
    shift=condition.pocket_translation.new_tensor([3.,-2.,1.])
    p=condition.pocket_mask
    return replace(condition,
        aatype=torch.where(p,(condition.aatype+1)%20,condition.aatype),
        pocket_translation=condition.pocket_translation+p[...,None]*shift,
        pocket_atom_xyz=condition.pocket_atom_xyz+(p[...,None]&condition.pocket_atom_mask)[...,None]*shift)


def paired_decomposition(clean,negative,positive,input_negative,input_positive):
    """[L,3] clean and [D,L,3] signed errors; all values per direction."""
    clean=clean.double();negative=negative.double();positive=positive.double();input_negative=input_negative.double();input_positive=input_positive.double()
    def energy(x):return x.square().sum(-1).mean(-1)
    def dot(x,y):return (x*y).sum(-1).mean(-1)
    even=(positive+negative)/2;odd=(positive-negative)/2;curvature=even-clean
    desired=(input_positive-input_negative)/2;correction=odd-desired
    mse=(energy(positive)+energy(negative))/2;ev=energy(even);od=energy(odd)
    bias=energy(clean).expand_as(mse);curve=energy(curvature);cross=2*dot(clean,curvature)
    denominator=(energy(correction)*energy(desired)).sqrt()
    cosine=torch.where(denominator>1e-20,dot(correction,-desired)/denominator,torch.nan)
    result=dict(pair_mse=mse,even_mse=ev,odd_mse=od,clean_bias_mse=bias,curvature_mse=curve,bias_curvature_cross=cross,
        input_pair_mse=(energy(input_positive)+energy(input_negative))/2,
        odd_gain=(od/energy(desired).clamp_min(1e-20)).sqrt(),odd_correction_cosine=cosine,
        clean_centroid_mse=clean.mean(-2).square().sum(-1).expand_as(mse),
        clean_internal_mse=energy(clean-clean.mean(-2,keepdim=True)).expand_as(mse))
    assert torch.allclose(mse,ev+od,atol=1e-9,rtol=1e-9)
    assert torch.allclose(mse,bias+curve+cross+od,atol=1e-9,rtol=1e-9)
    return result
