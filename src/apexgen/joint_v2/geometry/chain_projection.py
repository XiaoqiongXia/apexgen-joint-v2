"""Exploratory fixed-covalent-chain projection of a generated N/CA/C backbone.

Only generated coordinates, generated residue identities and receptor context
enter this module. No native peptide or external refold coordinates are inputs.
"""
import math
import torch
from torch.nn import functional as F
from apexgen.shared.geometry.backbone import place_atom
from apexgen.shared.geometry.residue_constants import BACKBONE_GEOMETRY
from apexgen.shared.geometry.joint_residue_constants import (
    AA3_TO_INDEX, BETWEEN_RES_BOND_LENGTH_C_N,
    BETWEEN_RES_COS_ANGLES_CA_C_N, BETWEEN_RES_COS_ANGLES_C_N_CA,
)
from apexgen.joint_v2.geometry.reconstruction import ideal_backbone_local
from apexgen.joint_v2.geometry.rotations import so3_exp


def dihedral(a, b, c, d):
    axis=F.normalize(c-b,dim=-1,eps=1e-12)
    v=a-b;w=d-c
    v=v-(v*axis).sum(-1,keepdim=True)*axis
    w=w-(w*axis).sum(-1,keepdim=True)*axis
    return torch.atan2((torch.cross(axis,v,dim=-1)*w).sum(-1),(v*w).sum(-1))


def internal_from_backbone(backbone, aatype):
    """Extract phi/psi, choose fixed cis/trans only at bonds before Pro.

    Non-Pro links are trans in this bounded projection family. At Pro, preserve
    the nearest input cis/trans basin; this is not a learned isomer distribution.
    """
    phi=torch.zeros_like(aatype,dtype=backbone.dtype)
    psi=torch.zeros_like(phi)
    phi[:,1:]=dihedral(backbone[:,:-1,2],backbone[:,1:,0],backbone[:,1:,1],backbone[:,1:,2])
    psi[:,:-1]=dihedral(backbone[:,:-1,0],backbone[:,:-1,1],backbone[:,:-1,2],backbone[:,1:,0])
    raw_omega=dihedral(backbone[:,:-1,1],backbone[:,:-1,2],backbone[:,1:,0],backbone[:,1:,1])
    cis=(aatype[:,1:]==AA3_TO_INDEX['PRO']) & (raw_omega.abs()<math.pi/2)
    omega=torch.where(cis,torch.zeros_like(raw_omega),torch.full_like(raw_omega,math.pi))
    return torch.stack([phi,psi],-1),omega


def build_chain(torsions, aatype, omega, mask):
    """Construct a continuous chain with the Joint-v2 local residue geometry."""
    if torsions.ndim!=3 or torsions.shape[-1]!=2 or torsions.shape[:2]!=aatype.shape or mask.shape!=aatype.shape:
        raise ValueError('Expected torsions [B,L,2], aatype/mask [B,L]')
    if omega.shape!=(aatype.shape[0],aatype.shape[1]-1):raise ValueError('Invalid omega layout')
    g=BACKBONE_GEOMETRY
    first=ideal_backbone_local(device=torsions.device,dtype=torsions.dtype).expand(len(torsions),3,3)
    n,ca,c=[first[:,0]],[first[:,1]],[first[:,2]]
    refs=torch.as_tensor(BETWEEN_RES_BOND_LENGTH_C_N,device=torsions.device,dtype=torsions.dtype)
    angle1=math.acos(float(BETWEEN_RES_COS_ANGLES_CA_C_N[0]))
    angle2=math.acos(float(BETWEEN_RES_COS_ANGLES_C_N_CA[0]))
    for j in range(aatype.shape[1]-1):
        cn=refs[(aatype[:,j+1]==AA3_TO_INDEX['PRO']).long()][:,None]
        nn=place_atom(n[-1],ca[-1],c[-1],cn,angle1,torsions[:,j,1])
        cc=place_atom(ca[-1],c[-1],nn,g.n_ca,angle2,omega[:,j])
        cnext=place_atom(c[-1],nn,cc,g.ca_c,g.n_ca_c,torsions[:,j+1,0])
        n.append(nn);ca.append(cc);c.append(cnext)
    bb=torch.stack([torch.stack(n,1),torch.stack(ca,1),torch.stack(c,1)],2)
    centroid=(bb[:,:,1]*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
    return bb-centroid[:,None,None,:]


def initial_pose(chain, reference, mask):
    """Fit the reconstructed chain to its own generated design, never to native."""
    weights=mask[:,:,None].expand(-1,-1,3).reshape(len(mask),-1).to(chain.dtype)
    a,b=chain.flatten(1,2),reference.flatten(1,2)
    ma=(a*weights[...,None]).sum(1)/weights.sum(1,keepdim=True)
    mb=(b*weights[...,None]).sum(1)/weights.sum(1,keepdim=True)
    covariance=(a-ma[:,None]).transpose(-1,-2)@((b-mb[:,None])*weights[...,None])
    u,_,vh=torch.linalg.svd(covariance)
    correction=torch.eye(3,device=a.device,dtype=a.dtype).repeat(len(a),1,1)
    correction[:,-1,-1]=torch.linalg.det(u@vh)
    rotation=u@correction@vh
    translation=mb-torch.einsum('bi,bij->bj',ma,rotation)
    return rotation,translation


def position_chain(chain, rotation_vector, initial_rotation, translation):
    rotation=so3_exp(rotation_vector).transpose(-1,-2)@initial_rotation
    return torch.einsum('blai,bij->blaj',chain,rotation)+translation[:,None,None,:]


def projection_terms(backbone, reference, mask, context_xyz, context_mask, core_mask):
    """Per-case tether, nonlocal clashes and observed-core attraction."""
    weights=mask.to(backbone.dtype)
    tether=((backbone-reference).square().sum(-1).mean(-1)*weights).sum(-1)/weights.sum(-1)
    atoms=backbone.flatten(1,2); valid=mask.repeat_interleave(3,dim=-1)
    owners=torch.arange(mask.shape[1],device=backbone.device).repeat_interleave(3)
    eligible=(owners[:,None]-owners[None,:]).abs()>1
    eligible=eligible[None]&valid[:,:,None]&valid[:,None,:]
    distances=torch.cdist(atoms,atoms,compute_mode='donot_use_mm_for_euclid_dist').masked_fill(~eligible,float('inf'))
    nearest=distances.min(-1).values
    self_clash=(F.relu(2.2-nearest).square()*valid).sum(-1)/valid.sum(-1)
    cross=torch.cdist(atoms,context_xyz.flatten(1,2),compute_mode='donot_use_mm_for_euclid_dist')
    cross=cross.masked_fill(~context_mask.flatten(1)[:,None,:],float('inf'))
    context_clash=(F.relu(2.2-cross.min(-1).values).square()*valid).sum(-1)/valid.sum(-1)
    core_atoms=core_mask[:,:,None].expand_as(context_mask)&context_mask
    core_d=cross.masked_fill(~core_atoms.flatten(1)[:,None,:],float('inf'))
    per_residue=core_d.reshape(len(mask),mask.shape[1],3,-1).flatten(2).min(-1).values
    core_distance=(F.relu(per_residue-4.5).square()*weights).sum(-1)/weights.sum(-1)/25
    core_residue_d=cross.masked_fill(~valid[:,:,None],float('inf')).reshape(len(mask),-1,context_mask.shape[1],context_mask.shape[2]).amin((1,3))
    second=core_residue_d.masked_fill(~core_mask,float('inf')).topk(2,largest=False).values[:,1]
    core_coverage=F.relu(second-4.5).square()/25
    return dict(tether=tether,self_clash=self_clash,context_clash=context_clash,
                site=core_distance+core_coverage)
