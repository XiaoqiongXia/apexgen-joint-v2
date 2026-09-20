"""Teacher-paired supervision of an actual jointly generated endpoint."""
import torch
from torch.nn import functional as F


def joint_sample_consistency(backbone, logits, teacher_backbone, teacher_aatype,
                             peptide_mask, *, coordinate_scale=2.0):
    """Return pose, internal-distance and sequence losses per sample.

    Coordinates share the receptor frame; there is no peptide alignment.
    Assignment of a noise base to its teacher is the runner's responsibility.
    """
    if coordinate_scale <= 0 or backbone.shape != teacher_backbone.shape:
        raise ValueError('Invalid coordinate scale or teacher shape')
    if backbone.shape != peptide_mask.shape + (3, 3) or logits.shape != peptide_mask.shape + (20,):
        raise ValueError('Expected unified N/CA/C backbone and 20-way sequence logits')
    if teacher_aatype.shape != peptide_mask.shape or not bool((peptide_mask.sum(-1)>=3).all()):
        raise ValueError('Expected matched sequence labels and at least three peptide residues')
    target=teacher_backbone.detach().float();labels=teacher_aatype.detach()
    scale=coordinate_scale**2;mask=peptide_mask.float()
    error=(backbone.float()-target).square().sum(-1).mean(-1)/scale
    pose=(error*mask).sum(-1)/mask.sum(-1)
    sequence=(F.cross_entropy(logits.float().transpose(1,2),labels.clamp(0,19),reduction='none')*mask).sum(-1)/mask.sum(-1)
    shape=[]
    for b,p in enumerate(peptide_mask):
        ca=backbone[b,p,1].float();ref=target[b,p,1]
        pred_dist=torch.cdist(ca,ca,compute_mode='donot_use_mm_for_euclid_dist')
        ref_dist=torch.cdist(ref,ref,compute_mode='donot_use_mm_for_euclid_dist')
        i=torch.arange(len(ca),device=ca.device)
        selected=(i[None,:]-i[:,None])>=2
        shape.append((pred_dist-ref_dist).square()[selected].mean()/scale)
    return dict(pose=pose,shape=torch.stack(shape),sequence=sequence)
