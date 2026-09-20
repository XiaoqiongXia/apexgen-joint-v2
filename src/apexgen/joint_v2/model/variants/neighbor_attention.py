"""Parameter-free directional peptide-neighbor routing for exploratory IPA."""

import torch


def peptide_neighbor_attention(condition, heads):
    """Heads 0/1/2 read self/left/right; remaining heads retain global access.

    Adjacency uses same chain and signed sequence-index difference, not tensor
    storage order. Missing neighbors fall back to self. Pocket queries retain
    their original global access. Cyclic wrap-around is not inferred.
    """
    if heads < 4:
        raise ValueError("neighbor routing requires at least four IPA heads")
    residue = condition.residue_mask
    peptide = condition.peptide_mask & residue
    size = residue.shape[1]
    valid = residue[:, :, None] & residue[:, None, :]
    allowed = valid[:, None].expand(-1, heads, -1, -1).clone()
    diagonal = torch.eye(size, device=residue.device, dtype=torch.bool)[None] & valid
    same_chain = condition.chain_index[:, :, None] == condition.chain_index[:, None, :]
    delta = condition.sequence_index[:, None, :] - condition.sequence_index[:, :, None]
    peptide_pair = peptide[:, :, None] & peptide[:, None, :] & same_chain
    for head, offset in enumerate([0, -1, 1]):
        neighbor = diagonal if offset == 0 else peptide_pair & (delta == offset)
        neighbor = neighbor | (~neighbor.any(-1, keepdim=True) & diagonal)
        allowed[:, head] = torch.where(peptide[:, :, None], neighbor, valid)
    return allowed
