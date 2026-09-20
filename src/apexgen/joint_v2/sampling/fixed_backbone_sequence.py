"""Inputs for an exploratory, frozen-backbone ProteinMPNN diagnostic.

ProteinMPNN is supplied externally by the experiment, not a v2 runtime dependency.
"""
import math
import numpy as np
import torch
from apexgen.shared.geometry.backbone import place_atom
from apexgen.joint_v2.geometry.chain_projection import dihedral

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def add_carbonyl_oxygen(backbone):
    """Add O opposite next N in the peptide plane, without moving N/CA/C.

    Terminal psi is undefined: use psi=pi, hence N-CA-C-O torsion zero.
    This deterministic virtual O is input featurization, not all-atom validation.
    """
    bb = torch.as_tensor(np.asarray(backbone), dtype=torch.float64)
    if bb.ndim != 3 or bb.shape[1:] != (3, 3) or len(bb) < 2 or not torch.isfinite(bb).all():
        raise ValueError('Expected finite N/CA/C [L>=2,3,3]')
    psi = torch.full((len(bb),), math.pi, dtype=bb.dtype)
    psi[:-1] = dihedral(bb[:-1, 0], bb[:-1, 1], bb[:-1, 2], bb[1:, 0])
    oxygen = place_atom(bb[:, 0], bb[:, 1], bb[:, 2], 1.231, math.radians(120.8), psi + math.pi)
    return torch.cat([bb, oxygen[:, None]], dim=1).numpy()


def mpnn_inputs(receptor_backbone, receptor_sequence, peptide_backbone, original_sequence, device='cpu'):
    """A fixed receptor, redesigned peptide, fixed Pro pattern; no other peptide labels."""
    nr, np_ = len(receptor_sequence), len(original_sequence)
    rec, pep = np.asarray(receptor_backbone), np.asarray(peptide_backbone)
    if rec.shape != (nr, 4, 3) or pep.shape != (np_, 4, 3):
        raise ValueError('Backbone length/atom layout mismatch')
    coords = np.concatenate([rec, pep])
    if not np.isfinite(coords).all():
        raise ValueError('Missing backbone atoms')
    placeholder = ''.join('P' if x == 'P' else 'A' for x in original_sequence)
    seq = receptor_sequence + placeholder
    x = torch.tensor(coords, dtype=torch.float32, device=device)[None]
    s = torch.tensor([[ALPHABET.index(a) for a in seq]], device=device)
    mask = torch.ones((1, nr + np_), device=device)
    chain_mask = torch.zeros_like(mask); chain_mask[:, nr:] = 1
    position_mask = torch.ones_like(mask)
    position_mask[:, nr:] = torch.tensor([a != 'P' for a in original_sequence], device=device)
    chain_encoding = torch.ones_like(s); chain_encoding[:, nr:] = 2
    residue_idx = torch.arange(nr + np_, device=device)[None]; residue_idx[:, nr:] += 100
    return dict(X=x, S_true=s, mask=mask, chain_mask=chain_mask,
                chain_M_pos=position_mask, chain_encoding_all=chain_encoding,
                residue_idx=residue_idx, bias_by_res=torch.zeros((1, nr + np_, 21), device=device))


def sample_sequence(model, inputs, receptor_sequence, original_sequence, seed, temperature):
    torch.manual_seed(seed)
    randn = torch.randn_like(inputs['mask'])
    with torch.no_grad():
        out = model.sample(**inputs, randn=randn, temperature=temperature,
                           omit_AAs_np=np.array([a in 'PX' for a in ALPHABET], dtype=np.float32),
                           bias_AAs_np=np.zeros(21, dtype=np.float32))
    full = ''.join(ALPHABET[i] for i in out['S'][0].tolist())
    nr = len(receptor_sequence)
    assert full[:nr] == receptor_sequence
    seq = full[nr:]
    assert [a == 'P' for a in seq] == [a == 'P' for a in original_sequence]
    return seq
