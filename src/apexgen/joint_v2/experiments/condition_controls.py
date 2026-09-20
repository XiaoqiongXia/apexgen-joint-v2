"""Matched G_s networks with native, constant, position or shuffled observations."""

import hashlib
import math

import torch

from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.contracts.state import sequence_endpoint_logits
from apexgen.joint_v2.contracts.task_contract import TaskObservation
from apexgen.joint_v2.model.task_factorization import observation_from_batch

CONDITION_MODES = ("native", "constant", "position", "shuffled", "normalized_position")
CONDITION_CONTRACT = dict(
    schema="apexgen.joint_v2.condition_controls.v1",
    network="G_s identical active modules; geometry objectives only",
    constant="aatype 0 code at every peptide position",
    position="global deterministic Gaussian 20D code per integer chain rank; centered and native norm matched; no sample/length/native inputs",
    shuffled="native multiset permuted per training step; fixed through refinement/rollout; local RNG",
    evaluation="fixed per-base condition seeds across steps; independent audit bank",
)
CONDITION_CONTRACT_SHA256 = canonical_sha256(CONDITION_CONTRACT)

# Keep the historical four-mode contract unchanged for existing checkpoints.
POSITION_ROUTE_CONTRACT = dict(
    schema="apexgen.joint_v2.position_route.v1",
    normalized_position="u=rank/(length-1); lift [u,1-u,0,...,0]; center and match native norm",
    position=CONDITION_CONTRACT["position"],
    encoder="reuse sequence norm/projection, add to peptide single before encoder blocks; initial decoder latent zero",
    decoder="historical sequence norm/projection into initial recurrent latent",
    shared="original encoder positions, full pocket, recurrent latent updates and all parameter tensors retained",
    restriction="G_s, equal encoder/decoder single widths; deterministic position conditions only",
)
POSITION_ROUTE_CONTRACT_SHA256 = canonical_sha256(POSITION_ROUTE_CONTRACT)


def normalized_position_code(mask):
    rank = (mask.long().cumsum(-1) - 1).clamp_min(0).float()
    u = rank / (mask.sum(-1, keepdim=True) - 1).clamp_min(1)
    code = torch.zeros(*mask.shape, 20, device=mask.device)
    code[..., 0], code[..., 1] = u, 1 - u
    code = code - code.mean(-1, keepdim=True)
    code = code * (math.log(381.0) * math.sqrt(19 / 20) / code.norm(dim=-1, keepdim=True))
    return torch.where(mask[..., None], code, 0.0)


def seed_for(seed, key):
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") % (
        2**63 - 1
    )


def controlled_observation(batch, task, protocol, *, bank, index):
    mode = protocol.get("condition_mode", "native")
    if mode not in CONDITION_MODES:
        raise ValueError(mode)
    if mode == "native":
        return observation_from_batch(batch, task)
    if task != "G_s":
        raise ValueError("condition controls require G_s with identical active modules")
    p = batch.condition.peptide_mask
    if mode == "normalized_position":
        z = normalized_position_code(p)
    elif mode == "constant":
        z = sequence_endpoint_logits(torch.zeros_like(p, dtype=torch.long), p)
    elif mode == "position":
        z = torch.zeros(*p.shape, 20, device=p.device)
        # Hash only the global study seed and integer rank, never sample identity.
        for i in range(int(p.sum(-1).max())):
            g = torch.Generator().manual_seed(seed_for(protocol["seed"], f"position:{i}"))
            code = torch.randn(20, generator=g)
            code -= code.mean()
            code *= math.log(381.0) * math.sqrt(19 / 20) / code.norm()
            for b in range(len(p)):
                idx = p[b].nonzero().flatten()
                if i < len(idx):
                    z[b, idx[i]] = code.to(p.device)
    else:
        native = batch.targets.endpoint_state(batch.condition).sequence_logits
        z = torch.zeros_like(native)
        for b, sid in enumerate(batch.sample_ids):
            idx = p[b].nonzero().flatten()
            g = torch.Generator().manual_seed(
                seed_for(protocol["seed"], f"{bank}:{index}:{sid}:shuffle")
            )
            perm = torch.randperm(len(idx), generator=g).to(p.device)
            z[b, idx] = native[b, idx[perm]]
    return TaskObservation("G_s", batch.condition, sequence_logits=z)
