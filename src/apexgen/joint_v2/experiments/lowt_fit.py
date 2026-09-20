"""Matched objective isolation for the exploratory fixed-sequence dynamic-pair model."""
from contextlib import nullcontext

import torch

from apexgen.joint_v2.experiments.clean_proposal import proposal_losses, training_objective
from apexgen.joint_v2.experiments.clean_support import denoise_losses
from apexgen.joint_v2.experiments.path_stability import path_losses
from apexgen.shared.training.precision import network_autocast

ARMS = ("path_only", "local_path")


def combine(terms, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return terms["path"] if arm == "path_only" else terms["local"] + terms["path"]


def objective(model, cases, arm, precision, *, local_grad=False):
    """Identical two forwards; an inactive local objective never enters backward."""
    if arm not in ARMS:
        raise ValueError(arm)
    obs, target = cases["obs"], cases["target"]
    mask = obs.pocket.peptide_mask
    terms, predictions = {}, {}
    for name in ("local", "path"):
        case, trace = cases[name], []
        enabled = name == "path" or arm == "local_path" or local_grad
        with (nullcontext() if enabled else torch.no_grad()):
            with network_autocast(target.translation.device, precision):
                pred = model(case["state"], case["time"], obs, trace=trace)
            assert torch.equal(pred.sequence_logits, target.sequence_logits)
            assert torch.equal(pred.translation[~mask], target.translation[~mask])
            assert torch.equal(pred.rotation[~mask], target.rotation[~mask])
            if name == "path":
                losses = path_losses(pred, target, obs.pocket)
                terms[name] = losses["total"].mean()
            else:
                losses = denoise_losses(pred, target, mask, case["severity"])
                proposals = proposal_losses(trace, mask, case["severity"],
                                            model.decoder.structure_module.translation_scale)
                terms[name] = training_objective(losses, proposals, case["clean"], 1.)
        predictions[name] = pred
    return combine(terms, arm), terms, predictions
