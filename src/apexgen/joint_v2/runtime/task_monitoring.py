"""Non-mutating gradient and actual optimizer-update measurements."""

import math

import torch


def module_group(name):
    if name.startswith("encoder.blocks."):
        parts = name.split(".")
        return ".".join(parts[:4])
    if name.startswith("encoder."):
        return ".".join(name.split(".")[:2])
    prefix = "decoder.structure_module."
    if name.startswith(prefix):
        return name[len(prefix) :].split(".")[0]
    raise ValueError(f"unmapped parameter {name}")


def parameter_manifest(model):
    return [
        {"name": n, "module": module_group(n), "numel": p.numel(), "active": p.requires_grad}
        for n, p in model.named_parameters()
    ]


def snapshot_before_clip(model):
    return {
        n: {
            "parameter": p.detach().clone(),
            "gradient": None if p.grad is None else p.grad.detach().float().clone(),
        }
        for n, p in model.named_parameters()
    }


def snapshot_after_clip(model, snapshot):
    for n, p in model.named_parameters():
        snapshot[n]["clipped_gradient"] = (
            None if p.grad is None else p.grad.detach().float().clone()
        )


def update_measurements(model, snapshot, optimizer):
    """Measure actual Adam delta against both gradients, including inactive parameters."""
    settings = {
        id(p): (group["lr"], group.get("weight_decay", 0.0))
        for group in optimizer.param_groups
        for p in group["params"]
    }
    groups = {}
    for name, p in model.named_parameters():
        group = groups.setdefault(module_group(name), [])
        before = snapshot[name]["parameter"].float()
        delta = p.detach().float() - before
        grad, clipped = snapshot[name]["gradient"], snapshot[name]["clipped_gradient"]
        lr, wd = settings.get(id(p), (0.0, 0.0))
        decay = -lr * wd * before if grad is not None else torch.zeros_like(before)
        zero = before.new_zeros(())
        group.append(
            torch.stack(
                [
                    before.new_tensor(p.numel()),
                    before.square().sum(),
                    delta.square().sum(),
                    zero if grad is None else grad.square().sum(),
                    zero if clipped is None else clipped.square().sum(),
                    zero if grad is None else -(grad * delta).sum(),
                    before.new_tensor(p.numel() if grad is None else 0),
                    zero if grad is None else (grad == 0).sum().float(),
                    decay.square().sum(),
                    before.new_tensor(p.numel() if p.requires_grad else 0),
                ]
            )
        )
    result = {}
    for group, values in groups.items():
        n, param2, delta2, grad2, clip2, dot, none, zeros, decay2, active = (
            torch.stack(values).sum(0).double().cpu().tolist()
        )
        cosine = None if delta2 == 0 or grad2 == 0 else dot / math.sqrt(delta2 * grad2)
        result[group] = {
            "parameter_count": int(n),
            "active_parameter_count": int(active),
            "parameter_rms": math.sqrt(param2 / n),
            "pre_clip_gradient_l2": math.sqrt(grad2),
            "pre_clip_gradient_rms": math.sqrt(grad2 / n),
            "post_clip_gradient_l2": math.sqrt(clip2),
            "post_clip_gradient_rms": math.sqrt(clip2 / n),
            "update_rms": math.sqrt(delta2 / n),
            "relative_update": None if param2 <= 1e-24 else math.sqrt(delta2 / param2),
            "update_alignment_to_negative_gradient": cosine,
            "grad_none_numel": int(none),
            "grad_zero_numel": int(zeros),
            "weight_decay_step_l2": math.sqrt(decay2),
        }
    return result


def objective_attribution(model, objectives):
    """Exact FP32 Gram matrices of the supplied, already weighted scalar objectives."""
    items = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names, parameters = zip(*items, strict=True)
    gradients = []
    for i, value in enumerate(objectives.values()):
        gradients.append(
            torch.autograd.grad(
                value,
                parameters,
                allow_unused=True,
                retain_graph=i + 1 < len(objectives),
            )
        )
    groups = {}
    for i, name in enumerate(names):
        groups.setdefault(module_group(name), []).append(i)
    result = {}
    device = parameters[0].device
    for group, indices in groups.items():
        count = sum(parameters[i].numel() for i in indices)
        gram = torch.zeros(len(gradients), len(gradients), device=device)
        none = [0] * len(gradients)
        with torch.autocast(device_type=device.type, enabled=False):
            for i in indices:
                values = []
                for j, gradient in enumerate(gradients):
                    v = gradient[i]
                    none[j] += parameters[i].numel() if v is None else 0
                    values.append(
                        torch.zeros_like(parameters[i], dtype=torch.float32).flatten()
                        if v is None
                        else v.float().flatten()
                    )
                matrix = torch.stack(values)
                gram += matrix @ matrix.T
        gg = gram.double().cpu()
        norms = gg.diag().clamp_min(0).sqrt()
        total = float(gg.sum().clamp_min(0).sqrt())
        labels = list(objectives)
        result[group] = {
            "parameter_count": count,
            "objectives": labels,
            "gram": gg.tolist(),
            "gradient_l2": dict(zip(labels, norms.tolist(), strict=True)),
            "gradient_rms": {
                label: float(norms[i]) / math.sqrt(count) for i, label in enumerate(labels)
            },
            "grad_none_numel": dict(zip(labels, none, strict=True)),
            "total_gradient_l2": total,
            "cancellation_ratio": None if float(norms.sum()) == 0 else total / float(norms.sum()),
            "cosine": [
                [
                    None
                    if float(norms[i] * norms[j]) == 0
                    else max(-1.0, min(1.0, float(gg[i, j] / (norms[i] * norms[j]))))
                    for j in range(len(labels))
                ]
                for i in range(len(labels))
            ],
        }
    return result


def trace_measurements(trace):
    result = []
    for entry in trace:
        row = {"iteration": entry["iteration"]}
        for name, value in entry.items():
            if not isinstance(value, torch.Tensor):
                continue
            row[name] = {
                "rms": float(value.detach().float().square().mean().sqrt()),
                "gradient_rms": None
                if value.grad is None
                else float(value.grad.detach().float().square().mean().sqrt()),
            }
        result.append(row)
    return result


def capture_objective_gradients(model, objectives):
    """Capture weighted task gradients on a separate forward graph, without touching .grad."""
    items = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names, parameters = zip(*items, strict=True)
    result = {}
    for i, (label, value) in enumerate(objectives.items()):
        gradients = torch.autograd.grad(
            value, parameters, allow_unused=True, retain_graph=i + 1 < len(objectives)
        )
        result[label] = {
            n: g.detach().float() for n, g in zip(names, gradients, strict=True) if g is not None
        }
    return result


def objective_update_dot(model, snapshot, captured):
    """First-order per-objective loss change along the actual optimizer delta."""
    result = {}
    params = dict(model.named_parameters())
    for label, gradients in captured.items():
        sums = {}
        for name, g in gradients.items():
            delta = params[name].detach().float() - snapshot[name]["parameter"].float()
            group = module_group(name)
            sums[group] = sums.get(group, 0.0) + (g * delta).sum()
        result[label] = {group: float(value) for group, value in sums.items()}
    return result
