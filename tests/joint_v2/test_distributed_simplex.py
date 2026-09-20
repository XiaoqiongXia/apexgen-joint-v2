"""Regression checks for unequal-rank weighting and nonduplicating assignments."""

import pytest
import torch

from apexgen.joint_v2.runtime.distributed import global_mean_loss, shard_indices


def test_uneven_rank_gradients_match_full_batch_mean():
    shards = [shard_indices(19, 3, rank) for rank in range(3)]
    assert [len(s) for s in shards] == [7, 6, 6]
    assert sorted(i for shard in shards for i in shard) == list(range(19))
    x = torch.arange(19, dtype=torch.float64)
    weight = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    expected, = torch.autograd.grad(((weight * x - 2) ** 2).mean(), weight)
    gradients = []
    for indices in shards:
        losses = (weight * x[indices] - 2) ** 2
        gradient, = torch.autograd.grad(global_mean_loss(losses, 19, 3), weight)
        gradients.append(gradient)
    torch.testing.assert_close(torch.stack(gradients).mean(), expected)


def test_reject_empty_rank():
    with pytest.raises(ValueError, match="distinct sample"):
        shard_indices(2, 3, 0)
