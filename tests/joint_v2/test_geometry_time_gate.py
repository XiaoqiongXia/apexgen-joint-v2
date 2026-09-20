"""Verify both refinement iterations and legacy checkpoint configuration semantics."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from apexgen.joint_v2.contracts.task_contract import TaskObservation
from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.runtime.portable import build_model, load_config, validate_config
from apexgen.joint_v2.sampling.simplex_runtime import sample_simplex_base


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("enabled", [False, True])
def test_geometry_updates_at_each_iteration_and_endpoint(enabled):
    torch.set_num_threads(2)
    config = load_config(ROOT / "configs/joint_v2/portable/simplex_tiny.yaml")
    config["model"]["architecture"]["structure_module"]["geometry_update_time_gate"] = enabled
    model = build_model(config, torch.device("cpu")).eval()
    module = model.network.decoder.structure_module
    with torch.no_grad():
        module.backbone_update.linear.weight.zero_()
        module.backbone_update.linear.bias.copy_(torch.tensor([0.1, -0.2, 0.15, 0.2, 0.1, -0.3]))
    dataset = JointV2Dataset(ROOT / "examples/boltzgen19/dataset", split="smoke")
    try:
        batch = collate_joint_v2_records([dataset[0]])
    finally:
        dataset.close()
    observation = TaskObservation("J", batch.condition)
    base = sample_simplex_base(batch.condition, generator=torch.Generator().manual_seed(17))
    with torch.no_grad():
        encoding = model.encode_complex(observation)
        for t in (0.0, 0.5, 1.0):
            trace = []
            prediction = model.network.decode(base.network_state(), torch.tensor([t]),
                                               observation, encoding, trace=trace)
            assert len(trace) == 2
            factor = 1 - t if enabled else 1
            for entry in trace:
                expected = (module.backbone_update.linear.bias * factor).expand_as(entry["update"])
                torch.testing.assert_close(entry["update"], expected)
            pocket = batch.condition.residue_mask & ~batch.condition.peptide_mask
            torch.testing.assert_close(prediction.translation[pocket], batch.condition.pocket_translation[pocket])
            torch.testing.assert_close(prediction.rotation[pocket], batch.condition.pocket_rotation[pocket])
            if t == 1:
                mask = batch.condition.peptide_mask
                if enabled:
                    torch.testing.assert_close(prediction.translation[mask], base.translation[mask])
                    torch.testing.assert_close(prediction.rotation[mask], base.rotation[mask])
                else:
                    assert not torch.allclose(prediction.translation[mask], base.translation[mask])
                    assert not torch.allclose(prediction.rotation[mask], base.rotation[mask])


def test_current_configs_disable_gate_and_legacy_configs_keep_it():
    for name in ("simplex_tiny.yaml", "simplex_tiny_ddp_overfit.yaml"):
        config = load_config(ROOT / "configs/joint_v2/portable" / name)
        assert config["model"]["architecture"]["structure_module"]["geometry_update_time_gate"] is False
        assert not build_model(config, torch.device("cpu")).network.decoder.structure_module.geometry_update_time_gate
    legacy = deepcopy(config)
    del legacy["model"]["architecture"]["structure_module"]["geometry_update_time_gate"]
    assert build_model(legacy, torch.device("cpu")).network.decoder.structure_module.geometry_update_time_gate
    config["model"]["architecture"]["structure_module"]["geometry_update_time_gate"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        validate_config(config)
