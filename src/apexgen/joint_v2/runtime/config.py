"""Fail-closed configuration loader for sequence--structure Joint-v2."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256


SCHEMA_VERSION = "apexgen.training.joint_v2.sequence_structure_endpoint.v5"
DATA_SCHEMA_VERSION = "apexgen.data.joint_v2.v2"
UNIFIED_DATA_SCHEMA_VERSION = "apexgen.data.joint_v2.v3"


class JointV2ConfigError(ValueError):
    pass


def _keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = set(value) if isinstance(value, dict) else set()
        raise JointV2ConfigError(
            f"{path} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _equal(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        raise JointV2ConfigError(f"{path} must be {expected!r}, got {value!r}")


def _positive_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JointV2ConfigError(f"{path} must be a positive integer")


def _finite(value: Any, path: str, *, minimum: float = 0.0, strict: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise JointV2ConfigError(f"{path} must be finite")
    if value < minimum or (strict and value == minimum):
        relation = ">" if strict else ">="
        raise JointV2ConfigError(f"{path} must be {relation} {minimum}")


def validate_joint_v2_data_config(config: dict[str, Any]) -> None:
    names = {
        "schema_version",
        "pocket_root",
        "target_root",
        "raw_structure_root",
        "train_split",
        "validation_split",
    }
    unified = config.get("schema_version") == UNIFIED_DATA_SCHEMA_VERSION
    if unified:
        names = (names - {"pocket_root", "target_root"}) | {"dataset_root"}
    _keys(config, names, "data config")
    _equal(
        config["schema_version"],
        UNIFIED_DATA_SCHEMA_VERSION if unified else DATA_SCHEMA_VERSION,
        "data schema_version",
    )
    for name in names - {"schema_version"}:
        if not isinstance(config[name], str) or not config[name].strip():
            raise JointV2ConfigError(f"data config {name} must be a non-empty string")
    if config["train_split"] == config["validation_split"]:
        raise JointV2ConfigError("training and validation splits must differ")


def load_joint_v2_data_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise JointV2ConfigError("Joint-v2 data config must be a mapping")
    validate_joint_v2_data_config(config)
    resolved = dict(config)
    for name in ("dataset_root", "pocket_root", "target_root", "raw_structure_root"):
        if name not in resolved:
            continue
        root = Path(resolved[name]).expanduser()
        resolved[name] = str((source.parent / root).resolve() if not root.is_absolute() else root)
    return resolved


def joint_v2_data_roots(config: dict[str, Any]) -> tuple[Path, Path | None]:
    """Resolve storage inputs without inventing a sidecar for unified records."""
    validate_joint_v2_data_config(config)
    if config["schema_version"] == UNIFIED_DATA_SCHEMA_VERSION:
        return Path(config["dataset_root"]), None
    return Path(config["pocket_root"]), Path(config["target_root"])


def resolve_codesign_data_config(
    config: dict[str, Any], *, base_dir: str | Path = "."
) -> dict[str, Any]:
    """Normalize codesign data settings through the same v2/v3 data contract.

    Accept a data_config path/mapping, a manifest's data mapping, or explicit
    top-level roots. Relative paths belong to the declaring config directory.
    Legacy double-root runs remain explicit; never synthesize a target store.
    """
    base_dir = Path(base_dir).resolve()
    selectors = [key for key in ("data_config", "data") if key in config]
    roots = {key for key in ("dataset_root", "pocket_root", "target_root") if key in config}
    if len(selectors) > 1 or (selectors and roots):
        raise JointV2ConfigError("ambiguous codesign data configuration")
    value = config[selectors[0]] if selectors else config
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        return load_joint_v2_data_config(path if path.is_absolute() else base_dir / path)
    if not isinstance(value, dict):
        raise JointV2ConfigError("codesign data configuration must be a path or mapping")
    unified = "dataset_root" in value
    if unified and ("pocket_root" in value or "target_root" in value):
        raise JointV2ConfigError("unified data must not include legacy roots")
    names = ("dataset_root",) if unified else ("pocket_root", "target_root")
    if any(key not in value for key in names):
        raise JointV2ConfigError("codesign requires dataset_root or both legacy roots")
    expected_schema = UNIFIED_DATA_SCHEMA_VERSION if unified else DATA_SCHEMA_VERSION
    declared_schema = value.get("schema_version")
    if declared_schema is not None and declared_schema != expected_schema:
        raise JointV2ConfigError("codesign data schema does not match its storage roots")
    resolved = dict(
        schema_version=expected_schema,
        **{key: value[key] for key in names},
        raw_structure_root=value.get("raw_structure_root", str(base_dir)),
        train_split=value.get("train_split", "train"),
        validation_split=value.get("validation_split", "valid"),
    )
    validate_joint_v2_data_config(resolved)
    for name in (*names, "raw_structure_root"):
        path = Path(resolved[name]).expanduser()
        resolved[name] = str((base_dir / path).resolve() if not path.is_absolute() else path)
    return resolved


def validate_joint_v2_config(config: dict[str, Any]) -> None:
    _keys(
        config,
        {
            "schema_version",
            "contracts",
            "architecture",
            "flow",
            "loss",
            "precision",
            "offpath",
            "runtime",
            "training",
        },
        "config",
    )
    _equal(config["schema_version"], SCHEMA_VERSION, "schema_version")
    _equal(
        config["contracts"],
        {"sequence_structure_endpoint": JOINT_V2_CONTRACT_SHA256},
        "contracts",
    )

    architecture = _keys(
        config["architecture"],
        {
            "name",
            "single_dim",
            "pair_dim",
            "encoder_single_dim",
            "encoder_pair_dim",
            "encoder_feature_mode",
            "encoder_blocks",
            "encoder_attention_heads",
            "decoder_parameter_sharing",
            "decoder_blocks",
            "rigid_update_parameterization",
            "translation_scale_factor",
            "structure_module",
            "sequence_module",
            "time_conditioning",
            "angle_head",
        },
        "architecture",
    )
    _equal(
        architecture["name"],
        "joint_v2_unified_complex_sequence_structure_endpoint_refinement",
        "architecture.name",
    )
    for name in (
        "single_dim",
        "pair_dim",
        "encoder_single_dim",
        "encoder_pair_dim",
        "encoder_blocks",
        "encoder_attention_heads",
        "decoder_blocks",
    ):
        _positive_int(architecture[name], f"architecture.{name}")
    if architecture["encoder_feature_mode"] not in {
        "full",
        "no_metadata_torsions",
        "backbone_atoms",
        "compact_geometry",
    }:
        raise JointV2ConfigError(
            "architecture.encoder_feature_mode must be full, no_metadata_torsions, "
            "backbone_atoms, or compact_geometry"
        )
    _equal(architecture["decoder_parameter_sharing"], True, "decoder_parameter_sharing")
    if architecture["encoder_single_dim"] % architecture["encoder_attention_heads"] != 0:
        raise JointV2ConfigError(
            "architecture.encoder_single_dim must be divisible by encoder_attention_heads"
        )
    _equal(
        architecture["rigid_update_parameterization"],
        "openfold_compose_q_update_vec",
        "rigid_update_parameterization",
    )
    _equal(architecture["translation_scale_factor"], 10.0, "translation_scale_factor")

    structure = _keys(
        architecture["structure_module"],
        {
            "c_ipa",
            "no_heads_ipa",
            "no_qk_points",
            "no_v_points",
            "dropout_rate",
            "no_transition_layers",
            "inf",
            "epsilon",
            "stop_rotation_gradient_between_blocks",
        },
        "architecture.structure_module",
    )
    for name in ("c_ipa", "no_heads_ipa", "no_qk_points", "no_v_points", "no_transition_layers"):
        _positive_int(structure[name], f"structure_module.{name}")
    _finite(structure["dropout_rate"], "structure_module.dropout_rate")
    if structure["dropout_rate"] >= 1.0:
        raise JointV2ConfigError("structure_module.dropout_rate must be less than 1.0")
    _equal(structure["inf"], 1e5, "structure_module.inf")
    _equal(structure["epsilon"], 1e-8, "structure_module.epsilon")
    _equal(structure["stop_rotation_gradient_between_blocks"], True, "stop_rotation_gradient")

    sequence = _keys(
        architecture["sequence_module"],
        {
            "types",
            "coordinate",
            "normalization",
            "projection",
            "projection_initialization",
            "context_injection",
            "latent_update",
            "latent_update_initialization",
            "update",
            "update_initialization",
            "final_head_blocks",
            "final_residual_gate",
            "parameter_sharing",
            "recurrent_feedback_between_blocks",
            "stop_gradient_between_blocks",
        },
        "architecture.sequence_module",
    )
    _positive_int(sequence["final_head_blocks"], "sequence_module.final_head_blocks")
    observed_sequence = {
        name: value for name, value in sequence.items() if name != "final_head_blocks"
    }
    _equal(
        observed_sequence,
        {
            "types": 20,
            "coordinate": "centered_logits",
            "normalization": "layer_norm",
            "projection": "linear_to_single_dim",
            "projection_initialization": "openfold_default",
            "context_injection": "recurrent_latent_before_every_ipa_block",
            "latent_update": "shared_geometry_conditioned_residual_mlp",
            "latent_update_initialization": "zero_final",
            "update": "final_only_resnet_linear_from_single_dim",
            "update_initialization": "zero",
            "final_residual_gate": "one_minus_t",
            "parameter_sharing": True,
            "recurrent_feedback_between_blocks": True,
            "stop_gradient_between_blocks": False,
        },
        "architecture.sequence_module",
    )

    time = _keys(
        architecture["time_conditioning"],
        {
            "type",
            "embedding_dim",
            "frequencies",
            "frequency_schedule",
            "film_final_initialization",
            "inject_before_every_ipa_block",
            "gate_every_block_frame_and_final_sequence_update_by_one_minus_t",
        },
        "architecture.time_conditioning",
    )
    _positive_int(time["embedding_dim"], "time_conditioning.embedding_dim")
    observed_time = {name: value for name, value in time.items() if name != "embedding_dim"}
    expected_time = {
        "type": "fourier_mlp_peptide_film",
        "frequencies": 16,
        "frequency_schedule": "integer_1_to_16",
        "film_final_initialization": "zero",
        "inject_before_every_ipa_block": True,
        "gate_every_block_frame_and_final_sequence_update_by_one_minus_t": True,
    }
    _equal(observed_time, expected_time, "architecture.time_conditioning")
    angle = _keys(
        architecture["angle_head"],
        {
            "slots",
            "slot_names",
            "hidden_dim",
            "blocks",
            "epsilon",
            "call",
            "output_projection",
            "sidechain_projection_initialization",
        },
        "architecture.angle_head",
    )
    _equal(angle["slots"], 7, "angle_head.slots")
    _equal(
        angle["slot_names"],
        ["phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4"],
        "angle_head.slot_names",
    )
    _positive_int(angle["hidden_dim"], "angle_head.hidden_dim")
    _positive_int(angle["blocks"], "angle_head.blocks")
    _equal(angle["epsilon"], 1e-8, "angle_head.epsilon")
    _equal(angle["call"], "final_block_only", "angle_head.call")
    _equal(
        angle["output_projection"],
        "shared_trunk_separate_backbone_and_sidechain_linear",
        "angle_head.output_projection",
    )
    _equal(
        angle["sidechain_projection_initialization"],
        "appended_after_ec6a13a_modules",
        "angle_head.sidechain_projection_initialization",
    )

    flow = _keys(
        config["flow"],
        {
            "state",
            "base_distribution",
            "base_translation_sigma_angstrom",
            "base_coordinate_origin",
            "pocket_state",
            "sequence_epsilon",
            "sequence_base_sigma",
            "training_time_distribution",
            "training_time_min",
            "training_time_max",
            "training_time_max_boundary",
            "inference_solver",
            "inference_steps",
            "inference_query_time_max",
        },
        "flow",
    )
    expected_flow = {
        "state": "translation_rotation_sequence_centered_logits",
        "base_distribution": "peptide_gaussian_translation_haar_rotation_centered_gaussian_logits",
        "base_translation_sigma_angstrom": 5.0,
        "base_coordinate_origin": "dataset_pocket_centered_origin",
        "pocket_state": "native_condition_fixed",
        "sequence_epsilon": 0.05,
        "sequence_base_sigma": 1.0,
        "training_time_distribution": "continuous_uniform",
        "training_time_min": 0.0,
        "training_time_max_boundary": "exclusive",
        "inference_solver": "joint_endpoint_fractional_geodesic_centered_logit",
        "inference_steps": 20,
        "inference_query_time_max": 0.95,
    }
    observed_flow = {name: value for name, value in flow.items() if name != "training_time_max"}
    _equal(observed_flow, expected_flow, "flow")
    _finite(flow["training_time_max"], "flow.training_time_max", strict=True)
    if flow["training_time_max"] > 1.0:
        raise JointV2ConfigError("flow.training_time_max must not exceed 1.0")
    _equal(config["offpath"], {"enabled": False, "probability": 0.0, "loss_weight": 0.0}, "offpath")
    _equal(
        config["precision"],
        {
            "network": "bfloat16",
            "geometry": "float32",
            "loss": "float32",
            "state_storage": "float32",
        },
        "precision",
    )
    _equal(
        config["runtime"],
        {
            "return_intermediate_endpoints_during_training": True,
            "return_intermediate_endpoints_during_inference": False,
        },
        "runtime",
    )

    loss = _keys(
        config["loss"],
        {
            "reduction",
            "trajectory_fape",
            "final_translation",
            "final_rotation",
            "final_backbone_n_ca_c",
            "backbone_angle",
            "backbone_angle_norm",
            "sidechain_angle",
            "sidechain_angle_norm",
            "sequence",
            "chain_continuity",
            "clash",
        },
        "loss",
    )
    _equal(loss["reduction"], "sample_first_batch_second", "loss.reduction")
    expected_loss = {
        "trajectory_fape": {
            "weight": 1.0,
            "peptide_weight": 1.0,
            "cross_weight": 1.0,
            "point_type": "residue_frame_origin",
            "length_scale_angstrom": 10.0,
            "distance_epsilon_angstrom_squared": 1e-4,
            "training_clamp_distance_angstrom": None,
            "validation_peptide_clamp_angstrom": 10.0,
            "validation_cross_clamp_angstrom": 30.0,
        },
        "final_translation": {"weight": 1.0, "squared_error_scale_angstrom": 10.0},
        "final_rotation": {"weight": 1.0, "squared_geodesic_scale_radian": math.pi},
        "final_backbone_n_ca_c": {"weight": 1.0, "squared_error_scale_angstrom": 10.0},
        "backbone_angle": {"weight": 1.0, "maximum_squared_sincos_distance": 4.0},
        "backbone_angle_norm": {"weight": 0.02},
        "sidechain_angle": {
            "weight": 1.0,
            "maximum_squared_sincos_distance": 4.0,
            "pi_periodic_symmetry": "minimum_equivalent_target",
        },
        "sidechain_angle_norm": {"weight": 0.02},
        "sequence": {
            "epsilon": 0.05,
            "sigma_logit": 1.0,
            "logit_weight": 1.0,
            "soft_ce_weight": 1.0,
            "total_weight": 1.0,
            "block_weighting": "final_only",
            "hard_nll_role": "metric_only",
        },
        "chain_continuity": {"weight": 0.0, "role": "validation_only"},
        "clash": {"weight": 0.0, "role": "validation_only"},
    }
    for name, expected in expected_loss.items():
        _equal(loss[name], expected, f"loss.{name}")

    training = _keys(
        config["training"],
        {
            "learning_rate",
            "minimum_learning_rate",
            "weight_decay",
            "gradient_clip",
            "ema_decay",
            "maximum_optimizer_steps",
            "warmup_steps",
            "encoder_joint_warmup_steps",
            "batch_size",
            "bucket_window_batches",
            "num_workers",
            "checkpoint_every_steps",
            "checkpoint_keep_last",
            "validation_every_steps",
            "rollout_every_steps",
            "rollout_panel_samples",
            "rollout_bases_per_sample",
            "rollout_clash_distance_angstrom",
            "seed",
        },
        "training",
    )
    for name in ("learning_rate", "gradient_clip", "ema_decay"):
        _finite(training[name], f"training.{name}", strict=True)
    for name in ("minimum_learning_rate", "weight_decay"):
        _finite(training[name], f"training.{name}")
    for name in (
        "maximum_optimizer_steps",
        "batch_size",
        "bucket_window_batches",
        "checkpoint_every_steps",
        "checkpoint_keep_last",
        "validation_every_steps",
        "rollout_every_steps",
        "rollout_panel_samples",
        "rollout_bases_per_sample",
    ):
        _positive_int(training[name], f"training.{name}")
    for name in ("warmup_steps", "num_workers", "seed"):
        value = training[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise JointV2ConfigError(f"training.{name} must be a non-negative integer")
    encoder_joint_warmup_steps = training["encoder_joint_warmup_steps"]
    if encoder_joint_warmup_steps is not None and (
        isinstance(encoder_joint_warmup_steps, bool)
        or not isinstance(encoder_joint_warmup_steps, int)
        or encoder_joint_warmup_steps < 0
    ):
        raise JointV2ConfigError(
            "training.encoder_joint_warmup_steps must be null or a non-negative integer"
        )
    if training["warmup_steps"] >= training["maximum_optimizer_steps"]:
        raise JointV2ConfigError("warmup_steps must be below maximum_optimizer_steps")
    if (
        encoder_joint_warmup_steps is not None
        and encoder_joint_warmup_steps >= training["maximum_optimizer_steps"]
    ):
        raise JointV2ConfigError("encoder_joint_warmup_steps must be below maximum_optimizer_steps")
    if not 0.0 < training["ema_decay"] < 1.0:
        raise JointV2ConfigError("ema_decay must be in (0, 1)")
    if not 0.0 <= training["minimum_learning_rate"] <= training["learning_rate"]:
        raise JointV2ConfigError("minimum_learning_rate must not exceed learning_rate")
    _equal(training["rollout_panel_samples"], 10, "rollout_panel_samples")
    _equal(training["rollout_bases_per_sample"], 4, "rollout_bases_per_sample")
    _equal(training["rollout_clash_distance_angstrom"], 2.0, "rollout_clash_distance_angstrom")


def load_joint_v2_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise JointV2ConfigError("Joint-v2 config must be a mapping")
    validate_joint_v2_config(config)
    return config
