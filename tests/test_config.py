"""Tests for algpronc.config: the strict nested-dataclass system every
experiment / config YAML depends on."""

from __future__ import annotations

import dataclasses

import pytest

from algpronc.config import (
    CLConfig,
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    FrameConfig,
    ImbalanceConfig,
    apply_override,
    apply_overrides,
    config_from_dict,
    to_dict,
)


def test_defaults_construct_and_keep_existing_fields():
    cfg = ExperimentConfig()
    
    
    assert cfg.run_id == "debug"
    assert cfg.cl.ridge_lambda == 1e-2
    assert cfg.cl.rp.enabled is False
    assert cfg.cl.etf.seed == 0
    assert cfg.checkpoint.enabled is True
    assert cfg.checkpoint.resume is True
    assert cfg.data.n_tasks == 10
    assert cfg.data.classes_per_task == 10


def test_new_fields_have_sane_defaults():
    cfg = ExperimentConfig()
    assert cfg.cl.heads == ["onehot", "etf_exact", "etf_stale"]
    assert isinstance(cfg.cl.frame, FrameConfig)
    assert cfg.cl.frame.kind == "etf"
    assert cfg.cl.frame.frame_m is None
    assert cfg.cl.frame.equinorm_mode == "column"
    assert cfg.cl.frame.compressed_m > 0
    assert isinstance(cfg.cl.imbalance, ImbalanceConfig)
    assert cfg.cl.imbalance.enabled is False
    assert cfg.cl.imbalance.imbalance_ratio == 100.0
    assert cfg.experiment == ""


def test_from_dict_round_trip_through_to_dict():
    cfg = ExperimentConfig()
    d = to_dict(cfg)
    cfg2 = config_from_dict(d)
    assert cfg2 == cfg


def test_from_dict_builds_new_dataclasses_from_yaml_shaped_dict():
    raw = {
        "run_id": "r1",
        "experiment": "exp1_main",
        "cl": {
            "ridge_lambda": 5e-3,
            "heads": ["onehot", "equinorm:etf_stale"],
            "frame": {"kind": "welch", "frame_m": 16, "equinorm_mode": "procrustes", "compressed_m": 8},
            "imbalance": {"enabled": True, "imbalance_ratio": 50.0, "seed": 7},
        },
    }
    cfg = config_from_dict(raw)
    assert cfg.run_id == "r1"
    assert cfg.experiment == "exp1_main"
    assert cfg.cl.ridge_lambda == 5e-3
    assert cfg.cl.heads == ["onehot", "equinorm:etf_stale"]
    assert cfg.cl.frame.kind == "welch"
    assert cfg.cl.frame.frame_m == 16
    assert cfg.cl.frame.equinorm_mode == "procrustes"
    assert cfg.cl.frame.compressed_m == 8
    assert cfg.cl.imbalance.enabled is True
    assert cfg.cl.imbalance.imbalance_ratio == 50.0
    assert cfg.cl.imbalance.seed == 7


def test_from_dict_still_rejects_unknown_keys():
    with pytest.raises(KeyError):
        config_from_dict({"cl": {"not_a_real_field": 1}})
    with pytest.raises(KeyError):
        config_from_dict({"totally_bogus_top_level_key": 1})


def test_frame_config_registered_in_dataclasses_map_for_nested_overrides():
    
    
    cfg = config_from_dict({"cl": {"frame": {"kind": "random"}}})
    assert isinstance(cfg.cl.frame, FrameConfig)
    assert cfg.cl.frame.kind == "random"
    
    assert cfg.cl.frame.frame_m is None
    assert cfg.cl.frame.compressed_m == FrameConfig().compressed_m


def test_imbalance_config_registered_in_dataclasses_map():
    cfg = config_from_dict({"cl": {"imbalance": {"enabled": True}}})
    assert isinstance(cfg.cl.imbalance, ImbalanceConfig)
    assert cfg.cl.imbalance.enabled is True
    assert cfg.cl.imbalance.imbalance_ratio == ImbalanceConfig().imbalance_ratio


def test_apply_override_dotted_path_for_new_fields():
    cfg = ExperimentConfig()
    apply_override(cfg, "cl.frame.frame_m", "8")
    assert cfg.cl.frame.frame_m == 8
    apply_override(cfg, "cl.frame.kind", "gs")
    assert cfg.cl.frame.kind == "gs"
    apply_override(cfg, "cl.imbalance.enabled", "true")
    assert cfg.cl.imbalance.enabled is True
    apply_override(cfg, "cl.imbalance.imbalance_ratio", "10.0")
    assert cfg.cl.imbalance.imbalance_ratio == 10.0


def test_apply_override_list_field_heads():
    cfg = ExperimentConfig()
    apply_overrides(cfg, ['cl.heads=["onehot", "etf_exact"]'])
    assert cfg.cl.heads == ["onehot", "etf_exact"]


def test_apply_overrides_experiment_field():
    cfg = ExperimentConfig()
    apply_overrides(cfg, ["experiment=exp2_compressed"])
    assert cfg.experiment == "exp2_compressed"


def test_exp_dict_fields_exist_and_are_independent_dicts():
    cfg = ExperimentConfig()
    for name in ("exp0", "exp1", "exp2", "exp3", "exp4"):
        assert hasattr(cfg, name)
        assert getattr(cfg, name) == {}
    cfg.exp2["m_values"] = [4, 8]
    assert cfg.exp0 == {}  


def test_all_dataclass_fields_are_dataclasses_or_scalars_or_containers():
    
    
    for f in dataclasses.fields(CLConfig):
        assert f.type in (
            "float",
            "list[str]",
        ) or dataclasses.is_dataclass(f.type) or f.type in (
            "RandomProjectionConfig",
            "ETFConfig",
            "FrameConfig",
            "ImbalanceConfig",
        )
