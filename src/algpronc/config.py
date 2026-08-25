"""Nested dataclass config system.

Configs are built from YAML plus `--set a.b=c` CLI overrides. Values are coerced
to the target dataclass field type, so `--set cl.ridge_lambda=1e-3` yields a float.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Sequence

import yaml


@dataclass
class DataConfig:
    dataset: str = "cifar100"  
    n_tasks: int = 10
    classes_per_task: int = 10  
    task_class_splits: list[list[int]] = field(default_factory=list)
    class_order_seed: int = 0  
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 0
    data_dir: str = ""  


@dataclass
class ModelConfig:
    backbone: str = "vit_b16_in21k"  
    pretrained: bool = True
    feature_dim: int = 768  
    freeze_backbone: bool = True


@dataclass
class RandomProjectionConfig:
    """Optional RanPAC-style frozen non-linear expansion before the ETF head."""

    enabled: bool = False
    proj_dim: int = 10000  
    activation: str = "relu"  
    seed: int = 0


@dataclass
class ETFConfig:
    """Dynamic Simplex ETF target-frame construction."""

    seed: int = 0  


@dataclass
class FrameConfig:
    """Target-frame construction knobs shared by the compressed/equinorm heads.

    `kind` selects the `algpronc.geometry.frames` builder: 'etf' -> truncated_etf
    (falls back to simplex_etf when m >= K-1), 'welch' -> welch_frame,
    'random' -> random_frame, 'gs' -> gs_expand (the original proposal's
    literal, non-ETF construction used by `heads.etf_stale`).
    """

    kind: str = "etf"  
    frame_m: int | None = None  
    equinorm_mode: str = "column"  
    compressed_m: int = 32  


@dataclass
class ImbalanceConfig:
    """Long-tailed class-incremental stream (exp3_imbalance)."""

    enabled: bool = False
    imbalance_ratio: float = 100.0
    seed: int = 0


@dataclass
class CLConfig:
    ridge_lambda: float = 1e-2
    rp: RandomProjectionConfig = field(default_factory=RandomProjectionConfig)
    etf: ETFConfig = field(default_factory=ETFConfig)
    
    
    heads: list[str] = field(default_factory=lambda: ["onehot", "etf_exact", "etf_stale"])
    frame: FrameConfig = field(default_factory=FrameConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)


@dataclass
class EvalConfig:
    every_task: bool = True  
    max_eval_batches: int | None = None


@dataclass
class CheckpointConfig:
    """Task-level checkpoint/resume, required for Spot VMs.

    Vertex `--restart-on-preemption` restarts the *process*, not the run, so a
    20-task job preempted after task 15 would otherwise start over. P_t and Q_t
    fully determine W_t, so checkpointing them is exact and cheap.
    """

    enabled: bool = True
    resume: bool = True  
    dir: str = ""  


@dataclass
class ExperimentConfig:
    run_id: str = "debug"
    seed: int = 0
    device: str = "auto"  
    out_dir: str = ""  
    
    
    experiment: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cl: CLConfig = field(default_factory=CLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    exp0: dict[str, Any] = field(default_factory=dict)
    exp1: dict[str, Any] = field(default_factory=dict)
    exp2: dict[str, Any] = field(default_factory=dict)
    exp3: dict[str, Any] = field(default_factory=dict)
    exp4: dict[str, Any] = field(default_factory=dict)

    def resolved_data_dir(self) -> str:
        return self.data.data_dir or os.environ.get("ALGPRONC_DATA_DIR", "./data")

    def resolved_out_dir(self) -> str:
        return self.out_dir or os.environ.get("ALGPRONC_OUT_DIR", "./runs")


_OPTIONAL_PREFIX = ("int | None", "float | None", "str | None")


def _coerce(value: Any, type_str: str) -> Any:
    """Coerce a YAML/CLI scalar to the annotated field type."""
    t = type_str.replace("typing.", "").strip()
    if value is None:
        return None
    if t in _OPTIONAL_PREFIX or t.startswith("Optional["):
        if isinstance(value, str) and value.lower() in ("none", "null", ""):
            return None
        t = t.split("|")[0].strip() if "|" in t else t[len("Optional[") : -1]
    if t == "bool":
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "y", "on")
        return bool(value)
    if t == "int":
        return int(value)
    if t == "float":
        return float(value)
    if t == "str":
        return str(value)
    return value


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    field_map = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in field_map:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        f = field_map[key]
        if is_dataclass(f.type) or (isinstance(f.type, str) and f.type in _DATACLASSES):
            sub = _DATACLASSES[f.type] if isinstance(f.type, str) else f.type
            kwargs[key] = _from_dict(sub, value or {})
        else:
            type_str = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
            if type_str.startswith("dict"):
                kwargs[key] = dict(value or {})
            elif type_str.startswith("list"):
                kwargs[key] = list(value or [])
            else:
                kwargs[key] = _coerce(value, type_str)
    return cls(**kwargs)


_DATACLASSES = {
    "DataConfig": DataConfig,
    "ModelConfig": ModelConfig,
    "RandomProjectionConfig": RandomProjectionConfig,
    "ETFConfig": ETFConfig,
    "FrameConfig": FrameConfig,
    "ImbalanceConfig": ImbalanceConfig,
    "CLConfig": CLConfig,
    "EvalConfig": EvalConfig,
    "CheckpointConfig": CheckpointConfig,
}


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return _from_dict(ExperimentConfig, data)


def load_config(path: str | os.PathLike) -> ExperimentConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(raw)


def apply_override(cfg: ExperimentConfig, dotted_key: str, raw_value: str) -> None:
    """Apply one `a.b=c` override in place."""
    parts = dotted_key.split(".")
    obj: Any = cfg
    for i, part in enumerate(parts[:-1]):
        if isinstance(obj, dict):
            obj = obj.setdefault(part, {})
            continue
        if not hasattr(obj, part):
            raise KeyError(f"unknown config path '{dotted_key}' (at '{'.'.join(parts[: i + 1])}')")
        obj = getattr(obj, part)
    leaf = parts[-1]
    if isinstance(obj, dict):
        obj[leaf] = yaml.safe_load(raw_value)
        return
    if not hasattr(obj, leaf):
        raise KeyError(f"unknown config path '{dotted_key}'")
    f = {fld.name: fld for fld in fields(obj)}[leaf]
    type_str = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
    if type_str.startswith("dict") or type_str.startswith("list"):
        setattr(obj, leaf, yaml.safe_load(raw_value) or type_str.startswith("dict") and {} or [])
    else:
        setattr(obj, leaf, _coerce(yaml.safe_load(raw_value), type_str))


def apply_overrides(cfg: ExperimentConfig, overrides: Sequence[str]) -> ExperimentConfig:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not of the form key=value")
        key, _, value = item.partition("=")
        apply_override(cfg, key.strip(), value.strip())
    return cfg


def to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def parse_cli_config(argv: Sequence[str] | None = None) -> ExperimentConfig:
    """Shared entrypoint arg parsing: `--config path [--set a.b=c ...]`."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--set",
        nargs="*",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="dotted-path overrides, e.g. cl.ridge_lambda=1e-3 data.n_tasks=20",
    )
    args = parser.parse_args(argv)
    overrides = [kv for group in args.overrides for kv in group]
    cfg = load_config(args.config)
    return apply_overrides(cfg, overrides)
