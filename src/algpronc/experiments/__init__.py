"""Shared plumbing for `algpronc.experiments.*` entrypoints.

Every module under this package is runnable as

    python -m algpronc.experiments.<name> --config path.yaml [--set a.b=c ...]

and follows the same four-step opening: `parse_cli_config`, seed, resolve the
device, open a `RunLogger`. `setup_experiment` below does exactly that so each
`exp*.py` file only has to state what makes it different.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import torch

from algpronc.config import ExperimentConfig, parse_cli_config, to_dict
from algpronc.utils.device import resolve_device
from algpronc.utils.logging import RunLogger
from algpronc.utils.seed import child_seed, seed_everything

if TYPE_CHECKING:  
    from algpronc.heads import AnalyticHead


_WRAPPER_HEADS = ("equinorm", "equiangular", "etf_project")


def setup_experiment(
    argv: list[str] | None = None, *, run_id_suffix: str | None = None
) -> tuple[ExperimentConfig, RunLogger, torch.device]:
    """`parse_cli_config` + seed + device + a `RunLogger` with the config dumped.

    Common to every experiment entrypoint; each `exp*.main()` calls this first.

    `run_id_suffix`, if given, is appended to `cfg.run_id` before the
    `RunLogger` (and therefore the checkpoint/summary/task-log paths, which
    are all keyed off `run_id`) is created. Use this when an experiment
    reuses another experiment's config file -- e.g. `exp2_compressed` sweeps
    over `configs/cifar100_t10.yaml`, the same file `exp1_main` uses -- so it
    doesn't write into the same run directory under a different head set:
    `run_cil`'s checkpoint resume trusts a loaded checkpoint's `task_index`
    once its head names match, so two experiments silently sharing one
    checkpoint file (matching heads or not) is a correctness hazard, not just
    a tidiness one.
    """
    cfg = parse_cli_config(argv)
    if run_id_suffix:
        cfg.run_id = f"{cfg.run_id}{run_id_suffix}"
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    logger = RunLogger(cfg)
    logger.dump_config(to_dict(cfg))
    return cfg, logger, device


def build_heads(
    cfg: ExperimentConfig,
    *,
    d: int,
    k_final: int | None = None,
    seed: int | None = None,
) -> "dict[str, AnalyticHead]":
    """Instantiate every head named in `cfg.cl.heads` via `algpronc.heads.build_head`,
    ready to be handed to `engine.run_cil` as the lockstep `heads` dict.

    `algpronc.heads.build_head(cl_cfg, d)` takes a *flat*, duck-typed
    `CLConfig`-shaped object (`.ridge_lambda`, `.etf.seed`, `.head`, `.proj_m`,
    `.frame`, `.equinorm_inner`, `.equinorm_mode` -- see its module docstring),
    one head name per call, and does the `equinorm` wrapping *internally*.
    This project's `CLConfig` (`config.py`) instead nests those knobs under
    `cl.frame.*` and lists several heads at once in `cl.heads`, so this
    function is the adapter between the two: for every entry in `cl.heads` it
    builds a small `types.SimpleNamespace` view exposing exactly the flat
    attributes `build_head` reads, then calls it.

    Head-name grammar (a convention of this glue code, not of `algpronc.heads`
    itself): a bare name ('onehot', 'etf_exact', 'etf_stale', 'compressed')
    builds that head; 'equinorm', 'equiangular' or 'etf_project', optionally
    suffixed ':<inner>', builds an `EquinormHead` wrapping `<inner>` (default
    'etf_stale'). The three wrapper names differ only in which half of the
    Simplex-ETF geometry they impose on the fitted `W` -- equal norms, equal
    angles, or both -- and because the name fixes the mode, all three can be
    listed in `cl.heads` and compared in one lockstep run. Plain 'equinorm'
    takes its mode from `cl.frame.equinorm_mode` instead.

    `k_final`: the final cumulative class count, needed to resolve
    `cl.frame.frame_m is None -> K_final - 1` (`build_head`'s own default,
    `proj_m=d`, is a *feature-dim* fallback, not a *class-count* one, so it is
    not what this codebase wants when `frame_m` is left unset). If not given,
    it is computed by loading features once via `engine.load_features` /
    `engine.build_task_splits` -- a cache hit for real datasets, a cheap
    deterministic regeneration for the synthetic path.
    """
    from algpronc.heads import build_head

    seed = cfg.seed if seed is None else seed

    if k_final is None and cfg.cl.frame.frame_m is None:
        from algpronc import engine

        n_classes, *_ = engine.load_features(cfg)
        task_splits = engine.build_task_splits(cfg, n_classes)
        k_final = sum(len(t) for t in task_splits)
    uncompressed_m = cfg.cl.frame.frame_m if cfg.cl.frame.frame_m is not None else max((k_final or d) - 1, 1)

    heads: "dict[str, AnalyticHead]" = {}
    for name in cfg.cl.heads:
        base_name, _, inner_name = name.partition(":")
        head_seed = child_seed(seed, name)
        proj_m = cfg.cl.frame.compressed_m if base_name == "compressed" else uncompressed_m
        view = types.SimpleNamespace(
            ridge_lambda=cfg.cl.ridge_lambda,
            etf=types.SimpleNamespace(seed=head_seed),
            head=base_name,
            proj_m=proj_m,
            frame=cfg.cl.frame.kind,
            equinorm_inner=(inner_name or "etf_stale") if base_name in _WRAPPER_HEADS else "onehot",
            equinorm_mode=cfg.cl.frame.equinorm_mode,
        )
        heads[name] = build_head(view, d)
    return heads


__all__ = ["setup_experiment", "build_heads"]
