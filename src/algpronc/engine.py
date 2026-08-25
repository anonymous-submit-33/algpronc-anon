"""The task-stream loop.

`run_cil` fits every head named in `cfg.cl.heads` **in lockstep**: one pass
over the task stream, and for each task the *same* `(Phi, y)` feature tensor
is handed to `observe_classes` + `partial_fit` on every head, in the same
Python `for` loop, before any head's `partial_fit` has been called on the
next task. This is what makes the paper's equivalence claim airtight -- the
heads cannot differ by data order, batch composition, feature noise, or RNG
state, because there is exactly one stream and every head sees the identical
tensor at the identical point in it. Any experiment that wants to claim "head
A equals / beats head B" must get both from one `run_cil` call; comparing two
separate calls (even with matching seeds) reintroduces exactly the ordering
risk this design eliminates.

Two feature sources are supported:

- Real cached features via `algpronc.data.features.cached_features`, split
  into tasks via `algpronc.data.splits.class_incremental_splits`.
- `cfg.data.dataset == "synthetic"`: a small class-conditional isotropic
  Gaussian generator, entirely local to this module. It exists so
  `exp0_equivalence` (and the test suite) can run with **no dataset
  download and no dependency on `algpronc.data`/`algpronc.models`
  landing** -- every tensor is a pure function of the config's seeds, so
  two calls with the same config reproduce bit-identical features.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Optional

import torch

from algpronc import metrics
from algpronc.config import ExperimentConfig
from algpronc.utils.logging import RunLogger
from algpronc.utils.seed import child_seed, seed_everything


try:  
    from algpronc.heads import AnalyticHead  
except ImportError:  
    AnalyticHead = Any  


def synthetic_features(
    n_classes: int,
    samples_per_class: int,
    d: int,
    *,
    seed: int,
    class_sep: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-conditional isotropic Gaussian features: `x_c ~ N(mu_c, I)`, with
    `mu_c ~ N(0, class_sep^2 I)` fixed by `seed`. A pure function of its
    arguments (uses a private `torch.Generator`, never global RNG state), so
    it is exactly reproducible across repeated calls -- required for
    checkpoint/resume correctness and for calling it twice (once from an
    experiment for diagnostics, once inside `run_cil`) without divergence.

    Returns `(X, y)` shuffled, `X` float32 `(n_classes * samples_per_class, d)`,
    `y` int64.
    """
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n_classes, d, generator=g) * class_sep
    feats, labels = [], []
    for c in range(n_classes):
        feats.append(means[c] + torch.randn(samples_per_class, d, generator=g))
        labels.append(torch.full((samples_per_class,), c, dtype=torch.long))
    X = torch.cat(feats, dim=0).to(torch.float32)
    y = torch.cat(labels, dim=0)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm]


def resolve_feature_dim(cfg: ExperimentConfig) -> int:
    """The head input width *before* any random projection: `cfg.model.feature_dim`.

    Declarative (does not touch data/backbone), so experiments can size heads
    before `run_cil` has loaded anything.
    """
    return cfg.model.feature_dim


def resolve_head_dim(cfg: ExperimentConfig) -> int:
    """The head input width *after* the optional random projection -- what
    heads should actually be constructed with."""
    if cfg.cl.rp.enabled:
        return cfg.cl.rp.proj_dim
    return resolve_feature_dim(cfg)


def resolve_frame_m(cfg: ExperimentConfig, k_final: int) -> int:
    """`cfg.cl.frame.frame_m`, or `K_final - 1` when unset."""
    return cfg.cl.frame.frame_m if cfg.cl.frame.frame_m is not None else max(k_final - 1, 1)


def load_features(cfg: ExperimentConfig) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(n_classes, train_X, train_y, test_X, test_y)`.

    Dispatches on `cfg.data.dataset`: `"synthetic"` uses the local generator
    above (no `algpronc.data` dependency); anything else goes through
    `algpronc.data.features.cached_features` (lazy-imported, since that module
    pulls in `algpronc.models`/`torchvision`/`timm`).
    """
    if cfg.data.dataset == "synthetic":
        d = resolve_feature_dim(cfg)
        if cfg.data.task_class_splits:
            n_classes = max(c for task in cfg.data.task_class_splits for c in task) + 1
        else:
            n_classes = cfg.data.n_tasks * cfg.data.classes_per_task
        seed = cfg.data.class_order_seed if cfg.data.class_order_seed else cfg.seed
        
        
        train_X, train_y = synthetic_features(n_classes, 64, d, seed=child_seed(seed, "synthetic"))
        test_X, test_y = synthetic_features(n_classes, 32, d, seed=child_seed(seed, "synthetic"))
        return n_classes, train_X, train_y, test_X, test_y

    from algpronc.data.features import cached_features

    train_X, train_y = cached_features(cfg, train=True)
    test_X, test_y = cached_features(cfg, train=False)
    n_classes = int(train_y.max().item()) + 1
    return n_classes, train_X, train_y, test_X, test_y


def build_task_splits(cfg: ExperimentConfig, n_classes: int) -> list[list[int]]:
    """`list[list[int]]` of length `cfg.data.n_tasks`: which classes each task
    introduces. Explicit `cfg.data.task_class_splits` wins; otherwise a
    contiguous chunking of a class order shuffled by `cfg.data.class_order_seed`.

    For the synthetic path this is computed locally (no `algpronc.data`
    dependency); for real datasets it delegates to
    `algpronc.data.splits.class_incremental_splits` so behaviour matches every
    other experiment exactly.
    """
    if cfg.data.task_class_splits:
        return [list(task) for task in cfg.data.task_class_splits]

    if cfg.data.dataset == "synthetic":
        classes = list(range(n_classes))
        random.Random(cfg.data.class_order_seed).shuffle(classes)
        per_task = cfg.data.classes_per_task
        return [classes[i * per_task : (i + 1) * per_task] for i in range(cfg.data.n_tasks)]

    from algpronc.data.splits import class_incremental_splits

    return class_incremental_splits(
        n_classes,
        cfg.data.n_tasks,
        classes_per_task=cfg.data.classes_per_task,
        order_seed=cfg.data.class_order_seed,
    )


def checkpoint_path(cfg: ExperimentConfig) -> str:
    d = cfg.checkpoint.dir or os.path.join(cfg.resolved_out_dir(), cfg.run_id, "checkpoints")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "ckpt.pt")


def _save_checkpoint(cfg: ExperimentConfig, *, task_index: int, heads: dict, eval_history: dict, class_counts: dict) -> None:
    """Atomic save (write to a temp file, then `os.replace`): a Spot VM
    preemption mid-write must never leave a half-written checkpoint that a
    resumed process would load."""
    path = checkpoint_path(cfg)
    state = {
        "task_index": task_index,
        "heads": {name: h.state_dict() for name, h in heads.items()},
        "eval_history": eval_history,
        "class_counts": class_counts,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_checkpoint(cfg: ExperimentConfig) -> Optional[dict]:
    path = checkpoint_path(cfg)
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


@dataclass
class _EvalHistory:
    preds: list
    ys: list
    task_ids: list


def run_cil(cfg: ExperimentConfig, *, logger: RunLogger, heads: dict[str, "AnalyticHead"]) -> dict:
    """One pass over the task stream, fitting EVERY head in `heads` on the SAME
    features in lockstep. This is what makes the equivalence claim airtight:
    the heads cannot differ by data order, seed, or feature noise -- see the
    module docstring.

    Per task: slice the cached features for that task's classes, optionally
    resample them to a long-tailed budget (`cfg.cl.imbalance`), apply the
    optional random projection, cast to float64, then call `observe_classes`
    + `partial_fit` on every head with the identical tensor, then evaluate
    every head on every class seen so far. Logs one JSONL row per
    `(task, head)` via `logger.log_task`.

    Checkpointing (`cfg.checkpoint`): after every task, every head's
    `state_dict()`, the completed task index, the running eval history, and
    the running per-class training counts are saved atomically. On start, if
    `cfg.checkpoint.resume` and a checkpoint exists, all of that is restored
    and training resumes at the first *incomplete* task -- so a Spot
    preemption loses at most the in-flight task.

    Returns a summary dict keyed by head name, each holding
    `average_accuracy` / `average_incremental_accuracy` / `forgetting` /
    `head_tail_accuracy` / `task_accuracy_matrix` (as a nested list).
    """
    seed_everything(cfg.seed)

    n_classes, train_X, train_y, test_X, test_y = load_features(cfg)
    task_splits = build_task_splits(cfg, n_classes)
    n_tasks = len(task_splits)
    class_to_task = {c: t for t, task in enumerate(task_splits) for c in task}

    proj = None
    if cfg.cl.rp.enabled:
        from algpronc.models import build_projection

        proj, _ = build_projection(cfg.cl.rp, train_X.shape[1])

    class_counts: dict[int, int] = {}
    eval_history: dict[str, _EvalHistory] = {name: _EvalHistory([], [], []) for name in heads}
    start_task = 0

    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        ckpt = _load_checkpoint(cfg)
        if ckpt is not None and set(ckpt["heads"]) != set(heads):
            
            
            logger.info(
                f"run_cil: ignoring checkpoint at {checkpoint_path(cfg)} -- its heads "
                f"{sorted(ckpt['heads'])} do not match this run's heads {sorted(heads)} "
                "(likely a run_id collision with a different experiment); starting fresh."
            )
            ckpt = None
        if ckpt is not None:
            for name, head in heads.items():
                if name in ckpt["heads"]:
                    head.load_state_dict(ckpt["heads"][name])
            for name in heads:
                hist = ckpt["eval_history"].get(name)
                if hist is not None:
                    eval_history[name] = _EvalHistory(list(hist["preds"]), list(hist["ys"]), list(hist["task_ids"]))
            class_counts = dict(ckpt.get("class_counts", {}))
            start_task = ckpt["task_index"] + 1
            logger.info(f"run_cil: resumed from checkpoint at task_index={ckpt['task_index']}, starting at task {start_task}")

    for t in range(start_task, n_tasks):
        classes_t = task_splits[t]
        classes_t_tensor = torch.as_tensor(classes_t, dtype=train_y.dtype)
        mask = torch.isin(train_y, classes_t_tensor)
        Phi_t = train_X[mask]
        y_t = train_y[mask]

        if cfg.cl.imbalance.enabled:
            from algpronc.data.splits import long_tail_counts, subsample_indices

            counts_t = long_tail_counts(
                classes_t,
                imbalance_ratio=cfg.cl.imbalance.imbalance_ratio,
                seed=child_seed(cfg.cl.imbalance.seed, t),
            )
            keep = subsample_indices(y_t.numpy(), counts_t, seed=child_seed(cfg.cl.imbalance.seed, t))
            keep_t = torch.as_tensor(keep, dtype=torch.long)
            Phi_t = Phi_t[keep_t]
            y_t = y_t[keep_t]

        if proj is not None:
            Phi_t = proj(Phi_t)
        Phi_t = Phi_t.to(torch.float64)
        y_t = y_t.to(torch.long)

        for c in classes_t:
            class_counts[int(c)] = class_counts.get(int(c), 0) + int((y_t == c).sum().item())

        for name, head in heads.items():
            head.observe_classes(classes_t)
            head.partial_fit(Phi_t, y_t)

        seen_classes = [c for task in task_splits[: t + 1] for c in task]
        seen_tensor = torch.as_tensor(seen_classes, dtype=test_y.dtype)
        eval_mask = torch.isin(test_y, seen_tensor)
        eval_X = test_X[eval_mask]
        eval_y = test_y[eval_mask]
        if proj is not None:
            eval_X = proj(eval_X)
        eval_X = eval_X.to(torch.float64)
        eval_task_ids = torch.as_tensor([class_to_task[int(c)] for c in eval_y.tolist()], dtype=torch.long)

        
        idx_to_class = torch.as_tensor(seen_classes, dtype=torch.long)

        for name, head in heads.items():
            pred_local = head.predict(eval_X)
            pred = idx_to_class[pred_local.to(torch.long)]
            acc = metrics.accuracy(pred, eval_y)
            logger.log_task({"task": t, "head": name, "n_seen_classes": len(seen_classes), "accuracy": acc})
            hist = eval_history[name]
            hist.preds.append(pred.detach().cpu() if isinstance(pred, torch.Tensor) else pred)
            hist.ys.append(eval_y.detach().cpu())
            hist.task_ids.append(eval_task_ids)

        if cfg.checkpoint.enabled:
            _save_checkpoint(
                cfg,
                task_index=t,
                heads=heads,
                eval_history={
                    name: {"preds": h.preds, "ys": h.ys, "task_ids": h.task_ids} for name, h in eval_history.items()
                },
                class_counts=class_counts,
            )

    summary: dict[str, Any] = {}
    for name in heads:
        hist = eval_history[name]
        A = metrics.task_accuracy_matrix(hist.preds, hist.ys, hist.task_ids, n_tasks)
        entry: dict[str, Any] = {
            "average_accuracy": metrics.average_accuracy(A),
            "average_incremental_accuracy": metrics.average_incremental_accuracy(A),
            "forgetting": metrics.forgetting(A),
            "task_accuracy_matrix": A.tolist(),
        }
        if hist.preds:
            entry["head_tail_accuracy"] = metrics.head_tail_accuracy(hist.preds[-1], hist.ys[-1], class_counts)
        summary[name] = entry
    summary["class_counts"] = class_counts
    summary["n_tasks"] = n_tasks
    summary["task_class_splits"] = task_splits
    return summary
