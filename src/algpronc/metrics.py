"""Class-incremental metrics.

The **task accuracy matrix** `A` is the primitive: `A[i, j]` is the accuracy on
task `j`'s classes, evaluated using the model state right after training task
`i` (so `A[i, j]` is undefined / NaN for `j > i`, since task `j` has not been
introduced yet). Every scalar summary (`average_accuracy`,
`average_incremental_accuracy`, `forgetting`) is a deterministic function of
`A` alone -- compute `A` once per head, then derive everything else from it,
rather than re-walking predictions per metric.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

try:  
    import torch
except ImportError:  
    torch = None  


def _to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def accuracy(pred: Any, y: Any) -> float:
    """Fraction of `pred == y`. NaN on an empty input (no samples to score)."""
    pred = _to_numpy(pred)
    y = _to_numpy(y)
    if y.size == 0:
        return float("nan")
    return float((pred == y).mean())


def per_class_accuracy(pred: Any, y: Any, n_classes: int) -> np.ndarray:
    """`(n_classes,)` per-class accuracy; NaN for classes absent from `y`."""
    pred = _to_numpy(pred)
    y = _to_numpy(y)
    out = np.full(n_classes, np.nan, dtype=np.float64)
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            out[c] = float((pred[mask] == y[mask]).mean())
    return out


def head_tail_accuracy(
    pred: Any,
    y: Any,
    class_counts: Mapping[int, int] | Sequence[int] | np.ndarray,
    frac: float = 0.5,
) -> dict[str, float]:
    """Split classes into 'head' (most frequent in training) and 'tail' (least
    frequent) and report accuracy within each split. Needed by `exp3_imbalance`
    to check whether `equinorm` recovers tail-class accuracy under long-tailed
    streams.

    Args:
        pred, y: predictions / ground-truth labels over the eval set.
        class_counts: training-sample counts, either `{class_id: n}` (as
            returned by `data.splits.long_tail_counts`) or an array indexed by
            class id.
        frac: fraction of classes (by count, descending) assigned to 'head';
            the remainder is 'tail'. `frac=0.5` is an even split.

    Returns:
        `{'head': acc, 'tail': acc}`, NaN for a split with no eval samples.
    """
    pred = _to_numpy(pred)
    y = _to_numpy(y)
    if isinstance(class_counts, Mapping):
        classes = np.array(sorted(class_counts.keys()))
        counts = np.array([class_counts[int(c)] for c in classes], dtype=np.float64)
    else:
        counts = _to_numpy(class_counts).astype(np.float64)
        classes = np.arange(len(counts))

    order = np.argsort(-counts, kind="stable")  
    n_head = max(1, int(round(len(classes) * frac)))
    n_head = min(n_head, len(classes) - 1) if len(classes) > 1 else len(classes)
    head_classes = classes[order[:n_head]]
    tail_classes = classes[order[n_head:]]

    head_mask = np.isin(y, head_classes)
    tail_mask = np.isin(y, tail_classes)
    return {
        "head": accuracy(pred[head_mask], y[head_mask]) if head_mask.any() else float("nan"),
        "tail": accuracy(pred[tail_mask], y[tail_mask]) if tail_mask.any() else float("nan"),
    }


def task_accuracy_matrix(
    preds: Sequence[Any],
    ys: Sequence[Any],
    task_ids: Sequence[Any],
    n_tasks: int,
) -> np.ndarray:
    """Build the `(n_tasks, n_tasks)` task accuracy matrix from a run's eval history.

    `preds[i]`, `ys[i]`, `task_ids[i]` are the predictions / labels / per-sample
    task index for the evaluation performed right after training task `i` (over
    every class introduced so far, i.e. tasks `0..i`); `task_ids[i][k]` is the
    task index that `ys[i][k]`'s class belongs to. `len(preds)` may be less
    than `n_tasks` for a partially completed (e.g. mid-resume) run -- rows
    beyond `len(preds) - 1` stay NaN.

    Returns:
        `A` with `A[i, j]` = accuracy restricted to task `j`'s classes after
        training through task `i`; NaN where `j > i` or task `i` was never run.
    """
    A = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
    for i in range(min(len(preds), n_tasks)):
        p = _to_numpy(preds[i])
        yy = _to_numpy(ys[i])
        tid = _to_numpy(task_ids[i])
        for j in range(i + 1):
            mask = tid == j
            if mask.any():
                A[i, j] = accuracy(p[mask], yy[mask])
    return A


def average_accuracy(A: np.ndarray) -> float:
    """Mean of the last completed row -- accuracy on every seen task, at the
    end of the stream."""
    finished = [i for i in range(A.shape[0]) if not np.all(np.isnan(A[i, : i + 1]))]
    if not finished:
        return float("nan")
    last = A[finished[-1], : finished[-1] + 1]
    seen = last[~np.isnan(last)]
    return float(seen.mean()) if seen.size else float("nan")


def average_incremental_accuracy(A: np.ndarray) -> float:
    """Mean, over every completed row `i`, of that row's mean accuracy over
    tasks `0..i` (the standard "average incremental accuracy" curve, collapsed
    to a scalar)."""
    row_means = []
    for i in range(A.shape[0]):
        row = A[i, : i + 1]
        row = row[~np.isnan(row)]
        if row.size:
            row_means.append(row.mean())
    return float(np.mean(row_means)) if row_means else float("nan")


def forgetting(A: np.ndarray) -> float:
    """`mean_j max_i A[i, j] - A[T, j]`, `T` = index of the last completed row.

    Positive values mean earlier tasks were, on average, better-classified at
    some point than they are by the end of the stream.
    """
    finished = [i for i in range(A.shape[0]) if not np.all(np.isnan(A[i, : i + 1]))]
    if not finished:
        return float("nan")
    T = finished[-1]
    vals = []
    for j in range(T + 1):
        col = A[: T + 1, j]
        seen = col[~np.isnan(col)]
        if seen.size == 0:
            continue
        vals.append(float(np.nanmax(col) - A[T, j]))
    return float(np.mean(vals)) if vals else 0.0
