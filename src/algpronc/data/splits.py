"""Class-incremental task splits and long-tail sub-sampling.

Two independent concerns live here:

- `class_incremental_splits` decides *which classes* belong to *which task*.
- `long_tail_counts` / `subsample_indices` decide *how many samples per class*
  survive, for the long-tailed-stream experiments (`exp3_imbalance`).

Both are pure functions of their seed argument so that a run is exactly
reproducible from its config alone.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _validate_partition(splits: list[list[int]], n_classes: int) -> None:
    """Every class in range(n_classes) must appear in exactly one task, exactly once."""
    seen: list[int] = [c for task in splits for c in task]
    if len(seen) != n_classes:
        raise ValueError(
            f"class_incremental_splits: expected {n_classes} classes total across all "
            f"tasks, got {len(seen)} (splits={splits})"
        )
    if sorted(seen) != list(range(n_classes)):
        dupes = sorted({c for c in seen if seen.count(c) > 1})
        missing = sorted(set(range(n_classes)) - set(seen))
        raise ValueError(
            "class_incremental_splits: every class in range(n_classes) must appear "
            f"exactly once. duplicates={dupes} missing={missing}"
        )


def class_incremental_splits(
    n_classes: int,
    n_tasks: int,
    *,
    classes_per_task: int | Sequence[int] | None = None,
    order_seed: int = 0,
    explicit: list[list[int]] | None = None,
) -> list[list[int]]:
    """Partition `range(n_classes)` into `n_tasks` contiguous chunks of a shuffled order.

    Args:
        n_classes: total number of classes in the dataset.
        n_tasks: number of tasks in the stream (ignored if `explicit` is given).
        classes_per_task: either
            - None: split n_classes evenly into n_tasks chunks (must divide evenly);
            - an int: every task gets exactly this many classes (`n_tasks *
              classes_per_task` must equal `n_classes`);
            - a sequence of `n_tasks` ints summing to `n_classes`: uneven chunk
              sizes over the shuffled order, e.g. `[50, 10, 10, 10, 10, 10]` for
              the common "B50-5step" base-then-incremental protocol.
        order_seed: seed for the class shuffle. Note: seed `1993` is the
            iCaRL / PODNet convention for the CIFAR-100 class order and is
            supported here like any other seed (no special-casing needed --
            `np.random.RandomState(1993).permutation` reproduces it).
        explicit: full override. A ready-made `list[list[int]]` assignment of
            class ids to tasks (as in `DataConfig.task_class_splits`). When
            given, `n_tasks`/`classes_per_task`/`order_seed` are ignored
            entirely; the only thing done is validating full coverage.

    Returns:
        `list[list[int]]` of length `n_tasks` (or `len(explicit)`), each inner
        list holding the (global) class ids assigned to that task.
    """
    if explicit:
        splits = [list(task) for task in explicit]
        _validate_partition(splits, n_classes)
        return splits

    if n_tasks <= 0:
        raise ValueError(f"n_tasks must be positive, got {n_tasks}")

    rng = np.random.RandomState(order_seed)
    order = rng.permutation(n_classes).tolist()

    if classes_per_task is None:
        if n_classes % n_tasks != 0:
            raise ValueError(
                f"n_classes={n_classes} does not divide evenly into n_tasks={n_tasks}; "
                "pass an explicit `classes_per_task` list for uneven splits."
            )
        chunk_sizes = [n_classes // n_tasks] * n_tasks
    elif isinstance(classes_per_task, int):
        chunk_sizes = [classes_per_task] * n_tasks
    else:
        chunk_sizes = list(classes_per_task)
        if len(chunk_sizes) != n_tasks:
            raise ValueError(
                f"classes_per_task has {len(chunk_sizes)} entries, expected n_tasks={n_tasks}"
            )

    if sum(chunk_sizes) != n_classes:
        raise ValueError(
            f"chunk sizes {chunk_sizes} sum to {sum(chunk_sizes)}, expected n_classes={n_classes}"
        )

    splits: list[list[int]] = []
    start = 0
    for size in chunk_sizes:
        splits.append(order[start : start + size])
        start += size

    _validate_partition(splits, n_classes)
    return splits


def long_tail_counts(
    class_ids: Sequence[int],
    *,
    imbalance_ratio: float = 100.0,
    n_max: int = 500,
    seed: int = 0,
) -> dict[int, int]:
    """Exponential-profile long-tailed per-class sample budget.

    `n_c = n_max * imbalance_ratio ** (-rank / (K - 1))` where `rank` runs
    `0 .. K-1` (rank 0 is the most frequent "head" class, rank K-1 is the
    rarest "tail" class) and `K = len(class_ids)`.

    Which class lands at which rank is itself decided by `seed` (a fixed
    permutation of `class_ids`), so the head/tail assignment does not trivially
    correlate with class id or task order, and the whole map is deterministic
    given `seed`.

    Args:
        class_ids: the class ids to assign counts to (order irrelevant --
            ranks are assigned via a seeded permutation, not input order).
        imbalance_ratio: `r` = n_max / n_min. `r=1` is balanced.
        n_max: sample budget of the most frequent class.
        seed: seed for the rank permutation.

    Returns:
        `{class_id: n_samples}`, `n_samples >= 1`.
    """
    ids = list(class_ids)
    K = len(ids)
    if K == 0:
        return {}
    rng = np.random.RandomState(seed)
    rank_of = rng.permutation(K)  
    counts: dict[int, int] = {}
    for i, cls in enumerate(ids):
        rank = int(rank_of[i])
        if K == 1 or imbalance_ratio == 1:
            n_c = float(n_max)
        else:
            n_c = n_max * (imbalance_ratio ** (-rank / (K - 1)))
        counts[cls] = max(1, round(n_c))
    return counts


def subsample_indices(
    labels: Sequence[int] | np.ndarray,
    counts: dict[int, int],
    seed: int = 0,
) -> np.ndarray:
    """Pick indices into `labels` realising the per-class budget in `counts`.

    For each class, samples `min(counts[c], n_available)` indices without
    replacement (deterministically, given `seed`). Classes not present in
    `counts` are dropped entirely. Returned indices are sorted ascending.

    Args:
        labels: (N,) array-like of class ids, one per dataset sample.
        counts: `{class_id: n_samples}` as returned by `long_tail_counts`.
        seed: seed for the per-class random choice.

    Returns:
        `np.ndarray` of int64 indices into `labels`.
    """
    labels_arr = np.asarray(labels)
    rng = np.random.RandomState(seed)
    selected: list[np.ndarray] = []
    for cls, n in counts.items():
        cls_idx = np.flatnonzero(labels_arr == cls)
        n_take = min(int(n), len(cls_idx))
        if n_take <= 0:
            continue
        if n_take < len(cls_idx):
            chosen = rng.choice(cls_idx, size=n_take, replace=False)
        else:
            chosen = cls_idx
        selected.append(chosen)
    if not selected:
        return np.array([], dtype=np.int64)
    idx = np.concatenate(selected).astype(np.int64)
    idx.sort()
    return idx
