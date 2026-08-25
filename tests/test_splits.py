"""Pure-logic tests for algpronc.data.splits -- no I/O, no downloads."""

from __future__ import annotations

import numpy as np
import pytest

from algpronc.data.splits import (
    class_incremental_splits,
    long_tail_counts,
    subsample_indices,
)


def test_uniform_split_covers_every_class_exactly_once():
    splits = class_incremental_splits(100, 10, order_seed=0)
    assert len(splits) == 10
    assert all(len(t) == 10 for t in splits)
    flat = sorted(c for task in splits for c in task)
    assert flat == list(range(100))


def test_split_is_deterministic_given_seed():
    a = class_incremental_splits(100, 10, order_seed=42)
    b = class_incremental_splits(100, 10, order_seed=42)
    assert a == b


def test_different_seeds_generally_give_different_orders():
    a = class_incremental_splits(100, 10, order_seed=0)
    b = class_incremental_splits(100, 10, order_seed=1)
    assert a != b


def test_seed_1993_icarl_podnet_convention_is_supported():
    
    
    splits = class_incremental_splits(100, 10, order_seed=1993)
    flat = sorted(c for task in splits for c in task)
    assert flat == list(range(100))
    again = class_incremental_splits(100, 10, order_seed=1993)
    assert splits == again


def test_classes_per_task_int_uniform():
    splits = class_incremental_splits(100, 10, classes_per_task=10, order_seed=0)
    assert [len(t) for t in splits] == [10] * 10


def test_classes_per_task_uneven_b50_5step_protocol():
    
    
    splits = class_incremental_splits(100, 6, classes_per_task=[50, 10, 10, 10, 10, 10], order_seed=7)
    assert [len(t) for t in splits] == [50, 10, 10, 10, 10, 10]
    flat = sorted(c for task in splits for c in task)
    assert flat == list(range(100))


def test_classes_per_task_uneven_wrong_length_raises():
    with pytest.raises(ValueError):
        class_incremental_splits(100, 6, classes_per_task=[50, 10, 10], order_seed=0)


def test_classes_per_task_uneven_wrong_sum_raises():
    with pytest.raises(ValueError):
        class_incremental_splits(100, 2, classes_per_task=[40, 40], order_seed=0)


def test_uneven_division_without_explicit_chunks_raises():
    with pytest.raises(ValueError):
        class_incremental_splits(100, 7, order_seed=0)  


def test_explicit_override_bypasses_order_seed():
    explicit = [[5, 1, 2], [0, 3, 4]]
    splits = class_incremental_splits(6, n_tasks=999, order_seed=123, explicit=explicit)
    assert splits == explicit


def test_explicit_override_validates_full_coverage():
    with pytest.raises(ValueError):
        class_incremental_splits(6, n_tasks=2, explicit=[[0, 1, 2], [3, 4, 4]])  


def test_explicit_override_validates_missing_class():
    with pytest.raises(ValueError):
        class_incremental_splits(4, n_tasks=2, explicit=[[0, 1], [2]])  


def test_long_tail_counts_balanced_when_ratio_is_one():
    counts = long_tail_counts(list(range(10)), imbalance_ratio=1, n_max=500, seed=0)
    assert all(v == 500 for v in counts.values())


def test_long_tail_counts_extremes_hit_n_max_and_n_max_over_ratio():
    class_ids = list(range(20))
    counts = long_tail_counts(class_ids, imbalance_ratio=100, n_max=500, seed=0)
    values = sorted(counts.values())
    assert values[0] == pytest.approx(500 / 100, abs=1)  
    assert values[-1] == 500  


def test_long_tail_counts_all_classes_present_and_positive():
    class_ids = list(range(50))
    counts = long_tail_counts(class_ids, imbalance_ratio=100, n_max=500, seed=3)
    assert set(counts.keys()) == set(class_ids)
    assert all(v >= 1 for v in counts.values())


def test_long_tail_counts_deterministic_given_seed():
    a = long_tail_counts(list(range(30)), imbalance_ratio=50, n_max=200, seed=17)
    b = long_tail_counts(list(range(30)), imbalance_ratio=50, n_max=200, seed=17)
    assert a == b


def test_long_tail_counts_different_seeds_permute_head_tail_assignment():
    a = long_tail_counts(list(range(30)), imbalance_ratio=50, n_max=200, seed=1)
    b = long_tail_counts(list(range(30)), imbalance_ratio=50, n_max=200, seed=2)
    assert a != b
    
    assert sorted(a.values()) == sorted(b.values())


def test_long_tail_counts_monotone_profile_shape():
    
    class_ids = list(range(11))
    counts = long_tail_counts(class_ids, imbalance_ratio=100, n_max=1000, seed=0)
    values_by_rank = sorted(counts.values(), reverse=True)
    expected = [1000 * (100 ** (-k / 10)) for k in range(11)]
    for got, exp in zip(values_by_rank, expected):
        assert got == pytest.approx(exp, rel=0.05, abs=1)


def test_subsample_indices_respects_counts():
    labels = np.array([0] * 20 + [1] * 5 + [2] * 100)
    counts = {0: 10, 1: 5, 2: 20}
    idx = subsample_indices(labels, counts, seed=0)
    selected_labels = labels[idx]
    assert (selected_labels == 0).sum() == 10
    assert (selected_labels == 1).sum() == 5  
    assert (selected_labels == 2).sum() == 20


def test_subsample_indices_deterministic_given_seed():
    labels = np.array([0] * 50 + [1] * 50)
    counts = {0: 10, 1: 10}
    a = subsample_indices(labels, counts, seed=5)
    b = subsample_indices(labels, counts, seed=5)
    assert np.array_equal(a, b)


def test_subsample_indices_drops_classes_not_in_counts():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    counts = {0: 5, 2: 5}  
    idx = subsample_indices(labels, counts, seed=0)
    assert set(labels[idx].tolist()) == {0, 2}


def test_subsample_indices_returns_sorted_unique_indices():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    counts = {0: 2, 1: 2}
    idx = subsample_indices(labels, counts, seed=1)
    assert list(idx) == sorted(set(idx.tolist()))
