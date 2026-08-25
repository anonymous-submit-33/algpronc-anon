"""Tests for algpronc.metrics.

The task accuracy matrix `A` is the primitive; average accuracy, average
incremental accuracy, and forgetting are pure functions of `A`. These tests
build small, hand-checkable `A` matrices (and, for `task_accuracy_matrix`
itself, small hand-checkable pred/y/task_id histories) so the expected
numbers can be verified by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algpronc.metrics import (
    accuracy,
    average_accuracy,
    average_incremental_accuracy,
    forgetting,
    head_tail_accuracy,
    per_class_accuracy,
    task_accuracy_matrix,
)


def test_accuracy_basic():
    pred = torch.tensor([0, 1, 2, 3])
    y = torch.tensor([0, 1, 1, 3])
    assert accuracy(pred, y) == pytest.approx(0.75)


def test_accuracy_accepts_numpy_too():
    pred = np.array([0, 1, 2])
    y = np.array([0, 0, 2])
    assert accuracy(pred, y) == pytest.approx(2 / 3)


def test_accuracy_empty_is_nan():
    assert np.isnan(accuracy(torch.tensor([]), torch.tensor([])))


def test_per_class_accuracy():
    pred = torch.tensor([0, 0, 1, 1, 2])
    y = torch.tensor([0, 1, 1, 1, 2])
    out = per_class_accuracy(pred, y, n_classes=4)
    assert out[0] == pytest.approx(1.0)  
    assert out[1] == pytest.approx(2 / 3)  
    assert out[2] == pytest.approx(1.0)  
    assert np.isnan(out[3])  


def test_head_tail_accuracy_splits_by_training_frequency():
    
    class_counts = {0: 1000, 1: 900, 2: 10, 3: 5}
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    pred = torch.tensor([0, 0, 1, 1, 2, 9, 3, 9])  
    out = head_tail_accuracy(pred, y, class_counts, frac=0.5)
    assert out["head"] == pytest.approx(1.0)
    assert out["tail"] == pytest.approx(0.5)


def test_head_tail_accuracy_accepts_array_counts():
    counts = np.array([100, 90, 5, 1])  
    y = torch.tensor([0, 1, 2, 3])
    pred = torch.tensor([0, 1, 2, 3])
    out = head_tail_accuracy(pred, y, counts, frac=0.5)
    assert out["head"] == pytest.approx(1.0)
    assert out["tail"] == pytest.approx(1.0)


def test_head_tail_accuracy_missing_split_is_nan():
    class_counts = {0: 10}
    y = torch.tensor([0, 0])
    pred = torch.tensor([0, 0])
    out = head_tail_accuracy(pred, y, class_counts, frac=0.5)
    assert out["head"] == pytest.approx(1.0)
    assert np.isnan(out["tail"])


def test_task_accuracy_matrix_shape_and_nan_upper_triangle():
    
    preds = [torch.tensor([0, 0]), torch.tensor([0, 1, 1])]
    ys = [torch.tensor([0, 0]), torch.tensor([0, 1, 1])]
    task_ids = [torch.tensor([0, 0]), torch.tensor([0, 1, 1])]
    A = task_accuracy_matrix(preds, ys, task_ids, n_tasks=2)
    assert A.shape == (2, 2)
    assert A[0, 0] == pytest.approx(1.0)
    assert np.isnan(A[0, 1])  
    assert A[1, 0] == pytest.approx(1.0)
    assert A[1, 1] == pytest.approx(1.0)


def test_task_accuracy_matrix_partial_row_accuracy():
    
    preds = [torch.tensor([0]), torch.tensor([9, 9, 1, 1])]
    ys = [torch.tensor([0]), torch.tensor([0, 0, 1, 1])]
    task_ids = [torch.tensor([0]), torch.tensor([0, 0, 1, 1])]
    A = task_accuracy_matrix(preds, ys, task_ids, n_tasks=2)
    assert A[1, 0] == pytest.approx(0.0)
    assert A[1, 1] == pytest.approx(1.0)


def test_task_accuracy_matrix_partial_history_leaves_unrun_rows_nan():
    
    preds = [torch.tensor([0, 0])]
    ys = [torch.tensor([0, 0])]
    task_ids = [torch.tensor([0, 0])]
    A = task_accuracy_matrix(preds, ys, task_ids, n_tasks=3)
    assert A.shape == (3, 3)
    assert A[0, 0] == pytest.approx(1.0)
    assert np.all(np.isnan(A[1:]))


def _hand_built_matrix() -> np.ndarray:
    
    
    A = np.full((3, 3), np.nan)
    A[0, 0] = 1.0
    A[1, 0] = 0.6
    A[1, 1] = 0.8
    A[2, 0] = 0.4
    A[2, 1] = 0.5
    A[2, 2] = 0.9
    return A


def test_average_accuracy_is_last_row_mean():
    A = _hand_built_matrix()
    
    assert average_accuracy(A) == pytest.approx((0.4 + 0.5 + 0.9) / 3)


def test_average_incremental_accuracy_is_mean_of_row_means():
    A = _hand_built_matrix()
    row0 = 1.0
    row1 = (0.6 + 0.8) / 2
    row2 = (0.4 + 0.5 + 0.9) / 3
    expected = (row0 + row1 + row2) / 3
    assert average_incremental_accuracy(A) == pytest.approx(expected)


def test_forgetting_is_mean_max_minus_final():
    A = _hand_built_matrix()
    
    
    expected = (0.6 + 0.3 + 0.0) / 3
    assert forgetting(A) == pytest.approx(expected)


def test_forgetting_zero_for_monotonically_improving_task():
    A = np.array([[1.0, np.nan], [1.0, 1.0]])
    assert forgetting(A) == pytest.approx(0.0)


def test_average_accuracy_all_nan_is_nan():
    A = np.full((2, 2), np.nan)
    assert np.isnan(average_accuracy(A))
    assert np.isnan(average_incremental_accuracy(A))
    assert np.isnan(forgetting(A))
