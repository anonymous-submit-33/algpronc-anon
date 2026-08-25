"""Tests for algpronc.engine.run_cil.

`algpronc.heads` (AnalyticHead + concrete heads) had not landed at the time
this test was written, so a tiny, self-contained ridge-regression head
(`_StubHead`, one-hot targets, textbook SMW recursion) stands in for it here
-- it exists ONLY in this test file, never in `src/`. It implements exactly
the `AnalyticHead` surface engine.py calls: `observe_classes`, `partial_fit`,
`scores`, `predict`, `state_dict`, `load_state_dict`, `.W`.

These tests exercise, against the real synthetic feature path in engine.py
(no dataset/backbone dependency) and the real (already-landed)
`algpronc.data.splits` / `algpronc.models.projection`:
  - the lockstep guarantee: every head in `heads` receives the identical
    feature tensor for a given task;
  - the checkpoint/resume contract: killing a run mid-stream and resuming it
    must reproduce the same final head state as an uninterrupted run;
  - basic shape/range sanity of the returned summary;
  - that the optional random projection and long-tail imbalance resampling
    plumb through without breaking the loop.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algpronc.config import ExperimentConfig
from algpronc.engine import run_cil
from algpronc.utils.logging import RunLogger


class _StubHead:
    """One-hot ridge head via explicit SMW recursion. Deliberately tiny --
    just enough to drive engine.run_cil's loop and prove the plumbing works.

    Mirrors the real `heads.base.AnalyticHead`'s indexing convention exactly
    (this matters! see `engine.run_cil`'s comment on `idx_to_class`):
    `scores`/`predict` are indexed by LOCAL class index -- order of first
    appearance across `observe_classes` calls -- not by global class id, so
    `partial_fit`'s `y` (global ids) is translated to local indices before
    building the one-hot target block, and `predict` returns a local index
    that `run_cil` itself translates back to a global class id.
    """

    def __init__(self, d: int, ridge_lambda: float = 1e-2):
        self.d = d
        self.ridge_lambda = ridge_lambda
        self.n_seen = 0
        self._classes: list[int] = []
        self._class_to_idx: dict[int, int] = {}
        self.P = torch.eye(d, dtype=torch.float64) / ridge_lambda
        self.Q = torch.zeros(d, 0, dtype=torch.float64)

    def observe_classes(self, new_classes) -> None:
        for c in new_classes:
            c = int(c)
            if c not in self._class_to_idx:
                self._class_to_idx[c] = len(self._classes)
                self._classes.append(c)
        self.n_seen = len(self._classes)
        if self.n_seen > self.Q.shape[1]:
            pad = torch.zeros(self.d, self.n_seen - self.Q.shape[1], dtype=torch.float64)
            self.Q = torch.cat([self.Q, pad], dim=1)

    def _to_local(self, y: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self._class_to_idx[int(c)] for c in y.tolist()], dtype=torch.long)

    def partial_fit(self, Phi: torch.Tensor, y: torch.Tensor) -> None:
        Phi = Phi.to(torch.float64)
        y_local = self._to_local(y)
        N = Phi.shape[0]
        Y = torch.zeros(N, self.n_seen, dtype=torch.float64)
        Y[torch.arange(N), y_local] = 1.0
        PPhiT = self.P @ Phi.T
        S = torch.eye(N, dtype=torch.float64) + Phi @ PPhiT
        gain = PPhiT @ torch.linalg.solve(S, torch.eye(N, dtype=torch.float64))
        self.P = self.P - gain @ Phi @ self.P
        self.Q = self.Q + Phi.T @ Y

    @property
    def W(self) -> torch.Tensor:
        return self.P @ self.Q

    def scores(self, Phi: torch.Tensor) -> torch.Tensor:
        return Phi.to(torch.float64) @ self.W

    def predict(self, Phi: torch.Tensor) -> torch.Tensor:
        return self.scores(Phi).argmax(dim=1)

    def state_dict(self) -> dict:
        return {
            "P": self.P.clone(),
            "Q": self.Q.clone(),
            "n_seen": self.n_seen,
            "classes": list(self._classes),
            "d": self.d,
            "ridge_lambda": self.ridge_lambda,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.P = sd["P"].clone()
        self.Q = sd["Q"].clone()
        self.n_seen = sd["n_seen"]
        self._classes = list(sd["classes"])
        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}


class _RecordingHead(_StubHead):
    """Same as _StubHead but remembers every Phi it was fit on, so tests can
    check that two heads in one run_cil call saw bit-identical tensors."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen_phi: list[torch.Tensor] = []

    def partial_fit(self, Phi: torch.Tensor, y: torch.Tensor) -> None:
        self.seen_phi.append(Phi.clone())
        super().partial_fit(Phi, y)


def _make_cfg(tmp_path, *, run_id: str, n_tasks: int = 4, classes_per_task: int = 2, d: int = 6) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.run_id = run_id
    cfg.seed = 0
    cfg.device = "cpu"
    cfg.out_dir = str(tmp_path)
    cfg.data.dataset = "synthetic"
    cfg.data.n_tasks = n_tasks
    cfg.data.classes_per_task = classes_per_task
    cfg.data.class_order_seed = 0
    cfg.model.feature_dim = d
    cfg.cl.ridge_lambda = 1e-2
    cfg.checkpoint.enabled = True
    cfg.checkpoint.resume = True
    return cfg


def test_run_cil_returns_well_shaped_summary(tmp_path):
    cfg = _make_cfg(tmp_path, run_id="basic")
    logger = RunLogger(cfg)
    heads = {"stub": _StubHead(d=cfg.model.feature_dim, ridge_lambda=cfg.cl.ridge_lambda)}
    summary = run_cil(cfg, logger=logger, heads=heads)

    assert summary["n_tasks"] == cfg.data.n_tasks
    s = summary["stub"]
    A = np.asarray(s["task_accuracy_matrix"])
    assert A.shape == (cfg.data.n_tasks, cfg.data.n_tasks)
    assert 0.0 <= s["average_accuracy"] <= 1.0
    assert 0.0 <= s["average_incremental_accuracy"] <= 1.0
    assert "head" in s["head_tail_accuracy"] and "tail" in s["head_tail_accuracy"]
    
    lines = logger.tasks_path.read_text().strip().splitlines()
    assert len(lines) == cfg.data.n_tasks  


def test_run_cil_multiple_heads_log_one_row_each(tmp_path):
    cfg = _make_cfg(tmp_path, run_id="multi")
    logger = RunLogger(cfg)
    heads = {
        "a": _StubHead(d=cfg.model.feature_dim, ridge_lambda=cfg.cl.ridge_lambda),
        "b": _StubHead(d=cfg.model.feature_dim, ridge_lambda=cfg.cl.ridge_lambda),
    }
    summary = run_cil(cfg, logger=logger, heads=heads)
    assert set(k for k in summary if k in heads) == {"a", "b"}
    lines = logger.tasks_path.read_text().strip().splitlines()
    assert len(lines) == cfg.data.n_tasks * 2


def test_lockstep_every_head_sees_identical_features_per_task(tmp_path):
    cfg = _make_cfg(tmp_path, run_id="lockstep")
    logger = RunLogger(cfg)
    heads = {
        "a": _RecordingHead(d=cfg.model.feature_dim, ridge_lambda=cfg.cl.ridge_lambda),
        "b": _RecordingHead(d=cfg.model.feature_dim, ridge_lambda=cfg.cl.ridge_lambda),
    }
    run_cil(cfg, logger=logger, heads=heads)

    assert len(heads["a"].seen_phi) == len(heads["b"].seen_phi) == cfg.data.n_tasks
    for t in range(cfg.data.n_tasks):
        assert torch.equal(heads["a"].seen_phi[t], heads["b"].seen_phi[t]), f"task {t}: heads saw different features"


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    
    cfg_full = _make_cfg(tmp_path, run_id="uninterrupted")
    heads_full = {"stub": _StubHead(d=cfg_full.model.feature_dim, ridge_lambda=cfg_full.cl.ridge_lambda)}
    summary_full = run_cil(cfg_full, logger=RunLogger(cfg_full), heads=heads_full)
    W_full = heads_full["stub"].W.clone()

    
    cfg_part = _make_cfg(tmp_path, run_id="interrupted")
    heads_part = {"stub": _StubHead(d=cfg_part.model.feature_dim, ridge_lambda=cfg_part.cl.ridge_lambda)}
    call_count = {"n": 0}
    real_partial_fit = heads_part["stub"].partial_fit

    def crashing_partial_fit(Phi, y):
        call_count["n"] += 1
        real_partial_fit(Phi, y)
        if call_count["n"] == 2:
            raise RuntimeError("simulated Spot preemption")

    heads_part["stub"].partial_fit = crashing_partial_fit
    with pytest.raises(RuntimeError, match="simulated Spot preemption"):
        run_cil(cfg_part, logger=RunLogger(cfg_part), heads=heads_part)

    
    heads_resumed = {"stub": _StubHead(d=cfg_part.model.feature_dim, ridge_lambda=cfg_part.cl.ridge_lambda)}
    summary_resumed = run_cil(cfg_part, logger=RunLogger(cfg_part), heads=heads_resumed)
    W_resumed = heads_resumed["stub"].W

    assert torch.allclose(W_resumed, W_full, atol=1e-9), "resumed run's final W does not match the uninterrupted run"
    
    A_full = np.asarray(summary_full["stub"]["task_accuracy_matrix"])
    A_resumed = np.asarray(summary_resumed["stub"]["task_accuracy_matrix"])
    assert A_full.shape == A_resumed.shape
    np.testing.assert_allclose(A_full, A_resumed, atol=1e-9, equal_nan=True)


def test_resume_without_checkpoint_starts_from_scratch(tmp_path):
    
    cfg = _make_cfg(tmp_path, run_id="fresh")
    heads = {"stub": _StubHead(d=cfg.model.feature_dim, ridge_lambda=cfg.cl.ridge_lambda)}
    summary = run_cil(cfg, logger=RunLogger(cfg), heads=heads)
    assert summary["stub"]["average_accuracy"] == summary["stub"]["average_accuracy"]  


def test_run_cil_with_random_projection(tmp_path):
    cfg = _make_cfg(tmp_path, run_id="rp", d=6)
    cfg.cl.rp.enabled = True
    cfg.cl.rp.proj_dim = 16
    cfg.cl.rp.seed = 0
    heads = {"stub": _StubHead(d=cfg.cl.rp.proj_dim, ridge_lambda=cfg.cl.ridge_lambda)}
    summary = run_cil(cfg, logger=RunLogger(cfg), heads=heads)
    assert 0.0 <= summary["stub"]["average_accuracy"] <= 1.0


def test_run_cil_with_imbalance_varies_class_counts(tmp_path):
    cfg = _make_cfg(tmp_path, run_id="imbalance", n_tasks=2, classes_per_task=4)
    cfg.cl.imbalance.enabled = True
    cfg.cl.imbalance.imbalance_ratio = 100.0
    cfg.cl.imbalance.seed = 0
    heads = {"stub": _StubHead(d=cfg.model.feature_dim, ridge_lambda=cfg.cl.ridge_lambda)}
    summary = run_cil(cfg, logger=RunLogger(cfg), heads=heads)
    counts = list(summary["class_counts"].values())
    assert len(counts) == 8
    assert max(counts) > min(counts)  
