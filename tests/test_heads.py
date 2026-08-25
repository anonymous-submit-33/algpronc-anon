"""Tests for `algpronc.heads` that don't depend on the cross-head equivalence
claim (that lives in tests/test_equivalence.py): batch-size invariance, the
SMW recursion vs. a direct ridge solve, `build_head` dispatch, and the
equinorm wrapper's escape from the invariance theorem.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from algpronc.heads import (
    CompressedHead,
    ETFExactHead,
    ETFStaleHead,
    EquinormHead,
    OneHotHead,
    build_head,
)


def _make_stream(d: int, K: int, n_per_class: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(K, d, generator=g, dtype=torch.float64) * 2.0
    Phis, ys = [], []
    for c in range(K):
        X = means[c] + torch.randn(n_per_class, d, generator=g, dtype=torch.float64)
        Phis.append(X)
        ys.append(torch.full((n_per_class,), c, dtype=torch.int64))
    Phi = torch.cat(Phis, dim=0)
    y = torch.cat(ys, dim=0)
    perm = torch.randperm(Phi.shape[0], generator=g)
    return Phi[perm], y[perm]


def test_partial_fit_is_batch_size_invariant():
    torch.manual_seed(0)
    d, K, n_per_class = 24, 5, 60
    Phi, y = _make_stream(d, K, n_per_class, seed=1)
    N = Phi.shape[0]

    head_big = OneHotHead(d, ridge_lambda=1e-2)
    head_big.observe_classes(list(range(K)))
    head_big.partial_fit(Phi, y)

    head_chunked = OneHotHead(d, ridge_lambda=1e-2)
    head_chunked.observe_classes(list(range(K)))
    bs = 32
    for i in range(0, N, bs):
        head_chunked.partial_fit(Phi[i : i + bs], y[i : i + bs])

    max_dW = (head_big.W - head_chunked.W).abs().max().item()
    assert max_dW < 1e-9, max_dW

    max_dP = (head_big.P - head_chunked.P).abs().max().item()
    assert max_dP < 1e-9, max_dP


def test_partial_fit_batch_size_invariance_when_batch_exceeds_d():
    """Exercises the n > d branch of the SMW update (invert d x d instead of
    n x n) against the n <= d branch, on the same data."""
    torch.manual_seed(0)
    d, K, n_per_class = 8, 4, 80  
    Phi, y = _make_stream(d, K, n_per_class, seed=2)
    N = Phi.shape[0]

    head_one_big_batch = OneHotHead(d, ridge_lambda=1e-1)
    head_one_big_batch.observe_classes(list(range(K)))
    head_one_big_batch.partial_fit(Phi, y)  

    head_small_chunks = OneHotHead(d, ridge_lambda=1e-1)
    head_small_chunks.observe_classes(list(range(K)))
    for i in range(0, N, 5):  
        head_small_chunks.partial_fit(Phi[i : i + 5], y[i : i + 5])

    max_dW = (head_one_big_batch.W - head_small_chunks.W).abs().max().item()
    assert max_dW < 1e-8, max_dW


def test_smw_P_matches_direct_ridge_solve():
    torch.manual_seed(0)
    d, K, n_per_class = 20, 6, 50
    ridge_lambda = 0.3
    Phi, y = _make_stream(d, K, n_per_class, seed=3)

    head = OneHotHead(d, ridge_lambda=ridge_lambda)
    head.observe_classes(list(range(K)))
    
    for i in range(0, Phi.shape[0], 40):
        head.partial_fit(Phi[i : i + 40], y[i : i + 40])

    R = Phi.T @ Phi + ridge_lambda * torch.eye(d, dtype=torch.float64)
    P_direct = torch.linalg.solve(R, torch.eye(d, dtype=torch.float64))

    assert (head.P - P_direct).abs().max().item() < 1e-9

    Y = torch.zeros(Phi.shape[0], K, dtype=torch.float64)
    Y.scatter_(1, y.view(-1, 1), 1.0)
    W_direct = torch.linalg.solve(R, Phi.T @ Y)
    assert (head.W - W_direct).abs().max().item() < 1e-9


def _cfg(**kwargs):
    base = dict(ridge_lambda=1e-2, etf=SimpleNamespace(seed=0))
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_build_head_dispatch():
    d = 10
    assert isinstance(build_head(_cfg(head="onehot"), d), OneHotHead)
    assert isinstance(build_head(_cfg(head="etf_exact", proj_m=d), d), ETFExactHead)
    assert isinstance(build_head(_cfg(head="etf_stale", proj_m=d), d), ETFStaleHead)
    assert isinstance(build_head(_cfg(head="compressed", proj_m=3, frame="welch"), d), CompressedHead)
    eq = build_head(_cfg(head="equinorm", equinorm_inner="onehot", equinorm_mode="procrustes"), d)
    assert isinstance(eq, EquinormHead)
    assert isinstance(eq.inner, OneHotHead)
    assert eq.mode == "procrustes"


def test_build_head_defaults_to_onehot():
    d = 10
    head = build_head(SimpleNamespace(), d)
    assert isinstance(head, OneHotHead)


def test_equinorm_column_mode_changes_predictions_under_imbalance():
    """Fit an imbalanced stream (a big head class, a tiny tail class) and
    check that column-normalising W actually moves predictions relative to
    the un-normalised inner head -- this is the live, positive result the
    paper reports (equinorm is *not* a no-op the way onehot/etf_exact are)."""
    torch.manual_seed(0)
    d = 16
    K = 4
    g = torch.Generator().manual_seed(11)
    means = torch.randn(K, d, generator=g, dtype=torch.float64) * 1.5

    
    counts = [300, 120, 60, 8]
    Phis, ys = [], []
    for c, n in enumerate(counts):
        X = means[c] + 0.8 * torch.randn(n, d, generator=g, dtype=torch.float64)
        Phis.append(X)
        ys.append(torch.full((n,), c, dtype=torch.int64))
    Phi = torch.cat(Phis, dim=0)
    y = torch.cat(ys, dim=0)
    perm = torch.randperm(Phi.shape[0], generator=g)
    Phi, y = Phi[perm], y[perm]

    inner = OneHotHead(d, ridge_lambda=1e-2)
    inner.observe_classes(list(range(K)))
    inner.partial_fit(Phi, y)

    eq = EquinormHead(inner, mode="column")

    
    inner_norms = inner.W.norm(dim=0)
    assert inner_norms.max().item() / inner_norms.min().item() > 1.2

    
    g2 = torch.Generator().manual_seed(99)
    Phi_eval = torch.cat([means[c] + 0.8 * torch.randn(200, d, generator=g2, dtype=torch.float64) for c in range(K)])
    y_eval = torch.cat([torch.full((200,), c, dtype=torch.int64) for c in range(K)])

    pred_inner = inner.predict(Phi_eval)
    pred_eq = eq.predict(Phi_eval)

    frac_changed = (pred_inner != pred_eq).float().mean().item()
    assert frac_changed > 0.0, "equinorm 'column' mode must change at least some predictions"

    acc_inner = (pred_inner == y_eval).float().mean().item()
    acc_eq = (pred_eq == y_eval).float().mean().item()
    
    
    assert acc_eq >= acc_inner - 0.02, (acc_inner, acc_eq)

    tail_mask = y_eval == (K - 1)
    tail_acc_inner = (pred_inner[tail_mask] == y_eval[tail_mask]).float().mean().item()
    tail_acc_eq = (pred_eq[tail_mask] == y_eval[tail_mask]).float().mean().item()
    assert tail_acc_eq >= tail_acc_inner, (tail_acc_inner, tail_acc_eq)


def test_equinorm_procrustes_mode_produces_unit_norm_columns():
    torch.manual_seed(0)
    d, K = 12, 5
    Phi, y = _make_stream(d, K, 40, seed=4)
    inner = OneHotHead(d, ridge_lambda=1e-2)
    inner.observe_classes(list(range(K)))
    inner.partial_fit(Phi, y)

    eq = EquinormHead(inner, mode="procrustes")
    norms = eq.W.norm(dim=0)
    assert torch.allclose(norms, torch.ones(K, dtype=torch.float64), atol=1e-8)


def test_equinorm_wrapping_etf_stale_still_delegates_fit_correctly():
    torch.manual_seed(0)
    d, K = 10, 4
    inner = ETFStaleHead(d, ridge_lambda=1e-2, m=d, seed=0)
    eq = EquinormHead(inner, mode="column")
    eq.observe_classes(list(range(K)))
    Phi, y = _make_stream(d, K, 20, seed=5)
    eq.partial_fit(Phi, y)
    assert eq.n_seen == K == inner.n_seen
    scores = eq.scores(Phi[:5])
    assert scores.shape == (5, K)


def _fitted_onehot(d=16, K=6, n=60, seed=7):
    Phi, y = _make_stream(d, K, n, seed=seed)
    inner = OneHotHead(d, ridge_lambda=1e-2)
    inner.observe_classes(list(range(K)))
    inner.partial_fit(Phi, y)
    return inner, Phi, y


def test_etf_project_mode_produces_an_exact_simplex_etf():
    """`etf_project` must satisfy the ETF definition exactly, not approximately:
    unit norms, every pairwise cosine -1/(K-1), columns summing to zero."""
    K = 6
    inner, _, _ = _fitted_onehot(K=K)
    W = EquinormHead(inner, mode="etf_project").W

    G = W.T @ W
    off = G[~torch.eye(K, dtype=torch.bool)]
    assert torch.allclose(torch.diagonal(G), torch.ones(K, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(off, torch.full_like(off, -1.0 / (K - 1)), atol=1e-10)
    assert W.sum(dim=1).abs().max().item() < 1e-10


def test_equiangular_mode_equalises_angles_but_preserves_the_fitted_norms():
    """The point of the `equiangular` head: it must differ from `etf_project`
    in exactly the equinorm property, so any accuracy gap between the two is
    attributable to norms rather than to angles."""
    K = 6
    inner, _, _ = _fitted_onehot(K=K)
    W_raw = inner.W
    W = EquinormHead(inner, mode="equiangular").W

    
    assert torch.allclose(W.norm(dim=0), W_raw.norm(dim=0), atol=1e-10)
    
    U = W / W.norm(dim=0, keepdim=True)
    G = U.T @ U
    off = G[~torch.eye(K, dtype=torch.bool)]
    assert torch.allclose(off, torch.full_like(off, -1.0 / (K - 1)), atol=1e-10)

    
    U_raw = W_raw / W_raw.norm(dim=0, keepdim=True)
    off_raw = (U_raw.T @ U_raw)[~torch.eye(K, dtype=torch.bool)]
    assert (off_raw.max() - off_raw.min()).item() > 1e-3


def test_etf_project_is_the_nearest_etf_not_merely_an_etf():
    """Guards the polar-factor solution: no random rotation of the projection
    may come closer to W than the projection itself."""
    K = 5
    inner, _, _ = _fitted_onehot(K=K)
    W_raw = inner.W
    W = EquinormHead(inner, mode="etf_project").W
    best = (W_raw - W).norm().item()

    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        A = torch.randn(W.shape[0], W.shape[0], generator=g, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        assert (W_raw - Q @ W).norm().item() >= best - 1e-9


def test_equiangular_and_etf_project_are_not_no_ops_and_differ_from_each_other():
    """Probes are drawn broadly rather than from the fitted class clusters:
    `_make_stream`'s clusters are separable enough that every head agrees on
    them, so agreement there says nothing about whether the decision
    boundaries moved."""
    d, K = 16, 6
    inner, _, _ = _fitted_onehot(d=d, K=K)
    g = torch.Generator().manual_seed(21)
    probes = torch.randn(500, d, generator=g, dtype=torch.float64)

    preds = {
        name: EquinormHead(inner, mode=name).predict(probes)
        for name in ("column", "equiangular", "etf_project")
    }
    base = inner.predict(probes)
    for name, p in preds.items():
        assert (p != base).any(), f"{name} must not be a no-op"
    
    assert (preds["equiangular"] != preds["column"]).any()
    assert (preds["etf_project"] != preds["column"]).any()


def test_wrapper_head_names_build_the_right_modes():
    d = 12
    for name in ("equiangular", "etf_project"):
        h = build_head(_cfg(head=name, equinorm_inner="onehot"), d)
        assert isinstance(h, EquinormHead)
        assert h.mode == name
        assert isinstance(h.inner, OneHotHead)
