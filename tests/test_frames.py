"""Tests for `algpronc.geometry.frames`."""

from __future__ import annotations

import math

import torch

from algpronc.geometry.frames import (
    frame_diagnostics,
    gs_expand,
    random_frame,
    simplex_etf,
    truncated_etf,
    welch_frame,
)


def test_simplex_etf_is_a_true_etf_for_various_K_m():
    for K in (2, 3, 5, 10, 17):
        for m in (K - 1, K, 2 * K, 40):
            U = simplex_etf(K, m, seed=1)
            assert U.dtype == torch.float64
            assert U.shape == (m, K)
            diag = frame_diagnostics(U)
            assert diag["is_etf"], (K, m, diag)
            assert diag["equinorm_err"] < 1e-10
            assert diag["equiangularity_err"] < 1e-10
            assert diag["sum_norm"] < 1e-10
            
            expected_offdiag = -1.0 / (K - 1)
            assert abs(diag["offdiag_min"] - expected_offdiag) < 1e-9
            assert abs(diag["offdiag_max"] - expected_offdiag) < 1e-9


def test_simplex_etf_gram_matches_closed_form():
    K, m = 8, 12
    U = simplex_etf(K, m, seed=0)
    G = U.T @ U
    ones = torch.ones(K, K, dtype=torch.float64)
    expected = (K / (K - 1)) * (torch.eye(K, dtype=torch.float64) - ones / K)
    assert torch.allclose(G, expected, atol=1e-9)


def test_simplex_etf_K1_and_K2_no_div_by_zero():
    U1 = simplex_etf(1, 5, seed=0)
    assert U1.shape == (5, 1)
    assert abs(U1.norm().item() - 1.0) < 1e-12

    U2 = simplex_etf(2, 3, seed=0)
    assert U2.shape == (3, 2)
    norms = U2.norm(dim=0)
    assert torch.allclose(norms, torch.ones(2, dtype=torch.float64), atol=1e-12)
    cos = (U2[:, 0] @ U2[:, 1]).item()
    assert abs(cos - (-1.0)) < 1e-9  


def test_simplex_etf_requires_m_ge_K_minus_1():
    try:
        simplex_etf(6, 3, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for m < K-1")


def test_simplex_etf_is_deterministic_given_seed():
    U1 = simplex_etf(5, 8, seed=42)
    U2 = simplex_etf(5, 8, seed=42)
    assert torch.equal(U1, U2)
    U3 = simplex_etf(5, 8, seed=43)
    assert not torch.equal(U1, U3)


def test_gs_expand_is_not_a_valid_etf():
    """`gs_expand` faithfully reproduces the original proposal's flawed
    construction and must NOT produce a Simplex ETF once the frame has
    actually grown."""
    U0 = simplex_etf(4, 16, seed=0)
    assert frame_diagnostics(U0)["is_etf"]

    U1 = gs_expand(U0, delta_K=3, seed=1)  
    assert U1.shape == (16, 7)
    diag1 = frame_diagnostics(U1)
    assert not diag1["is_etf"], diag1

    
    U2 = gs_expand(U1, delta_K=2, seed=2)  
    diag2 = frame_diagnostics(U2)
    assert not diag2["is_etf"], diag2
    
    
    assert diag2["equiangularity_err"] > 1e-3


def test_gs_expand_preserves_old_vertex_directions_up_to_global_rescale():
    """`gs_expand` explicitly keeps the *directions* of `U_prev`'s columns
    (only a single global scalar is applied to every column, old and new
    alike) -- this is the specific, literal behaviour asked for, even though
    the result is not equiangular."""
    U0 = simplex_etf(4, 16, seed=0)
    U1 = gs_expand(U0, delta_K=2, seed=1)
    old_block = U1[:, :4]
    
    
    ratios = (old_block / U0).mean(dim=0)
    reconstructed = U0 * ratios
    assert torch.allclose(old_block, reconstructed, atol=1e-9)
    
    
    assert torch.allclose(ratios, ratios[0] * torch.ones_like(ratios), atol=1e-9)
    expected_scale = math.sqrt(6 / 5)  
    assert abs(ratios[0].item() - expected_scale) < 1e-9


def test_welch_frame_beats_random_frame_and_reports_bound():
    K, m, seed = 24, 6, 0
    U_w = welch_frame(K, m, seed=seed, iters=1500)
    U_r = random_frame(K, m, seed=seed)
    diag_w = frame_diagnostics(U_w)
    diag_r = frame_diagnostics(U_r)

    assert diag_w["welch_bound"] == diag_r["welch_bound"]
    expected_bound = math.sqrt((K - m) / (m * (K - 1)))
    assert abs(diag_w["welch_bound"] - expected_bound) < 1e-12

    
    assert not diag_w["is_etf"]
    
    
    assert diag_w["coherence"] <= diag_r["coherence"]
    assert diag_w["coherence"] >= diag_w["welch_bound"] - 1e-9
    
    assert diag_w["equinorm_err"] < 1e-8


def test_welch_frame_is_deterministic_and_terminates():
    U1 = welch_frame(20, 5, seed=7, iters=300)
    U2 = welch_frame(20, 5, seed=7, iters=300)
    assert torch.equal(U1, U2)
    U3 = welch_frame(20, 5, seed=8, iters=300)
    assert not torch.equal(U1, U3)


def test_welch_frame_falls_back_to_simplex_etf_when_m_ge_K_minus_1():
    U = welch_frame(6, 5, seed=0)  
    assert frame_diagnostics(U)["is_etf"]
    U2 = welch_frame(6, 10, seed=0)  
    assert frame_diagnostics(U2)["is_etf"]


def test_truncated_etf_falls_back_to_simplex_etf_when_m_ge_K_minus_1():
    U = truncated_etf(6, 5, seed=0)  
    assert frame_diagnostics(U)["is_etf"]
    U2 = truncated_etf(6, 10, seed=0)  
    assert frame_diagnostics(U2)["is_etf"]
    
    assert torch.equal(U, simplex_etf(6, 5, seed=0))


def test_truncated_etf_shape_and_unit_norm_when_compressed():
    for K, m in [(10, 4), (10, 8), (24, 6), (50, 32)]:
        U = truncated_etf(K, m, seed=0)
        assert U.dtype == torch.float64
        assert U.shape == (m, K)
        norms = U.norm(dim=0)
        assert torch.allclose(norms, torch.ones(K, dtype=torch.float64), atol=1e-9)


def test_truncated_etf_is_not_a_valid_etf_when_compressed():
    """m < K-1 means K exactly-equiangular unit vectors don't fit -- a random
    projection of the true ETF necessarily breaks equiangularity, unlike
    `simplex_etf` itself. If this ever reports `is_etf=True`, the projection
    step is accidentally degenerate (e.g. m rows not actually independent)."""
    U = truncated_etf(24, 6, seed=0)
    diag = frame_diagnostics(U)
    assert not diag["is_etf"]
    assert diag["equiangularity_err"] > 1e-6


def test_truncated_etf_is_deterministic_given_seed():
    U1 = truncated_etf(20, 5, seed=7)
    U2 = truncated_etf(20, 5, seed=7)
    assert torch.equal(U1, U2)
    U3 = truncated_etf(20, 5, seed=8)
    assert not torch.equal(U1, U3)


def test_truncated_etf_requires_m_ge_1():
    try:
        truncated_etf(10, 0, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for m < 1")


def test_random_frame_unit_norm_and_deterministic():
    U1 = random_frame(10, 4, seed=5)
    assert U1.shape == (4, 10)
    norms = U1.norm(dim=0)
    assert torch.allclose(norms, torch.ones(10, dtype=torch.float64), atol=1e-12)
    U2 = random_frame(10, 4, seed=5)
    assert torch.equal(U1, U2)


def test_frame_diagnostics_handles_K1_without_div_by_zero():
    U = simplex_etf(1, 4, seed=0)
    diag = frame_diagnostics(U)
    assert diag["welch_bound"] == 0.0
    assert diag["coherence"] == 0.0
    assert diag["equiangularity_err"] == 0.0
    assert not math.isnan(diag["min_norm"])
