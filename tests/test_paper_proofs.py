"""Step-by-step verification of every proof in the paper's Appendix A.

This file exists because the paper's central claims are *proved*, not merely
measured, and a proof is only as good as its weakest individual step. The
repo's other tests (notably `test_equivalence.py`) check the theorem
**end-to-end** on a task stream -- which passes even when an intermediate
lemma is wrong. Here each lemma is checked **in isolation**, in the same order
the appendix proves them, so a broken step is localised rather than masked by
a correct conclusion.

Two kinds of check appear:

  * `test_lemma*` / `test_theorem*` / `test_corollary*` / `test_prop*`
    verify a step as stated, symbolically (exact rationals, small K) where the
    claim is an algebraic identity, numerically in float64 otherwise.
  * `test_adversarial_*` verify each hypothesis is **load-bearing** by breaking
    it and confirming the conclusion then *fails*. Numerics alone cannot
    distinguish a sound proof from a vacuous one; these can.

Naming maps 1:1 onto the appendix: Lemma 1 (target linearity), Lemma 2 (order
invariance), Theorem 1 (target-geometry invariance), Corollary 1 (ETF targets
are a no-op), Proposition 2 (simplex ETFs are not nested), Corollary 3
(nearest-ETF projection of W is orthonormalisation).
"""

from __future__ import annotations

import math

import pytest
import sympy as sp
import torch

from algpronc.geometry.frames import simplex_etf
from algpronc.heads import ETFExactHead, OneHotHead


def _ridge_M(Phi: torch.Tensor, lam: float, omega: torch.Tensor | None = None) -> torch.Tensor:
    """`M = (Phi^T Omega Phi + lam I)^-1 Phi^T Omega`, the target-independent
    factor of the ridge solution. `omega=None` means unweighted."""
    d = Phi.shape[1]
    Id = torch.eye(d, dtype=torch.float64)
    if omega is None:
        R = Phi.T @ Phi + lam * Id
        return torch.linalg.solve(R, Phi.T)
    W_om = torch.diag(omega)
    R = Phi.T @ W_om @ Phi + lam * Id
    return torch.linalg.solve(R, Phi.T @ W_om)


def _synthetic(d=24, N=400, K=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    Phi = torch.randn(N, d, generator=g, dtype=torch.float64)
    y = torch.randint(0, K, (N,), generator=g)
    Y = torch.zeros(N, K, dtype=torch.float64)
    Y[torch.arange(N), y] = 1.0
    return Phi, y, Y


def _argmax_sets(S: torch.Tensor, tol: float = 0.0) -> list[frozenset[int]]:
    """Argmax as a *set* per row, so ties are compared honestly rather than
    through whatever tie-break `torch.argmax` happens to use."""
    out = []
    for row in S:
        top = row.max()
        out.append(frozenset(torch.nonzero(row >= top - tol).flatten().tolist()))
    return out


def _etf_gram(K: int) -> torch.Tensor:
    return (K / (K - 1.0)) * (
        torch.eye(K, dtype=torch.float64) - torch.ones(K, K, dtype=torch.float64) / K
    )


@pytest.mark.parametrize("lam", [1e-4, 1e-2, 1.0])
@pytest.mark.parametrize("m", [6, 7, 20])
def test_lemma1_target_linearity_unweighted(lam, m):
    """`W_A = M (Y A^T) = (M Y) A^T = W_onehot A^T` -- the whole content of the
    lemma is that `M` does not depend on the targets."""
    K = 7
    Phi, _, Y = _synthetic(K=K)
    A = simplex_etf(K, m, seed=3)
    M = _ridge_M(Phi, lam)
    W_A = M @ (Y @ A.T)
    W_onehot = M @ Y
    assert torch.allclose(W_A, W_onehot @ A.T, atol=1e-12, rtol=0)


@pytest.mark.parametrize("lam", [1e-4, 1.0])
def test_lemma1_holds_under_arbitrary_sample_weighting(lam):
    """The lemma only needs `M` to be target-free, so *any* left factor works.
    Weights are drawn per-sample and non-uniform across classes on purpose --
    the paper claims invariance under sample weighting, and this is the step
    that carries that claim."""
    K = 7
    Phi, _, Y = _synthetic(K=K)
    g = torch.Generator().manual_seed(11)
    omega = torch.rand(Phi.shape[0], generator=g, dtype=torch.float64) * 9.0 + 0.1
    A = simplex_etf(K, 12, seed=5)
    M = _ridge_M(Phi, lam, omega=omega)
    assert torch.allclose(M @ (Y @ A.T), (M @ Y) @ A.T, atol=1e-12, rtol=0)


def test_lemma1_holds_for_the_actual_smw_recursion():
    """Lemma 1 is stated for a batch solve, but the paper applies it to ACIL's
    *recursive* head. This checks the recursion really does realise the same
    target-free `M`: the fitted `W` of the ETF head equals the one-hot head's
    `W A^T` after a multi-task stream, not just after one batch."""
    d, K, m, lam = 16, 6, 9, 1e-2
    onehot = OneHotHead(d, lam)
    etf = ETFExactHead(d, lam, m=m, seed=1)
    g = torch.Generator().manual_seed(7)
    for t in range(3):
        classes = list(range(2 * t, 2 * t + 2))
        onehot.observe_classes(classes)
        etf.observe_classes(classes)
        Phi = torch.randn(50, d, generator=g, dtype=torch.float64)
        y = torch.tensor([classes[i % 2] for i in range(50)])
        onehot.partial_fit(Phi, y)
        etf.partial_fit(Phi, y)
    
    
    A_head = simplex_etf(K, m, seed=1)
    W_A = etf.P @ (etf.Q @ A_head.T)
    assert torch.allclose(W_A, onehot.W @ A_head.T, atol=1e-10, rtol=0)
    
    assert torch.allclose(etf.W, onehot.W @ (A_head.T @ A_head), atol=1e-10, rtol=0)
    
    Phi_test = torch.randn(80, d, generator=g, dtype=torch.float64)
    assert _argmax_sets(etf.scores(Phi_test)) == _argmax_sets(onehot.scores(Phi_test))


def test_adversarial_lemma1_fails_for_a_target_dependent_fit():
    """If the fitting map is allowed to look at the targets, the lemma must
    break. A target-dependent ridge weight is the simplest witness; without
    this check, `test_lemma1_*` would pass for reasons weaker than claimed."""
    K = 7
    Phi, _, Y = _synthetic(K=K)
    A = simplex_etf(K, 10, seed=2)

    def target_dependent_fit(E):
        lam = 1e-2 * float(E.abs().sum())  
        return _ridge_M(Phi, lam) @ E

    W_A = target_dependent_fit(Y @ A.T)
    assert not torch.allclose(W_A, target_dependent_fit(Y) @ A.T, atol=1e-8, rtol=0)


def test_lemma2_order_invariance():
    """A positive rescale plus a *class-independent* shift preserves the whole
    ordering, hence the argmax set. `gamma` varies per row (per sample), which
    is the case the theorem actually needs."""
    g = torch.Generator().manual_seed(4)
    S = torch.randn(200, 9, generator=g, dtype=torch.float64)
    alpha = 0.37
    gamma = torch.randn(200, 1, generator=g, dtype=torch.float64)
    S2 = alpha * S + gamma
    assert _argmax_sets(S) == _argmax_sets(S2)
    
    
    for i in range(S.shape[0]):
        d1 = S[i].unsqueeze(0) - S[i].unsqueeze(1)
        d2 = S2[i].unsqueeze(0) - S2[i].unsqueeze(1)
        assert torch.all(torch.sign(d1) == torch.sign(d2))


def test_lemma2_preserves_tie_sets_exactly():
    """Ties must map to ties -- otherwise 'same argmax' would be a statement
    about a tie-break rule rather than about the scores."""
    S = torch.tensor([[1.0, 1.0, 0.5], [2.0, 2.0, 2.0]], dtype=torch.float64)
    S2 = 3.0 * S + 7.0
    assert _argmax_sets(S) == _argmax_sets(S2) == [frozenset({0, 1}), frozenset({0, 1, 2})]


def test_adversarial_lemma2_needs_alpha_positive():
    """alpha < 0 reverses the order: the hypothesis alpha > 0 is load-bearing,
    not decoration. This is the single hypothesis that makes Corollary 1 a
    statement about ETFs rather than about all linear recodings."""
    g = torch.Generator().manual_seed(6)
    S = torch.randn(200, 9, generator=g, dtype=torch.float64)
    assert _argmax_sets(S) != _argmax_sets(-1.0 * S)


def test_adversarial_lemma2_needs_the_shift_class_independent():
    """A per-class shift (beta not a multiple of 11^T) breaks it. This is why
    the theorem needs `A^T A = alpha I + beta 11^T` specifically, and not just
    'some' symmetric positive-definite Gram."""
    g = torch.Generator().manual_seed(8)
    S = torch.randn(200, 9, generator=g, dtype=torch.float64)
    per_class = torch.randn(1, 9, generator=g, dtype=torch.float64) * 2.0
    assert _argmax_sets(S) != _argmax_sets(S + per_class)


@pytest.mark.parametrize("lam", [1e-4, 1e-2, 1.0])
@pytest.mark.parametrize("m", [6, 7, 13, 40])
def test_theorem1_score_identity(lam, m):
    K = 7
    Phi, _, Y = _synthetic(K=K)
    A = simplex_etf(K, m, seed=m)
    M = _ridge_M(Phi, lam)
    s_onehot = Phi @ (M @ Y)
    s_A = (Phi @ (M @ (Y @ A.T))) @ A
    assert torch.allclose(s_A, s_onehot @ (A.T @ A), atol=1e-10, rtol=0)


def test_theorem1_score_identity_holds_for_a_non_etf_recoding():
    """The score identity is claimed for *any* `A`, with only the argmax half
    needing the `alpha I + beta 11^T` structure. A random (non-ETF) `A` must
    therefore still satisfy the identity."""
    K, lam = 7, 1e-2
    Phi, _, Y = _synthetic(K=K)
    g = torch.Generator().manual_seed(21)
    A = torch.randn(11, K, generator=g, dtype=torch.float64)
    M = _ridge_M(Phi, lam)
    s_onehot = Phi @ (M @ Y)
    s_A = (Phi @ (M @ (Y @ A.T))) @ A
    assert torch.allclose(s_A, s_onehot @ (A.T @ A), atol=1e-10, rtol=0)


def test_adversarial_theorem1_argmax_moves_for_a_general_recoding():
    """...and for that same random `A`, the argmax must actually *move*.
    Otherwise Corollary 1 would be a triviality about all recodings rather
    than a consequence of ETF structure."""
    K, lam = 7, 1e-2
    Phi, _, Y = _synthetic(K=K)
    g = torch.Generator().manual_seed(21)
    A = torch.randn(11, K, generator=g, dtype=torch.float64)
    M = _ridge_M(Phi, lam)
    s_onehot = Phi @ (M @ Y)
    s_A = s_onehot @ (A.T @ A)
    assert _argmax_sets(s_A) != _argmax_sets(s_onehot)


@pytest.mark.parametrize("K", [3, 5, 10, 100])
def test_corollary1_gram_has_the_required_form(K):
    m = K - 1
    A = simplex_etf(K, m, seed=K)
    G = A.T @ A
    alpha = K / (K - 1.0)
    beta = -1.0 / (K - 1.0)
    target = alpha * torch.eye(K, dtype=torch.float64) + beta * torch.ones(
        K, K, dtype=torch.float64
    )
    assert torch.allclose(G, target, atol=1e-12, rtol=0)
    assert alpha > 0


def test_corollary1_gram_form_symbolically():
    """Exact-rational check that `alpha = K/(K-1)`, `beta = -1/(K-1)` really do
    reproduce the ETF's defining Gram (diagonal 1, off-diagonal -1/(K-1)) for
    symbolic `K` -- no floating point anywhere."""
    K = sp.symbols("K", positive=True, integer=True)
    alpha = K / (K - 1)
    beta = -sp.Rational(1, 1) / (K - 1)
    assert sp.simplify(alpha + beta - 1) == 0  
    assert sp.simplify(beta - (-1 / (K - 1))) == 0  
    
    assert sp.simplify(alpha + K * beta) == 0


@pytest.mark.parametrize("K", [3, 6, 25])
def test_corollary1_columns_sum_to_zero_is_implied_not_assumed(K):
    """The appendix claims `sum_c a_c = 0` *follows* from the Gram conditions
    rather than being an extra axiom: `A^T(A 1) = G 1 = 0` and `A 1` lies in
    range(A), so `A 1 = 0`. Checked here on frames built only from the norm
    and angle conditions."""
    A = simplex_etf(K, K - 1, seed=K + 1)
    G = A.T @ A
    ones = torch.ones(K, dtype=torch.float64)
    assert torch.allclose(G @ ones, torch.zeros(K, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(A @ ones, torch.zeros(A.shape[0], dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("K", [3, 6, 25])
def test_corollary1_rank_forces_m_at_least_K_minus_1(K):
    """`rank(A) = rank(A^T A) = rank(G) = K-1`, so `m >= K-1` is *forced*, not
    an assumption. This is the step that makes `m < K-1` (the paper's
    compressed-frame regime) provably outside the theorem's scope."""
    G = _etf_gram(K)
    assert torch.linalg.matrix_rank(G, tol=1e-10).item() == K - 1
    for m in (K - 1, K, K + 7):
        A = simplex_etf(K, m, seed=m)
        assert torch.linalg.matrix_rank(A, tol=1e-10).item() == K - 1


@pytest.mark.parametrize("K", [4, 9])
def test_corollary1_is_independent_of_which_etf_realisation_is_chosen(K):
    """Any two ETFs of the same (K, m) differ by an orthogonal transform of
    R^m, which cancels in `A^T A`. So the no-op conclusion cannot depend on
    the arbitrary frame choice -- checked across seeds and against an explicit
    random rotation."""
    m = K + 3
    A1 = simplex_etf(K, m, seed=1)
    A2 = simplex_etf(K, m, seed=2)
    assert torch.allclose(A1.T @ A1, A2.T @ A2, atol=1e-12, rtol=0)
    g = torch.Generator().manual_seed(13)
    Q, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    assert torch.allclose(A1.T @ A1, (Q @ A1).T @ (Q @ A1), atol=1e-12, rtol=0)


@pytest.mark.parametrize("lam", [1e-4, 1e-2, 1.0])
def test_corollary1_end_to_end_argmax_identity(lam):
    """The corollary as the paper uses it: ETF-target scores and one-hot scores
    have identical argmax sets, at several ridge strengths."""
    K, m = 7, 12
    Phi, _, Y = _synthetic(K=K)
    A = simplex_etf(K, m, seed=9)
    M = _ridge_M(Phi, lam)
    s_onehot = Phi @ (M @ Y)
    s_A = (Phi @ (M @ (Y @ A.T))) @ A
    assert _argmax_sets(s_onehot) == _argmax_sets(s_A)


@pytest.mark.parametrize("K", [3, 5, 12])
def test_prop2_at_most_one_vertex_can_survive_an_expansion(K):
    """Proof: if two distinct vertices `a_c, a_c'` were shared between a
    K-ETF and a (K+1)-ETF, their inner product would have to equal both
    `-1/(K-1)` and `-1/K`. Since those differ for every K >= 2, at most one
    vertex can be preserved. Checked here as the numerical statement that the
    two required off-diagonal values are never equal."""
    old = -1.0 / (K - 1)
    new = -1.0 / K
    assert old != new
    assert abs(old - new) > 1e-12
    
    
    A_old, A_new = simplex_etf(K, K, seed=1), simplex_etf(K + 1, K, seed=1)
    shared = [
        c
        for c in range(K)
        if min(
            (A_old[:, c] - A_new[:, j]).norm().item() for j in range(K + 1)
        )
        < 1e-8
    ]
    assert len(shared) <= 1


@pytest.mark.parametrize("K", [3, 5, 9])
def test_prop2_the_bound_of_one_shared_vertex_is_tight(K):
    """The appendix claims 'at most one' is tight -- one vertex *can* always be
    preserved, because O(m) acts transitively on the unit sphere and rotating
    an ETF leaves it an ETF. Constructed explicitly here; without this the
    proposition would be weaker than stated."""
    m = K + 2
    A = simplex_etf(K, m, seed=K)
    A_next = simplex_etf(K + 1, m, seed=K + 5)
    
    a, b = A_next[:, 0], A[:, 0]
    v = b - a
    if v.norm() > 1e-12:  
        v = v / v.norm()
        Q = torch.eye(m, dtype=torch.float64) - 2.0 * torch.outer(v, v)
    else:
        Q = torch.eye(m, dtype=torch.float64)
    A_rot = Q @ A_next
    assert torch.allclose(A_rot[:, 0], A[:, 0], atol=1e-9, rtol=0)
    
    assert torch.allclose(A_rot.T @ A_rot, _etf_gram(K + 1), atol=1e-9, rtol=0)


def test_prop2_symbolic_no_common_offdiagonal():
    """`-1/(K-1) = -1/K` has no solution for integer K >= 2 -- the whole proof,
    checked symbolically rather than at sampled K."""
    K = sp.symbols("K", integer=True)
    sols = sp.solve(sp.Eq(-1 / (K - 1), -1 / K), K)
    assert sols == []


def _G_half(K: int) -> torch.Tensor:
    P = torch.eye(K, dtype=torch.float64) - torch.ones(K, K, dtype=torch.float64) / K
    return math.sqrt(K / (K - 1.0)) * P


def _project_onto_etf(W: torch.Tensor):
    K = W.shape[1]
    Gh = _G_half(K)
    U, S, Vt = torch.linalg.svd(W @ Gh, full_matrices=False)
    O_star = U @ Vt
    return O_star @ Gh, O_star, S


@pytest.mark.parametrize("K", [3, 5, 20])
def test_corollary3_step_a_G_half_is_exactly_c_times_the_centering_projector(K):
    """Step (a): `I - 11^T/K` is idempotent, so `G^(1/2) = sqrt(K/(K-1))
    (I - 11^T/K)` exactly -- no eigendecomposition, and `G^(1/2)` is singular
    with null vector `1` (the fact steps (e)-(f) rely on)."""
    Gh = _G_half(K)
    assert torch.allclose(Gh @ Gh, _etf_gram(K), atol=1e-12, rtol=0)
    assert torch.allclose(Gh, Gh.T, atol=1e-14, rtol=0)
    ones = torch.ones(K, dtype=torch.float64)
    assert torch.allclose(Gh @ ones, torch.zeros(K, dtype=torch.float64), atol=1e-12)
    assert torch.linalg.matrix_rank(Gh, tol=1e-10).item() == K - 1


def test_corollary3_step_a_symbolically():
    """Same step in exact arithmetic for K = 4: `(cP)^2 = G` with c = sqrt(K/(K-1)),
    verified by sympy rather than float64."""
    K = 4
    P = sp.eye(K) - sp.ones(K, K) / K
    c = sp.sqrt(sp.Rational(K, K - 1))
    G = sp.Rational(K, K - 1) * P
    assert sp.simplify((c * P) * (c * P) - G) == sp.zeros(K, K)


@pytest.mark.parametrize("K", [3, 6, 15])
def test_corollary3_step_b_frobenius_norm_is_constant_on_the_feasible_set(K):
    """Step (b): `||M||_F^2 = tr(M^T M) = tr(G) = K` for every feasible `M`, so
    minimising `||W - M||_F` is *equivalent* to maximising `<W, M>`. Without
    this the reduction to Procrustes would not be valid."""
    assert abs(torch.trace(_etf_gram(K)).item() - K) < 1e-10
    g = torch.Generator().manual_seed(K)
    Gh = _G_half(K)
    for _ in range(5):
        Q, _ = torch.linalg.qr(torch.randn(K + 4, K, generator=g, dtype=torch.float64))
        M = Q @ Gh
        assert abs((M.norm() ** 2).item() - K) < 1e-9


@pytest.mark.parametrize("K", [3, 6])
def test_corollary3_step_c_feasible_set_equals_the_O_G_half_parametrisation(K):
    """Step (c): every `M` with `M^T M = G` can be written `O G^(1/2)` with
    `O^T O = I`. The appendix gives the explicit construction
    `O = (1/c) M + q 1^T/sqrt(K)` for any unit `q` orthogonal to range(M);
    it is checked here directly (this is the step the old proof sketch merely
    asserted)."""
    d = K + 3  
    g = torch.Generator().manual_seed(K * 5)
    Gh = _G_half(K)
    c = math.sqrt(K / (K - 1.0))
    for _ in range(5):
        Q, _ = torch.linalg.qr(torch.randn(d, K, generator=g, dtype=torch.float64))
        M = Q @ Gh  
        assert torch.allclose(M.T @ M, _etf_gram(K), atol=1e-10, rtol=0)
        
        ones = torch.ones(K, dtype=torch.float64)
        assert torch.allclose(M @ ones, torch.zeros(d, dtype=torch.float64), atol=1e-10)
        
        Um, Sm, _ = torch.linalg.svd(M, full_matrices=True)
        q = Um[:, K - 1 :][:, -1]  
        assert (M.T @ q).abs().max().item() < 1e-9
        O = (1.0 / c) * M + torch.outer(q, ones) / math.sqrt(K)
        assert torch.allclose(O.T @ O, torch.eye(K, dtype=torch.float64), atol=1e-9, rtol=0)
        assert torch.allclose(O @ Gh, M, atol=1e-9, rtol=0)


@pytest.mark.parametrize("K", [3, 6, 12])
def test_corollary3_step_d_polar_factor_beats_every_other_feasible_point(K):
    """Step (d): the Procrustes solution `O* = U V^T` really is the maximiser --
    checked against many random feasible competitors, and against the value
    `sum_i sigma_i` the proof predicts."""
    d = K + 10
    g = torch.Generator().manual_seed(K * 7 + 1)
    W = torch.randn(d, K, generator=g, dtype=torch.float64)
    Gh = _G_half(K)
    M_star, O_star, S = _project_onto_etf(W)
    best = torch.trace(W.T @ M_star).item()
    assert abs(best - S.sum().item()) < 1e-9  
    for _ in range(50):
        Q, _ = torch.linalg.qr(torch.randn(d, K, generator=g, dtype=torch.float64))
        assert torch.trace(W.T @ (Q @ Gh)).item() <= best + 1e-9
    
    for _ in range(50):
        Q, _ = torch.linalg.qr(torch.randn(d, K, generator=g, dtype=torch.float64))
        assert (W - Q @ Gh).norm().item() >= (W - M_star).norm().item() - 1e-9


@pytest.mark.parametrize("K", [4, 9])
def test_corollary3_step_e_M_star_is_unique_though_O_star_is_not(K):
    """Step (e), the subtlety the old sketch missed: `W G^(1/2)` is rank
    `K-1`, so its smallest singular value is 0 and `O*` is *not* unique -- the
    free direction pairs with `v = 1/sqrt(K)`, which `G^(1/2)` annihilates.
    Perturbing that direction must therefore leave `M*` untouched."""
    d = K + 8
    g = torch.Generator().manual_seed(K * 3)
    W = torch.randn(d, K, generator=g, dtype=torch.float64)
    Gh = _G_half(K)
    U, S, Vt = torch.linalg.svd(W @ Gh, full_matrices=False)
    assert S[-1].item() < 1e-10  
    assert abs(abs(Vt[-1] @ (torch.ones(K, dtype=torch.float64) / math.sqrt(K))).item() - 1.0) < 1e-8
    M_star = (U @ Vt) @ Gh
    
    for _ in range(10):
        z = torch.randn(d, generator=g, dtype=torch.float64)
        z = z - U[:, :-1] @ (U[:, :-1].T @ z)
        z = z / z.norm()
        U_alt = U.clone()
        U_alt[:, -1] = z
        O_alt = U_alt @ Vt
        assert torch.allclose(O_alt.T @ O_alt, torch.eye(K, dtype=torch.float64), atol=1e-9)
        assert torch.allclose(O_alt @ Gh, M_star, atol=1e-9, rtol=0)


@pytest.mark.parametrize("K", [4, 9, 16])
def test_corollary3_step_e_closed_form_for_M_star(K):
    """Step (e) states `M* = c B (B^T B)^{+1/2}` with `B = W G^(1/2)` -- the
    partial-isometry factor of the polar decomposition, manifestly a function
    of `B` alone and therefore independent of the SVD chosen. That closed form
    is asserted in the appendix, so it is checked here directly."""
    d = K + 12
    g = torch.Generator().manual_seed(K * 29)
    W = torch.randn(d, K, generator=g, dtype=torch.float64)
    Gh = _G_half(K)
    c = math.sqrt(K / (K - 1.0))
    M_star, _, _ = _project_onto_etf(W)
    B = W @ Gh
    evals, evecs = torch.linalg.eigh(B.T @ B)
    inv_sqrt = torch.where(evals > 1e-10, evals.clamp(min=1e-30) ** -0.5, torch.zeros_like(evals))
    pinv_half = evecs @ torch.diag(inv_sqrt) @ evecs.T
    assert torch.allclose(M_star, c * (B @ pinv_half), atol=1e-9, rtol=0)


@pytest.mark.parametrize("K", [4, 9, 20])
def test_corollary3_step_f_score_identity_and_argmax_equality(K):
    """Step (f): `phi^T M* = c (s_polar - mean_j(s_polar))` -- Lemma 2's
    `alpha s + gamma 1` form once more, hence identical argmax. Note `s_polar`
    is read off the polar factor of `W G^(1/2)` (the *column-centred* W), not
    of `W`."""
    d = K + 30
    g = torch.Generator().manual_seed(K * 11)
    W = torch.randn(d, K, generator=g, dtype=torch.float64)
    Phi = torch.randn(150, d, generator=g, dtype=torch.float64)
    M_star, O_star, _ = _project_onto_etf(W)
    c = math.sqrt(K / (K - 1.0))
    s_etf = Phi @ M_star
    s_polar = Phi @ O_star
    predicted = c * (s_polar - s_polar.mean(dim=1, keepdim=True))
    assert torch.allclose(s_etf, predicted, atol=1e-10, rtol=0)
    assert _argmax_sets(s_etf) == _argmax_sets(s_polar)


@pytest.mark.parametrize("K", [5, 12])
def test_corollary3_the_polar_factor_is_of_the_centred_W_not_of_W(K):
    """The correction this session made to the paper's wording: `U V^T` comes
    from the SVD of `W G^(1/2) = c W (I - 11^T/K)`, and calling it 'the polar
    factor of W' is wrong -- the polar factor of `W` itself is a *different*
    matrix and does *not* give the same predictions in general."""
    d = K + 20
    g = torch.Generator().manual_seed(K * 17)
    W = torch.randn(d, K, generator=g, dtype=torch.float64)
    Phi = torch.randn(200, d, generator=g, dtype=torch.float64)
    _, O_centred, _ = _project_onto_etf(W)
    Uw, _, Vtw = torch.linalg.svd(W, full_matrices=False)
    O_raw = Uw @ Vtw  
    assert not torch.allclose(O_centred, O_raw, atol=1e-6, rtol=0)
    assert _argmax_sets(Phi @ O_centred) != _argmax_sets(Phi @ O_raw)


@pytest.mark.parametrize("K", [4, 8])
def test_corollary3_M_star_satisfies_the_etf_constraint_exactly(K):
    """Sanity closure: the minimiser is genuinely feasible -- unit-norm
    columns, all pairwise cosines `-1/(K-1)`, columns summing to zero."""
    d = K + 15
    g = torch.Generator().manual_seed(K * 23)
    W = torch.randn(d, K, generator=g, dtype=torch.float64)
    M_star, _, _ = _project_onto_etf(W)
    assert torch.allclose(M_star.T @ M_star, _etf_gram(K), atol=1e-9, rtol=0)
    assert torch.allclose(
        M_star @ torch.ones(K, dtype=torch.float64),
        torch.zeros(d, dtype=torch.float64),
        atol=1e-9,
    )


def test_adversarial_corollary3_uniqueness_needs_full_rank_centred_W():
    """The uniqueness claim in step (e) assumes `rank(W (I - 11^T/K)) = K-1`.
    Break that -- feed a `W` whose centred version is rank-deficient beyond the
    structural drop -- and `M*` becomes genuinely non-unique, confirming the
    hypothesis is load-bearing rather than cosmetic."""
    K, d = 6, 12
    g = torch.Generator().manual_seed(99)
    B = torch.randn(d, K - 3, generator=g, dtype=torch.float64)
    C = torch.randn(K - 3, K, generator=g, dtype=torch.float64)
    W = B @ C  
    Gh = _G_half(K)
    assert torch.linalg.matrix_rank(W @ Gh, tol=1e-9).item() < K - 1
    U, S, Vt = torch.linalg.svd(W @ Gh, full_matrices=False)
    assert (S < 1e-9).sum().item() >= 2  
    M1 = (U @ Vt) @ Gh
    
    idx = int((S > 1e-9).sum().item())
    R = torch.eye(K, dtype=torch.float64)
    R[idx, idx], R[idx, idx + 1] = 0.0, 1.0
    R[idx + 1, idx], R[idx + 1, idx + 1] = 1.0, 0.0
    M2 = (U @ R @ Vt) @ Gh
    obj1 = (W - M1).norm().item()
    obj2 = (W - M2).norm().item()
    assert abs(obj1 - obj2) < 1e-8  
    assert not torch.allclose(M1, M2, atol=1e-6, rtol=0)  
