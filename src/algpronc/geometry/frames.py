"""Simplex Equiangular Tight Frames (ETF), the original proposal's flawed
Gram-Schmidt expansion, and Welch-bound-seeking compressed frames.

Everything here is `torch.float64` on CPU: frame geometry is exactly what
the paper's algebra is checked against, so it must not carry float32
rounding.

Three constructions, three roles:
  - `simplex_etf`   the *correct* Simplex ETF, regenerated fresh from a fixed
                     seed at any `K` -- used by `heads.etf_exact`.
  - `gs_expand`      the original proposal's Step 2/4, implemented literally,
                      warts and all -- it does NOT produce an ETF. Used by
                      `heads.etf_stale`. Do not "fix" it.
  - `welch_frame`    a genuine (approximate) Welch-bound-minimising frame for
                      the compressed regime `m < K-1`, where a one-hot/ETF
                      target does not even exist.
  - `truncated_etf`  the exact Simplex ETF, built in its natural K-1 dims and
                      projected onto a random m-dim subspace for m < K-1 --
                      the "what if we just cut the real ETF down to size"
                      baseline against `welch_frame`/`random_frame`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _random_orthogonal(n: int, *, seed: int, device: torch.device) -> Tensor:
    """A deterministic (given `seed`) orthogonal matrix in R^{n x n}, float64."""
    if n == 0:
        return torch.zeros(0, 0, dtype=torch.float64, device=device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    A = torch.randn(n, n, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    
    
    sign = torch.sign(torch.diagonal(R))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    Q = Q * sign.unsqueeze(0)
    return Q.to(device)


def simplex_etf(K: int, m: int, *, seed: int = 0, device: torch.device | str | None = None) -> Tensor:
    """Unit-norm Simplex ETF vertices, shape (m, K).

    Guarantees (exactly, to float64 precision): ``||u_c|| = 1``,
    ``<u_c, u_c'> = -1/(K-1)`` for ``c != c'``, and ``sum_c u_c = 0``.

    Construction: build the K x K Gram ``G = K/(K-1) (I - 11^T/K)`` implicitly
    via the projector ``P = I - 11^T/K`` (eigenvalues {0 (x1), 1 (x K-1)}),
    take the K-1 nonzero-eigenvalue directions as the raw simplex, normalise
    each column to unit norm (equivalent to the textbook `sqrt(K/(K-1))`
    rescale, since all raw columns have identical norm by symmetry), zero-pad
    up to R^m, and apply a seeded random orthogonal rotation of R^m (so the
    frame is not axis-aligned -- axis alignment would make the one-hot
    equivalence visually obvious, which defeats the point of the paper).

    Requires ``m >= K - 1``.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if m < K - 1:
        raise ValueError(f"simplex_etf requires m >= K-1, got m={m}, K={K}")
    dev = torch.device(device) if device is not None else torch.device("cpu")

    if K == 1:
        
        
        U0 = torch.ones(1, 1, dtype=torch.float64, device=dev)
    else:
        ones_over_K = torch.full((K, K), 1.0 / K, dtype=torch.float64, device=dev)
        P = torch.eye(K, dtype=torch.float64, device=dev) - ones_over_K
        P = (P + P.T) / 2
        eigvals, eigvecs = torch.linalg.eigh(P)  
        V = eigvecs[:, 1:]  
        U0 = V.T  
        norms = U0.norm(dim=0, keepdim=True).clamp(min=1e-300)
        U0 = U0 / norms  

    if m > U0.shape[0]:
        pad = torch.zeros(m - U0.shape[0], K, dtype=torch.float64, device=dev)
        U0 = torch.cat([U0, pad], dim=0)

    R = _random_orthogonal(m, seed=seed, device=dev)
    return R @ U0


def gs_expand(U_prev: Tensor, delta_K: int, *, seed: int = 0) -> Tensor:
    """The original proposal's Step 2/4 Gram-Schmidt frame expansion,
    literally.

    The original proposal says: "Retain the existing K_{t-1} ETF basis
    vectors. Sample delta_K linearly independent vectors. Apply Gram-Schmidt
    orthogonalization ... with respect to the subspace spanned by U_{K_{t-1}},
    forcing new target vectors into the orthogonal complement. Re-normalize
    all K_t vectors ... scaling by sqrt(K_t / (K_t - 1))."

    This is implemented exactly as written, including the bug: the "Gram-
    Schmidt" projection removes components along `U_prev`'s columns using
    ``V - U_prev @ (U_prev.T @ V)``, i.e. it treats `U_prev`'s columns as an
    orthonormal basis. They are not (a Simplex ETF is only *equiangular*,
    off-diagonal Gram `-1/(K_prev-1) != 0`), so the residual is not actually
    orthogonal to `span(U_prev)`. The final uniform rescale by
    `sqrt(K_new/(K_new-1))` is then applied to *every* column, old and new
    alike, which silently changes the norm/inner-products of the columns kept
    from the previous call too (and compounds across repeated expansions).

    NET EFFECT: this does **not** return a Simplex ETF. Old-old inner products
    drift with every expansion, old-new pairs are only approximately zero, and
    new-new pairs are uncorrelated Gaussian residual angles -- so the off-
    diagonal Gram is not constant (it empirically spreads over roughly
    [-0.38, +0.05] instead of a constant -1/(K-1)), and the vertices no
    longer sum to zero. This is deliberate: it is the object the paper
    studies (see `heads.etf_stale`). Do NOT "fix" this function.
    """
    U_prev = U_prev.to(torch.float64)
    m, K_prev = U_prev.shape
    if delta_K <= 0:
        raise ValueError(f"delta_K must be positive, got {delta_K}")
    K_new = K_prev + delta_K
    device = U_prev.device

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    V = torch.randn(m, delta_K, generator=gen, dtype=torch.float64).to(device)

    if K_prev > 0:
        
        
        V_perp = V - U_prev @ (U_prev.T @ V)
    else:
        V_perp = V

    U_full = torch.cat([U_prev, V_perp], dim=1)  
    scale = math.sqrt(K_new / (K_new - 1))
    return U_full * scale


def welch_frame(K: int, m: int, *, seed: int = 0, iters: int = 2000) -> Tensor:
    """Unit-norm frame for the compressed regime ``m < K-1``.

    Minimises the maximum coherence ``max_{c!=c'} |<u_c,u_c'>|`` toward the
    Welch bound ``sqrt((K-m)/(m(K-1)))`` by alternating projection on the
    Gram matrix: project onto rank-<=m PSD matrices (truncated eigendecomp,
    clipping negative eigenvalues), then project onto the "unit diagonal,
    bounded off-diagonal" constraint set (clip off-diagonals to +-mu, reset
    the diagonal to 1). This is a standard POCS scheme for near-Welch-bound
    ("Grassmannian") frame design; it need not reach the bound exactly (that
    is only achievable for special (K, m)), but it reliably improves on a
    random frame. Deterministic given `seed`; always runs exactly `iters`
    iterations, so it always terminates.

    Falls back to `simplex_etf` when ``m >= K - 1`` (a true ETF is available
    and is itself Welch-bound-achieving).
    """
    if m >= K - 1:
        return simplex_etf(K, m, seed=seed)
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if K < 2:
        raise ValueError(f"welch_frame needs K >= 2 in the compressed regime, got K={K}")

    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    U = torch.randn(m, K, generator=gen, dtype=torch.float64, device=device)
    U = U / U.norm(dim=0, keepdim=True).clamp(min=1e-300)
    G = U.T @ U

    mu = math.sqrt((K - m) / (m * (K - 1)))

    for _ in range(iters):
        
        eigvals, eigvecs = torch.linalg.eigh(G)
        top = eigvals[-m:].clamp(min=0.0)
        vecs = eigvecs[:, -m:]
        G = vecs @ torch.diag(top) @ vecs.T
        G = (G + G.T) / 2
        
        G = G.clamp(min=-mu, max=mu)
        G.fill_diagonal_(1.0)
        G = (G + G.T) / 2

    eigvals, eigvecs = torch.linalg.eigh(G)
    vals = eigvals[-m:].clamp(min=1e-15)
    vecs = eigvecs[:, -m:]
    U_out = torch.diag(vals.sqrt()) @ vecs.T  
    norms = U_out.norm(dim=0, keepdim=True).clamp(min=1e-300)
    return U_out / norms


def truncated_etf(K: int, m: int, *, seed: int = 0, device: torch.device | str | None = None) -> Tensor:
    """Genuine truncated-ETF baseline for the compressed regime ``m < K-1``.

    Builds the *exact* Simplex ETF in its natural ``K-1`` dimensions
    (`simplex_etf(K, K-1, ...)`), then projects it onto a random
    ``m``-dimensional subspace via a Haar-random semi-orthogonal ``(m, K-1)``
    matrix (the first `m` rows of a random orthogonal `(K-1, K-1)` matrix,
    `seed`-shifted by 1 so the projection doesn't reuse the ETF's own
    rotation), and renormalises columns back to unit norm.

    This is deliberately different from `welch_frame` (numerically optimises
    directly for near-equiangularity/near-Welch-bound in `m` dims, with no
    reference to the true ETF) and `random_frame` (no ETF structure at all):
    `truncated_etf` starts from the actual ETF geometry and only loses
    information via the dimensionality cut. A random projection preserves
    pairwise angles only approximately (Johnson-Lindenstrauss-style), so the
    result is genuinely *not* equiangular once ``m < K-1`` -- there is no
    way to fit `K` exactly-equiangular unit vectors into fewer than `K-1`
    dimensions, full stop. That's the point of the comparison: does starting
    from real ETF structure and truncating it beat a Welch-bound-optimised
    frame built from scratch, or a plain random one?

    Falls back to `simplex_etf(K, m, seed=seed)` when ``m >= K-1`` (no
    truncation needed, and none is possible to skip: the projection step
    requires a strict dimensionality cut).
    """
    if m >= K - 1:
        return simplex_etf(K, m, seed=seed, device=device)
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    dev = torch.device(device) if device is not None else torch.device("cpu")

    U_full = simplex_etf(K, K - 1, seed=seed, device=dev)  
    R = _random_orthogonal(K - 1, seed=seed + 1, device=dev)
    P = R[:m, :]  

    U = P @ U_full  
    norms = U.norm(dim=0, keepdim=True).clamp(min=1e-300)
    return U / norms


def random_frame(K: int, m: int, *, seed: int = 0) -> Tensor:
    """Unit-norm i.i.d. Gaussian columns, shape (m, K). Control for `welch_frame`."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    U = torch.randn(m, K, generator=gen, dtype=torch.float64)
    return U / U.norm(dim=0, keepdim=True).clamp(min=1e-300)


def frame_diagnostics(U: Tensor) -> dict:
    """Geometry diagnostics for a frame ``U`` of shape (m, K).

    Returns
        min_norm, max_norm         column-norm range
        equinorm_err                max_norm - min_norm
        offdiag_min, offdiag_max    range of pairwise *cosine* similarities
        equiangularity_err          offdiag_max - offdiag_min (0 for a true ETF)
        coherence                   max |cosine similarity| over c != c'
        welch_bound                 sqrt((K-m)/(m(K-1))); 0 if K <= 1
        sum_norm                   ``|| sum_c u_c ||`` (0 for a true ETF)
        is_etf                     True iff equinorm_err < 1e-8 and
                                    equiangularity_err < 1e-8 and sum_norm < 1e-8
    """
    U = U.to(torch.float64)
    m, K = U.shape
    norms = U.norm(dim=0)
    min_norm = norms.min().item()
    max_norm = norms.max().item()
    equinorm_err = max_norm - min_norm
    sum_norm = U.sum(dim=1).norm().item()

    safe_norms = norms.clamp(min=1e-300)
    Un = U / safe_norms
    G = Un.T @ Un
    if K > 1:
        offdiag_mask = ~torch.eye(K, dtype=torch.bool, device=G.device)
        offdiag_vals = G[offdiag_mask]
        offdiag_min = offdiag_vals.min().item()
        offdiag_max = offdiag_vals.max().item()
        equiangularity_err = offdiag_max - offdiag_min
        coherence = offdiag_vals.abs().max().item()
        welch_bound = math.sqrt(max(K - m, 0) / (m * (K - 1)))
    else:
        offdiag_min = offdiag_max = 0.0
        equiangularity_err = 0.0
        coherence = 0.0
        welch_bound = 0.0

    is_etf = equinorm_err < 1e-8 and equiangularity_err < 1e-8 and sum_norm < 1e-8
    return {
        "min_norm": min_norm,
        "max_norm": max_norm,
        "equinorm_err": equinorm_err,
        "coherence": coherence,
        "welch_bound": welch_bound,
        "offdiag_min": offdiag_min,
        "offdiag_max": offdiag_max,
        "equiangularity_err": equiangularity_err,
        "sum_norm": sum_norm,
        "is_etf": bool(is_etf),
    }
