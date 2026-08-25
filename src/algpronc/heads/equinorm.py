"""`EquinormHead`: a wrapper that non-linearly post-processes any inner head's
`W`, escaping the onehot/ETF invariance theorem.

Every other head in this package produces `W = P @ (linear function of Y)`,
so its argmax is a fixed linear image of the one-hot solution (a provably
no-op result). `EquinormHead` breaks that by applying a
*non-linear* transform to `W` after fitting, which is exactly the
Neural-Collapse geometry (equal-norm classifier columns, equiangular class
directions, self-duality) that the linear theorem cannot see. This is where
the paper's other positive result lives: under a class-imbalanced stream,
minority classes get systematically smaller-norm ridge columns, and
renormalising them changes (and, empirically, improves) predictions relative
to the wrapped head.

A Simplex ETF is *equinorm* and *equiangular* at once, and the two are
separately testable on the weight side even though C1 makes them jointly
inert on the target side. `mode` selects which half is imposed -- `column`
(norms only), `equiangular` (angles only), `etf_project` (both) -- so a
lockstep run can attribute an effect to one property rather than to "ETF
geometry" in the abstract.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from .base import AnalyticHead


_MODES = ("column", "procrustes", "equiangular", "etf_project")


def _etf_gram_sqrt(K: int, *, device: torch.device) -> Tensor:
    """`G^(1/2)` for the Simplex-ETF Gram `G = K/(K-1) (I - 11^T/K)`, shape (K, K).

    `I - 11^T/K` is a projector, so its square root is itself and
    `G^(1/2) = sqrt(K/(K-1)) (I - 11^T/K)` exactly -- no eigendecomposition
    needed. `G` has rank `K-1` with null vector `1`, which is what makes an
    ETF's columns sum to zero.
    """
    P = torch.eye(K, dtype=torch.float64, device=device) - 1.0 / K
    return math.sqrt(K / (K - 1)) * P


def _project_onto_etf(W: Tensor) -> Tensor:
    """The nearest Simplex ETF to `W` in Frobenius norm: `argmin_M ||W - M||_F`
    subject to `M^T M = G`, the Simplex-ETF Gram.

    Writing `M = P G^(1/2)` with `P^T P = I` (valid since `G` is PSD), the
    objective reduces to maximising `tr(P^T W G^(1/2))`, whose solution is the
    orthogonal polar factor of `A = W G^(1/2)`: if `A = U S V^T` then `P = U V^T`
    and `M = U V^T G^(1/2)`.

    `A` is rank-deficient by construction (`A 1 = 0`, since `G^(1/2)` centres the
    columns), so the singular vector for the zero singular value is arbitrary --
    but it is paired with `v = 1/sqrt(K)`, and `1^T G^(1/2) = 0` annihilates it,
    so `M` is still well defined. The result satisfies `M^T M = G` exactly:
    unit-norm columns, all pairwise cosines `-1/(K-1)`, columns summing to zero.
    """
    K = W.shape[1]
    if K < 2:
        
        return W / W.norm(dim=0, keepdim=True).clamp(min=1e-12)
    G_half = _etf_gram_sqrt(K, device=W.device)
    A = W @ G_half
    U, _, Vt = torch.linalg.svd(A, full_matrices=False)
    return (U @ Vt) @ G_half


class EquinormHead(AnalyticHead):
    """Wraps `inner: AnalyticHead` and post-processes its `W` non-linearly.

    A Simplex ETF is two properties at once -- *equinorm* columns and
    *equiangular* pairwise angles -- and the modes here separate them so a
    lockstep run can attribute any effect to one or the other:

      `column`      equinorm only: rescale each `W[:, c]` to unit norm; angles
                    are left exactly as fitted. This is the mode behind the
                    paper's C5 long-tail result.
      `equiangular` equiangular only: snap the *directions* to a Simplex ETF
                    (all pairwise cosines `-1/(K-1)`) but restore each column's
                    original norm afterwards, so the norm profile the fit
                    produced survives untouched.
      `etf_project` both: replace `W` with the nearest Simplex ETF to it, which
                    is equinorm and equiangular simultaneously.
      `procrustes`  solves `min_R ||W R - U||_F` over orthogonal `R` (with `U`
                    equal to `W` column-normalised), then snaps to unit norm.
                    Note this rotates in *class-index space*, so it mixes class
                    directions and leaves the Gram at `R^T G R` -- it does *not*
                    equalise angles. Predates the two modes above.

    Fitting is delegated unchanged to `inner`; only readout (`scores`, `W`) is
    post-processed.
    """

    def __init__(self, inner: AnalyticHead, *, mode: str = "column") -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown mode '{mode}', expected one of {_MODES}")
        self.inner = inner
        self.mode = mode
        self.d = inner.d
        self.ridge_lambda = inner.ridge_lambda

    
    @property
    def n_seen(self) -> int:  
        return self.inner.n_seen

    @n_seen.setter
    def n_seen(self, value: int) -> None:
        
        
        pass

    def observe_classes(self, new_classes: Sequence[int]) -> None:
        self.inner.observe_classes(new_classes)

    def _on_observe_classes(self, new_classes: Sequence[int]) -> None:  
        
        raise NotImplementedError

    def partial_fit(self, Phi: Tensor, y: Tensor, sample_weight: Tensor | None = None) -> None:
        self.inner.partial_fit(Phi, y, sample_weight=sample_weight)

    def _update_Q(self, *args, **kwargs) -> None:  
        raise NotImplementedError

    def _postprocess(self, W: Tensor) -> Tensor:
        if W.shape[1] == 0:
            return W
        if self.mode == "column":
            norms = W.norm(dim=0, keepdim=True).clamp(min=1e-12)
            return W / norms

        if self.mode == "etf_project":
            return _project_onto_etf(W)

        if self.mode == "equiangular":
            
            
            norms = W.norm(dim=0, keepdim=True).clamp(min=1e-12)
            return _project_onto_etf(W / norms) * norms

        
        U = W / W.norm(dim=0, keepdim=True).clamp(min=1e-12)
        M = W.T @ U  
        Us, _, Vt = torch.linalg.svd(M)
        R = Us @ Vt  
        W_rot = W @ R
        norms = W_rot.norm(dim=0, keepdim=True).clamp(min=1e-12)
        return W_rot / norms

    @property
    def W(self) -> Tensor:
        return self._postprocess(self.inner.W)

    def scores(self, Phi: Tensor) -> Tensor:
        Phi = Phi.to(torch.float64)
        return Phi @ self.W

    def predict(self, Phi: Tensor) -> Tensor:
        return self.scores(Phi).argmax(dim=1)

    def state_dict(self) -> dict:
        return {"inner": self.inner.state_dict(), "mode": self.mode}

    def load_state_dict(self, sd: dict) -> None:
        self.inner.load_state_dict(sd["inner"])
        self.mode = sd["mode"]
