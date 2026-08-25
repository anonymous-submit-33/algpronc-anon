"""`AnalyticHead`: the shared recursive-ridge machinery every head builds on.

All heads maintain one covariance inverse ``P_t = (Phi_{1:t}^T Phi_{1:t} + lambda I)^-1``
via the Sherman-Morrison-Woodbury (SMW) identity. They
differ only in what they accumulate as the cross-correlation `Q` and how they read
scores out of `(P, Q)`. Concrete heads implement `_on_observe_classes`, `_update_Q`,
`scores`, and the `W` property; `partial_fit`, the SMW recursion, and `predict` live
here so every head gets the exact same (tested) numerics.
"""

from __future__ import annotations

import abc
from typing import Sequence

import torch
from torch import Tensor


def _smw_update(P: Tensor, Phi_b: Tensor) -> Tensor:
    """One SMW step: P <- (P^-1 + Phi_b^T Phi_b)^-1, inverting whichever of
    (n x n) or (d x d) is smaller.

    `P` is the one explicit inverse this codebase maintains; every other
    inverse here goes through `torch.linalg.solve`, never `torch.linalg.inv`
    followed by a matmul.
    """
    n, d = Phi_b.shape
    if n == 0:
        return P
    if n <= d:
        
        PhiP = Phi_b @ P  
        S = torch.eye(n, dtype=torch.float64, device=P.device) + Phi_b @ PhiP.T
        S = (S + S.T) / 2
        sol = torch.linalg.solve(S, PhiP)  
        P_new = P - PhiP.T @ sol
    else:
        
        
        Id = torch.eye(d, dtype=torch.float64, device=P.device)
        R_prev = torch.linalg.solve(P, Id)
        R_new = R_prev + Phi_b.T @ Phi_b
        P_new = torch.linalg.solve(R_new, Id)
    return (P_new + P_new.T) / 2


class AnalyticHead(abc.ABC):
    """Base class for all closed-form ridge-regression heads.

    d: int                    input feature dim (post random projection)
    ridge_lambda: float
    n_seen: int                number of classes observed so far
    """

    def __init__(self, d: int, ridge_lambda: float) -> None:
        self.d = int(d)
        self.ridge_lambda = float(ridge_lambda)
        self.n_seen = 0
        self._classes: list[int] = []
        self._class_to_idx: dict[int, int] = {}
        self.P = torch.eye(self.d, dtype=torch.float64) / self.ridge_lambda

    
    def observe_classes(self, new_classes: Sequence[int]) -> None:
        """Called once per task, before `partial_fit`, with that task's novel labels."""
        added = False
        for c in new_classes:
            c = int(c)
            if c in self._class_to_idx:
                continue
            self._class_to_idx[c] = len(self._classes)
            self._classes.append(c)
            added = True
        self.n_seen = len(self._classes)
        if added:
            self._on_observe_classes(new_classes)

    @abc.abstractmethod
    def _on_observe_classes(self, new_classes: Sequence[int]) -> None:
        """Hook: grow whatever per-class state (Q columns, frames, ...) this head keeps."""

    def _local_labels(self, y: Tensor) -> Tensor:
        idx = [self._class_to_idx[int(c)] for c in y.tolist()]
        return torch.tensor(idx, dtype=torch.int64)

    @staticmethod
    def _onehot(y_local: Tensor, K: int) -> Tensor:
        n = y_local.shape[0]
        Y = torch.zeros(n, K, dtype=torch.float64)
        if n > 0:
            Y.scatter_(1, y_local.view(-1, 1), 1.0)
        return Y

    
    def partial_fit(self, Phi: Tensor, y: Tensor, sample_weight: Tensor | None = None) -> None:
        """Phi (N, d), y (N,) global class ids. Casts to float64 on entry
        (features may arrive float32, head state never is). Exactly
        batch-size invariant: the SMW recursion is an
        associative accumulation of Phi^T Phi and Phi^T Y, so splitting one
        call into several smaller ones changes nothing but float rounding.

        `sample_weight` (optional, N,) implements weighted ridge regression by
        pre-scaling both the feature rows fed into the P update and the target
        rows fed into each head's own `_update_Q`, by `sqrt(w)`: since
        weighted ridge minimises `sum_i w_i ||phi_i^T W - y_i||^2 + lambda||W||^2
        == ||diag(sqrt(w))(Phi W - Y)||_F^2 + lambda||W||^2`, scaling both
        sides by `sqrt(w)` reduces it to ordinary (unweighted) ridge.
        """
        Phi = Phi.to(torch.float64)
        y = y.to(torch.int64)
        y_local = self._local_labels(y)

        if sample_weight is not None:
            w = sample_weight.to(torch.float64).view(-1)
            sqrt_w = w.clamp(min=0).sqrt().view(-1, 1)
            Phi_eff = Phi * sqrt_w
        else:
            sqrt_w = None
            Phi_eff = Phi

        self.P = _smw_update(self.P, Phi_eff)
        self._update_Q(Phi, Phi_eff, y_local, sqrt_w)

    @abc.abstractmethod
    def _update_Q(
        self,
        Phi: Tensor,
        Phi_eff: Tensor,
        y_local: Tensor,
        sqrt_w: Tensor | None,
    ) -> None:
        """Hook: accumulate this head's cross-correlation term(s)."""

    
    @abc.abstractmethod
    def scores(self, Phi: Tensor) -> Tensor:  
        ...

    def predict(self, Phi: Tensor) -> Tensor:  
        return self.scores(Phi).argmax(dim=1)

    @property
    @abc.abstractmethod
    def W(self) -> Tensor:  
        ...

    
    def state_dict(self) -> dict:
        return {
            "d": self.d,
            "ridge_lambda": self.ridge_lambda,
            "n_seen": self.n_seen,
            "classes": list(self._classes),
            "P": self.P.clone(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.d = sd["d"]
        self.ridge_lambda = sd["ridge_lambda"]
        self.n_seen = sd["n_seen"]
        self._classes = list(sd["classes"])
        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}
        self.P = sd["P"].clone().to(torch.float64)


class EvolvingFrameHead(AnalyticHead):
    """Shared machinery for heads that target `y -> U_Kt[:, y]` for an evolving
    (K-growing) frame `U_Kt` in R^{m x K}, accumulating Q *directly* in the
    m-dimensional frame space rather than retaining the one-hot `Q_oh`.

    This is what gives `etf_stale` and `compressed` their constant, O(d*m)
    memory footprint regardless of how many classes have been seen -- and it
    is also exactly what makes them "stale": each task's contribution to `Q`
    is baked in using whatever frame `U_Kt` was current *at the time it was
    fit*. When a later task grows the frame again, earlier contributions are
    not (and, without retaining raw data or `Q_oh`, cannot cheaply be)
    re-expressed in the new frame. `heads.etf_exact` avoids this by keeping
    `Q_oh` and re-projecting at readout instead (see heads/etf.py) -- at the
    cost of O(d*K) memory. Subclasses only need to implement `_next_frame`,
    which decides how the frame grows from task to task (a literal
    `geometry.gs_expand` for `etf_stale`, a fresh regeneration from a chosen
    frame family for `compressed`).
    """

    def __init__(self, d: int, ridge_lambda: float, m: int) -> None:
        super().__init__(d, ridge_lambda)
        self.m = int(m)
        self.U = torch.zeros(self.m, 0, dtype=torch.float64)  
        self.Q = torch.zeros(self.d, self.m, dtype=torch.float64)  

    @abc.abstractmethod
    def _next_frame(self, K_prev: int, K_new: int, U_prev: Tensor) -> Tensor:
        """Return the new (m, K_new) frame given the previous (m, K_prev) one."""

    def _on_observe_classes(self, new_classes: Sequence[int]) -> None:
        K_prev = self.U.shape[1]
        K_new = self.n_seen
        if K_new > K_prev:
            self.U = self._next_frame(K_prev, K_new, self.U)

    def _update_Q(self, Phi, Phi_eff, y_local, sqrt_w) -> None:
        K = self.n_seen
        Y = self._onehot(y_local, K)  
        if sqrt_w is not None:
            Y = Y * sqrt_w
        E = Y @ self.U.T  
        self.Q = self.Q + Phi_eff.T @ E  

    def scores(self, Phi: Tensor) -> Tensor:
        Phi = Phi.to(torch.float64)
        return (Phi @ self.P @ self.Q) @ self.U  

    @property
    def W(self) -> Tensor:
        return (self.P @ self.Q) @ self.U  

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["m"] = self.m
        sd["U"] = self.U.clone()
        sd["Q"] = self.Q.clone()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.m = sd["m"]
        self.U = sd["U"].clone().to(torch.float64)
        self.Q = sd["Q"].clone().to(torch.float64)
