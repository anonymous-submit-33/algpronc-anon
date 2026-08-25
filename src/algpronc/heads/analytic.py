"""`OneHotHead`: the plain ACIL / GACL baseline.

Q grows by zero-padding then accumulating `Phi^T Y_onehot`; readout is the
ordinary ridge prediction `Phi @ P @ Q`. This is the reference every other
head in this package is compared against.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .base import AnalyticHead


class OneHotHead(AnalyticHead):
    """`Q <- [Q | 0] + Phi^T Y`, readout `Phi @ P @ Q`."""

    def __init__(self, d: int, ridge_lambda: float) -> None:
        super().__init__(d, ridge_lambda)
        self.Q = torch.zeros(self.d, 0, dtype=torch.float64)

    def _on_observe_classes(self, new_classes: Sequence[int]) -> None:
        K_prev = self.Q.shape[1]
        K_new = self.n_seen
        if K_new > K_prev:
            pad = torch.zeros(self.d, K_new - K_prev, dtype=torch.float64)
            self.Q = torch.cat([self.Q, pad], dim=1)

    def _update_Q(self, Phi, Phi_eff, y_local, sqrt_w) -> None:
        K = self.n_seen
        Y = self._onehot(y_local, K)  
        if sqrt_w is not None:
            Y = Y * sqrt_w
        self.Q = self.Q + Phi_eff.T @ Y

    def scores(self, Phi: Tensor) -> Tensor:
        Phi = Phi.to(torch.float64)
        return Phi @ self.P @ self.Q

    @property
    def W(self) -> Tensor:
        return self.P @ self.Q

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["Q"] = self.Q.clone()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.Q = sd["Q"].clone().to(torch.float64)
