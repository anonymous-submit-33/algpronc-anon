"""`ETFExactHead` (retro-corrected Alg-ProNC) and `ETFStaleHead` (the original
proposal's literal algorithm).

`ETFExactHead` is expected to match `OneHotHead` bit-for-bit in argmax: a test
failing that is a bug in this code, not a discovery. `ETFStaleHead` is
expected to *deviate* from it once more than one task has been seen -- that
deviation is the paper's actual finding about the original proposal, not a
bug either. Do not "fix" either result.
"""

from __future__ import annotations

from torch import Tensor

from ..geometry.frames import gs_expand, simplex_etf
from .analytic import OneHotHead
from .base import EvolvingFrameHead


class ETFExactHead(OneHotHead):
    """Retro-corrected Alg-ProNC: exact algebraic equivalent of `onehot`.

    `etf_exact` does NOT store past raw data, and it does not even store past
    target blocks in a fixed frame: it keeps exactly the same one-hot Q_oh
    that `OneHotHead` accumulates (inherited unchanged from `OneHotHead`) and
    re-expresses it in the *current* frame only at readout time:
    `Q_etf = Q_oh @ U_{K_t}^T`. Because this re-expression is one (d,K)x(K,m)
    matmul done on demand, the retro-correction that fixes `etf_stale`'s
    staleness bug is *free*: no extra state, no revisiting past batches.

    `U_{K_t}` is `simplex_etf(K_t, m)` regenerated fresh at each call from a
    fixed seed.
    """

    def __init__(self, d: int, ridge_lambda: float, m: int, *, seed: int = 0) -> None:
        super().__init__(d, ridge_lambda)
        self.m = int(m)
        self.seed = int(seed)

    def _current_frame(self) -> Tensor:
        K = self.n_seen
        return simplex_etf(K, self.m, seed=self.seed)

    def scores(self, Phi: Tensor) -> Tensor:
        Phi = Phi.to(self.Q.dtype)
        U = self._current_frame()  
        Q_etf = self.Q @ U.T  
        return (Phi @ self.P @ Q_etf) @ U  

    @property
    def W(self) -> Tensor:
        U = self._current_frame()
        Q_etf = self.Q @ U.T
        return (self.P @ Q_etf) @ U

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["m"] = self.m
        sd["seed"] = self.seed
        return sd

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.m = sd["m"]
        self.seed = sd["seed"]


class ETFStaleHead(EvolvingFrameHead):
    """The original proposal's literal algorithm (Alg-ProNC as originally
    proposed).

    Uses `geometry.gs_expand` (the original construction's flawed
    Gram-Schmidt expansion) to grow the frame task over task, and accumulates
    `Q` directly in the m-dim frame space (`EvolvingFrameHead`), so past
    tasks' contributions are left expressed in whatever frame was current
    when they were fit. This is NOT retro-corrected, unlike `ETFExactHead`:
    it is exactly the "staleness" artifact that separates the literal
    algorithm from ACIL, and it measurably *hurts* accuracy once more than
    one task has been seen. This is a deliberate object of study -- do not
    "fix" it.
    """

    def __init__(self, d: int, ridge_lambda: float, m: int, *, seed: int = 0) -> None:
        super().__init__(d, ridge_lambda, m)
        self.seed = int(seed)

    def _next_frame(self, K_prev: int, K_new: int, U_prev: Tensor) -> Tensor:
        if K_prev == 0:
            return simplex_etf(K_new, self.m, seed=self.seed)
        return gs_expand(U_prev, K_new - K_prev, seed=self.seed)

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["seed"] = self.seed
        return sd

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.seed = sd["seed"]
