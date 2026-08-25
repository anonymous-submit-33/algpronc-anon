"""`CompressedHead`: a constant-memory head for the compressed regime `m < K-1`.

One-hot targets do not exist below `m = K-1` (there is no way to embed `K`
mutually-equidistant points in fewer than `K-1` dimensions), so the
onehot/etf_exact invariance theorem has nothing to apply to here -- this is
one of the two places (`heads.equinorm` is the other) where the paper's
positive results live.
"""

from __future__ import annotations

from typing import Callable

from torch import Tensor

from ..geometry.frames import random_frame, truncated_etf, welch_frame
from .base import EvolvingFrameHead

_FRAME_BUILDERS: dict[str, Callable[..., Tensor]] = {
    "welch": welch_frame,
    "random": random_frame,
    "etf": truncated_etf,
}


class CompressedHead(EvolvingFrameHead):
    """Like `etf_stale`, accumulates Q directly in the m-dim frame space (so
    memory is O(d*m), constant in the number of classes seen) rather than
    retaining a one-hot Q_oh -- so it necessarily inherits the same kind of
    staleness artifact when the frame changes across tasks (there is no
    (d,K)-sized Q_oh to cheaply re-project from, the way `etf_exact` does).

    `frame` selects the target-frame family used at each task, regenerated
    fresh (not incrementally expanded) from `K_t` and a fixed seed:
      - 'welch'  `geometry.welch_frame`     (near-Welch-bound compressed frame)
      - 'random' `geometry.random_frame`    (i.i.d. Gaussian control)
      - 'etf'    `geometry.truncated_etf`   (the exact ETF, built in K_t-1 dims
                 and projected down to `proj_m` -- meaningful at any `proj_m`,
                 including the compressed regime `proj_m < K_t - 1`)
    """

    def __init__(
        self,
        d: int,
        ridge_lambda: float,
        m: int,
        *,
        frame: str = "welch",
        seed: int = 0,
    ) -> None:
        super().__init__(d, ridge_lambda, m)
        if frame not in _FRAME_BUILDERS:
            raise ValueError(f"unknown frame '{frame}', expected one of {sorted(_FRAME_BUILDERS)}")
        self.frame = frame
        self.seed = int(seed)

    def _next_frame(self, K_prev: int, K_new: int, U_prev: Tensor) -> Tensor:
        builder = _FRAME_BUILDERS[self.frame]
        return builder(K_new, self.m, seed=self.seed)

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["frame"] = self.frame
        sd["seed"] = self.seed
        return sd

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.frame = sd["frame"]
        self.seed = sd["seed"]
