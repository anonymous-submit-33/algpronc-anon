"""RanPAC-style frozen random projection.

`x -> act(x @ Wr)`, `Wr ~ N(0, 1/d_in)` of shape `(d_in, proj_dim)`, fixed by a
seed, no bias. The whole point is that fitting and evaluation see the *exact
same* `Wr` -- so `Wr` is generated once from `seed` with a private CPU
`torch.Generator`, never from the module-global RNG, and is a pure function
of `(seed, d_in, proj_dim)`.

Note on the return type: this returns `(callable_or_None, output_dim)`
rather than a bare `Callable | None`, since callers otherwise have no way to
learn `proj_dim` (or that it falls back to `d_in` when the projection is
disabled) without reaching back into `cfg_rp`. Downstream code (`engine.py`,
`heads/`) should unpack the pair.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
}


class RandomProjection:
    """Callable frozen projection `x -> act(x @ Wr)`.

    `Wr` is materialised once at construction time (float64, CPU) from a
    private generator seeded with `seed`, so two `RandomProjection`s built
    with the same `(d_in, proj_dim, seed)` are identical -- this is what
    lets "fit" and "eval" agree. At call time `Wr` is cast to the input's
    dtype/device (never the other way around), so float64 heads keep full
    precision while an eval pass on float32 features still works.
    """

    def __init__(self, d_in: int, proj_dim: int, *, seed: int = 0, activation: str = "relu") -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'; supported: {sorted(_ACTIVATIONS)}")
        self.d_in = d_in
        self.proj_dim = proj_dim
        self.seed = seed
        self.activation = activation
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        self.Wr = torch.randn(d_in, proj_dim, generator=generator, dtype=torch.float64) / math.sqrt(d_in)
        self._act = _ACTIVATIONS[activation]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_in:
            raise ValueError(f"expected last dim {self.d_in}, got {x.shape[-1]}")
        Wr = self.Wr.to(device=x.device, dtype=x.dtype)
        return self._act(x @ Wr)

    def __repr__(self) -> str:  
        return (
            f"RandomProjection(d_in={self.d_in}, proj_dim={self.proj_dim}, "
            f"seed={self.seed}, activation='{self.activation}')"
        )


def build_projection(cfg_rp: Any, d_in: int) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, int]:
    """Build the frozen random projection described by `cfg_rp` (a
    `RandomProjectionConfig`-shaped object: `.enabled`, `.proj_dim`,
    `.activation`, `.seed`).

    Returns:
        `(None, d_in)` if `cfg_rp.enabled` is falsy (identity: no projection).
        `(RandomProjection(...), cfg_rp.proj_dim)` otherwise.
    """
    if not getattr(cfg_rp, "enabled", False):
        return None, d_in
    proj = RandomProjection(
        d_in,
        cfg_rp.proj_dim,
        seed=cfg_rp.seed,
        activation=cfg_rp.activation,
    )
    return proj, cfg_rp.proj_dim
