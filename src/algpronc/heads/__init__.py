"""Analytic (closed-form) ridge-regression heads.

All heads share one recursive covariance inverse `P`, updated via
Sherman-Morrison-Woodbury (see `heads.base.AnalyticHead`), and differ only in
what they accumulate as the cross-correlation `Q` and how they read scores
out of `(P, Q)`:

  onehot       ACIL / GACL baseline: plain one-hot targets.
  etf_exact    retro-corrected Alg-ProNC; provably matches `onehot`'s argmax.
  etf_stale    the original proposal's literal (flawed) algorithm; deviates
               from `onehot` once more than one task has been seen, and
               *hurts*.
  compressed   constant-memory head for `m < K-1` (no one-hot target exists).
  equinorm     wraps any of the above with a non-linear W post-process.

`build_head(cl_cfg, d)` is the single dispatch point. `CLConfig`
(src/algpronc/config.py) does not yet carry a dedicated "which head / which
frame" sub-schema, so every knob here is
read defensively with `getattr(cl_cfg, name, default)`: anything missing
falls back to the plain ACIL baseline. This lets `cl_cfg` be either a real
`CLConfig` or any duck-typed config object (e.g. from experiments/tests)
exposing the same optional attributes:

  cl_cfg.ridge_lambda      float, default 1e-2
  cl_cfg.etf.seed          int, default 0 (or cl_cfg.seed as a fallback)
  cl_cfg.head              str, one of the names above, default 'onehot'
  cl_cfg.proj_m            int, frame width `m`, default `d`
  cl_cfg.frame             str, 'welch' | 'random' | 'etf', default 'welch'
                           (only consulted when cl_cfg.head == 'compressed')
  cl_cfg.equinorm_inner    str, head name to wrap, default 'onehot'
                           (only consulted when cl_cfg.head == 'equinorm')
  cl_cfg.equinorm_mode     str, 'column' | 'procrustes', default 'column'
"""

from __future__ import annotations

from typing import Any

from .analytic import OneHotHead
from .base import AnalyticHead, EvolvingFrameHead
from .compressed import CompressedHead
from .equinorm import EquinormHead
from .etf import ETFExactHead, ETFStaleHead

__all__ = [
    "AnalyticHead",
    "EvolvingFrameHead",
    "OneHotHead",
    "ETFExactHead",
    "ETFStaleHead",
    "CompressedHead",
    "EquinormHead",
    "build_head",
]

_SIMPLE_HEADS = ("onehot", "etf_exact", "etf_stale", "compressed")


def _build_simple(name: str, cl_cfg: Any, d: int, ridge_lambda: float, seed: int) -> AnalyticHead:
    if name == "onehot":
        return OneHotHead(d, ridge_lambda)
    if name in ("etf_exact", "etf_stale", "compressed"):
        m = int(getattr(cl_cfg, "proj_m", d) or d)
        if name == "etf_exact":
            return ETFExactHead(d, ridge_lambda, m, seed=seed)
        if name == "etf_stale":
            return ETFStaleHead(d, ridge_lambda, m, seed=seed)
        frame = str(getattr(cl_cfg, "frame", "welch"))
        return CompressedHead(d, ridge_lambda, m, frame=frame, seed=seed)
    raise ValueError(f"unknown head '{name}', expected one of {_SIMPLE_HEADS + ('equinorm',)}")


def build_head(cl_cfg: Any, d: int) -> AnalyticHead:
    """Construct one `AnalyticHead` from a `CLConfig`-like object. See the
    module docstring for which (optional) attributes are consulted."""
    ridge_lambda = float(getattr(cl_cfg, "ridge_lambda", 1e-2))
    etf_cfg = getattr(cl_cfg, "etf", None)
    if etf_cfg is not None and hasattr(etf_cfg, "seed"):
        seed = int(etf_cfg.seed)
    else:
        seed = int(getattr(cl_cfg, "seed", 0))
    head_name = str(getattr(cl_cfg, "head", "onehot")).lower()

    
    if head_name in ("equinorm", "equiangular", "etf_project"):
        inner_name = str(getattr(cl_cfg, "equinorm_inner", "onehot")).lower()
        if head_name == "equinorm":
            mode = str(getattr(cl_cfg, "equinorm_mode", "column")).lower()
        else:
            mode = head_name
        inner = _build_simple(inner_name, cl_cfg, d, ridge_lambda, seed)
        return EquinormHead(inner, mode=mode)
    return _build_simple(head_name, cl_cfg, d, ridge_lambda, seed)
