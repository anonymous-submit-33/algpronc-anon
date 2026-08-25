"""exp2_compressed -- sweep `proj_m` below `K-1`, where one-hot targets don't
even exist (a one-hot target matrix needs one column per class; a compressed
frame with `m < K-1` has no such counterpart). Compares `welch` vs `random`
vs truncated-`etf` compressed frames.

All sweep points are fit **in lockstep in a single `engine.run_cil` call**:
they differ only in the frame passed to the `compressed` head, so they can
(and must, per the lockstep design in `engine.py`) share one pass over the
identical feature stream.

Sweep points come from `cfg.exp2`:
    m_values: list[int]   -- default [4, 8, 16, 32]
    frames:   list[str]   -- default ["welch", "random", "etf"]
"""

from __future__ import annotations

import types

from algpronc import engine
from algpronc.experiments import setup_experiment
from algpronc.utils.seed import child_seed


def _build_sweep_heads(cfg, *, d: int) -> dict:
    """One `compressed` head per (frame, m) sweep point. `algpronc.heads.build_head`
    takes a flat, duck-typed `CLConfig`-shaped view (`.ridge_lambda`, `.etf.seed`,
    `.head`, `.proj_m`, `.frame`) and one head name per call -- see
    `experiments.build_heads`'s docstring for the full adapter rationale."""
    from algpronc.heads import build_head

    m_values = cfg.exp2.get("m_values", [4, 8, 16, 32])
    frames = cfg.exp2.get("frames", ["welch", "random", "etf"])

    heads = {}
    for m in m_values:
        for frame in frames:
            name = f"{frame}_m{m}"
            view = types.SimpleNamespace(
                ridge_lambda=cfg.cl.ridge_lambda,
                etf=types.SimpleNamespace(seed=child_seed(cfg.seed, name)),
                head="compressed",
                proj_m=m,
                frame=frame,
            )
            heads[name] = build_head(view, d)
    return heads


def main(argv: list[str] | None = None) -> int:
    cfg, logger, device = setup_experiment(argv, run_id_suffix="__exp2_compressed")

    d = engine.resolve_head_dim(cfg)
    heads = _build_sweep_heads(cfg, d=d)
    
    
    cfg.cl.heads = list(heads)

    summary = engine.run_cil(cfg, logger=logger, heads=heads)
    logger.write_summary(summary)

    for name in heads:
        s = summary[name]
        logger.info(f"exp2_compressed[{name}]: avg_acc={s['average_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
