"""exp1_main -- the main CIL benchmark table (CIFAR-100 T=10/20, TinyImageNet T=10).

Fits every head in `cfg.cl.heads` in lockstep via `engine.run_cil` and
reports average accuracy, average incremental accuracy, and forgetting per
head -- the standard CIL scoreboard.
"""

from __future__ import annotations

from algpronc import engine
from algpronc.experiments import build_heads, setup_experiment


def main(argv: list[str] | None = None) -> int:
    cfg, logger, device = setup_experiment(argv)

    d = engine.resolve_head_dim(cfg)
    heads = build_heads(cfg, d=d)

    summary = engine.run_cil(cfg, logger=logger, heads=heads)
    logger.write_summary(summary)

    for name in cfg.cl.heads:
        s = summary[name]
        logger.info(
            f"exp1_main[{name}]: avg_acc={s['average_accuracy']:.4f} "
            f"avg_incremental_acc={s['average_incremental_accuracy']:.4f} "
            f"forgetting={s['forgetting']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
