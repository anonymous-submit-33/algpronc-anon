"""exp3_imbalance -- long-tailed class-incremental streams. Does `equinorm`
recover tail-class accuracy that a plain analytic head loses to spectral
collapse under imbalance?

`cfg.cl.imbalance.enabled=true` makes `engine.run_cil` resample each task's
training features to an exponential-profile long-tailed budget (see
`data.splits.long_tail_counts`) before fitting -- identically for every head,
since imbalance is applied once per task before the lockstep fit loop.

Compares the head/tail accuracy split (`metrics.head_tail_accuracy`, via
`engine.run_cil`'s summary) between a baseline head and its `equinorm`-wrapped
counterpart. Which two heads are compared is `cfg.cl.heads` itself (default:
`["onehot", "equinorm:onehot"]` in `configs/cifar100_longtail.yaml` -- deliberately
*not* `etf_stale`, whose staleness collapse (see exp0/C3) would swamp the imbalance
effect this experiment exists to isolate).
"""

from __future__ import annotations

from algpronc import engine
from algpronc.experiments import build_heads, setup_experiment


def main(argv: list[str] | None = None) -> int:
    cfg, logger, device = setup_experiment(argv)

    if not cfg.cl.imbalance.enabled:
        logger.info("exp3_imbalance: cfg.cl.imbalance.enabled is False -- this run is a balanced control, not the long-tail claim")

    d = engine.resolve_head_dim(cfg)
    heads = build_heads(cfg, d=d)

    summary = engine.run_cil(cfg, logger=logger, heads=heads)
    logger.write_summary(summary)

    baseline = next((n for n in heads if "equinorm" not in n), None)
    equinorm_name = next((n for n in heads if "equinorm" in n), None)
    if baseline and equinorm_name:
        tail_base = summary[baseline]["head_tail_accuracy"]["tail"]
        tail_eq = summary[equinorm_name]["head_tail_accuracy"]["tail"]
        logger.info(
            f"exp3_imbalance: tail accuracy {baseline}={tail_base:.4f} "
            f"{equinorm_name}={tail_eq:.4f} (delta={tail_eq - tail_base:+.4f})"
        )

    for name in heads:
        s = summary[name]
        logger.info(f"exp3_imbalance[{name}]: avg_acc={s['average_accuracy']:.4f} head_tail={s['head_tail_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
