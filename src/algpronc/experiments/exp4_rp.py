"""exp4_rp -- random-projection dimension sweep, with and without the ETF head.

Unlike exp2's frame sweep, the random projection changes the *feature space*
itself (`x -> act(x @ Wr)`, `Wr` of shape `(d, proj_dim)`), so different
`proj_dim`s cannot share one lockstep pass -- each is a genuinely different
feature stream. Within one `proj_dim`, though, `onehot` and `etf_exact` (or
whichever heads `cfg.cl.heads` names) *are* fit in lockstep, exactly like
every other experiment.

Sweep points come from `cfg.exp4.proj_dims` (default `[0, 1000, 5000, 10000]`;
`0` means "no projection", i.e. `cl.rp.enabled=False`). Each point gets its
own sub-run (`<run_id>__dim<K>`) so `engine.run_cil`'s fixed per-task logging
schema (task, head, accuracy) doesn't need a third axis; this experiment's own
`summary.json` aggregates across dims.
"""

from __future__ import annotations

import copy

from algpronc import engine
from algpronc.config import to_dict
from algpronc.experiments import build_heads, setup_experiment
from algpronc.utils.logging import RunLogger


def main(argv: list[str] | None = None) -> int:
    cfg, logger, device = setup_experiment(argv, run_id_suffix="__exp4_rp")

    proj_dims = cfg.exp4.get("proj_dims", [0, 1000, 5000, 10000])
    by_dim = {}
    for dim in proj_dims:
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.run_id = f"{cfg.run_id}__dim{dim}"
        if dim <= 0:
            sub_cfg.cl.rp.enabled = False
        else:
            sub_cfg.cl.rp.enabled = True
            sub_cfg.cl.rp.proj_dim = dim

        sub_logger = RunLogger(sub_cfg)
        sub_logger.dump_config(to_dict(sub_cfg))

        d = engine.resolve_head_dim(sub_cfg)
        heads = build_heads(sub_cfg, d=d)
        summary = engine.run_cil(sub_cfg, logger=sub_logger, heads=heads)
        sub_logger.write_summary(summary)

        by_dim[str(dim)] = {name: summary[name]["average_accuracy"] for name in heads}
        logger.info(f"exp4_rp[dim={dim}]: {by_dim[str(dim)]}")

    full_summary = {"proj_dims": proj_dims, "average_accuracy_by_dim": by_dim}
    logger.write_summary(full_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
