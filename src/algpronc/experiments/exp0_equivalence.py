"""exp0_equivalence -- the paper's headline experiment.

Thm 1 (exactness): for a ridge head on frozen features, `etf_exact`'s scores
equal `onehot`'s scores rescaled by the ETF's Gram matrix, `s_etf_exact =
s_onehot @ G` with `G = U^T U = alpha*I + beta*11^T`, `alpha > 0` -- so argmax
(and therefore every classification decision) is identical. This experiment
fits `onehot` / `etf_exact` / `etf_stale` **in lockstep on the same real (or
synthetic) features** via `engine.run_cil`, then checks that claim directly:

    max |s_etf_exact - s_onehot @ G|            (expected: ~0, to 1e-10)
    argmax(s_etf_exact) == argmax(s_onehot)     (expected: agreement == 1.0)

Thm 2 (staleness hurts): `etf_stale` uses the original proposal's literal
Gram-Schmidt frame expansion (`geometry.frames.gs_expand`), which leaves earlier tasks'
`Q` blocks expressed in an outdated frame. Unlike `etf_exact` this is *not* a
re-expression of the one-hot solution, so its accuracy trajectory is expected
to be strictly worse than `onehot`'s at every task after the first frame
expansion. This experiment also dumps `frame_diagnostics` for both
`simplex_etf` and `gs_expand` at every task size, to make concrete *why*:
`gs_expand`'s frame is not an ETF (its Gram matrix's off-diagonal entries are
not constant, and its column sum is not zero).

Runs on real cached features by default, or with no dataset download at all
via `--set data.dataset=synthetic` (see `engine.synthetic_features`).
"""

from __future__ import annotations

import torch

from algpronc import engine
from algpronc.config import ExperimentConfig
from algpronc.experiments import build_heads, setup_experiment
from algpronc.utils.seed import child_seed


def _score_equivalence(cfg: ExperimentConfig, heads: dict, eval_X: torch.Tensor, k_final: int) -> dict:
    """max|s_etf_exact - s_onehot @ G| and argmax agreement, on one eval batch."""
    from algpronc.geometry.frames import simplex_etf

    s_onehot = heads["onehot"].scores(eval_X)
    s_etf_exact = heads["etf_exact"].scores(eval_X)

    m = engine.resolve_frame_m(cfg, k_final)
    seed = child_seed(cfg.seed, "etf_exact")
    U = simplex_etf(k_final, m, seed=seed)
    G = (U.T @ U).to(s_onehot.dtype)

    rescaled = s_onehot @ G
    diff = (s_etf_exact - rescaled).abs()
    max_abs_diff = float(diff.max().item())

    argmax_etf = s_etf_exact.argmax(dim=1)
    argmax_onehot = s_onehot.argmax(dim=1)
    agreement = float((argmax_etf == argmax_onehot).float().mean().item())
    return {"max_abs_score_diff": max_abs_diff, "argmax_agreement_rate": agreement}


def _frame_diagnostics_by_task_size(cfg: ExperimentConfig, task_splits: list[list[int]]) -> dict:
    """`frame_diagnostics` for `simplex_etf(K_t, m)` and the doc-faithful
    `gs_expand` progression, at every cumulative task size K_t. `gs_expand` is
    bootstrapped from `simplex_etf` at the first task (there is no U_prev to
    expand from yet) and then literally Gram-Schmidt-expanded at every
    subsequent task -- mirroring what `heads.etf_stale` does internally."""
    from algpronc.geometry.frames import frame_diagnostics, gs_expand, simplex_etf

    k_final = sum(len(t) for t in task_splits)
    m = engine.resolve_frame_m(cfg, k_final)
    seed = child_seed(cfg.seed, "frame_diag")

    out = {"simplex_etf": [], "gs_expand": []}
    U_gs = None
    k_cum = 0
    for t, classes_t in enumerate(task_splits):
        k_cum += len(classes_t)
        U_etf = simplex_etf(k_cum, m, seed=seed)
        out["simplex_etf"].append({"task": t, "K": k_cum, **frame_diagnostics(U_etf)})

        if U_gs is None:
            U_gs = simplex_etf(k_cum, m, seed=seed)
        else:
            U_gs = gs_expand(U_gs, len(classes_t), seed=seed)
        out["gs_expand"].append({"task": t, "K": k_cum, **frame_diagnostics(U_gs)})
    return out


def _accuracy_trajectory(summary: dict, name: str) -> list[float]:
    A = summary[name]["task_accuracy_matrix"]
    traj = []
    for i, row in enumerate(A):
        seen = [v for v in row[: i + 1] if v == v]  
        traj.append(sum(seen) / len(seen) if seen else float("nan"))
    return traj


def _markdown_table(summary: dict, equivalence: dict, traj_onehot: list[float], traj_stale: list[float]) -> str:
    lines = [
        "# exp0_equivalence",
        "",
        f"max |s_etf_exact - s_onehot @ G| = **{equivalence['max_abs_score_diff']:.3e}**",
        f"argmax agreement rate = **{equivalence['argmax_agreement_rate']:.6f}**",
        "",
        "| task | onehot acc | etf_exact acc | etf_stale acc | stale worse? |",
        "|---|---|---|---|---|",
    ]
    traj_exact = _accuracy_trajectory(summary, "etf_exact")
    for i, (a, b, c) in enumerate(zip(traj_onehot, traj_exact, traj_stale)):
        worse = "yes" if c < a - 1e-12 else "no"
        lines.append(f"| {i} | {a:.4f} | {b:.4f} | {c:.4f} | {worse} |")
    lines += [
        "",
        "| head | avg acc | avg incremental acc | forgetting |",
        "|---|---|---|---|",
    ]
    for name in ("onehot", "etf_exact", "etf_stale"):
        s = summary[name]
        lines.append(
            f"| {name} | {s['average_accuracy']:.4f} | {s['average_incremental_accuracy']:.4f} | {s['forgetting']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    cfg, logger, device = setup_experiment(argv)

    required = ["onehot", "etf_exact", "etf_stale"]
    if list(cfg.cl.heads) != required:
        logger.info(f"exp0_equivalence: overriding cl.heads {cfg.cl.heads!r} -> {required!r} (exp0 needs exactly these)")
        cfg.cl.heads = list(required)

    d = engine.resolve_head_dim(cfg)

    n_classes, _, _, test_X, test_y = engine.load_features(cfg)
    task_splits = engine.build_task_splits(cfg, n_classes)
    k_final = sum(len(t) for t in task_splits)

    heads = build_heads(cfg, d=d, k_final=k_final)
    summary = engine.run_cil(cfg, logger=logger, heads=heads)

    final_classes = sorted(c for t in task_splits for c in t)
    eval_mask = torch.isin(test_y, torch.as_tensor(final_classes, dtype=test_y.dtype))
    eval_X = test_X[eval_mask].to(torch.float64)
    if cfg.cl.rp.enabled:
        from algpronc.models import build_projection

        proj, _ = build_projection(cfg.cl.rp, test_X.shape[1])
        if proj is not None:
            eval_X = proj(test_X[eval_mask]).to(torch.float64)

    equivalence = _score_equivalence(cfg, heads, eval_X, k_final)
    frame_diag = _frame_diagnostics_by_task_size(cfg, task_splits)
    traj_onehot = _accuracy_trajectory(summary, "onehot")
    traj_stale = _accuracy_trajectory(summary, "etf_stale")

    full_summary = {
        "heads": summary,
        "equivalence": equivalence,
        "frame_diagnostics": frame_diag,
        "accuracy_trajectory": {
            "onehot": traj_onehot,
            "etf_exact": _accuracy_trajectory(summary, "etf_exact"),
            "etf_stale": traj_stale,
        },
        "etf_stale_strictly_worse": all(b <= a + 1e-12 for a, b in zip(traj_onehot, traj_stale)),
    }
    logger.write_json("summary.json", full_summary)
    logger.write_summary(full_summary)

    md = _markdown_table(summary, equivalence, traj_onehot, traj_stale)
    (logger.dir / "summary.md").write_text(md)

    logger.info(
        f"exp0_equivalence: max_abs_score_diff={equivalence['max_abs_score_diff']:.3e} "
        f"argmax_agreement_rate={equivalence['argmax_agreement_rate']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
