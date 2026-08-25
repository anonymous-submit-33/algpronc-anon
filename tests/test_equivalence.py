"""The headline test the paper rests on: on a synthetic Gaussian
class-conditional multi-task stream,

  * `etf_exact` and `onehot` produce IDENTICAL argmax predictions, and their
    score matrices satisfy `s_etf == s_onehot @ G` with `G = U^T U`, to
    max abs error < 1e-9 -- for several `m`, several `lambda`, and under
    per-sample weighting;
  * `etf_stale` DIFFERS (strictly lower accuracy) once more than one task has
    been seen -- that deviation is a *result*, not a failure;
  * the recursive SMW `P` matches a direct batch ridge solve.

A failure of the first two bullets is a bug in this code, not a discovery;
a failure of the third bullet (etf_stale NOT differing) would mean the
staleness artifact isn't being reproduced faithfully.
"""

from __future__ import annotations

import torch

from algpronc.geometry.frames import simplex_etf
from algpronc.heads import ETFExactHead, ETFStaleHead, OneHotHead


def _make_class_incremental_stream(
    d: int,
    n_tasks: int,
    classes_per_task: int,
    n_per_class: int,
    *,
    seed: int,
    class_scale: float = 2.5,
    noise_scale: float = 1.0,
):
    """Synthetic Gaussian class-conditional features over a task stream.
    Returns (tasks, K_total) where tasks is a list of (classes, Phi, y)."""
    g = torch.Generator().manual_seed(seed)
    K_total = n_tasks * classes_per_task
    means = torch.randn(K_total, d, generator=g, dtype=torch.float64) * class_scale
    tasks = []
    for t in range(n_tasks):
        classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        Phis, ys = [], []
        for c in classes:
            X = means[c] + noise_scale * torch.randn(n_per_class, d, generator=g, dtype=torch.float64)
            Phis.append(X)
            ys.append(torch.full((n_per_class,), c, dtype=torch.int64))
        Phi_t = torch.cat(Phis, dim=0)
        y_t = torch.cat(ys, dim=0)
        perm = torch.randperm(Phi_t.shape[0], generator=g)
        tasks.append((classes, Phi_t[perm], y_t[perm]))
    return tasks, K_total, means


def _eval_set(means: torch.Tensor, K: int, n_per_class: int, *, seed: int, noise_scale: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    d = means.shape[1]
    Phis, ys = [], []
    for c in range(K):
        X = means[c] + noise_scale * torch.randn(n_per_class, d, generator=g, dtype=torch.float64)
        Phis.append(X)
        ys.append(torch.full((n_per_class,), c, dtype=torch.int64))
    return torch.cat(Phis, dim=0), torch.cat(ys, dim=0)


def test_etf_exact_matches_onehot_across_m_and_lambda():
    d = 24
    n_tasks, classes_per_task, n_per_class = 3, 4, 30
    K_total = n_tasks * classes_per_task
    tasks, _, means = _make_class_incremental_stream(
        d, n_tasks, classes_per_task, n_per_class, seed=0
    )
    Phi_eval, y_eval = _eval_set(means, K_total, 25, seed=1000)

    ms = sorted({K_total - 1, K_total, 2 * K_total, d})
    lambdas = [1e-4, 1e-2, 1.0, 1e2]
    etf_seed = 7

    max_abs_diff_overall = 0.0
    for m in ms:
        for ridge_lambda in lambdas:
            oh = OneHotHead(d, ridge_lambda=ridge_lambda)
            ex = ETFExactHead(d, ridge_lambda=ridge_lambda, m=m, seed=etf_seed)
            for classes, Phi_t, y_t in tasks:
                oh.observe_classes(classes)
                ex.observe_classes(classes)
                oh.partial_fit(Phi_t, y_t)
                ex.partial_fit(Phi_t, y_t)

            s_oh = oh.scores(Phi_eval)
            s_ex = ex.scores(Phi_eval)

            pred_oh = s_oh.argmax(dim=1)
            pred_ex = s_ex.argmax(dim=1)
            assert torch.equal(pred_oh, pred_ex), (m, ridge_lambda)

            U = simplex_etf(K_total, m, seed=etf_seed)
            G = U.T @ U
            diff = (s_ex - s_oh @ G).abs().max().item()
            max_abs_diff_overall = max(max_abs_diff_overall, diff)
            assert diff < 1e-9, (m, ridge_lambda, diff)

    print(f"[test_etf_exact_matches_onehot_across_m_and_lambda] "
          f"max|s_etf - s_onehot @ G| over all (m, lambda) = {max_abs_diff_overall:.3e}")


def test_etf_exact_matches_onehot_under_per_sample_weighting():
    d = 20
    n_tasks, classes_per_task, n_per_class = 3, 3, 25
    K_total = n_tasks * classes_per_task
    tasks, _, means = _make_class_incremental_stream(
        d, n_tasks, classes_per_task, n_per_class, seed=2
    )
    Phi_eval, y_eval = _eval_set(means, K_total, 20, seed=2000)

    m = d
    ridge_lambda = 1e-2
    etf_seed = 11
    g = torch.Generator().manual_seed(123)

    oh = OneHotHead(d, ridge_lambda=ridge_lambda)
    ex = ETFExactHead(d, ridge_lambda=ridge_lambda, m=m, seed=etf_seed)
    for classes, Phi_t, y_t in tasks:
        oh.observe_classes(classes)
        ex.observe_classes(classes)
        w = torch.rand(Phi_t.shape[0], generator=g, dtype=torch.float64) * 0.9 + 0.1
        oh.partial_fit(Phi_t, y_t, sample_weight=w)
        ex.partial_fit(Phi_t, y_t, sample_weight=w)

    s_oh = oh.scores(Phi_eval)
    s_ex = ex.scores(Phi_eval)
    assert torch.equal(s_oh.argmax(dim=1), s_ex.argmax(dim=1))

    U = simplex_etf(K_total, m, seed=etf_seed)
    G = U.T @ U
    diff = (s_ex - s_oh @ G).abs().max().item()
    assert diff < 1e-9, diff
    print(f"[test_etf_exact_matches_onehot_under_per_sample_weighting] max abs diff = {diff:.3e}")


def test_etf_stale_deviates_and_hurts_accuracy_after_multiple_tasks():
    d = 24
    n_tasks, classes_per_task, n_per_class = 4, 4, 40
    K_total = n_tasks * classes_per_task
    tasks, _, means = _make_class_incremental_stream(
        d, n_tasks, classes_per_task, n_per_class, seed=3
    )
    Phi_eval, y_eval = _eval_set(means, K_total, 40, seed=3000)

    ridge_lambda = 1e-2
    m = d
    seed = 5

    oh = OneHotHead(d, ridge_lambda=ridge_lambda)
    st = ETFStaleHead(d, ridge_lambda=ridge_lambda, m=m, seed=seed)

    accs_oh, accs_st = [], []
    for t, (classes, Phi_t, y_t) in enumerate(tasks):
        oh.observe_classes(classes)
        st.observe_classes(classes)
        oh.partial_fit(Phi_t, y_t)
        st.partial_fit(Phi_t, y_t)

        K_seen = oh.n_seen
        mask = y_eval < K_seen
        acc_oh = (oh.predict(Phi_eval[mask]) == y_eval[mask]).float().mean().item()
        acc_st = (st.predict(Phi_eval[mask]) == y_eval[mask]).float().mean().item()
        accs_oh.append(acc_oh)
        accs_st.append(acc_st)

        if t == 0:
            
            
            assert abs(acc_oh - acc_st) < 1e-6, (acc_oh, acc_st)
        else:
            
            
            assert acc_st < acc_oh - 1e-6, (t, acc_oh, acc_st)

    print(f"[test_etf_stale_deviates_and_hurts_accuracy_after_multiple_tasks] "
          f"per-task accuracy onehot={accs_oh} etf_stale={accs_st}")


def test_recursive_smw_P_matches_direct_solve_across_the_full_stream():
    d = 24
    n_tasks, classes_per_task, n_per_class = 3, 4, 30
    tasks, K_total, _ = _make_class_incremental_stream(d, n_tasks, classes_per_task, n_per_class, seed=4)

    ridge_lambda = 0.05
    head = OneHotHead(d, ridge_lambda=ridge_lambda)
    all_Phi = []
    for classes, Phi_t, y_t in tasks:
        head.observe_classes(classes)
        head.partial_fit(Phi_t, y_t)
        all_Phi.append(Phi_t)
    Phi_all = torch.cat(all_Phi, dim=0)

    R = Phi_all.T @ Phi_all + ridge_lambda * torch.eye(d, dtype=torch.float64)
    P_direct = torch.linalg.solve(R, torch.eye(d, dtype=torch.float64))

    max_abs_diff = (head.P - P_direct).abs().max().item()
    assert max_abs_diff < 1e-9, max_abs_diff
    print(f"[test_recursive_smw_P_matches_direct_solve_across_the_full_stream] "
          f"max|P_recursive - P_direct| = {max_abs_diff:.3e}")
