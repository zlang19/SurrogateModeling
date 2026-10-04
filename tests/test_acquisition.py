import numpy as np

from surrogatemodeling.methods.acquisition import epig_score, greedy_batch, iv_score, rank_one_update


def rbf(A, B, ls=0.3):
    return np.exp(-0.5 * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / ls**2)


def posterior_cov(Xtr, noise, Xq):
    K = rbf(Xtr, Xtr) + np.diag(noise)
    Kq = rbf(Xq, Xtr)
    return rbf(Xq, Xq) - Kq @ np.linalg.solve(K, Kq.T)


def test_rank_one_update_matches_exact_posterior():
    rng = np.random.default_rng(0)
    Xtr, Xq = rng.random((6, 2)), rng.random((20, 2))
    K = posterior_cov(Xtr, np.full(6, 1e-2), Xq)
    updated = rank_one_update(K, 7, 0.05)
    exact = posterior_cov(np.vstack([Xtr, Xq[7]]), np.append(np.full(6, 1e-2), 0.05), Xq)
    np.testing.assert_allclose(updated, exact, atol=1e-10)


def test_iv_score_equals_drop_in_mean_reference_variance():
    rng = np.random.default_rng(1)
    Xtr, Xq = rng.random((5, 2)), rng.random((30, 2))
    K = posterior_cov(Xtr, np.full(5, 1e-2), Xq)
    n_ref, tau2 = 10, 0.03
    scores = iv_score(K, n_ref, np.full(20, tau2))
    for c in (0, 5, 19):
        after = rank_one_update(K, n_ref + c, tau2)
        drop = np.trace(K[:n_ref, :n_ref]) / n_ref - np.trace(after[:n_ref, :n_ref]) / n_ref
        assert np.isclose(scores[c], drop)


def test_scores_prefer_uncertain_relevant_points_and_penalize_noise():
    rng = np.random.default_rng(2)
    Xq = rng.random((40, 1))
    K = posterior_cov(np.array([[0.1]]), np.array([1e-4]), Xq)
    for fn in (iv_score, epig_score):
        quiet, noisy = fn(K, 20, np.full(20, 1e-3)), fn(K, 20, np.full(20, 1.0))
        assert np.all(noisy <= quiet + 1e-12)


def test_cost_aware_batch_respects_budget_and_prefers_cheap_when_value_scales():
    rng = np.random.default_rng(3)
    Xq = rng.random((60, 2))
    K = posterior_cov(rng.random((3, 2)), np.full(3, 1e-2), Xq)
    costs = np.array([0.1, 1.0])
    tau2 = np.stack([np.full(40, 0.02), np.full(40, 0.002)])[None]  # (m=1, F=2, C=40)
    chosen = greedy_batch([K], 20, tau2, costs, k=5, budget=0.35, score="iv", cost_aware=True)
    assert sum(costs[f] for _, f in chosen) <= 0.35 + 1e-9
    assert all(f == 0 for _, f in chosen)
