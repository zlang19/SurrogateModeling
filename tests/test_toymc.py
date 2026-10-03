import numpy as np
import pytest

from surrogatemodeling.problems.toymc.physics import run_kcode
from surrogatemodeling.problems.toymc.problem import DIST, ToyMCProblem, _params

NOMINAL = np.array([m.rv.mean() for m in DIST.marginals])


def replicates(n_particles, n_reps=16):
    return np.array([run_kcode(_params(NOMINAL), n_particles, np.random.default_rng(s)).mean for s in range(n_reps)])


def test_spread_scales_as_inverse_sqrt_histories():
    lo, hi = replicates(200, n_reps=48), replicates(800, n_reps=48)
    ratio = lo.std(axis=0, ddof=1) / hi.std(axis=0, ddof=1)
    # 4x histories -> 2x smaller spread; with 48 replicates each std is good to ~10%.
    assert np.all((1.5 < ratio) & (ratio < 2.6))


def test_low_fidelity_is_unbiased():
    lo, hi = replicates(100, n_reps=48), replicates(1600, n_reps=12)
    # Difference of means within 3 standard errors, for every output (incl. the ratios).
    se = np.sqrt(lo.var(axis=0, ddof=1) / len(lo) + hi.var(axis=0, ddof=1) / len(hi))
    assert np.all(np.abs(lo.mean(axis=0) - hi.mean(axis=0)) < 3 * se)


def test_reported_sigma_matches_spread_for_k_eff():
    p = ToyMCProblem()
    X = np.tile(NOMINAL, (16, 1))
    obs = p.evaluate(X, np.full(16, 0.5), np.random.default_rng(0))
    assert 0.6 < obs.y[:, 0].std(ddof=1) / obs.sigma[:, 0].mean() < 1.6


def test_physics_sensitivities_have_expected_signs():
    def k(**changes):
        x = NOMINAL.copy()
        for name, v in changes.items():
            x[DIST.names.index(name)] = v
        return run_kcode(_params(x), 2000, np.random.default_rng(0)).mean

    base = k()
    assert k(boron_ppm=1500)[0] < base[0] < k(boron_ppm=0)[0]
    assert k(enr_inner=5.0, enr_outer=5.0)[0] > base[0]
    # More enrichment inside than outside pushes power inward.
    assert k(enr_inner=5.0, enr_outer=2.0)[1] > base[1] > k(enr_inner=2.0, enr_outer=5.0)[1]


def test_fidelity_spec():
    spec = ToyMCProblem().spec
    assert spec.dim == 23 and spec.n_outputs == 3
    assert spec.cost(1.0) == pytest.approx(1.0)
