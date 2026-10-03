import numpy as np
import pytest

from surrogatemodeling.problems.analytic import borehole, mc_cost, padded, wing_weight
from surrogatemodeling.registry import PROBLEMS


@pytest.mark.parametrize("name", [n for n in PROBLEMS if n != "toymc"])  # toymc truth is slow; see test_toymc
def test_problem_outputs_are_finite_and_vary(name):
    p = PROBLEMS[name]()
    X, Y = p.test_set()
    assert X.shape == (2000, p.spec.dim)
    assert Y.shape == (2000, p.spec.n_outputs)
    assert np.all(np.isfinite(Y))
    assert np.all(np.std(Y, axis=0) > 0)
    assert p.spec.cost(1.0) == pytest.approx(1.0)


def test_cost_has_overhead_and_is_monotone():
    f = np.linspace(0.01, 1, 50)
    c = np.array([mc_cost(x) for x in f])
    assert np.all(np.diff(c) > 0)
    assert mc_cost(1e-9) == pytest.approx(0.01, abs=1e-6)


def test_padding_preserves_base_and_has_requested_weak_variance():
    base = wing_weight()
    p = padded(base, 40, weak_fraction=0.5, weak_var_frac=0.02)
    rng = np.random.default_rng(0)
    X = p.spec.dist.sample(40000, rng)
    d0 = base.spec.dim
    y = p.truth(X)
    weak_part = y - base.truth(X[:, :d0])
    assert abs(np.var(weak_part) / np.var(base.truth(X[:, :d0])) - 0.02) < 0.002
    assert abs(weak_part.mean()) < 0.05 * np.sqrt(np.var(weak_part))

    # Inert inputs (the last half of the padding) have no effect at all.
    X2 = X.copy()
    X2[:, d0 + 15 :] = rng.uniform(size=X2[:, d0 + 15 :].shape)
    np.testing.assert_array_equal(p.truth(X2), y)


def test_noise_floor_keeps_sigma_positive_near_zero_output():
    p = borehole()
    obs = p.noise.observe(np.zeros((10, 1)), np.ones(10), np.random.default_rng(0))
    assert np.all(obs.sigma > 0)
