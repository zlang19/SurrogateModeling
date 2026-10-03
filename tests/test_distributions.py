import numpy as np

from surrogatemodeling.core import distributions as D


def make():
    return D.Distribution([D.uniform("a", 2, 5), D.normal("b", 1, 0.5, k=3), D.lognormal("c", 0, 0.5, k=3)])


def test_bounds_and_unit_round_trip():
    dist = make()
    lo, hi = dist.bounds()
    np.testing.assert_allclose(lo, [2, -0.5, np.exp(-1.5)])
    np.testing.assert_allclose(hi, [5, 2.5, np.exp(1.5)])
    X = dist.sample(100, np.random.default_rng(0))
    np.testing.assert_allclose(dist.from_unit(dist.to_unit(X)), X)


def test_samples_mostly_inside_box():
    dist = make()
    X = dist.sample(20000, np.random.default_rng(1))
    lo, hi = dist.bounds()
    inside = np.all((X >= lo) & (X <= hi), axis=1).mean()
    assert inside > 0.99
