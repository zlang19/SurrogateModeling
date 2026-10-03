import numpy as np
import pytest

from surrogatemodeling.core import distributions as D
from surrogatemodeling.core.runner import run
from surrogatemodeling.methods.gp_fixed import SobolGP
from surrogatemodeling.problems.analytic import AnalyticProblem
from surrogatemodeling.problems.base import NoiseModel

METHODS = [SobolGP]


def smooth_2d():
    dist = D.Distribution([D.uniform("x0", -1, 1), D.normal("x1", 0, 0.5, k=3)])
    fn = lambda X: np.column_stack([np.sin(2 * X[:, 0]) + X[:, 1] ** 2, np.exp(0.5 * X[:, 0]) * X[:, 1]])
    return AnalyticProblem("smooth2d", dist, ["a", "b"], fn, NoiseModel(rel=1e-4, abs=1e-4))


@pytest.mark.parametrize("method_cls", METHODS)
def test_fits_smooth_2d_function(method_cls):
    df = run(smooth_2d(), method_cls(), seed=0, budget=60)
    final = df.iloc[-1]
    assert final["nrmse/a"] < 0.01
    assert final["nrmse/b"] < 0.01
