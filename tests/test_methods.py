import numpy as np
import pytest

from surrogatemodeling.core import distributions as D
from surrogatemodeling.core.runner import run
from surrogatemodeling.problems.analytic import AnalyticProblem
from surrogatemodeling.problems.base import NoiseModel
from surrogatemodeling.registry import METHODS


def smooth_2d():
    dist = D.Distribution([D.uniform("x0", -1, 1), D.normal("x1", 0, 0.5, k=3)])
    fn = lambda X: np.column_stack([np.sin(2 * X[:, 0]) + X[:, 1] ** 2, np.exp(0.5 * X[:, 0]) * X[:, 1]])
    return AnalyticProblem("smooth2d", dist, ["a", "b"], fn, NoiseModel(rel=1e-4, abs=1e-4))


# A sanity bar, not a ranking: EPIG levels off near 1% here because in the near-noiseless
# limit it scores correlation with the targets rather than remaining variance.
TOLERANCE = 0.02


@pytest.mark.parametrize("name", list(METHODS))
def test_fits_smooth_2d_function(name):
    df = run(smooth_2d(), METHODS[name](), seed=0, budget=60)
    final = df.iloc[-1]
    assert final["cost"] <= 60 + 1e-9
    assert final["nrmse/a"] < TOLERANCE
    assert final["nrmse/b"] < TOLERANCE
