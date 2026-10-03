import numpy as np
import pytest

from surrogatemodeling.core import distributions as D
from surrogatemodeling.core.protocols import Prediction
from surrogatemodeling.core.runner import BudgetExceededError, run
from surrogatemodeling.problems.analytic import AnalyticProblem


def tiny_problem():
    dist = D.Distribution([D.uniform("x0", 0, 1), D.uniform("x1", 0, 1)])
    return AnalyticProblem("tiny", dist, ["y"], lambda X: np.sin(3 * X[:, 0]) + X[:, 1] ** 2, overhead=0.0)


class RandomHalfFidelity:
    """Random points at fidelity 0.5; predicts the running mean."""

    def __init__(self, overspend=False):
        self.overspend = overspend

    def setup(self, spec, budget, rng):
        self.spec, self.rng, self.ys = spec, rng, []

    def ask(self, n, budget_remaining):
        k = n if self.overspend else min(n, int(budget_remaining / 0.5 + 1e-9))
        return self.spec.dist.sample(k, self.rng), np.full(k, 0.5)

    def tell(self, X, fidelity, y, sigma):
        self.ys.append(y)

    def predict(self, X):
        return Prediction(mean=np.full((len(X), 1), np.vstack(self.ys).mean()))


def test_budget_accounting_is_exact():
    df = run(tiny_problem(), RandomHalfFidelity(), seed=0, budget=7.0, batch_size=5)
    # 5 pts x 0.5, 5 pts x 0.5, then only 4 pts fit in the remaining 2.0
    np.testing.assert_allclose(df["cost"], [2.5, 5.0, 7.0])
    assert df["n_evals"].tolist() == [5, 10, 14]


def test_overspending_batch_is_an_error():
    with pytest.raises(BudgetExceededError):
        run(tiny_problem(), RandomHalfFidelity(overspend=True), seed=0, budget=3.0, batch_size=5)


def test_runs_are_reproducible():
    a = run(tiny_problem(), RandomHalfFidelity(), seed=3, budget=5.0)
    b = run(tiny_problem(), RandomHalfFidelity(), seed=3, budget=5.0)
    np.testing.assert_array_equal(a["nrmse/y"], b["nrmse/y"])
