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
    """Random points at the quarter-histories config (cost 0.25); predicts the running mean."""

    def __init__(self, overspend=False):
        self.overspend = overspend

    def setup(self, spec, budget, rng):
        self.spec, self.rng, self.ys = spec, rng, []
        self.idx = next(i for i, f in enumerate(spec.fidelities) if f.knobs["histories"] == 0.25)

    def ask(self, n, budget_remaining):
        k = n if self.overspend else min(n, int(budget_remaining / 0.25 + 1e-9))
        return self.spec.dist.sample(k, self.rng), np.full(k, self.idx)

    def tell(self, X, fidelity, y, sigma):
        self.ys.append(y)

    def predict(self, X):
        return Prediction(mean=np.full((len(X), 1), np.vstack(self.ys).mean()))


def test_budget_accounting_is_exact():
    df = run(tiny_problem(), RandomHalfFidelity(), seed=0, budget=7.0, batch_size=5)
    # 5 pts x 0.25 per batch; the last batch only fits 3 pts in the remaining 0.75
    np.testing.assert_allclose(df["cost"], [1.25, 2.5, 3.75, 5.0, 6.25, 7.0])
    assert df["n_evals"].tolist() == [5, 10, 15, 20, 25, 28]


def test_overspending_batch_is_an_error():
    with pytest.raises(BudgetExceededError):
        run(tiny_problem(), RandomHalfFidelity(overspend=True), seed=0, budget=2.0, batch_size=5)


def test_runs_are_reproducible():
    a = run(tiny_problem(), RandomHalfFidelity(), seed=3, budget=5.0)
    b = run(tiny_problem(), RandomHalfFidelity(), seed=3, budget=5.0)
    np.testing.assert_array_equal(a["nrmse/y"], b["nrmse/y"])


def test_fidelity_must_be_a_menu_index():
    class FloatFidelity(RandomHalfFidelity):
        def ask(self, n, budget_remaining):
            return self.spec.dist.sample(n, self.rng), np.full(n, 0.5)

    with pytest.raises(ValueError, match="config indices"):
        run(tiny_problem(), FloatFidelity(), seed=0, budget=2.0)
