"""Method #1: GP with ARD lengthscales and fixed per-point noise on a fixed Sobol design.

All points are run at high fidelity. Each output gets an independent GP.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import qmc

from surrogatemodeling.core.protocols import Prediction, ProblemSpec
from surrogatemodeling.methods.gp_common import IndependentGPs


def sobol_design(spec: ProblemSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    """n scrambled Sobol points over the training box, in input units."""
    sobol = qmc.Sobol(spec.dim, scramble=True, seed=rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # balance warning for non-power-of-2 n
        return spec.dist.from_unit(sobol.random(n))


class Dataset:
    """Accumulated observations: X (n, d), fidelity (n,), y and var (n, m)."""

    def __init__(self, spec: ProblemSpec):
        self.X = np.empty((0, spec.dim))
        self.fidelity = np.empty(0)
        self.y = np.empty((0, spec.n_outputs))
        self.var = np.empty((0, spec.n_outputs))

    def add(self, X, fidelity, y, sigma) -> None:
        self.X = np.vstack([self.X, X])
        self.fidelity = np.concatenate([self.fidelity, fidelity])
        self.y = np.vstack([self.y, y])
        self.var = np.vstack([self.var, sigma**2])

    def __len__(self) -> int:
        return len(self.X)


class FixedDesignMethod:
    """Base for fixed-design methods: the whole high-fidelity Sobol design is queued at setup."""

    def setup(self, spec: ProblemSpec, budget: float, rng: np.random.Generator) -> None:
        self.spec = spec
        self.unit_cost = spec.cost(1.0)
        self.queue = sobol_design(spec, int(np.floor(budget / self.unit_cost + 1e-9)), rng)
        self.data = Dataset(spec)
        self.stale = True

    def ask(self, n: int, budget_remaining: float) -> tuple[np.ndarray, np.ndarray]:
        k = min(n, len(self.queue), int(np.floor(budget_remaining / self.unit_cost + 1e-9)))
        X, self.queue = self.queue[:k], self.queue[k:]
        return X, np.ones(k)

    def tell(self, X, fidelity, y, sigma) -> None:
        self.data.add(X, fidelity, y, sigma)
        self.stale = True


class SobolGP(FixedDesignMethod):
    def setup(self, spec: ProblemSpec, budget: float, rng: np.random.Generator) -> None:
        super().setup(spec, budget, rng)
        self.gp = IndependentGPs(refit_growth=1.0)  # baseline: full hyperparameter fit every batch

    def predict(self, X: np.ndarray) -> Prediction:
        if self.stale:
            self.gp.fit(self.spec.dist.to_unit(self.data.X), self.data.y, self.data.var)
            self.stale = False
        mean, var = self.gp.predict(self.spec.dist.to_unit(X))
        return Prediction(mean=mean, var=var)

    def diagnostics(self) -> dict:
        return self.gp.diagnostics(self.spec.dist.names, self.spec.output_names)
