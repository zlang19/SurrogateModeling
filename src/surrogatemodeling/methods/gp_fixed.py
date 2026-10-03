"""Method #1: GP with ARD lengthscales and fixed per-point noise on a fixed Sobol design.

All points are run at high fidelity. Inputs are mapped to the unit training box; outputs
are standardized; each output gets an independent GP (botorch batches them).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

from surrogatemodeling.core.protocols import Prediction, ProblemSpec

_MIN_VAR = 1e-12


class SobolGP:
    def setup(self, spec: ProblemSpec, budget: float, rng: np.random.Generator) -> None:
        self.spec = spec
        self.unit_cost = spec.cost(1.0)
        n_total = int(np.floor(budget / self.unit_cost + 1e-9))
        sobol = qmc.Sobol(spec.dim, scramble=True, seed=rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # balance warning for non-power-of-2 n
            self.queue = spec.dist.from_unit(sobol.random(n_total))
        self.X = np.empty((0, spec.dim))
        self.y = np.empty((0, spec.n_outputs))
        self.var = np.empty((0, spec.n_outputs))
        self.model = None

    def ask(self, n: int, budget_remaining: float) -> tuple[np.ndarray, np.ndarray]:
        k = min(n, len(self.queue), int(np.floor(budget_remaining / self.unit_cost + 1e-9)))
        X, self.queue = self.queue[:k], self.queue[k:]
        return X, np.ones(k)

    def tell(self, X: np.ndarray, fidelity: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> None:
        self.X = np.vstack([self.X, X])
        self.y = np.vstack([self.y, y])
        self.var = np.vstack([self.var, sigma**2])
        self.model = None

    def _fit(self) -> SingleTaskGP:
        t = lambda a: torch.as_tensor(a, dtype=torch.float64)
        model = SingleTaskGP(
            t(self.spec.dist.to_unit(self.X)),
            t(self.y),
            train_Yvar=t(np.maximum(self.var, _MIN_VAR)),
            outcome_transform=Standardize(m=self.spec.n_outputs),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        return model

    def predict(self, X: np.ndarray) -> Prediction:
        if self.model is None:
            self.model = self._fit()
        with torch.no_grad():
            post = self.model.posterior(torch.as_tensor(self.spec.dist.to_unit(X), dtype=torch.float64))
            return Prediction(mean=post.mean.numpy(), var=post.variance.numpy())
