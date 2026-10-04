"""Method #3: SAAS GP (fully Bayesian, sparse axis-aligned subspace prior) on a fixed Sobol design.

The half-Cauchy prior on inverse lengthscales shrinks irrelevant inputs toward zero, which
suits many-inputs/few-points problems (Eriksson & Jankowiak, 2021). Hyperparameters are
sampled with NUTS; predictions are the moment-matched mixture over the samples. One model
per output, fixed per-point noise from the reported sigma.
"""

from __future__ import annotations

import jax
import numpy as np
import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from surrogatemodeling.core.protocols import Prediction
from surrogatemodeling.methods.gp_fixed import FixedDesignMethod

_MIN_VAR = 1e-6

# NUTS budget per fit; botorch's defaults (512 / 256 / 16) cost ~4x more for a small gain.
WARMUP, NUM_SAMPLES, THINNING = 256, 128, 16


class SobolSAAS(FixedDesignMethod):
    def predict(self, X: np.ndarray) -> Prediction:
        if self.stale:
            self._fit()
            self.stale = False
        Xt = torch.as_tensor(self.spec.dist.to_unit(X), dtype=torch.float64)
        means, vars_ = [], []
        with torch.no_grad():
            for model, loc, scale in zip(self.models, self.loc, self.scale):
                post = model.posterior(Xt)
                means.append(post.mixture_mean.reshape(-1).numpy() * scale + loc)
                vars_.append(post.mixture_variance.reshape(-1).numpy() * scale**2)
        return Prediction(mean=np.column_stack(means), var=np.column_stack(vars_))

    def _fit(self) -> None:
        y, var = self.data.y, self.data.var
        sd = y.std(axis=0, ddof=1) if len(y) > 1 else np.ones(y.shape[1])
        self.loc, self.scale = y.mean(axis=0), np.where(sd > 0, sd, 1.0)
        X = torch.as_tensor(self.spec.dist.to_unit(self.data.X), dtype=torch.float64)
        self.models = []
        for j in range(y.shape[1]):
            yj = torch.as_tensor((y[:, j : j + 1] - self.loc[j]) / self.scale[j], dtype=torch.float64)
            vj = torch.as_tensor(np.maximum(var[:, j : j + 1] / self.scale[j] ** 2, _MIN_VAR), dtype=torch.float64)
            model = SaasFullyBayesianSingleTaskGP(X, yj, train_Yvar=vj, outcome_transform=None)
            fit_fully_bayesian_model_nuts(
                model, warmup_steps=WARMUP, num_samples=NUM_SAMPLES, thinning=THINNING,
                disable_progbar=True, seed=int(self.seed),
            )
            self.models.append(model)
        # Each fit has a new data size, so JAX compiles a new NUTS kernel; without this the
        # compiled kernels accumulate to several GB per run. (Outputs share one compile.)
        jax.clear_caches()

    def setup(self, spec, budget, rng) -> None:
        super().setup(spec, budget, rng)
        self.seed = rng.integers(2**31)
