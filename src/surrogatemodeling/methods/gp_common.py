"""Shared GP machinery: one independent GP per output on unit-box inputs.

Noise is fixed per point from the reported sigma; with `extra_noise` a homoscedastic term
is learned on top (useful when σ is under-reported or ignored inputs act as noise).
Outputs are standardized here so all hyperparameter priors see unit-scale data.

Hyperparameters are re-optimized only when the data has grown by `refit_growth` since the
last full fit; in between, the previous hyperparameters are reused on the new data.
"""

from __future__ import annotations

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

_MIN_VAR = 1e-8  # in standardized units
_DTYPE = torch.float64


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=_DTYPE)


def _hyper_modules(model: SingleTaskGP) -> dict[str, torch.nn.Module]:
    mods = {"covar": model.covar_module, "mean": model.mean_module}
    if hasattr(model.likelihood, "second_noise_covar") and model.likelihood.second_noise_covar is not None:
        mods["extra_noise"] = model.likelihood.second_noise_covar
    return mods


class IndependentGPs:
    def __init__(self, extra_noise: bool = False, refit_growth: float = 1.2):
        self.extra_noise = extra_noise
        self.refit_growth = refit_growth
        self.models: list[SingleTaskGP] = []
        self.loc = self.scale = None
        self.active: np.ndarray | None = None  # input columns the GPs see; None = all
        self._n_at_full_fit = 0
        self._hypers: list[dict[str, dict]] = []

    def fit(self, Xu: np.ndarray, Y: np.ndarray, Var: np.ndarray, active: np.ndarray | None = None) -> None:
        if active is not None and (self.active is None or not np.array_equal(active, self.active)):
            self._hypers = []  # a different input set invalidates saved lengthscales
        self.active = active
        full_fit = not self._hypers or len(Xu) >= self.refit_growth * self._n_at_full_fit
        sd = Y.std(axis=0, ddof=1) if len(Y) > 1 else np.ones(Y.shape[1])
        self.loc, self.scale = Y.mean(axis=0), np.where(sd > 0, sd, 1.0)
        X = _t(self._cols(Xu))
        self.models = []
        for j in range(Y.shape[1]):
            y = _t((Y[:, j : j + 1] - self.loc[j]) / self.scale[j])
            v = _t(np.maximum(Var[:, j] / self.scale[j] ** 2, _MIN_VAR))
            if self.extra_noise:
                lik = FixedNoiseGaussianLikelihood(noise=v, learn_additional_noise=True)
                model = SingleTaskGP(X, y, likelihood=lik, outcome_transform=None)
            else:
                model = SingleTaskGP(X, y, train_Yvar=v[:, None], outcome_transform=None)
            if full_fit:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            else:
                for name, mod in _hyper_modules(model).items():
                    mod.load_state_dict(self._hypers[j][name])
            model.eval()
            self.models.append(model)
        if full_fit:
            self._n_at_full_fit = len(Xu)
            self._hypers = [{k: m.state_dict() for k, m in _hyper_modules(model).items()} for model in self.models]

    def _cols(self, Xu: np.ndarray) -> np.ndarray:
        return Xu if self.active is None else Xu[:, self.active]

    def predict(self, Xu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Latent (noise-free) mean and variance, (n, m) each, in output units."""
        X = _t(self._cols(Xu))
        means, vars_ = [], []
        with torch.no_grad():
            for j, model in enumerate(self.models):
                post = model.posterior(X)
                means.append(post.mean.squeeze(-1).numpy() * self.scale[j] + self.loc[j])
                vars_.append(post.variance.squeeze(-1).numpy() * self.scale[j] ** 2)
        return np.column_stack(means), np.column_stack(vars_)

    def joint_cov(self, Xu: np.ndarray) -> list[np.ndarray]:
        """Per output, the latent posterior covariance over Xu, in standardized units."""
        X = _t(self._cols(Xu))
        with torch.no_grad():
            return [model.posterior(X).mvn.covariance_matrix.numpy() for model in self.models]

    def inverse_lengthscales_sq(self) -> np.ndarray:
        """(m, d_active) ARD relevance 1/ℓ² per output."""
        out = []
        for model in self.models:
            kern = model.covar_module
            ls = getattr(kern, "base_kernel", kern).lengthscale
            out.append(1.0 / ls.detach().numpy().reshape(-1) ** 2)
        return np.array(out)

    def extra_noise_var(self) -> np.ndarray | None:
        """(m,) learned additional noise variance in output units, if enabled."""
        if not self.extra_noise:
            return None
        return np.array(
            [float(m.likelihood.second_noise_covar.noise.detach().reshape(-1)[0]) * s**2 for m, s in zip(self.models, self.scale)]
        )
