"""Shared GP machinery: one independent GP per output on unit-box inputs.

Noise is fixed per point from the reported sigma; with `extra_noise` a homoscedastic term
is learned on top (useful when σ is under-reported or ignored inputs act as noise).
Outputs are standardized here so all hyperparameter priors see unit-scale data.

Hyperparameters are re-optimized only when the data has grown by `refit_growth` since the
last full fit; in between, the previous hyperparameters are reused on the new data.
"""

from __future__ import annotations

import time

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

_MIN_VAR = 1e-8  # in standardized units
_DTYPE = torch.float64
# Exact Cholesky for every solve. gpytorch otherwise switches to CG/Lanczos above 800
# points, whose caches made cost-aware runs (~2000 cheap points) grow to 6-7 GB.
_EXACT = (gpytorch.settings.max_cholesky_size(100_000), gpytorch.settings.fast_pred_var(False))


class _exact:
    def __enter__(self):
        for ctx in _EXACT:
            ctx.__enter__()

    def __exit__(self, *exc):
        for ctx in reversed(_EXACT):
            ctx.__exit__(*exc)


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=_DTYPE)


def _hyper_modules(model: SingleTaskGP) -> dict[str, torch.nn.Module]:
    mods = {"covar": model.covar_module, "mean": model.mean_module}
    if hasattr(model.likelihood, "second_noise_covar") and model.likelihood.second_noise_covar is not None:
        mods["extra_noise"] = model.likelihood.second_noise_covar
    noise = getattr(model.likelihood, "noise_covar", None)
    if noise is not None and any(True for _ in noise.parameters()):  # e.g. learned per-config noise scales
        mods["noise_model"] = noise
    warp = getattr(model, "input_transform", None)
    if warp is not None and any(True for _ in warp.parameters()):  # e.g. learned input warping
        mods["input_transform"] = warp
    return mods


class IndependentGPs:
    """
    Options:
      log_outputs   model outputs that are positive (in the first fit's data) on the log
                    scale; predictions map back as lognormal mean/variance
      jitter_seed   randomize the starting hyperparameters of every full fit (for ensembles)
    `var_scale` (m,) may be set by a method to rescale predictive sd (e.g. prequential calibration).
    """

    def __init__(self, extra_noise: bool = False, refit_growth: float = 1.2, calibrate: bool = False,
                 log_outputs: bool = False, jitter_seed: int | None = None):
        self.extra_noise = extra_noise
        self.refit_growth = refit_growth
        self.calibrate = calibrate
        self.log_outputs = log_outputs
        self._log: np.ndarray | None = None  # (m,) which outputs are modeled on the log scale
        self._ref_y: np.ndarray | None = None  # (m,) typical |y| for mapping variances to the log scale
        self._jitter = torch.Generator().manual_seed(jitter_seed) if jitter_seed is not None else None
        self.cal = None  # (m,) predictive-sd scale from LOO calibration; None = off
        self.var_scale = None  # (m,) external predictive-sd scale; None = off
        self.models: list[SingleTaskGP] = []
        self.loc = self.scale = None
        self.active: np.ndarray | None = None  # input columns the GPs see; None = all
        self._n_at_full_fit = 0
        self._hypers: list[dict[str, dict]] = []
        self.last_fit = {"reoptimized": False, "fit_time": 0.0}

    def fit(self, Xu: np.ndarray, Y: np.ndarray, Var: np.ndarray, active: np.ndarray | None = None) -> None:
        with _exact():
            self._fit(Xu, Y, Var, active)

    def _fit(self, Xu: np.ndarray, Y: np.ndarray, Var: np.ndarray, active: np.ndarray | None) -> None:
        if active is not None and (self.active is None or not np.array_equal(active, self.active)):
            self._hypers = []  # a different input set invalidates saved lengthscales
        self.active = active
        t0 = time.perf_counter()
        full_fit = not self._hypers or len(Xu) >= self.refit_growth * self._n_at_full_fit
        Y, Var = self._to_model_scale(Y, Var)
        sd = Y.std(axis=0, ddof=1) if len(Y) > 1 else np.ones(Y.shape[1])
        self.loc, self.scale = Y.mean(axis=0), np.where(sd > 0, sd, 1.0)
        X = _t(self._cols(Xu))
        self.models = []
        for j in range(Y.shape[1]):
            y = _t((Y[:, j : j + 1] - self.loc[j]) / self.scale[j])
            v = _t(np.maximum(Var[:, j] / self.scale[j] ** 2, _MIN_VAR))
            model = self._make_model(X, y, v)
            if full_fit:
                if self._jitter is not None:  # random restart: perturb the starting lengthscales
                    for name, p in model.named_parameters():
                        if "raw_lengthscale" in name:
                            p.data += torch.randn(p.shape, generator=self._jitter, dtype=p.dtype)
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            else:
                for name, mod in _hyper_modules(model).items():
                    mod.load_state_dict(self._hypers[j][name])
            model.eval()
            self.models.append(model)
        if full_fit:
            self._n_at_full_fit = len(Xu)
            self._hypers = [{k: m.state_dict() for k, m in _hyper_modules(model).items()} for model in self.models]
        if self.calibrate:
            self.cal = np.array([self._loo_scale(m) for m in self.models])
        self.last_fit = {"reoptimized": full_fit, "fit_time": time.perf_counter() - t0}

    def _to_model_scale(self, Y: np.ndarray, Var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Log-transform positive outputs (decided on the first fit); delta-method variances."""
        if self._log is None:
            self._log = (Y.min(axis=0) > 0) if self.log_outputs else np.zeros(Y.shape[1], dtype=bool)
            self._ref_y = np.where(self._log, np.median(np.abs(Y), axis=0), 1.0)
        if not self._log.any():
            return Y, Var
        Y, Var = Y.copy(), Var.copy()
        pos = np.maximum(Y[:, self._log], 1e-6 * self._ref_y[self._log])
        Y[:, self._log] = np.log(pos)
        Var[:, self._log] = Var[:, self._log] / pos**2
        return Y, Var

    def model_var(self, var: np.ndarray) -> np.ndarray:
        """Map output-unit variances (m, ...) to the standardized model scale."""
        ref = (np.where(self._log, self._ref_y, 1.0) if self._log is not None else 1.0) * self.scale
        return var / (np.asarray(ref) ** 2).reshape(-1, *([1] * (var.ndim - 1)))

    def _covar_module(self, d: int):
        """Kernel over the model inputs; None = botorch's default (ARD RBF). Subclasses override."""
        return None

    def _input_transform(self, d: int):
        """Optional botorch input transform (e.g. warping); None = identity. Subclasses override."""
        return None

    def _make_model(self, X: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> SingleTaskGP:
        covar, warp = self._covar_module(X.shape[-1]), self._input_transform(X.shape[-1])
        if self.extra_noise:
            lik = FixedNoiseGaussianLikelihood(noise=v, learn_additional_noise=True)
            return SingleTaskGP(X, y, likelihood=lik, covar_module=covar, outcome_transform=None, input_transform=warp)
        return SingleTaskGP(X, y, train_Yvar=v[:, None], covar_module=covar, outcome_transform=None, input_transform=warp)

    @staticmethod
    def _loo_scale(model: SingleTaskGP, level: float = 0.95) -> float:
        """Conformal-style sd scale from closed-form leave-one-out residuals.

        z_i = [K^-1 (y - m)]_i / sqrt([K^-1]_ii) is training point i's residual against the
        model fit without it, in units of that prediction's sd (noise included). The scale
        makes `level` of |z| fall inside the nominal interval; clamped to [0.5, 5].
        """
        X, y = model.train_inputs[0], model.train_targets
        with torch.no_grad():
            K = model.covar_module(X).to_dense()
            noise = model.likelihood._shaped_noise_covar(X.shape[:-1], X).diagonal(dim1=-2, dim2=-1)
            K = K + torch.diag(noise.reshape(-1).expand(len(y)))
            Kinv = torch.cholesky_inverse(torch.linalg.cholesky(K))
            alpha = Kinv @ (y - model.mean_module(X))
            z = (alpha / torch.sqrt(torch.diagonal(Kinv))).abs().numpy()
        from scipy import stats

        q = np.quantile(z, level) / stats.norm.ppf(0.5 + level / 2)
        return float(np.clip(q, 0.5, 5.0))

    def _cols(self, Xu: np.ndarray) -> np.ndarray:
        return Xu if self.active is None else Xu[:, self.active]

    def predict(self, Xu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Latent (noise-free) mean and variance, (n, m) each, in output units."""
        X = _t(self._cols(Xu))
        means, vars_ = [], []
        with torch.no_grad(), _exact():
            for j, model in enumerate(self.models):
                post = model.posterior(X)
                mu = post.mean.squeeze(-1).numpy() * self.scale[j] + self.loc[j]
                v = post.variance.squeeze(-1).numpy() * self.scale[j] ** 2
                for s in (self.cal, self.var_scale):
                    if s is not None:
                        v = v * s[j] ** 2
                if self._log is not None and self._log[j]:  # lognormal moments
                    mu, v = np.exp(mu + v / 2), np.expm1(v) * np.exp(2 * mu + v)
                means.append(mu)
                vars_.append(v)
        return np.column_stack(means), np.column_stack(vars_)

    def joint_cov(self, Xu: np.ndarray) -> list[np.ndarray]:
        """Per output, the latent posterior covariance over Xu, in standardized units."""
        X = _t(self._cols(Xu))
        with torch.no_grad(), _exact():
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

    def diagnostics(self, input_names: list[str], output_names: list[str]) -> dict:
        """Last fit's timing, whether hyperparameters were re-optimized, and ARD lengthscales
        (unit-box units) per output; plus the learned extra noise sd when enabled."""
        if not self.models:
            return {}
        names = list(np.asarray(input_names)[self.active]) if self.active is not None else list(input_names)
        ls = 1.0 / np.sqrt(self.inverse_lengthscales_sq())
        out = {
            **self.last_fit,
            "lengthscales": {o: dict(zip(names, map(float, row), strict=True)) for o, row in zip(output_names, ls)},
        }
        if self.cal is not None:
            out["calibration_scale"] = dict(zip(output_names, map(float, self.cal), strict=True))
        extra = self.extra_noise_var()
        if extra is not None:
            out["extra_noise_sd"] = dict(zip(output_names, map(float, np.sqrt(extra)), strict=True))
        return out
