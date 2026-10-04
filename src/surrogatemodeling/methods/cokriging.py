"""Method #7 `adaptive_iv_mf_ck`: cost-aware IV with a co-kriging (per-config bias) GP.

Once cycles are fidelity knobs, a cheap config can be biased (unconverged source,
population bias), not just noisy. The model, per output:

    y_c(x) = f(x) + b_c(x) + noise,      b_hf = 0,
    b_c ~ GP(0, s_c * RBF_iso(x, x'))    independent across configs

f gets botorch's usual ARD RBF; the bias processes share one isotropic lengthscale and
each config learns its own variance s_c (high fidelity's is pinned to 0). Predictions are
of f, i.e. the high-fidelity output.

Acquisition: the joint posterior covariance is taken over the reference points at high
fidelity (i.e. f) plus every (candidate, config) pair, so each row carries its config's bias
process. The greedy rank-one updates then capture that the bias is *correlated*: many cheap
runs near each other share one bias and stop teaching us about f, which independent extra
noise would not. Rows are scored per unit cost exactly as in #5.
"""

from __future__ import annotations

import numpy as np
import torch
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.kernels import AdditiveKernel, Kernel, RBFKernel
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import Noise
from gpytorch.priors import LogNormalPrior
from linear_operator.operators import DiagLinearOperator

from surrogatemodeling.methods.acquisition import greedy_batch
from surrogatemodeling.methods.adaptive import N_CAND, N_REF, AdaptiveGP
from surrogatemodeling.methods.gp_common import IndependentGPs


class ConfigBiasKernel(Kernel):
    """k((x, c), (x', c')) = s_c * [c == c'] * RBF_iso(x, x'); the last input column is c."""

    def __init__(self, n_configs: int, hf: int, x_dims: int, **kwargs):
        super().__init__(**kwargs)
        self.hf, self.x_dims = hf, x_dims
        self.rbf = RBFKernel(lengthscale_prior=LogNormalPrior(0.0, 1.0))
        self.register_parameter("raw_scales", torch.nn.Parameter(torch.full((n_configs,), -3.0)))
        self.register_constraint("raw_scales", Positive())
        mask = torch.ones(n_configs)
        mask[hf] = 0.0
        self.register_buffer("mask", mask)

    @property
    def scales(self) -> torch.Tensor:
        return self.raw_scales_constraint.transform(self.raw_scales) * self.mask

    def forward(self, x1, x2, diag: bool = False, **params):
        c1, c2 = x1[..., -1].long(), x2[..., -1].long()
        k = self.rbf.forward(x1[..., : self.x_dims], x2[..., : self.x_dims], diag=diag)
        s = self.scales
        if diag:
            return k * s[c1]
        same = (c1.unsqueeze(-1) == c2.unsqueeze(-2)).to(k.dtype)
        return k * same * s[c1].unsqueeze(-1)


class ConfigScaledNoise(Noise):
    """Per-point noise = s_c * reported sigma^2, with one learned scale per fidelity config.

    Under-reported batch sigmas (generation correlation) are a roughly *multiplicative*
    error that differs by config and output; this learns it from the data. The fixed
    per-point variances are a plain attribute, not a buffer, so warm-started
    hyperparameters (state dicts) don't depend on the number of points.
    """

    def __init__(self, noise: torch.Tensor, n_configs: int):
        super().__init__()
        self.fixed = noise
        self.register_parameter("raw_scale", torch.nn.Parameter(torch.zeros(n_configs)))
        self.register_constraint("raw_scale", GreaterThan(0.2))  # allow modest over-reporting

    @property
    def scale(self) -> torch.Tensor:
        return self.raw_scale_constraint.transform(self.raw_scale)

    def forward(self, *params, shape=None, **kwargs):
        X = params[0] if params else None
        if isinstance(X, (list, tuple)):  # ExactGP passes train_inputs as a list
            X = X[0]
        if X is not None and X.shape[-2] == self.fixed.shape[-1]:
            return DiagLinearOperator(self.fixed * self.scale[X[..., -1].long()])
        n = shape[-1] if shape is not None else X.shape[-2]
        return DiagLinearOperator(self.fixed.mean() * self.scale.mean() * torch.ones(n, dtype=self.fixed.dtype))


class CoKrigingGPs(IndependentGPs):
    """IndependentGPs whose inputs carry the fidelity config index as a last column."""

    def __init__(self, n_configs: int, hf: int, extra_noise: bool = True, refit_growth: float = 1.2,
                 calibrate: bool = False, noise_scale: bool = False):
        super().__init__(extra_noise=extra_noise and not noise_scale, refit_growth=refit_growth, calibrate=calibrate)
        self.n_configs, self.hf, self.noise_scale = n_configs, hf, noise_scale

    def _make_model(self, X, y, v):
        if not self.noise_scale:
            return super()._make_model(X, y, v)
        lik = _GaussianLikelihoodBase(noise_covar=ConfigScaledNoise(v, self.n_configs))
        return SingleTaskGP(X, y, likelihood=lik, covar_module=self._covar_module(X.shape[-1]), outcome_transform=None)

    def noise_scales(self) -> np.ndarray | None:
        """(m, n_configs) learned multiplier on reported variance, if enabled."""
        if not self.noise_scale:
            return None
        return np.array([m.likelihood.noise_covar.scale.detach().numpy() for m in self.models])

    def _covar_module(self, D: int):
        d = D - 1
        f = get_covar_module_with_dim_scaled_prior(ard_num_dims=d, active_dims=list(range(d)))
        return AdditiveKernel(f, ConfigBiasKernel(self.n_configs, self.hf, d))

    def inverse_lengthscales_sq(self) -> np.ndarray:
        return np.array([1.0 / m.covar_module.kernels[0].lengthscale.detach().numpy().reshape(-1) ** 2 for m in self.models])

    def bias_var(self) -> np.ndarray:
        """(m, n_configs) bias prior variance per config, standardized units."""
        return np.array([m.covar_module.kernels[1].scales.detach().numpy() for m in self.models])


class CoKrigingAdaptive(AdaptiveGP):
    """With calibrate=True (#7c), predictive variance is rescaled by closed-form LOO
    calibration (see IndependentGPs._loo_scale); the acquisition is unaffected."""

    def __init__(self, calibrate: bool = False, noise_scale: bool = False):
        super().__init__(score="iv", cost_aware=True, extra_noise=not noise_scale)
        self.calibrate, self.noise_scale = calibrate, noise_scale

    def setup(self, spec, budget, rng) -> None:
        super().setup(spec, budget, rng)
        self.gp = CoKrigingGPs(len(spec.fidelities), spec.hf, calibrate=self.calibrate, noise_scale=self.noise_scale)

    def _fit(self, U: np.ndarray) -> None:
        self.gp.fit(np.column_stack([U, self.data.fidelity]), self.data.y, self.data.var)

    def _query(self, U: np.ndarray) -> np.ndarray:
        return np.column_stack([U, np.full(len(U), self.spec.hf)])

    def ask(self, n: int, budget_remaining: float) -> tuple[np.ndarray, np.ndarray]:
        if len(self.queue):
            return super().ask(n, budget_remaining)  # Sobol seed at high fidelity
        self._refresh()
        F = len(self.options)
        n_cand = max(128, N_CAND // F)  # rows grow with the menu; keep the matrix ~N_CAND
        cand = self._candidates()[:n_cand]
        U_ref, U_cand = self.spec.dist.to_unit(self.ref), self.spec.dist.to_unit(cand)
        rows = [np.column_stack([U_cand, np.full(n_cand, opt)]) for opt in self.options]
        covs = self.gp.joint_cov(np.vstack([self._query(U_ref), *rows]))
        tau2_opt = super()._noise_options(1)[:, :, 0]  # (m, F) observation noise, standardized
        scales = self.gp.noise_scales()
        if scales is not None:  # #7s: correct each config's reported noise by its learned scale
            tau2_opt = tau2_opt * scales[:, self.options]
        tau2 = np.repeat(tau2_opt, n_cand, axis=1)[:, None, :]  # one "option", F*C rows
        costs = np.repeat(self.costs, n_cand)[None, :]
        hf = int(np.flatnonzero(self.options == self.spec.hf)[0])
        scores: list[float] = []
        chosen = greedy_batch(covs, N_REF, tau2, costs, n, budget_remaining, score="iv", cost_aware=True,
                              target_tau2=tau2_opt[:, hf], scores_out=scores)
        idx = [r % n_cand for r, _ in chosen]
        fids = self.options[[r // n_cand for r, _ in chosen]]
        self.last_batch = {
            "phase": "adaptive",
            "fidelities": [self.spec.fidelities[i].name for i in fids],
            "best_score": scores[0] if scores else None,
            "hf_noise_sd": dict(zip(self.spec.output_names, map(float, np.sqrt(self._hf_var)), strict=True)),
        }
        return cand[idx].reshape(-1, self.spec.dim), fids

    def diagnostics(self) -> dict:
        names = [f.name for f in self.spec.fidelities]
        d = super().diagnostics()
        scales = self.gp.noise_scales() if self.gp.models else None
        if scales is not None:
            d["noise_scale"] = {o: dict(zip(names, map(float, row), strict=True)) for o, row in zip(self.spec.output_names, scales)}
        if self.gp.models:
            d["bias_sd"] = {
                o: dict(zip(names, map(float, np.sqrt(row) * s), strict=True))
                for o, row, s in zip(self.spec.output_names, self.gp.bias_var(), self.gp.scale)
            }
        return d
