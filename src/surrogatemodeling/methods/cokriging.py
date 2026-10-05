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
from botorch.models.transforms.input import Warp
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.kernels import AdditiveKernel, Kernel, RBFKernel
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import Noise
from gpytorch.priors import LogNormalPrior
from linear_operator.operators import DiagLinearOperator

from surrogatemodeling.methods.acquisition import greedy_batch
from surrogatemodeling.core.protocols import Prediction
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

    def __init__(self, noise: torch.Tensor, n_configs: int, pooled: bool = False):
        super().__init__()
        self.fixed, self.n_configs = noise, n_configs
        # pooled: one scale shared by all configs (under-reporting is a property of the output)
        self.register_parameter("raw_scale", torch.nn.Parameter(torch.zeros(1 if pooled else n_configs)))
        self.register_constraint("raw_scale", GreaterThan(0.2))  # allow modest over-reporting

    @property
    def scale(self) -> torch.Tensor:
        """(n_configs,) multiplier per config (all equal when pooled)."""
        return self.raw_scale_constraint.transform(self.raw_scale).expand(self.n_configs)

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
                 calibrate: bool = False, noise_scale: bool = False, pooled_scale: bool = False,
                 kernel: str = "rbf", warp: bool = False, log_outputs: bool = False, jitter_seed: int | None = None):
        super().__init__(extra_noise=extra_noise and not noise_scale, refit_growth=refit_growth, calibrate=calibrate,
                         log_outputs=log_outputs, jitter_seed=jitter_seed)
        self.n_configs, self.hf, self.noise_scale = n_configs, hf, noise_scale
        self.pooled_scale, self.kernel, self.warp = pooled_scale, kernel, warp

    def _make_model(self, X, y, v):
        if not self.noise_scale:
            return super()._make_model(X, y, v)
        lik = _GaussianLikelihoodBase(noise_covar=ConfigScaledNoise(v, self.n_configs, pooled=self.pooled_scale))
        D = X.shape[-1]
        return SingleTaskGP(X, y, likelihood=lik, covar_module=self._covar_module(D), outcome_transform=None,
                            input_transform=self._input_transform(D))

    def _input_transform(self, D: int):
        if not self.warp:
            return None
        return Warp(d=D, indices=list(range(D - 1)))  # learned Kumaraswamy CDF per input; not the config column

    def noise_scales(self) -> np.ndarray | None:
        """(m, n_configs) learned multiplier on reported variance, if enabled."""
        if not self.noise_scale:
            return None
        return np.array([m.likelihood.noise_covar.scale.detach().numpy() for m in self.models])

    def _covar_module(self, D: int):
        d = D - 1
        f = get_covar_module_with_dim_scaled_prior(ard_num_dims=d, active_dims=list(range(d)), use_rbf_kernel=self.kernel == "rbf")
        return AdditiveKernel(f, ConfigBiasKernel(self.n_configs, self.hf, d))

    def inverse_lengthscales_sq(self) -> np.ndarray:
        return np.array([1.0 / m.covar_module.kernels[0].lengthscale.detach().numpy().reshape(-1) ** 2 for m in self.models])

    def bias_var(self) -> np.ndarray:
        """(m, n_configs) bias prior variance per config, standardized units."""
        return np.array([m.covar_module.kernels[1].scales.detach().numpy() for m in self.models])


PQ_WINDOW = 300  # most recent out-of-sample residuals used for prequential calibration / weights
PQ_MIN = 20
_PQ_GRID = np.geomspace(0.5, 10.0, 200)


class CoKrigingAdaptive(AdaptiveGP):
    """Co-kriging adaptive method and its variants.

    calibrate     (#7c) LOO calibration of predictive variance
    noise_scale   (#7s) learned per-config multiplier on reported sigma^2; pooled_scale: one per output
    kernel        "rbf" (default) or "matern" (5/2) for f
    warp          learned input warping
    log_outputs   model positive outputs on the log scale
    ensemble      number of GPs (>1: extra members start from jittered hyperparameters; predictions mix)
    prequential   scale predictive sd so recent *out-of-sample* residuals (each batch predicted before
                  training on it) cover 95%
    weighted      weight each output's variance reduction by its estimated error (from those residuals)
    """

    def __init__(self, calibrate: bool = False, noise_scale: bool = False, pooled_scale: bool = False,
                 kernel: str = "rbf", warp: bool = False, log_outputs: bool = False, ensemble: int = 1,
                 prequential: bool = False, weighted: bool = False):
        super().__init__(score="iv", cost_aware=True, extra_noise=not noise_scale)
        self.calibrate, self.noise_scale, self.pooled_scale = calibrate, noise_scale, pooled_scale
        self.kernel, self.warp, self.log_outputs = kernel, warp, log_outputs
        self.ensemble, self.prequential, self.weighted = ensemble, prequential, weighted

    def setup(self, spec, budget, rng) -> None:
        super().setup(spec, budget, rng)
        make = lambda seed: CoKrigingGPs(
            len(spec.fidelities), spec.hf, calibrate=self.calibrate, noise_scale=self.noise_scale,
            pooled_scale=self.pooled_scale, kernel=self.kernel, warp=self.warp, log_outputs=self.log_outputs,
            jitter_seed=seed,
        )
        self.members = [make(None)] + [make(int(rng.integers(2**31))) for _ in range(self.ensemble - 1)]
        self.gp = self.members[0]  # drives the acquisition
        self._resid: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (r, v_latent, tau2) per point
        self.pq_scale = np.ones(spec.n_outputs)
        self.out_weights = np.ones(spec.n_outputs)

    def _fit(self, U: np.ndarray) -> None:
        Xc = np.column_stack([U, self.data.fidelity])
        for gp in self.members:
            gp.fit(Xc, self.data.y, self.data.var)

    # --- out-of-sample residuals ---------------------------------------------------------------

    def _obs_noise(self, fid: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """(n, m) the model's own estimate of each new observation's noise variance (output units)."""
        tau2 = sigma**2
        scales = self.gp.noise_scales()
        if scales is not None:
            tau2 = tau2 * scales[:, fid].T
        extra = self.gp.extra_noise_var()
        if extra is not None:
            tau2 = tau2 + extra[None, :]
        return tau2

    def tell(self, X, fidelity, y, sigma) -> None:
        if self.gp.models and not self.stale:  # model trained on everything *before* this batch
            U = self.spec.dist.to_unit(X)
            mu, v = self.gp.predict(np.column_stack([U, fidelity]))  # latent f + bias of each point's config
            tau2 = self._obs_noise(np.asarray(fidelity), sigma)
            for i in range(len(X)):
                self._resid.append((y[i] - mu[i], v[i], tau2[i]))
            self._resid = self._resid[-PQ_WINDOW:]
            self._update_from_residuals()
        super().tell(X, fidelity, y, sigma)

    def _update_from_residuals(self) -> None:
        if len(self._resid) < PQ_MIN:
            return
        r, v, t = (np.array(a) for a in zip(*self._resid))  # (n, m) each
        for o in range(r.shape[1]):
            if self.prequential:
                cover = np.array([np.mean(np.abs(r[:, o]) <= 1.96 * np.sqrt(c**2 * v[:, o] + t[:, o])) for c in _PQ_GRID])
                ok = np.flatnonzero(cover >= 0.95)
                self.pq_scale[o] = _PQ_GRID[ok[0]] if len(ok) else _PQ_GRID[-1]
        if self.weighted:  # estimated latent error per output, relative to its spread
            err = np.sqrt(np.maximum(np.mean(r**2 - t, axis=0), 0.0)) / self.gp.scale
            self.out_weights = err / err.mean() if err.mean() > 0 else np.ones_like(err)

    def predict(self, X: np.ndarray) -> Prediction:
        self._refresh()
        Q = self._query(self.spec.dist.to_unit(X))
        preds = [gp.predict(Q) for gp in self.members]
        mean = np.mean([m for m, _ in preds], axis=0)
        var = np.mean([v + m**2 for m, v in preds], axis=0) - mean**2  # mixture moments
        if self.prequential:
            var = var * self.pq_scale[None, :] ** 2
        return Prediction(mean=mean, var=var)

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
                              target_tau2=tau2_opt[:, hf], scores_out=scores,
                              output_weights=self.out_weights if self.weighted else None)
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
        if self.prequential:
            d["prequential_scale"] = dict(zip(self.spec.output_names, map(float, self.pq_scale), strict=True))
        if self.weighted:
            d["output_weights"] = dict(zip(self.spec.output_names, map(float, self.out_weights), strict=True))
        if self.gp.models:
            d["bias_sd"] = {
                o: dict(zip(names, map(float, np.sqrt(row) * s), strict=True))
                for o, row, s in zip(self.spec.output_names, self.gp.bias_var(), self.gp.scale)
            }
        return d
