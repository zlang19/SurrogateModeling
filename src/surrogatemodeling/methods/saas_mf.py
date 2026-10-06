"""Method #8 `adaptive_iv_mf_saas`: cost-aware sampling with a SAAS surrogate.

Point selection is the cost-aware co-kriging acquisition of `pq_matern` (#7s + prequential
calibration + Matérn), unchanged. What changes is the surrogate whose predictions are
evaluated: a fully Bayesian SAAS GP, which in the SAAS trial beat the plain GP everywhere at
equal budget and was far better calibrated.

SAAS fits hyperparameters by NUTS, whose cost grows ~n^3; cost-aware runs reach ~2,000 points.
So:
- hyperparameters are sampled on a subsample of at most SUBSAMPLE points (every HF point plus
  a random draw of the rest), refit only when the data has grown by REFIT_GROWTH;
- each MCMC sample's GP is then conditioned on *all* the data for prediction (cheap).

Fidelity enters SAAS as extra inputs: one feature per knob that varies across the menu, scaled
so high fidelity is 0 and the cheapest setting is 1; predictions are made at high fidelity.
SAAS can then learn how cheap configs differ (bias) as a smooth function of the knobs. Noise is
the reported sigma^2 times the per-config scale learned by the co-kriging model (#7s), since
batch sigmas are under-reported and SAAS can only take fixed noise.
"""

from __future__ import annotations

import time

import jax
import numpy as np
import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from surrogatemodeling.core.protocols import Prediction, ProblemSpec
from surrogatemodeling.methods.cokriging import CoKrigingAdaptive

SUBSAMPLE = 250
REFIT_GROWTH = 1.5
WARMUP, NUM_SAMPLES, THINNING = 128, 64, 8  # 8 hyperparameter samples per fit
_MIN_VAR = 1e-6
PREDICT_CHUNK = 250


class _CapturingSaas(SaasFullyBayesianSingleTaskGP):
    """Keeps the MCMC samples botorch would otherwise only load into this model."""

    def load_mcmc_samples(self, mcmc_samples):
        self.captured = {k: v.detach().clone() for k, v in mcmc_samples.items()}
        super().load_mcmc_samples(mcmc_samples)


def knob_features(spec: ProblemSpec) -> np.ndarray:
    """(n_configs, k): per varying knob, log-distance from HF scaled so HF = 0 and the farthest
    setting = 1. Knobs constant across the menu are dropped."""
    names = sorted(spec.fidelities[spec.hf].knobs)
    hf = spec.fidelities[spec.hf].knobs
    cols = []
    for k in names:
        v = np.array([f.knobs[k] for f in spec.fidelities], dtype=float)
        d = np.abs(np.log(v / hf[k]))
        if d.max() > 0:
            cols.append(d / d.max())
    return np.column_stack(cols) if cols else np.zeros((len(spec.fidelities), 0))


class SaasCostAware(CoKrigingAdaptive):
    """knobs=True feeds the fidelity knobs to SAAS as inputs (bias as a smooth function of the
    knobs); knobs=False treats every fidelity as the same function with config-corrected noise."""

    def __init__(self, knobs: bool = True):
        super().__init__(noise_scale=True, prequential=True, kernel="matern")
        self.use_knobs = knobs

    def setup(self, spec, budget, rng) -> None:
        super().setup(spec, budget, rng)
        self.kfeat = knob_features(spec) if self.use_knobs else np.zeros((len(spec.fidelities), 0))
        self.samples: list[dict | None] = [None] * spec.n_outputs
        self._n_at_saas_fit = 0
        self.saas_stale = True
        self.nuts_time = 0.0
        self._saas_models: list[SaasFullyBayesianSingleTaskGP] = []
        self._seed = int(rng.integers(2**31))

    def tell(self, X, fidelity, y, sigma) -> None:
        super().tell(X, fidelity, y, sigma)
        self.saas_stale = True

    # --- SAAS surrogate ------------------------------------------------------------------------

    def _inputs(self, U: np.ndarray, fid: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.column_stack([U, self.kfeat[fid]]), dtype=torch.float64)

    def _refresh_saas(self) -> None:
        if not self.saas_stale:
            return
        self._refresh()  # co-kriging model: needed for the learned noise scales
        U = self.spec.dist.to_unit(self.data.X)
        fid = self.data.fidelity
        n = len(U)
        scales = self.gp.noise_scales()  # (m, F)
        var = self.data.var * (scales[:, fid].T if scales is not None else 1.0)
        sd = self.data.y.std(axis=0, ddof=1) if n > 1 else np.ones(self.spec.n_outputs)
        self.loc, self.scale = self.data.y.mean(axis=0), np.where(sd > 0, sd, 1.0)
        X = self._inputs(U, fid)

        if self.samples[0] is None or n >= REFIT_GROWTH * self._n_at_saas_fit:
            idx = self._subsample(fid)
            t0 = time.perf_counter()
            for j in range(self.spec.n_outputs):
                y = (self.data.y[idx, j : j + 1] - self.loc[j]) / self.scale[j]
                v = np.maximum(var[idx, j : j + 1] / self.scale[j] ** 2, _MIN_VAR)
                sub = _CapturingSaas(X[idx], torch.as_tensor(y), torch.as_tensor(v), outcome_transform=None)
                fit_fully_bayesian_model_nuts(sub, warmup_steps=WARMUP, num_samples=NUM_SAMPLES,
                                              thinning=THINNING, disable_progbar=True, seed=self._seed)
                self.samples[j] = sub.captured
            jax.clear_caches()  # one compile per fit size; see saas.py
            self.nuts_time = time.perf_counter() - t0
            self._n_at_saas_fit = n

        self._saas_models = []
        for j in range(self.spec.n_outputs):  # condition every hyperparameter sample on all the data
            y = torch.as_tensor((self.data.y[:, j : j + 1] - self.loc[j]) / self.scale[j])
            v = torch.as_tensor(np.maximum(var[:, j : j + 1] / self.scale[j] ** 2, _MIN_VAR))
            model = SaasFullyBayesianSingleTaskGP(X, y, v, outcome_transform=None)
            model.load_mcmc_samples(self.samples[j])
            model.eval()
            self._saas_models.append(model)
        self.saas_stale = False

    def _subsample(self, fid: np.ndarray) -> np.ndarray:
        n = len(fid)
        if n <= SUBSAMPLE:
            return np.arange(n)
        hf = np.flatnonzero(fid == self.spec.hf)
        rest = np.setdiff1d(np.arange(n), hf)
        take = max(0, SUBSAMPLE - len(hf))
        return np.sort(np.concatenate([hf[-SUBSAMPLE:], self.rng.choice(rest, size=min(take, len(rest)), replace=False)]))

    def predict(self, X: np.ndarray) -> Prediction:
        self._refresh_saas()
        Xt = self._inputs(self.spec.dist.to_unit(X), np.full(len(X), self.spec.hf))
        means, vars_ = [], []
        with torch.no_grad():
            for j, model in enumerate(self._saas_models):
                # Chunked: the batched (per-MCMC-sample) posterior forms a full covariance over its
                # inputs, ~8 x 2000^2 doubles per output; whole-test-set calls pushed 4 runs past 20 GB.
                m_parts, v_parts = [], []
                for k in range(0, len(Xt), PREDICT_CHUNK):
                    post = model.posterior(Xt[k : k + PREDICT_CHUNK])
                    m_parts.append(post.mixture_mean.reshape(-1).numpy())
                    v_parts.append(post.mixture_variance.reshape(-1).numpy())
                means.append(np.concatenate(m_parts) * self.scale[j] + self.loc[j])
                vars_.append(np.concatenate(v_parts) * self.scale[j] ** 2)
        return Prediction(mean=np.column_stack(means), var=np.column_stack(vars_))

    def diagnostics(self) -> dict:
        d = super().diagnostics()
        d["saas"] = {"nuts_time": self.nuts_time, "fit_at_n": self._n_at_saas_fit,
                     "knob_features": int(self.kfeat.shape[1])}
        return d
