"""Neural surrogates: deep ensembles of small MLPs (Lakshminarayanan et al., 2017).

Two methods share one model:
- `sobol_de`: the high-fidelity Sobol design of `sobol_gp`, with a deep ensemble instead of a GP.
- `adaptive_iv_mf_de_safe`: the leader's point selection (`safe_pooled`: co-kriging GP, safe menu,
  pooled sigma scales), unchanged, with the deep ensemble making the predictions. The design is
  the same, so any difference is the surrogate. Fidelity enters as extra input features (one per
  knob that varies on the menu; HF = 0), as in #8, so the network can learn config bias.

The ensemble is M MLPs trained as one batched network (weights carry a leading member axis), so
training costs about as much as one network. Each member has its own random init and mini-batch
order. Loss is the Gaussian NLL with a per-point noise variance a_o * sigma_i^2 + b_o, where
sigma_i is the reported MC error and a_o, b_o are learned per output (a pooled sigma scale plus
extra noise, as in the GP methods). The members' means predict the noise-free output; the spread
of the members' means is the predictive (epistemic) variance. Deep-ensemble spread is usually too
narrow, so it is widened by prequential calibration (each new batch predicted before training on
it, sd scaled so 95% of recent residuals are covered), as in `pq` but allowed to narrow too.
"""

from __future__ import annotations

import numpy as np
import torch

from surrogatemodeling.core.protocols import Prediction, ProblemSpec
from surrogatemodeling.methods.cokriging import CoKrigingAdaptive
from surrogatemodeling.methods.gp_fixed import FixedDesignMethod
from surrogatemodeling.methods.saas_mf import knob_features

MEMBERS, WIDTH = 5, 32
COLD_EPOCHS, WARM_EPOCHS = 1500, 300
REFIT_GROWTH = 1.5  # cold retrain when the data has grown this much since the last one
# Tuned on Borehole-30D seed 0 only (toymc_axial and OpenMC held out): 0.250 -> 0.108 NRMSE.
LR, WEIGHT_DECAY = 1e-2, 1.0
BATCH = 256
# The ensemble's spread doesn't shrink in step with its error as data grows, so only recent
# residuals are representative: the last 5 batches (Borehole-30D NCRPS 0.12 -> 0.06-0.08 vs 300).
PQ_WINDOW, PQ_MIN = 25, 20
# Unlike a GP posterior, ensemble spread has no natural scale (at the tuned weight decay it is
# too wide: coverage ~1.0), so calibration may also narrow it.
_PQ_GRID = np.geomspace(0.2, 10.0, 300)


def _peek_seed(rng: np.random.Generator) -> int:
    """A seed derived from rng *without advancing it*, so the point selection (and thus the
    OpenMC evaluation cache) matches the GP method this one shadows."""
    return int(np.random.default_rng(rng.bit_generator.state["state"]["state"] % 2**63).integers(2**31))


class DeepEnsemble:
    """M two-hidden-layer SiLU MLPs, d -> WIDTH -> WIDTH -> m, trained jointly."""

    def __init__(self, d: int, m: int, seed: int, members: int = MEMBERS, width: int = WIDTH):
        self.d, self.m, self.M, self.w = d, m, members, width
        self.gen = torch.Generator().manual_seed(seed)
        self.params: list[torch.Tensor] | None = None
        self.n_at_cold = 0

    def _init(self) -> None:
        def layer(i, o):
            W = torch.randn(self.M, i, o, generator=self.gen, dtype=torch.float64) * np.sqrt(2.0 / i)
            return [W, torch.zeros(self.M, 1, o, dtype=torch.float64)]

        self.params = [*layer(self.d, self.w), *layer(self.w, self.w), *layer(self.w, self.m)]
        self.log_a = torch.zeros(self.m, dtype=torch.float64)  # noise variance = a*sigma^2 + b
        self.log_b = torch.full((self.m,), -6.0, dtype=torch.float64)
        for p in [*self.params, self.log_a, self.log_b]:
            p.requires_grad_(True)

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        """X (M, n, d) or (n, d) -> (M, n, m)."""
        h = X if X.dim() == 3 else X.expand(self.M, *X.shape)
        W1, b1, W2, b2, W3, b3 = self.params
        h = torch.nn.functional.silu(torch.bmm(h, W1) + b1)
        h = torch.nn.functional.silu(torch.bmm(h, W2) + b2)
        return torch.bmm(h, W3) + b3

    def fit(self, X: np.ndarray, y: np.ndarray, noise: np.ndarray) -> None:
        """X (n, d) scaled inputs; y (n, m) standardized; noise (n, m) reported variance, standardized."""
        n = len(X)
        cold = self.params is None or n >= REFIT_GROWTH * self.n_at_cold
        if cold:
            self._init()
            self.n_at_cold = n
        Xt, yt, vt = (torch.as_tensor(a, dtype=torch.float64) for a in (X, y, noise))
        opt = torch.optim.AdamW([*self.params, self.log_a, self.log_b], lr=LR, weight_decay=0.0)
        bs = min(BATCH, n)
        for _ in range(COLD_EPOCHS if cold else WARM_EPOCHS):
            # Each member sees its own random mini-batch: (M, bs) indices.
            idx = torch.stack([torch.randperm(n, generator=self.gen)[:bs] for _ in range(self.M)])
            mu = self._forward(Xt[idx])
            s2 = self.log_a.exp() * vt[idx] + self.log_b.exp() + 1e-8
            nll = 0.5 * (torch.log(s2) + (yt[idx] - mu) ** 2 / s2).mean()
            l2 = sum((W**2).sum() for W in self.params[0::2]) / n  # weight decay scaled by data size
            opt.zero_grad()
            (nll + WEIGHT_DECAY * l2).backward()
            opt.step()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = self._forward(torch.as_tensor(X, dtype=torch.float64)).numpy()  # (M, n, m)
        return mu.mean(axis=0), mu.var(axis=0, ddof=1)

    @torch.no_grad()
    def noise_params(self) -> tuple[np.ndarray, np.ndarray]:
        return self.log_a.exp().numpy(), self.log_b.exp().numpy()


class _EnsembleSurrogate:
    """Standardization, refits, and prequential calibration around a DeepEnsemble."""

    def __init__(self, spec: ProblemSpec, d: int, seed: int):
        self.spec, self.net = spec, DeepEnsemble(d, spec.n_outputs, seed)
        self.loc = np.zeros(spec.n_outputs)
        self.scale = np.ones(spec.n_outputs)
        self.resid: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.pq_scale = np.ones(spec.n_outputs)
        self.trained = False

    def fit(self, F: np.ndarray, y: np.ndarray, var: np.ndarray) -> None:
        self.loc = y.mean(axis=0)
        sd = y.std(axis=0, ddof=1) if len(y) > 1 else np.ones(y.shape[1])
        self.scale = np.where(sd > 0, sd, 1.0)
        self.net.fit(F, (y - self.loc) / self.scale, var / self.scale**2)
        self.trained = True

    def latent(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, v = self.net.predict(F)
        return mu * self.scale + self.loc, v * self.scale**2

    def obs_noise(self, var: np.ndarray) -> np.ndarray:
        a, b = self.net.noise_params()
        return a * var + b * self.scale**2

    def record(self, F: np.ndarray, y: np.ndarray, var: np.ndarray) -> None:
        """Out-of-sample residuals of a new batch, before training on it."""
        if not self.trained:
            return
        mu, v = self.latent(F)
        t = self.obs_noise(var)
        self.resid = (self.resid + list(zip(y - mu, v, t, strict=True)))[-PQ_WINDOW:]
        if len(self.resid) < PQ_MIN:
            return
        r, v, t = (np.array(a) for a in zip(*self.resid))
        for o in range(r.shape[1]):
            cover = np.array([np.mean(np.abs(r[:, o]) <= 1.96 * np.sqrt(c**2 * v[:, o] + t[:, o])) for c in _PQ_GRID])
            ok = np.flatnonzero(cover >= 0.95)
            self.pq_scale[o] = _PQ_GRID[ok[0]] if len(ok) else _PQ_GRID[-1]

    def predict(self, F: np.ndarray) -> Prediction:
        mu, v = self.latent(F)
        return Prediction(mean=mu, var=np.maximum(v, 1e-12) * self.pq_scale**2)

    def diagnostics(self) -> dict:
        names = self.spec.output_names
        if not self.trained:
            return {}
        a, b = self.net.noise_params()
        return {"de_noise_scale": dict(zip(names, map(float, a), strict=True)),
                "de_extra_noise_sd": dict(zip(names, map(float, np.sqrt(b) * self.scale), strict=True)),
                "de_prequential_scale": dict(zip(names, map(float, self.pq_scale), strict=True))}


class SobolDeepEnsemble(FixedDesignMethod):
    """`sobol_de`: the baseline's HF Sobol design, deep-ensemble surrogate."""

    def setup(self, spec, budget, rng) -> None:
        super().setup(spec, budget, rng)  # draws the Sobol design first, so it matches sobol_gp
        self.de = _EnsembleSurrogate(spec, spec.dim, _peek_seed(rng))

    def tell(self, X, fidelity, y, sigma) -> None:
        self.de.record(self.spec.dist.to_unit(X), y, sigma**2)
        super().tell(X, fidelity, y, sigma)

    def predict(self, X: np.ndarray) -> Prediction:
        if self.stale:
            self.de.fit(self.spec.dist.to_unit(self.data.X), self.data.y, self.data.var)
            self.stale = False
        return self.de.predict(self.spec.dist.to_unit(X))

    def diagnostics(self) -> dict:
        return self.de.diagnostics()


class CostAwareDeepEnsemble(CoKrigingAdaptive):
    """`adaptive_iv_mf_de_safe`: safe_pooled's sampling, deep-ensemble predictions."""

    def __init__(self):
        super().__init__(noise_scale=True, prequential=True, kernel="matern", safe_menu=True, pooled_scale=True)

    def setup(self, spec, budget, rng) -> None:
        super().setup(spec, budget, rng)
        self.kfeat = knob_features(spec)
        self.de = _EnsembleSurrogate(spec, spec.dim + self.kfeat.shape[1], _peek_seed(rng))
        self.de_stale = True

    def _features(self, X: np.ndarray, fid: np.ndarray) -> np.ndarray:
        return np.column_stack([self.spec.dist.to_unit(X), self.kfeat[np.asarray(fid, dtype=int)]])

    def tell(self, X, fidelity, y, sigma) -> None:
        self.de.record(self._features(X, fidelity), y, sigma**2)
        super().tell(X, fidelity, y, sigma)  # GP bookkeeping (drives the acquisition)
        self.de_stale = True

    def predict(self, X: np.ndarray) -> Prediction:
        if self.de_stale:
            self.de.fit(self._features(self.data.X, self.data.fidelity), self.data.y, self.data.var)
            self.de_stale = False
        return self.de.predict(self._features(X, np.full(len(X), self.spec.hf)))

    def diagnostics(self) -> dict:
        return super().diagnostics() | self.de.diagnostics()
