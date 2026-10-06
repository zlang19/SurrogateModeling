"""Hybrid methods: a high-fidelity Sobol seed, then GP-driven greedy batches.

#4  adaptive_iv     integrated-variance reduction, high fidelity only
#4b adaptive_epig   EPIG, high fidelity only
#5  adaptive_iv_mf  integrated-variance reduction per unit cost over (x, fidelity)
#5b adaptive_iv_mf_xn  #5 with a learned extra noise term on top of the reported σ
#6  screen_gp       ARD screening after the seed, then #4 on the active inputs
"""

from __future__ import annotations

import numpy as np

from surrogatemodeling.core.protocols import Prediction, ProblemSpec
from surrogatemodeling.methods.acquisition import greedy_batch
from surrogatemodeling.methods.gp_common import IndependentGPs
from surrogatemodeling.methods.gp_fixed import Dataset, sobol_design

SEED_FRACTION = 0.25
N_REF = 512
N_CAND = 1024


class AdaptiveGP:
    def __init__(self, score: str = "iv", cost_aware: bool = False, extra_noise: bool = False,
                 seed_fraction: float = SEED_FRACTION):
        self.seed_fraction = seed_fraction
        self.score = score
        self.cost_aware = cost_aware
        self.extra_noise = extra_noise

    def setup(self, spec: ProblemSpec, budget: float, rng: np.random.Generator) -> None:
        self.spec, self.rng = spec, rng
        # Options are fidelity config indices: the whole menu when cost-aware, else HF only.
        self.options = np.arange(len(spec.fidelities)) if self.cost_aware else np.array([spec.hf])
        self.costs = np.array([spec.cost(i) for i in self.options])
        self.unit_cost = spec.cost(spec.hf)
        n_seed = max(2, int(np.floor(self.seed_fraction * budget / self.unit_cost)))
        self.queue = sobol_design(spec, n_seed, rng)
        self.ref = spec.dist.sample(N_REF, rng)
        self.data = Dataset(spec)
        self.gp = IndependentGPs(extra_noise=self.extra_noise)
        self.stale = True
        self.last_batch: dict = {}

    # --- protocol ------------------------------------------------------------------------

    def ask(self, n: int, budget_remaining: float) -> tuple[np.ndarray, np.ndarray]:
        if len(self.queue):
            k = min(n, len(self.queue), int(np.floor(budget_remaining / self.unit_cost + 1e-9)))
            X, self.queue = self.queue[:k], self.queue[k:]
            self.last_batch = {"phase": "seed", "fidelities": [self.spec.fidelities[self.spec.hf].name] * k}
            return X, np.full(k, self.spec.hf)
        self._refresh()
        cand = self._candidates()
        U = self._query(self.spec.dist.to_unit(np.vstack([self.ref, cand])))
        tau2 = self._noise_options(len(cand))
        hf = int(np.flatnonzero(self.options == self.spec.hf)[0])
        scores: list[float] = []
        chosen = greedy_batch(
            self.gp.joint_cov(U), N_REF, tau2, self.costs, n, budget_remaining,
            score=self.score, cost_aware=self.cost_aware, target_tau2=tau2[:, hf, 0], scores_out=scores,
        )
        idx = [c for c, _ in chosen]
        fids = self.options[[f for _, f in chosen]]
        self.last_batch = {
            "phase": "adaptive",
            "fidelities": [self.spec.fidelities[i].name for i in fids],
            "best_score": scores[0] if scores else None,
            "hf_noise_sd": dict(zip(self.spec.output_names, map(float, np.sqrt(self._hf_var)), strict=True)),
        }
        return cand[idx].reshape(-1, self.spec.dim), fids

    def diagnostics(self) -> dict:
        return {**self.gp.diagnostics(self.spec.dist.names, self.spec.output_names), "batch": self.last_batch}

    def tell(self, X, fidelity, y, sigma) -> None:
        self.data.add(X, fidelity, y, sigma)
        self.stale = True

    def predict(self, X: np.ndarray) -> Prediction:
        self._refresh()
        mean, var = self.gp.predict(self._query(self.spec.dist.to_unit(X)))
        return Prediction(mean=mean, var=var)

    def _query(self, U: np.ndarray) -> np.ndarray:
        """Model inputs for predicting the high-fidelity output at unit-box points U."""
        return U

    # --- internals -----------------------------------------------------------------------

    def _refresh(self) -> None:
        if self.stale:
            self._fit(self.spec.dist.to_unit(self.data.X))
            self.stale = False

    def _fit(self, U: np.ndarray) -> None:
        self.gp.fit(U, self.data.y, self.data.var)

    def _candidates(self) -> np.ndarray:
        """Half uniform over the box, half from the input distribution (clipped to the box)."""
        lo, hi = self.spec.dist.bounds()
        half = N_CAND // 2
        uniform = self.spec.dist.from_unit(self.rng.random((half, self.spec.dim)))
        from_dist = np.clip(self.spec.dist.sample(N_CAND - half, self.rng), lo, hi)
        return np.vstack([uniform, from_dist])

    def _noise_options(self, n_cand: int) -> np.ndarray:
        """(m, F, C) expected noise variance in standardized units, per fidelity option.

        A config already run uses the median reported variance of its own points (so
        cycle-dependent noise is learned, not assumed). An untried config falls back to
        MC scaling: the HF-equivalent variance (median of var x relative histories over all
        data) divided by its relative histories.
        """
        rel = np.array([self.spec.relative_histories(i) for i in self.data.fidelity])
        hf_equiv = np.median(self.data.var * rel[:, None], axis=0)  # (m,)
        tau2 = np.empty((self.spec.n_outputs, len(self.options)))
        for j, opt in enumerate(self.options):
            seen = self.data.fidelity == opt
            tau2[:, j] = (
                np.median(self.data.var[seen], axis=0) if seen.any() else hf_equiv / self.spec.relative_histories(opt)
            )
        self._hf_var = hf_equiv
        tau2 = self.gp.model_var(tau2)  # reported variances: output units -> model scale
        extra = self.gp.extra_noise_var()
        if extra is not None:  # learned extra noise is already on the model's (possibly log) scale
            tau2 = tau2 + extra[:, None] / self.gp.scale[:, None] ** 2
        return np.repeat(tau2[:, :, None], n_cand, axis=2)


class ScreenedGP(AdaptiveGP):
    """ARD screening once the seed design is in, then an extra-noise GP on the active inputs.

    Until the seed is complete the GP uses every input (a handful of points cannot rank
    20+ inputs). Re-screens whenever the data has doubled since the last screen. Ignored
    inputs act as noise, which the learned extra-noise term absorbs.
    """

    COVERAGE = 0.99  # keep inputs until this share of ARD relevance is covered
    MAX_ACTIVE = 12

    def __init__(self, score: str = "iv"):
        super().__init__(score=score, extra_noise=True)

    def setup(self, spec, budget, rng) -> None:
        super().setup(spec, budget, rng)
        self.n_seed = len(self.queue)
        self.active = None
        self._n_at_screen = 0

    def _fit(self, U: np.ndarray) -> None:
        if len(U) < self.n_seed:
            self.gp.fit(U, self.data.y, self.data.var)
            return
        if self.active is None or len(U) >= 2 * self._n_at_screen:
            self.active = self._screen(U)
            self._n_at_screen = len(U)
        self.gp.fit(U, self.data.y, self.data.var, active=self.active)

    def _screen(self, U: np.ndarray) -> np.ndarray:
        full = IndependentGPs(extra_noise=True)
        full.fit(U, self.data.y, self.data.var)
        rel = full.inverse_lengthscales_sq()
        share = (rel / rel.sum(axis=1, keepdims=True)).max(axis=0)  # most relevant to any output
        order = np.argsort(-share)
        n_keep = int(np.searchsorted(np.cumsum(share[order]) / share.sum(), self.COVERAGE) + 1)
        n_keep = int(np.clip(n_keep, min(2, U.shape[1]), self.MAX_ACTIVE))
        return np.sort(order[:n_keep])

    def diagnostics(self) -> dict:
        names = self.spec.dist.names
        active = [names[i] for i in self.active] if self.active is not None else list(names)
        return {**super().diagnostics(), "active_inputs": active}

    def predict(self, X: np.ndarray) -> Prediction:
        # The extra-noise term mostly stands in for the screened-out inputs, which are part
        # of the true function, so it belongs in the predictive variance of f.
        pred = super().predict(X)
        if self.active is None or len(self.active) == self.spec.dim:
            return pred
        return Prediction(mean=pred.mean, var=pred.var + self.gp.extra_noise_var()[None, :])
