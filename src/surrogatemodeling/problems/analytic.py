"""Analytic benchmark functions wrapped with MC-style noise.

Truth is exact. Cost follows MC: proportional to histories plus a small fixed per-run
overhead, so cost(fidelity) = overhead + (1 - overhead) * fidelity and cost(1) = 1.

Function definitions and input distributions follow the Virtual Library of Simulation
Experiments (Surjanovic & Bingham, sfu.ca/~ssurjano).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from surrogatemodeling.core import distributions as D
from surrogatemodeling.core.protocols import FidelityConfig, Observation, ProblemSpec
from surrogatemodeling.problems.base import CachedTestSet, NoiseModel

DEFAULT_LADDER = [1 / 16, 1 / 4, 1.0]
DEFAULT_OVERHEAD = 0.01
DEFAULT_REL_NOISE = 0.01
DEFAULT_ABS_NOISE_FRAC = 1e-3  # absolute noise floor, as a fraction of each output's std
_STATS_SAMPLES, _STATS_SEED = 20000, 7


def mc_cost(fidelity: float, overhead: float = DEFAULT_OVERHEAD) -> float:
    return overhead + (1 - overhead) * float(fidelity)


def history_menu(ladder: list[float] = DEFAULT_LADDER, overhead: float = DEFAULT_OVERHEAD) -> tuple[list[FidelityConfig], int]:
    """One-knob menu: fractions of high-fidelity histories, cheapest first; returns (menu, hf index)."""
    menu = [FidelityConfig(f"h={f:g}", {"histories": float(f)}, mc_cost(f, overhead)) for f in sorted(ladder)]
    return menu, len(menu) - 1


class AnalyticProblem(CachedTestSet):
    def __init__(
        self,
        name: str,
        dist: D.Distribution,
        output_names: list[str],
        fn: Callable[[np.ndarray], np.ndarray],
        noise: NoiseModel | None = None,
        overhead: float = DEFAULT_OVERHEAD,
    ):
        menu, hf = history_menu(overhead=overhead)
        self.spec = ProblemSpec(name, dist, output_names, menu, hf)
        self.fn = fn
        self.overhead = overhead
        if noise is None:
            noise = NoiseModel(rel=DEFAULT_REL_NOISE, abs=tuple(DEFAULT_ABS_NOISE_FRAC * self.output_std()))
        self.noise = noise

    def truth(self, X: np.ndarray) -> np.ndarray:
        return self.fn(X).reshape(len(X), -1)

    def output_std(self) -> np.ndarray:
        """Std of each output under the input distribution (fixed-seed estimate)."""
        X = self.spec.dist.sample(_STATS_SAMPLES, np.random.default_rng(_STATS_SEED))
        return np.std(self.truth(X), axis=0)

    def evaluate(self, X: np.ndarray, fidelity: np.ndarray, rng: np.random.Generator) -> Observation:
        frac = np.array([self.spec.fidelities[i].knobs["histories"] for i in fidelity])
        return self.noise.observe(self.truth(X), frac, rng)


# --- Borehole (8D) ---------------------------------------------------------------------


def _borehole(X: np.ndarray) -> np.ndarray:
    rw, r, Tu, Hu, Tl, Hl, L, Kw = X.T
    log_r = np.log(r / rw)
    return 2 * np.pi * Tu * (Hu - Hl) / (log_r * (1 + 2 * L * Tu / (log_r * rw**2 * Kw) + Tu / Tl))


def borehole(noise: NoiseModel | None = None) -> AnalyticProblem:
    """Borehole water flow, with the UQ-literature input distributions."""
    dist = D.Distribution(
        [
            D.normal("rw", 0.10, 0.0161812, k=3),
            D.lognormal("r", 7.71, 1.0056, k=3),
            D.uniform("Tu", 63070, 115600),
            D.uniform("Hu", 990, 1110),
            D.uniform("Tl", 63.1, 116),
            D.uniform("Hl", 700, 820),
            D.uniform("L", 1120, 1680),
            D.uniform("Kw", 9855, 12045),
        ]
    )
    return AnalyticProblem("borehole", dist, ["flow"], _borehole, noise)


# --- Wing weight (10D) -----------------------------------------------------------------


def _wing_weight(X: np.ndarray) -> np.ndarray:
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = X.T
    cos = np.cos(np.deg2rad(Lam))
    return (
        0.036 * Sw**0.758 * Wfw**0.0035 * (A / cos**2) ** 0.6 * q**0.006 * lam**0.04
        * (100 * tc / cos) ** -0.3 * (Nz * Wdg) ** 0.49
        + Sw * Wp
    )


def wing_weight(noise: NoiseModel | None = None) -> AnalyticProblem:
    """Light aircraft wing weight; a few inputs (Nz, A, Wdg, tc) dominate."""
    dist = D.Distribution(
        [
            D.uniform("Sw", 150, 200),
            D.uniform("Wfw", 220, 300),
            D.uniform("A", 6, 10),
            D.uniform("Lambda", -10, 10),
            D.uniform("q", 16, 45),
            D.uniform("lambda", 0.5, 1),
            D.uniform("tc", 0.08, 0.18),
            D.uniform("Nz", 2.5, 6),
            D.uniform("Wdg", 1700, 2500),
            D.uniform("Wp", 0.025, 0.08),
        ]
    )
    return AnalyticProblem("wing_weight", dist, ["weight"], _wing_weight, noise)


# --- OTL circuit (6D) ------------------------------------------------------------------


def _otl(X: np.ndarray) -> np.ndarray:
    Rb1, Rb2, Rf, Rc1, Rc2, beta = X.T
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    b = beta * (Rc2 + 9)
    denom = b + Rf
    return (Vb1 + 0.74) * b / denom + 11.35 * Rf / denom + 0.74 * Rf * b / (denom * Rc1)


def otl(noise: NoiseModel | None = None) -> AnalyticProblem:
    """Output transformerless push-pull circuit midpoint voltage."""
    dist = D.Distribution(
        [
            D.uniform("Rb1", 50, 150),
            D.uniform("Rb2", 25, 70),
            D.uniform("Rf", 0.5, 3),
            D.uniform("Rc1", 1.2, 2.5),
            D.uniform("Rc2", 0.25, 1.2),
            D.uniform("beta", 50, 300),
        ]
    )
    return AnalyticProblem("otl", dist, ["vm"], _otl, noise)


# --- Morris (20D) ----------------------------------------------------------------------


def _morris_coefficients(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """First- and second-order coefficients: fixed large ones plus fixed-seed N(0,1) rest."""
    rng = np.random.default_rng(seed)
    b1 = rng.standard_normal(20)
    b1[:10] = 20
    b2 = np.triu(rng.standard_normal((20, 20)), k=1)
    b2[:6, :6] = np.triu(np.full((6, 6), -15.0), k=1)
    return b1, b2


_MORRIS_B1, _MORRIS_B2 = _morris_coefficients()


def _morris(X: np.ndarray) -> np.ndarray:
    w = 2 * (X - 0.5)
    for i in (2, 4, 6):  # x3, x5, x7 (1-based) are warped
        w[:, i] = 2 * (1.1 * X[:, i] / (X[:, i] + 0.1) - 0.5)
    y = w @ _MORRIS_B1 + np.einsum("ni,ij,nj->n", w, _MORRIS_B2, w)
    # Third order: -10 over i<j<l in the first 5; fourth order: +5 over i<j<l<s in the first 4.
    for i in range(5):
        for j in range(i + 1, 5):
            for k in range(j + 1, 5):
                y -= 10 * w[:, i] * w[:, j] * w[:, k]
    y += 5 * w[:, 0] * w[:, 1] * w[:, 2] * w[:, 3]
    return y


def morris(noise: NoiseModel | None = None) -> AnalyticProblem:
    """Morris (1991) screening function: 10 important inputs with interactions, 10 weak."""
    dist = D.Distribution([D.uniform(f"x{i + 1}", 0, 1) for i in range(20)])
    return AnalyticProblem("morris", dist, ["y"], _morris, noise)


# --- Padding to high dimension ---------------------------------------------------------


def padded(
    base: AnalyticProblem,
    total_dim: int,
    weak_fraction: float = 0.5,
    weak_var_frac: float = 0.005,
    seed: int = 0,
) -> AnalyticProblem:
    """Append U[0,1] inputs until `total_dim`: some weak, the rest inert.

    The weak inputs add zero-mean, unit-variance linear or quadratic terms whose summed
    variance is `weak_var_frac` of each output's variance. A model that ignores them
    entirely is floored at NRMSE ≈ sqrt(weak_var_frac).
    """
    d0 = base.spec.dim
    n_pad = total_dim - d0
    if n_pad <= 0:
        raise ValueError(f"total_dim {total_dim} must exceed base dim {d0}")
    n_weak = round(weak_fraction * n_pad)
    rng = np.random.default_rng(seed)
    std = base.output_std()  # (m,)
    shares = rng.dirichlet(np.ones(n_weak)) if n_weak else np.empty(0)  # each weak input's share of the variance
    coef = np.sqrt(weak_var_frac * shares)[:, None] * std[None, :]  # (n_weak, m)
    quadratic = np.arange(n_weak) % 2 == 1

    def fn(X: np.ndarray) -> np.ndarray:
        Z = X[:, d0 : d0 + n_weak] - 0.5
        H = np.where(quadratic, np.sqrt(180) * (Z**2 - 1 / 12), np.sqrt(12) * Z)
        return base.truth(X[:, :d0]) + H @ coef

    dist = D.Distribution(base.spec.dist.marginals + [D.uniform(f"pad{k}", 0, 1) for k in range(n_pad)])
    return AnalyticProblem(f"{base.spec.name}_d{total_dim}", dist, base.spec.output_names, fn, base.noise, base.overhead)
