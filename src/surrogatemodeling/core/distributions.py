"""Input distributions: independent marginals over a declared training box.

The box (`bounds()`) is where methods may sample; it covers the bulk of the probability
mass. Test sets are drawn from the distribution itself, which is what pdf-weights the error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Marginal:
    """A scipy frozen distribution plus the training interval for that input."""

    name: str
    rv: Any  # scipy frozen continuous distribution
    lower: float
    upper: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.rv.rvs(size=n, random_state=rng)


def uniform(name: str, lo: float, hi: float) -> Marginal:
    return Marginal(name, stats.uniform(lo, hi - lo), lo, hi)


def normal(name: str, mean: float, sd: float, k: float = 4.0) -> Marginal:
    """Normal with a training box of mean ± k·sd."""
    return Marginal(name, stats.norm(mean, sd), mean - k * sd, mean + k * sd)


def lognormal(name: str, mu: float, s: float, k: float = 4.0) -> Marginal:
    """Lognormal with log-mean mu and log-sd s; box is exp(mu ± k·s)."""
    return Marginal(name, stats.lognorm(s, scale=np.exp(mu)), float(np.exp(mu - k * s)), float(np.exp(mu + k * s)))


def truncnormal(name: str, mean: float, sd: float, lo: float, hi: float) -> Marginal:
    a, b = (lo - mean) / sd, (hi - mean) / sd
    return Marginal(name, stats.truncnorm(a, b, loc=mean, scale=sd), lo, hi)


@dataclass(frozen=True)
class Distribution:
    marginals: list[Marginal] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.marginals)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.marginals]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.column_stack([m.sample(n, rng) for m in self.marginals])

    def bounds(self) -> np.ndarray:
        """(2, d) array: row 0 lower, row 1 upper."""
        return np.array([[m.lower for m in self.marginals], [m.upper for m in self.marginals]])

    def to_unit(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds()
        return (X - lo) / (hi - lo)

    def from_unit(self, U: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds()
        return lo + U * (hi - lo)
