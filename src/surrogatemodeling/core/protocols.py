"""The contracts between problems, methods and the runner.

Arrays are numpy throughout: X is (n, d), fidelity is (n,), y/sigma/mean/var are (n, m).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from surrogatemodeling.core.distributions import Distribution


@dataclass(frozen=True)
class ProblemSpec:
    """Everything a Method may know about a Problem. Never includes the truth function."""

    name: str
    dist: Distribution
    output_names: list[str]
    cost: Callable[[float], float]  # fidelity in (0, 1] -> cost; cost(1.0) == 1
    fidelity_ladder: list[float] | None = None  # optional hint; methods may ignore

    @property
    def dim(self) -> int:
        return self.dist.dim

    @property
    def n_outputs(self) -> int:
        return len(self.output_names)


@dataclass(frozen=True)
class Observation:
    y: np.ndarray  # (n, m) noisy outputs
    sigma: np.ndarray  # (n, m) reported standard error; itself an estimate


@dataclass(frozen=True)
class Prediction:
    mean: np.ndarray  # (n, m)
    var: np.ndarray | None = None  # (n, m) predictive variance of the noise-free output


class Problem(Protocol):
    spec: ProblemSpec

    def evaluate(self, X: np.ndarray, fidelity: np.ndarray, rng: np.random.Generator) -> Observation: ...

    def test_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached (X, Y_true) drawn from spec.dist."""
        ...


class Method(Protocol):
    def setup(self, spec: ProblemSpec, budget: float, rng: np.random.Generator) -> None: ...

    def ask(self, n: int, budget_remaining: float) -> tuple[np.ndarray, np.ndarray]:
        """Return up to n points and their fidelities. Returning zero points ends the run."""
        ...

    def tell(self, X: np.ndarray, fidelity: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> Prediction: ...
