"""The contracts between problems, methods and the runner.

Interface v2 (2026-10-04): fidelity is a choice from a menu of configurations declared by
the problem (e.g. MC particles per cycle, inactive and active cycles), not a scalar. A
method asks for a config *index* per point; costs are in high-fidelity run units.

Arrays are numpy throughout: X is (n, d), fidelity is an (n,) int array of config indices,
y/sigma/mean/var are (n, m).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from surrogatemodeling.core.distributions import Distribution


@dataclass(frozen=True)
class FidelityConfig:
    """One runnable fidelity setting. Knob names are problem-specific, e.g. {"histories": 0.25}
    or {"particles": 1000, "inactive": 25, "active": 100}."""

    name: str
    knobs: dict[str, float] = field(default_factory=dict)
    cost: float = 1.0  # in high-fidelity run units


@dataclass(frozen=True)
class ProblemSpec:
    """Everything a Method may know about a Problem. Never includes the truth function."""

    name: str
    dist: Distribution
    output_names: list[str]
    fidelities: list[FidelityConfig]
    hf: int  # index of the high-fidelity config (cost 1)

    @property
    def dim(self) -> int:
        return self.dist.dim

    @property
    def n_outputs(self) -> int:
        return len(self.output_names)

    def cost(self, idx: int) -> float:
        return self.fidelities[int(idx)].cost

    def relative_histories(self, idx: int) -> float:
        """Scored histories of config `idx` relative to high fidelity (MC variance scales as
        its inverse). Uses a "histories" knob, else particles x active cycles; 1 if unknown."""
        k, h = self.fidelities[int(idx)].knobs, self.fidelities[self.hf].knobs
        if "histories" in k:
            return k["histories"] / h["histories"]
        if "particles" in k and "active" in k:
            return (k["particles"] * k["active"]) / (h["particles"] * h["active"])
        return 1.0


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
        """Return up to n points and their fidelity config indices. Zero points ends the run."""
        ...

    def tell(self, X: np.ndarray, fidelity: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> Prediction: ...
