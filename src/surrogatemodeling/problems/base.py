"""Shared problem machinery: MC-like noise and cached test sets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from surrogatemodeling.core.protocols import Observation, ProblemSpec

TEST_SET_SIZE = 2000
TEST_SET_SEED = 20261003


def cache_dir() -> Path:
    return Path(os.environ.get("SM_CACHE_DIR", "results/cache"))


@dataclass(frozen=True)
class NoiseModel:
    """MC-style noise at fidelity f (fraction of high-fidelity histories):

        sigma = sqrt((rel·|f(x)|)² + abs²) / sqrt(f)

    The reported sigma is itself an estimate, scaled by sqrt(chi²_dof / dof) as a
    batch-statistics standard error would be.
    """

    rel: float = 0.01
    abs: float | tuple[float, ...] = 0.0  # scalar, or one per output
    sigma_dof: int = 20

    def true_sigma(self, y: np.ndarray, fidelity: np.ndarray) -> np.ndarray:
        return np.sqrt((self.rel * y) ** 2 + np.asarray(self.abs) ** 2) / np.sqrt(fidelity)[:, None]

    def observe(self, y: np.ndarray, fidelity: np.ndarray, rng: np.random.Generator) -> Observation:
        sigma = self.true_sigma(y, fidelity)
        y_obs = y + sigma * rng.standard_normal(y.shape)
        sigma_hat = sigma * np.sqrt(rng.chisquare(self.sigma_dof, y.shape) / self.sigma_dof)
        return Observation(y=y_obs, sigma=sigma_hat)


class CachedTestSet:
    """Mixin: draws the test set from spec.dist with a fixed seed and caches it to disk."""

    spec: ProblemSpec

    def truth(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def test_set(self) -> tuple[np.ndarray, np.ndarray]:
        path = cache_dir() / "testsets" / f"{self.spec.name}_n{TEST_SET_SIZE}_s{TEST_SET_SEED}.npz"
        if path.exists():
            data = np.load(path)
            return data["X"], data["Y"]
        X = self.spec.dist.sample(TEST_SET_SIZE, np.random.default_rng(TEST_SET_SEED))
        Y = self.truth(X)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X, Y=Y)
        return X, Y
