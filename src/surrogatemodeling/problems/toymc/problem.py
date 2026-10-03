"""The toy MC neutronics problem behind the Problem protocol.

Fidelity scales histories per generation (generation counts are fixed), so cost is linear
in fidelity plus a fixed overhead, and noise is genuine MC noise with batch-statistics σ.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from surrogatemodeling.core import distributions as D
from surrogatemodeling.core.protocols import Observation, ProblemSpec
from surrogatemodeling.problems.analytic import DEFAULT_LADDER, DEFAULT_OVERHEAD, mc_cost
from surrogatemodeling.problems.base import TEST_SET_SEED, TEST_SET_SIZE, cache_dir
from surrogatemodeling.problems.toymc.physics import run_kcode

HF_PARTICLES = 1000  # histories per generation at fidelity 1
MIN_PARTICLES = 50
TRUTH_FIDELITY = 100  # test-set truth runs at 100x high-fidelity histories
_TRUTH_CHUNK = 50
_TRUTH_STREAM = 2

DIST = D.Distribution(
    [
        # Strong / moderate
        D.uniform("enr_inner", 2.0, 5.0),
        D.uniform("enr_outer", 2.0, 5.0),
        D.uniform("boron_ppm", 0.0, 1500.0),
        D.normal("mod_density", 0.72, 0.03),
        D.uniform("core_width", 40.0, 80.0),
        D.uniform("inner_fraction", 0.3, 0.7),
        D.uniform("fuel_vf", 0.30, 0.40),
        D.normal("fuel_temp", 900.0, 100.0),
        # Weak engineering inputs
        D.normal("mod_temp", 580.0, 10.0),
        D.uniform("refl_thickness", 5.0, 30.0),
        D.normal("refl_density", 0.72, 0.03),
        D.uniform("gd_inner", 0.0, 0.3),
        D.normal("clad_abs", 1.0, 0.1),
        # Nuclear-data multipliers
        D.normal("u235_fis", 1.0, 0.01),
        D.normal("u235_cap", 1.0, 0.02),
        D.normal("u238_cap", 1.0, 0.02),
        D.normal("u238_fastfis", 1.0, 0.03),
        D.normal("h_scat", 1.0, 0.01),
        D.normal("h_cap", 1.0, 0.02),
        D.normal("o_scat", 1.0, 0.02),
        D.normal("boron_abs", 1.0, 0.01),
        D.normal("nu_thermal", 1.0, 0.003),
        D.normal("nu_fast", 1.0, 0.005),
    ]
)
OUTPUTS = ["k_eff", "power_ratio", "capture_to_fission"]


def _params(x: np.ndarray) -> dict[str, float]:
    return dict(zip(DIST.names, map(float, x), strict=True))


def _particles(fidelity: float) -> int:
    return max(MIN_PARTICLES, round(HF_PARTICLES * fidelity))


def _run_point(x: np.ndarray, n_particles: int, seed: list[int]) -> tuple[np.ndarray, np.ndarray]:
    est = run_kcode(_params(x), n_particles, np.random.default_rng(seed))
    return est.mean, est.std_err


def _truth_chunk(X: np.ndarray, start: int) -> np.ndarray:
    n = _particles(TRUTH_FIDELITY)
    return np.array([_run_point(x, n, [TEST_SET_SEED, _TRUTH_STREAM, start + i])[0] for i, x in enumerate(X)])


class ToyMCProblem:
    def __init__(self, overhead: float = DEFAULT_OVERHEAD):
        self.spec = ProblemSpec("toymc", DIST, OUTPUTS, partial(mc_cost, overhead=overhead), DEFAULT_LADDER)

    def evaluate(self, X: np.ndarray, fidelity: np.ndarray, rng: np.random.Generator) -> Observation:
        seeds = rng.integers(0, 2**63, size=len(X))
        results = [_run_point(x, _particles(f), [int(s)]) for x, f, s in zip(X, fidelity, seeds, strict=True)]
        return Observation(y=np.array([r[0] for r in results]), sigma=np.array([r[1] for r in results]))

    def test_set(self, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Cached truth at 100x high-fidelity histories, built in parallel and resumable by chunk."""
        root = cache_dir() / "testsets" / f"{self.spec.name}_n{TEST_SET_SIZE}_s{TEST_SET_SEED}"
        final = root.with_suffix(".npz")
        if final.exists():
            data = np.load(final)
            return data["X"], data["Y"]

        X = self.spec.dist.sample(TEST_SET_SIZE, np.random.default_rng(TEST_SET_SEED))
        root.mkdir(parents=True, exist_ok=True)
        starts = list(range(0, TEST_SET_SIZE, _TRUTH_CHUNK))
        todo = [s for s in starts if not (root / f"{s:05d}.npy").exists()]
        if todo:
            print(f"building toymc truth: {len(todo)} chunks of {_TRUTH_CHUNK} points (cached, resumable)")
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futures = {s: pool.submit(_truth_chunk, X[s : s + _TRUTH_CHUNK], s) for s in todo}
                for i, (s, fut) in enumerate(futures.items(), 1):
                    np.save(root / f"{s:05d}.npy", fut.result())
                    print(f"  chunk {i}/{len(todo)}")
        Y = np.vstack([np.load(root / f"{s:05d}.npy") for s in starts])
        np.savez(final, X=X, Y=Y)
        return X, Y
