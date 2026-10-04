"""`toymc_axial` behind the Problem protocol, calibrated to the OpenMC fidelity study.

Calibration (studies/toy_calibration.py vs results/openmc_fidelity/):
- High fidelity is 2000 particles x (70 inactive + 140 active): the flat-start source
  settles in ~35 generations, and HF uses 2x that, as the OpenMC HF does (settle ~50, HF 100).
- Fidelity menu: the same 12 relative configs as the OpenMC grid.
- Cost: OpenMC's measured shape, a fixed 3% of HF plus particles x (inactive + active).
  Cutting active cycles alone therefore still pays for the inactive ones.
- Matches OpenMC's shapes: axial offset sigma under-reported ~5x (OpenMC ~4x), peaking ~2x,
  k / capture-to-fission ~1.2-1.4x; strong bias only at joint-low settings, peaking
  biased upward at few particles. The toy is somewhat harsher than OpenMC (about 2x the
  bias at the joint-low config).
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from surrogatemodeling.core import distributions as D
from surrogatemodeling.core.protocols import FidelityConfig, Observation, ProblemSpec
from surrogatemodeling.problems.base import TEST_SET_SEED, TEST_SET_SIZE, cache_dir
from surrogatemodeling.problems.openmc.problem import DIST as OPENMC_DIST
from surrogatemodeling.problems.toymc.axial import run_axial

OUTPUTS = ["k_eff", "axial_offset", "axial_peaking", "capture_to_fission"]
HF = {"particles": 2000, "inactive": 70, "active": 140}
OVERHEAD_FRAC = 0.03  # OpenMC: 11.2 s of a 374 s HF run
REF = {"particles": HF["particles"] * 10, "inactive": HF["inactive"] * 2, "active": HF["active"]}
_TRUTH_CHUNK, _TRUTH_STREAM = 50, 3

DIST = D.Distribution(
    OPENMC_DIST.marginals
    + [
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
        D.normal("clad_abs", 1.0, 0.1),
    ]
)


def _cost(knobs: dict) -> float:
    work = lambda k: k["particles"] * (k["inactive"] + k["active"])
    o = OVERHEAD_FRAC * work(HF)
    return (o + work(knobs)) / (o + work(HF))


def menu() -> tuple[list[FidelityConfig], int]:
    """The OpenMC grid's 12 configs, relative to HF; cheapest first."""
    configs = {"hf": dict(HF)}
    for knob, fracs in {"particles": (0.1, 0.25, 0.5), "inactive": (0.1, 0.25, 0.5), "active": (0.125, 0.25, 0.5)}.items():
        for f in fracs:
            configs[f"{knob}={f:g}"] = {**HF, knob: max(1, round(HF[knob] * f))}
    configs["joint_low"] = {"particles": round(HF["particles"] * 0.1), "inactive": round(HF["inactive"] * 0.1), "active": round(HF["active"] * 0.125)}
    configs["low_particles_low_active"] = {**HF, "particles": round(HF["particles"] * 0.1), "active": round(HF["active"] * 0.125)}
    items = sorted(configs.items(), key=lambda kv: _cost(kv[1]))
    fidelities = [FidelityConfig(name, {k: float(v) for k, v in knobs.items()}, _cost(knobs)) for name, knobs in items]
    return fidelities, next(i for i, (name, _) in enumerate(items) if name == "hf")


def _params(x: np.ndarray) -> dict[str, float]:
    return dict(zip(DIST.names, map(float, x), strict=True))


def _run(x: np.ndarray, knobs: dict, seed) -> tuple[np.ndarray, np.ndarray]:
    o = run_axial(_params(x), int(knobs["particles"]), int(knobs["inactive"]), int(knobs["active"]), np.random.default_rng(seed))
    return o["outputs"], o["sigma"]


def _truth_chunk(X: np.ndarray, start: int) -> np.ndarray:
    return np.array([_run(x, REF, [TEST_SET_SEED, _TRUTH_STREAM, start + i])[0] for i, x in enumerate(X)])


class ToyMCAxialProblem:
    def __init__(self):
        fidelities, hf = menu()
        self.spec = ProblemSpec("toymc_axial", DIST, OUTPUTS, fidelities, hf)

    def evaluate(self, X: np.ndarray, fidelity: np.ndarray, rng: np.random.Generator) -> Observation:
        seeds = rng.integers(0, 2**63, size=len(X))
        res = [_run(x, self.spec.fidelities[i].knobs, [int(s)]) for x, i, s in zip(X, fidelity, seeds, strict=True)]
        return Observation(y=np.array([r[0] for r in res]), sigma=np.array([r[1] for r in res]))

    def test_set(self, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Truth = reference runs (10x particles, 2x inactive), cached and resumable by chunk."""
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
            print(f"building toymc_axial truth: {len(todo)} chunks of {_TRUTH_CHUNK} (cached, resumable)", flush=True)
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {s: pool.submit(_truth_chunk, X[s : s + _TRUTH_CHUNK], s) for s in todo}
                for i, (s, fut) in enumerate(futures.items(), 1):
                    np.save(root / f"{s:05d}.npy", fut.result())
                    print(f"  chunk {i}/{len(todo)}", flush=True)
        Y = np.vstack([np.load(root / f"{s:05d}.npy") for s in starts])
        np.savez(final, X=X, Y=Y)
        return X, Y
