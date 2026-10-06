"""Project-side wrapper for the OpenMC pin-column model (model.py).

model.py runs in a separate conda env (Python 3.10, OpenMC 0.15); this module only writes
its JSON input, runs it as a subprocess and reads the JSON output. Cost is the child
process's CPU time, so interpreter start-up and cross-section loading — the fixed
per-run overhead — are included, as they would be for MCNP.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from surrogatemodeling.core import distributions as D

OPENMC_PYTHON = Path(os.environ.get("SM_OPENMC_PYTHON", Path.home() / "miniconda3/envs/openmc-env/bin/python"))
CROSS_SECTIONS = Path(
    os.environ.get(
        "SM_OPENMC_CROSS_SECTIONS",
        Path.home() / "Documents/Projects/openMC/endfb81/endfb-viii.1-hdf5/cross_sections.xml",
    )
)
MODEL = Path(__file__).with_name("model.py")
OUTPUTS = ["k_eff", "axial_offset", "axial_peaking", "capture_to_fission"]

DIST = D.Distribution(
    [
        # Fuel
        D.uniform("enr_bottom", 2.5, 4.5),
        D.uniform("enr_middle", 3.0, 5.0),
        D.uniform("enr_top", 2.5, 4.5),
        D.normal("fuel_temp", 900.0, 100.0),
        D.uniform("pellet_radius", 0.400, 0.420),
        # Coolant
        D.normal("mod_density_in", 0.74, 0.01),
        D.uniform("density_drop", 0.03, 0.10),
        D.uniform("boron_ppm", 0.0, 1500.0),
        # Lattice
        D.uniform("pitch", 1.24, 1.30),
        D.uniform("clad_thickness", 0.055, 0.065),
        # Axial
        D.uniform("rod_insertion", 0.0, 100.0),
        D.uniform("refl_top", 10.0, 30.0),
        D.uniform("refl_bottom", 10.0, 30.0),
        # Absorber
        D.uniform("gd_wt", 0.0, 8.0),
    ]
)


def params_from_vector(x: np.ndarray) -> dict[str, float]:
    return dict(zip(DIST.names, map(float, x), strict=True))


def nominal() -> dict[str, float]:
    return {m.name: float(m.rv.mean()) for m in DIST.marginals}


def run_point(
    params: dict[str, float], fidelity: dict[str, int], seed: int, record_batches: bool = False, keep_dir: Path | None = None
) -> dict:
    """Run one OpenMC calculation; returns model.py's output plus `cpu_total` (child CPU s)."""
    with tempfile.TemporaryDirectory(prefix="omc_") as tmp:
        run_dir = Path(keep_dir) if keep_dir else Path(tmp)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "input.json").write_text(
            json.dumps({"params": params, "fidelity": fidelity, "seed": int(seed), "threads": 1, "record_batches": record_batches})
        )
        env = {**os.environ, "OPENMC_CROSS_SECTIONS": str(CROSS_SECTIONS), "OMP_NUM_THREADS": "1"}
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        proc = subprocess.run(
            [str(OPENMC_PYTHON), str(MODEL), str(run_dir)], env=env, capture_output=True, text=True
        )
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        if proc.returncode != 0:
            raise RuntimeError(f"OpenMC run failed ({proc.returncode}): {proc.stderr[-2000:] or proc.stdout[-2000:]}")
        out = json.loads((run_dir / "output.json").read_text())
    out["cpu_total"] = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return out


# --- The OpenMC column behind the Problem protocol (final validation) --------------------------

import hashlib  # noqa: E402
import multiprocessing  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402

from surrogatemodeling.core.protocols import FidelityConfig, Observation, ProblemSpec  # noqa: E402
from surrogatemodeling.problems.base import cache_dir  # noqa: E402

# Measured by studies/fidelity_characterization.py (results/openmc_fidelity/cost_fit.json).
OMC_HF = {"particles": 10000, "inactive": 100, "active": 200}
OMC_OVERHEAD_S, OMC_S_PER_PARTICLE_BATCH = 11.2, 1.193e-4
# Validation truth: 2x particles and 2x inactive cycles of HF (~16 CPU-min each); a 100-point
# test set keeps the reference build to ~27 CPU-h.
OMC_REF = {"particles": 2 * OMC_HF["particles"], "inactive": 2 * OMC_HF["inactive"], "active": OMC_HF["active"]}
OMC_TEST_SIZE, OMC_TEST_SEED = 100, 20261005
PARALLEL_EVALS = 5  # a batch's points run as concurrent single-thread OpenMC processes


def _omc_cost_s(k: dict) -> float:
    return OMC_OVERHEAD_S + OMC_S_PER_PARTICLE_BATCH * k["particles"] * (k["inactive"] + k["active"])


def omc_menu() -> tuple[list[FidelityConfig], int]:
    """The characterization grid's 12 configs (absolute knobs), costs from the measured model."""
    hf = OMC_HF
    configs = {"hf": dict(hf)}
    for knob, values in {"particles": (1000, 2500, 5000), "inactive": (10, 25, 50), "active": (25, 50, 100)}.items():
        for v in values:
            configs[f"{knob}={v}"] = {**hf, knob: v}
    configs["joint_low"] = {"particles": 1000, "inactive": 10, "active": 25}
    configs["low_particles_low_active"] = {**hf, "particles": 1000, "active": 25}
    hf_cost = _omc_cost_s(hf)
    items = sorted(configs.items(), key=lambda kv: _omc_cost_s(kv[1]))
    menu = [FidelityConfig(n, {k: float(v) for k, v in kn.items()}, _omc_cost_s(kn) / hf_cost) for n, kn in items]
    return menu, next(i for i, (n, _) in enumerate(items) if n == "hf")


def _cached_run(params: dict, knobs: dict, seed: int) -> tuple[list, list]:
    """run_point with an on-disk cache keyed by (inputs, knobs, seed). Methods that share a seed
    design (same Sobol points, same noise seeds) then reuse each other's expensive runs."""
    key = hashlib.sha256(json.dumps([params, knobs, int(seed)], sort_keys=True).encode()).hexdigest()[:32]
    path = cache_dir() / "openmc_evals" / f"{key}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return d["y"], d["sigma"]
    out = run_point(params, {k: int(v) for k, v in knobs.items()}, seed)
    y = [out["outputs"][o] for o in OUTPUTS]
    s = [out["sigma"][o] for o in OUTPUTS]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"y": y, "sigma": s, "cpu": out["cpu_total"], "knobs": knobs}))
    os.replace(tmp, path)
    return y, s


def _truth_point(args) -> list:
    params, seed = args
    return _cached_run(params, OMC_REF, seed)[0]


class OpenMCProblem:
    """The 2 m pin column with the 12-config cycle/particle menu, for final validation."""

    def __init__(self):
        fidelities, hf = omc_menu()
        self.spec = ProblemSpec("openmc", DIST, OUTPUTS, fidelities, hf)

    def evaluate(self, X: np.ndarray, fidelity: np.ndarray, rng: np.random.Generator) -> Observation:
        seeds = rng.integers(0, 2**31, size=len(X))
        jobs = [(params_from_vector(x), self.spec.fidelities[i].knobs, int(s)) for x, i, s in zip(X, fidelity, seeds, strict=True)]
        with ThreadPoolExecutor(max_workers=PARALLEL_EVALS) as pool:  # each job is its own OpenMC process
            res = list(pool.map(lambda j: _cached_run(*j), jobs))
        return Observation(y=np.array([r[0] for r in res]), sigma=np.array([r[1] for r in res]))

    def test_set(self, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reference truth for OMC_TEST_SIZE points (cached per point, so the build resumes)."""
        final = cache_dir() / "testsets" / f"openmc_n{OMC_TEST_SIZE}_s{OMC_TEST_SEED}.npz"
        if final.exists():
            d = np.load(final)
            return d["X"], d["Y"]
        X = DIST.sample(OMC_TEST_SIZE, np.random.default_rng(OMC_TEST_SEED))
        jobs = [(params_from_vector(x), OMC_TEST_SEED + i) for i, x in enumerate(X)]
        print(f"building openmc truth: {len(jobs)} reference runs (cached per point)", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            Y = np.array(list(pool.map(_truth_point, jobs)))
        final.parent.mkdir(parents=True, exist_ok=True)
        np.savez(final, X=X, Y=Y)
        return X, Y
