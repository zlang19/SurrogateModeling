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
