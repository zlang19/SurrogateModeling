"""Calibrate toymc_axial against the OpenMC fidelity study (shapes, not absolute values).

    uv run python studies/toy_calibration.py converge   # flat-start convergence at the 4 study points
    uv run python studies/toy_calibration.py grid HF_PARTICLES HF_INACTIVE HF_ACTIVE   # then compares
    uv run python studies/toy_calibration.py compare
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from surrogatemodeling.problems.toymc.axial import run_axial

sys.path.insert(0, "studies")
from fidelity_characterization import study_points  # noqa: E402  (same 4 points as OpenMC)

MULTIPLIERS = ["u235_fis", "u235_cap", "u238_cap", "u238_fastfis", "h_scat", "h_cap", "o_scat", "boron_abs", "nu_thermal", "nu_fast", "clad_abs"]
OUTPUTS = ["k_eff", "axial_offset", "axial_peaking", "capture_to_fission"]


def points():
    return [{**p, **{m: 1.0 for m in MULTIPLIERS}} for p in study_points()]


def _converge(args):
    p_idx, rep = args
    o = run_axial(points()[p_idx], 1000, 0, 300, np.random.default_rng([p_idx, rep]), record=True)
    return p_idx, o["ao_generation"]


def converge():
    with ProcessPoolExecutor(12) as pool:
        res = list(pool.map(_converge, [(p, r) for p in range(4) for r in range(20)]))
    for p in range(4):
        ao = np.mean([a for q, a in res if q == p], axis=0)
        print(f"point {p} mean AO, 10-gen blocks:", np.round(ao.reshape(30, 10).mean(1)[:12], 3), "... tail", round(ao[-100:].mean(), 3))


def _grid_run(args):
    p_idx, cfg, fid, seed = args
    o = run_axial(points()[p_idx], fid["particles"], fid["inactive"], fid["active"], np.random.default_rng([p_idx, seed, 7]))
    return {"point": p_idx, "config": cfg, **{f"knob/{k}": v for k, v in fid.items()},
            **dict(zip(OUTPUTS, o["outputs"])), **{f"sigma/{k}": v for k, v in zip(OUTPUTS, o["sigma"])}}


def grid(n_hf: int, i_hf: int, a_hf: int, reps: int = 25):
    """Same relative design as the OpenMC grid: knobs at 1/10, 1/4, 1/2 of HF (active 1/8, 1/4, 1/2)."""
    hf = {"particles": n_hf, "inactive": i_hf, "active": a_hf}
    configs = {"hf": hf}
    for knob, fracs in {"particles": (0.1, 0.25, 0.5), "inactive": (0.1, 0.25, 0.5), "active": (0.125, 0.25, 0.5)}.items():
        for f in fracs:
            configs[f"{knob}={f:g}"] = {**hf, knob: max(1, round(hf[knob] * f))}
    configs["joint_low"] = {"particles": round(n_hf * 0.1), "inactive": round(i_hf * 0.1), "active": round(a_hf * 0.125)}
    configs["low_particles_low_active"] = {**hf, "particles": round(n_hf * 0.1), "active": round(a_hf * 0.125)}
    tasks = [(p, c, f, r) for p in range(4) for c, f in configs.items() for r in range(reps)]
    tasks += [(p, "reference", {"particles": n_hf * 10, "inactive": i_hf * 2, "active": a_hf}, 999) for p in range(4)]
    with ProcessPoolExecutor(12) as pool:
        df = pd.DataFrame(list(pool.map(_grid_run, tasks, chunksize=4)))
    df.to_parquet("results/openmc_fidelity/toy_grid.parquet")
    return df




def compare(toy: pd.DataFrame) -> None:
    """Under-reporting and bias tables, toy vs OpenMC, by matching relative config."""
    import json

    from fidelity_analysis import by_config, summarize

    toy = toy.copy()
    toy["kind"] = np.where(toy["config"] == "reference", "reference", "grid")
    toy["fidelity"] = [json.dumps({"particles": int(a), "inactive": int(b), "active": int(c)})
                       for a, b, c in zip(toy["knob/particles"], toy["knob/inactive"], toy["knob/active"])]
    toy["cpu_total"] = 1.0
    t = by_config(summarize(toy))
    o = by_config(summarize(pd.read_parquet("results/openmc_fidelity/grid.parquet")))
    # OpenMC config names are absolute ("inactive=10"); map to the toy's relative names by order.
    rename = {}
    for knob in ("particles", "inactive", "active"):
        om = sorted(c for c in o.index if c.startswith(knob + "="))
        om = sorted(om, key=lambda c: float(c.split("=")[1]))
        ty = sorted((c for c in t.index if c.startswith(knob + "=")), key=lambda c: float(c.split("=")[1]))
        rename.update(dict(zip(om, ty)))
    o = o.rename(index=rename)
    for metric, label in (("underreport", "spread ÷ reported σ"), ("bias_hf_sigma", "|bias| in HF-noise units")):
        rows = []
        for cfg in t.index:
            if cfg in o.index:
                rows.append({"config": cfg, **{f"toy {x}": t.loc[cfg, f"{metric}/{x}"] for x in OUTPUTS},
                             **{f"omc {x}": o.loc[cfg, f"{metric}/{x}"] for x in OUTPUTS}})
        print(f"\n## {label}")
        print(pd.DataFrame(rows).set_index("config").round(2).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "converge":
        converge()
    elif sys.argv[1] == "grid":
        compare(grid(*map(int, sys.argv[2:5])))
    else:  # compare
        compare(pd.read_parquet("results/openmc_fidelity/toy_grid.parquet"))
