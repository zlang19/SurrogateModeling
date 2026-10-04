"""OpenMC fidelity characterization: what do particles, inactive and active cycles cost,
and what noise, sigma under-reporting and bias do they produce?

    uv run python studies/fidelity_characterization.py pilot      # ~10 min: cost fit + convergence
    uv run python studies/fidelity_characterization.py grid       # main study (size set from the pilot)
    uv run python studies/fidelity_characterization.py analyze    # tables + plots -> results/openmc_fidelity/

Runs are independent OpenMC subprocesses (one thread each) in a process pool. Each stage
writes one row per run to results/openmc_fidelity/<stage>.parquet and skips rows already
done, so an interrupted stage resumes. Launch long stages as a memory-capped systemd
service (see README, "Running long experiments").
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from surrogatemodeling.problems.openmc.problem import DIST, OUTPUTS, nominal, params_from_vector, run_point

OUT = Path("results/openmc_fidelity")
WORKERS = int(os.environ.get("SM_WORKERS", "12"))


def study_points() -> list[dict[str, float]]:
    """Four fixed input points: nominal, rod out, rod deep, and a random draw."""
    rng = np.random.default_rng(20261004)
    draws = [params_from_vector(x) for x in DIST.sample(3, rng)]
    draws[0]["rod_insertion"] = 0.0
    draws[1]["rod_insertion"] = 90.0
    return [nominal(), *draws]


# --- task running ---------------------------------------------------------------------------


def _task(task: dict) -> dict:
    t0 = time.time()
    out = run_point(task["params"], task["fidelity"], task["seed"], record_batches=task.get("record", False))
    row = {k: v for k, v in task.items() if k != "params"}
    row.update(fidelity=json.dumps(task["fidelity"]), wall=time.time() - t0, cpu_total=out["cpu_total"], cpu_model=out["cpu_time"])
    for name in OUTPUTS:
        row[name] = out["outputs"][name]
        row[f"sigma/{name}"] = out["sigma"][name]
    for key in ("k_generation", "entropy", "ao_batches", "axial_power"):
        if key in out:
            row[key] = json.dumps(out[key])
    row.update({f"knob/{k}": v for k, v in task["fidelity"].items()})
    return row


def run_tasks(stage: str, tasks: list[dict]) -> pd.DataFrame:
    path = OUT / f"{stage}.parquet"
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = set(done["task_id"]) if len(done) else set()
    todo = [t for t in tasks if t["task_id"] not in have]
    print(f"[{stage}] {len(todo)} runs to do ({len(have)} already done)", flush=True)
    rows, last_save = [], time.time()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        futures = {pool.submit(_task, t): t for t in sorted(todo, key=lambda t: -t.get("est_cpu", 0))}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                rows.append(fut.result())
                print(f"[{stage}] {i}/{len(todo)} done {t['task_id']}", flush=True)
            except Exception as e:  # keep going; a re-run retries it
                print(f"[{stage}] {i}/{len(todo)} FAILED {t['task_id']}: {e!r}", flush=True)
            if rows and (time.time() - last_save > 60 or i == len(todo)):
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                OUT.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                done.to_parquet(tmp)
                tmp.rename(path)
                rows, last_save = [], time.time()
    return done


# --- stages ---------------------------------------------------------------------------------


def pilot_tasks() -> list[dict]:
    points = study_points()
    tasks = []
    # Cost: CPU vs particles x batches (fixed overhead = intercept).
    for n in (500, 2000, 8000, 32000):
        for b in (20, 80):
            tasks.append({"task_id": f"cost_n{n}_b{b}", "kind": "cost", "point": 0, "seed": 1, "params": points[0],
                          "fidelity": {"particles": n, "inactive": b // 2, "active": b - b // 2}, "est_cpu": n * b * 6e-5})
    # Convergence: flat start, every batch active, per-batch entropy / axial offset recorded.
    for p, params in enumerate(points):
        for n in (2000, 10000):
            for seed in (1, 2):
                tasks.append({"task_id": f"conv_p{p}_n{n}_s{seed}", "kind": "convergence", "point": p, "seed": seed,
                              "params": params, "record": True,
                              "fidelity": {"particles": n, "inactive": 0, "active": 400}, "est_cpu": n * 400 * 6e-5})
    return tasks


def grid_tasks(hf: dict, levels: dict, replicates: int, ref_mult: int) -> list[dict]:
    """One-at-a-time knob sweeps around HF plus joint low settings, replicated, plus references."""
    points = study_points()
    configs = {"hf": dict(hf)}
    for knob, values in levels.items():
        for v in values:
            if v != hf[knob]:
                configs[f"{knob}={v}"] = {**hf, knob: v}
    lo = {k: min(v) for k, v in levels.items()}
    configs["joint_low"] = dict(lo)
    configs["low_particles_low_active"] = {**hf, "particles": lo["particles"], "active": lo["active"]}
    tasks = []
    for p, params in enumerate(points):
        for name, fid in configs.items():
            for r in range(replicates):
                tasks.append({"task_id": f"grid_p{p}_{name}_r{r}", "kind": "grid", "config": name, "point": p, "seed": 100 + r,
                              "params": params, "fidelity": fid,
                              "est_cpu": fid["particles"] * (fid["inactive"] + fid["active"]) * 6e-5})
        ref = {"particles": hf["particles"] * ref_mult, "inactive": hf["inactive"] * 2, "active": hf["active"]}
        tasks.append({"task_id": f"ref_p{p}", "kind": "reference", "config": "reference", "point": p, "seed": 999,
                      "params": params, "fidelity": ref, "est_cpu": ref["particles"] * (ref["inactive"] + ref["active"]) * 6e-5})
    return tasks


GRID_FILE = OUT / "grid_spec.json"


def est_cpu(fid: dict, spec: dict) -> float:
    c = spec["cost"]
    return c["overhead_s"] + c["s_per_particle_batch"] * fid["particles"] * (fid["inactive"] + fid["active"])  # written once the pilot has been analyzed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "grid", "analyze"])
    args = ap.parse_args()
    if args.stage == "pilot":
        run_tasks("pilot", pilot_tasks())
    elif args.stage == "grid":
        spec = json.loads(GRID_FILE.read_text())
        tasks = grid_tasks(spec["hf"], spec["levels"], spec["replicates"], spec["ref_mult"])
        for t in tasks:
            t["est_cpu"] = est_cpu(t["fidelity"], spec)
        est = sum(t["est_cpu"] for t in tasks)
        print(f"[grid] {len(tasks)} runs, estimated {est / 3600:.1f} CPU-h = {est / 3600 / WORKERS:.1f} h on {WORKERS} workers", flush=True)
        run_tasks("grid", tasks)
    else:
        from fidelity_analysis import analyze  # sibling module; studies/ is on sys.path when run as a script

        analyze(OUT)


if __name__ == "__main__":
    main()
