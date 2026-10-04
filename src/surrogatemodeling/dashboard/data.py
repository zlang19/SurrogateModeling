"""Read experiment results and live logs; derive run states, curves and ETAs.

Everything here is read-only and works on whatever is on disk: finished runs' Parquet
files, live `.jsonl`/`.status.json` files, and per-invocation manifests (see core/live.py).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from surrogatemodeling.registry import METHODS
from surrogatemodeling.report.plots import curves_on_grid

STALE_S = 120.0  # heartbeat age after which a "running" run is considered dead
GRID_POINTS = 80
METRICS = ("nrmse", "coverage", "nll", "max_error")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_run_id(rid: str) -> tuple[str, str, int]:
    problem, method, seed = rid.split("__")
    return problem, method, int(seed.removeprefix("s"))


def method_order(methods) -> list[str]:
    present = set(methods)
    return [m for m in METHODS if m in present] + sorted(present - set(METHODS))


def method_index(method: str) -> int:
    """Fixed palette slot: registry position, so colors never depend on what is shown."""
    names = list(METHODS)
    return names.index(method) if method in names else -1


def jsonable(obj):
    """Replace NaN/inf with None, recursively, so browsers can JSON.parse the result."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def systemd_services(pattern: str = "sm-*") -> list[dict]:
    try:
        out = subprocess.run(
            ["systemctl", "--user", "list-units", pattern, "--all", "--plain", "--no-legend", "--no-pager"],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    services = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) >= 4:
            services.append({"unit": parts[0], "active": parts[2], "sub": parts[3]})
    return services


@dataclass
class Snapshot:
    """One consistent read of an experiment directory."""

    rows: pd.DataFrame  # every batch row: finished runs + partial live runs
    runs: pd.DataFrame  # one row per run: state, progress, timing
    manifests: list[dict]
    now: float


class Store:
    """Caches file reads by (mtime, size) so polling stays cheap."""

    def __init__(self, results: Path):
        self.results = results
        self._files: dict[Path, tuple[tuple[float, int], object]] = {}

    def _cached(self, path: Path, loader):
        try:
            st = path.stat()
        except FileNotFoundError:
            return None
        key = (st.st_mtime, st.st_size)
        hit = self._files.get(path)
        if hit and hit[0] == key:
            return hit[1]
        try:
            value = loader(path)
        except (ValueError, OSError, json.JSONDecodeError):
            return hit[1] if hit else None  # mid-write; keep the last good read
        self._files[path] = (key, value)
        return value

    @staticmethod
    def _read_jsonl(path: Path) -> pd.DataFrame:
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(lines)

    # --- experiments -----------------------------------------------------------------------

    def experiment_dir(self, name: str) -> Path:
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"bad experiment name {name!r}")
        path = self.results / name
        if not path.is_dir():
            raise ValueError(f"no experiment {name!r}")
        return path

    def experiments(self) -> list[dict]:
        out = []
        for d in sorted(p for p in self.results.iterdir() if p.is_dir() and ((p / "runs").is_dir() or (p / "live").is_dir())):
            manifests = [self._cached(m, lambda p: json.loads(p.read_text())) for m in sorted((d / "live" / "manifests").glob("*.json"))]
            live = any(m and pid_alive(m.get("pid")) for m in manifests)
            out.append({"name": d.name, "live": live, "done": len(list((d / "runs").glob("*.parquet")))})
        return out

    # --- snapshot --------------------------------------------------------------------------

    def snapshot(self, name: str) -> Snapshot:
        d = self.experiment_dir(name)
        now = time.time()
        live = d / "live"

        done_frames = {}
        for p in sorted((d / "runs").glob("*.parquet")):
            df = self._cached(p, pd.read_parquet)
            if df is not None:
                done_frames[p.stem] = df
        statuses = {}
        for p in live.glob("*.status.json"):
            st = self._cached(p, lambda p: json.loads(p.read_text()))
            if st:
                statuses[st["run"]] = st
        manifests = []
        for p in sorted((live / "manifests").glob("*.json")):
            m = self._cached(p, lambda p: json.loads(p.read_text()))
            if m:
                manifests.append({**m, "id": p.stem, "alive": pid_alive(m.get("pid"))})

        partial_frames = {}
        for p in live.glob("*.jsonl"):
            rid = p.name.removesuffix(".jsonl")
            if rid in done_frames:
                continue
            df = self._cached(p, self._read_jsonl)
            if df is not None and len(df):
                problem, method, seed = parse_run_id(rid)
                partial_frames[rid] = df.assign(problem=problem, method=method, seed=seed)

        frames = list(done_frames.values()) + list(partial_frames.values())
        rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        runs = self._run_table(done_frames, partial_frames, statuses, manifests, now)
        return Snapshot(rows=rows, runs=runs, manifests=manifests, now=now)

    def _run_table(self, done, partial, statuses, manifests, now) -> pd.DataFrame:
        planned: dict[str, list[dict]] = {}
        for m in manifests:
            for problem, method, seed in m["runs"]:
                planned.setdefault(f"{problem}__{method}__s{seed}", []).append(m)
        ids = set(done) | set(statuses) | set(planned)
        records = []
        for rid in ids:
            problem, method, seed = parse_run_id(rid)
            st = statuses.get(rid)
            frame = done.get(rid, partial.get(rid))
            rec = {
                "run": rid, "problem": problem, "method": method, "seed": seed,
                "state": None, "cost": None, "budget": None, "batches": None,
                "elapsed": None, "heartbeat_age": None, "error": None, "started": None,
            }
            if frame is not None and len(frame):
                last = frame.sort_values("cost").iloc[-1]
                rec.update(cost=float(last["cost"]), batches=int(len(frame)))
                if "elapsed" in frame and pd.notna(last.get("elapsed")):
                    rec["elapsed"] = float(last["elapsed"])
            if st:
                rec.update(budget=st.get("budget"), started=st.get("started"), error=st.get("error"))
                rec["heartbeat_age"] = now - st["heartbeat"]
                if st["state"] == "running":
                    rec["elapsed"] = now - st["started"]
                    rec["cost"], rec["batches"] = st.get("cost"), st.get("batches")
            pending_by_live = any(m["alive"] and (not st or m["started"] > st["started"]) for m in planned.get(rid, []))

            if rid in done:
                rec["state"] = "done"
            elif st and st["state"] == "running" and pid_alive(st.get("pid")) and now - st["heartbeat"] < STALE_S:
                rec["state"] = "running"
            elif pending_by_live:
                rec["state"] = "queued"
            elif st and st["state"] == "failed":
                rec["state"] = "failed"
            elif st and st["state"] == "running":
                rec["state"] = "stale"
            elif st and st["state"] == "done":
                rec["state"] = "done"
            else:
                rec["state"] = "not_scheduled"
            if rec["budget"] is None:
                budgets = [m["budget"] for m in planned.get(rid, [])]
                rec["budget"] = budgets[-1] if budgets else None
            records.append(rec)
        runs = pd.DataFrame.from_records(records)
        if len(runs):
            runs = runs.sort_values(["problem", "method", "seed"], key=lambda c: c.map(method_index) if c.name == "method" else c)
        return runs.reset_index(drop=True)


# --- ETA -------------------------------------------------------------------------------------


def reference_histories(rows: pd.DataFrame, runs: pd.DataFrame) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, float]]]:
    """Per (problem, method): finished runs' (cost fraction, elapsed) curves and total duration."""
    if rows.empty or "elapsed" not in rows:
        return {}
    done = set(runs.loc[runs["state"] == "done", "run"])
    refs: dict[tuple[str, str], list] = {}
    for (problem, method, seed), r in rows.dropna(subset=["elapsed"]).groupby(["problem", "method", "seed"]):
        if f"{problem}__{method}__s{seed}" not in done:
            continue
        r = r.sort_values("cost")
        frac = np.concatenate([[0.0], r["cost"].to_numpy() / r["cost"].iloc[-1]])
        t = np.concatenate([[0.0], r["elapsed"].to_numpy()])
        refs.setdefault((problem, method), []).append((frac, t, float(t[-1])))
    return refs


def run_eta(elapsed: float, frac: float, refs: list) -> float | None:
    """Remaining seconds for a run `frac` of the way through its budget after `elapsed` s."""
    if refs:
        if frac <= 0:
            return max(0.0, float(np.median([total for *_, total in refs])) - elapsed)
        ratios = [total / max(np.interp(frac, f, t), 1e-9) for f, t, total in refs]
        return max(0.0, elapsed * (float(np.median(ratios)) - 1))
    if frac > 0:
        return elapsed * (1 - frac) / frac
    return None


def add_etas(snap: Snapshot) -> tuple[pd.DataFrame, dict]:
    """Per-run ETA column plus an experiment-level estimate."""
    runs = snap.runs.copy()
    refs = reference_histories(snap.rows, runs)
    by_method: dict[str, list[float]] = {}
    for (_, method), lst in refs.items():
        by_method.setdefault(method, []).extend(total for *_, total in lst)
    all_durations = [d for lst in by_method.values() for d in lst]

    def typical(problem: str, method: str) -> float | None:
        for pool in ([total for *_, total in refs.get((problem, method), [])], by_method.get(method, []), all_durations):
            if pool:
                return float(np.median(pool))
        return None

    etas, remaining, unknown = [], [], 0
    for rec in runs.itertuples():
        eta = None
        if rec.state == "running":
            frac = (rec.cost or 0.0) / rec.budget if rec.budget else 0.0
            eta = run_eta(rec.elapsed or 0.0, frac, refs.get((rec.problem, rec.method), []))
            if eta is None:
                typ = typical(rec.problem, rec.method)
                eta = None if typ is None else max(0.0, typ - (rec.elapsed or 0.0))
        elif rec.state == "queued":
            eta = typical(rec.problem, rec.method)
        etas.append(eta if rec.state == "running" else None)
        if rec.state in ("running", "queued"):
            if eta is None:
                unknown += 1
            else:
                remaining.append(eta)
    runs["eta"] = etas

    workers = sum(int(m.get("workers", 1)) for m in snap.manifests if m["alive"]) or 1
    pending = int(runs["state"].isin(["running", "queued"]).sum()) if len(runs) else 0
    exp_eta = max(max(remaining), sum(remaining) / workers) if remaining else None
    summary = {"eta": exp_eta if pending else 0.0, "eta_partial": unknown > 0, "workers": workers}
    return runs, summary


# --- curves ----------------------------------------------------------------------------------


def problem_outputs(rows: pd.DataFrame) -> dict[str, list[str]]:
    out = {}
    if rows.empty:
        return out
    cols = [c for c in rows.columns if c.startswith("nrmse/")]
    for problem, sub in rows.groupby("problem"):
        out[problem] = [c.split("/", 1)[1] for c in cols if sub[c].notna().any()]
    return dict(sorted(out.items()))


def curves(
    snap: Snapshot, problem: str, output: str, metric: str = "nrmse", x: str = "cost",
    seeds: bool = False, methods: list[str] | None = None,
) -> dict:
    """Median + IQR across seeds per method on a shared log grid, optionally with seed lines."""
    if metric not in METRICS or x not in ("cost", "elapsed"):
        raise ValueError("bad metric or x")
    col = f"{metric}/{output}"
    rows = snap.rows
    if rows.empty or col not in rows or x not in rows:
        return {"methods": []}
    sub = rows[(rows["problem"] == problem) & rows[col].notna() & rows[x].notna() & (rows[x] > 0)]
    if methods:
        sub = sub[sub["method"].isin(methods)]
    if sub.empty:
        return {"methods": []}
    grid = np.geomspace(sub[x].min(), sub[x].max(), GRID_POINTS)
    live_runs = set(snap.runs.loc[snap.runs["state"] != "done", "run"]) if len(snap.runs) else set()

    out = []
    for method in method_order(sub["method"].unique()):
        m = sub[sub["method"] == method]
        # Finished runs hold their last value to the end of the grid; live ones stop at their last row.
        parts = []
        for seed, r in m.groupby("seed"):
            extend = f"{problem}__{method}__s{seed}" not in live_runs
            parts.append(curves_on_grid(r, col, grid, x=x, extend=extend))
        Y = np.vstack(parts)
        n = np.sum(~np.isnan(Y), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN grid columns -> NaN
            q25, q50, q75 = np.nanpercentile(Y, [25, 50, 75], axis=0)
        # While seeds are still running, a median over the one or two furthest-along seeds
        # is noise: draw the median only where at least half the seeds have data, the band
        # only where at least two do.
        q50 = np.where(n >= max(1, math.ceil(len(parts) / 2)), q50, np.nan)
        q25, q75 = (np.where(n >= 2, q, np.nan) for q in (q25, q75))
        entry = {
            "method": method, "index": method_index(method), "x": grid, "median": q50, "q25": q25, "q75": q75,
            "n": n, "seeds_total": int(m["seed"].nunique()),
        }
        if seeds:
            entry["seed_lines"] = [
                {"seed": int(seed), "x": r.sort_values(x)[x].to_numpy(), "y": r.sort_values(x)[col].to_numpy()}
                for seed, r in m.groupby("seed")
            ]
        out.append(entry)
    return {"methods": out, "x": x, "metric": metric}


def run_detail(snap: Snapshot, rid: str) -> dict:
    problem, method, seed = parse_run_id(rid)
    rows = snap.rows
    if rows.empty:
        r = rows
    else:
        r = rows[(rows["problem"] == problem) & (rows["method"] == method) & (rows["seed"] == seed)].sort_values("cost")
    info = snap.runs[snap.runs["run"] == rid]
    diagnostics = []
    if "diagnostics" in r:
        diagnostics = [json.loads(d) if isinstance(d, str) else None for d in r["diagnostics"]]
    metric_cols = [c for c in r.columns if c.split("/")[0] in METRICS and r[c].notna().any()]
    keep = [c for c in ("batch", "n_evals", "cost", "elapsed", "ask_time", "sim_time", "predict_time") if c in r]
    return {
        "run": info.to_dict("records")[0] if len(info) else {"run": rid},
        "rows": r[keep + metric_cols].to_dict("list"),
        "diagnostics": diagnostics,
    }
