"""Paired-by-seed tables for one experiment/problem: final metrics, AUCs and run time per method.

    uv run python studies/validation_tables.py [experiment[,experiment...]] [problem]   # default openmc_validation openmc

Prints Markdown: per output, the seed-median final NRMSE / coverage / NCRPS, AUC-NRMSE and AUC-NCRPS
(shared log-cost grid, as in the ranking report), and how many seeds each method beats the first
listed method on (the `sobol_gp` baseline when present). Also run time and the number of evaluations per run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surrogatemodeling.dashboard.data import Store  # noqa: E402
from surrogatemodeling.report.plots import GRID_POINTS, curves_on_grid  # noqa: E402


def tables(experiment: str = "openmc_validation", problem: str = "openmc") -> str:
    snaps = [Store(Path("results")).snapshot(e) for e in experiment.split(",")]  # "a,b" pools experiments
    all_runs = pd.concat([s.runs for s in snaps], ignore_index=True)
    all_rows = pd.concat([s.rows for s in snaps], ignore_index=True)
    runs = all_runs[(all_runs["problem"] == problem) & (all_runs["state"] == "done")]
    rows = all_rows[all_rows["problem"] == problem]
    rows = rows[rows.set_index(["method", "seed"]).index.isin(runs.set_index(["method", "seed"]).index)]
    order = [m for m in dict.fromkeys(runs["method"])]
    last = rows.sort_values("cost").groupby(["method", "seed"]).tail(1)
    start = rows.groupby(["method", "seed"])["cost"].min().max()
    stop = rows.groupby(["method", "seed"])["cost"].max().min()
    grid = np.geomspace(start, stop, GRID_POINTS)
    outputs = [c.split("/", 1)[1] for c in rows.columns if c.startswith("nrmse/") and rows[c].notna().any()]
    ref = "sobol_gp" if "sobol_gp" in order else order[0]  # the success bar is defined against the baseline
    out = [f"## {experiment} · {problem}", "", f"Seeds per method: {last.groupby('method')['seed'].nunique().to_dict()}", ""]
    for o in outputs:
        lines = ["| method | final NRMSE | final coverage | final NCRPS | AUC-NRMSE | AUC-NCRPS | seeds beating " + f"`{ref}` (NRMSE) |",
                 "|---|---|---|---|---|---|---|"]
        piv = last.pivot(index="seed", columns="method", values=f"nrmse/{o}")
        for m in order:
            r, lm = rows[rows["method"] == m], last[last["method"] == m]
            auc = {k: 10 ** np.mean(np.log10(np.nanmedian(curves_on_grid(r, f"{k}/{o}", grid), axis=0))) for k in ("nrmse", "ncrps")}
            wins = "—" if m == ref else f"{int((piv[m] < piv[ref]).sum())}/{int(piv[[m, ref]].dropna().shape[0])}"
            cov = lm[f"coverage/{o}"].median()
            lines.append(f"| `{m}` | {lm[f'nrmse/{o}'].median():.3f} | {cov:.2f} | {lm[f'ncrps/{o}'].median():.3f} | "
                         f"{auc['nrmse']:.3f} | {auc['ncrps']:.3f} | {wins} |")
        out += [f"### {o}", "", *lines, ""]
    out += ["### Run time and evaluations", "", "| method | median run time (min) | evaluations per run |", "|---|---|---|"]
    for m in order:
        lm = last[last["method"] == m]
        out.append(f"| `{m}` | {lm['elapsed'].median() / 60:.0f} | {int(lm['n_evals'].median())} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(tables(*sys.argv[1:]))
