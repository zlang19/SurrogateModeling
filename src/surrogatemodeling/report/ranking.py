"""Cross-problem ranking by normalized area under the error-vs-cost curve.

For each (problem, output, method) the seed-median curve is put on a shared log-cost grid
and its mean log10 value is taken: the area under the log-log curve divided by the
log-cost span, reported as a geometric mean ("AUC-<metric>"). Two criteria are ranked:

- AUC-NRMSE: accuracy
- AUC-NCRPS: probabilistic quality (accuracy *and* honest error bars), when recorded

Methods are ranked per (problem, output) and the ranks averaged; ties share the mean rank.
The calibration gate is the share of (problem, output) pairs whose final coverage is at
least COVERAGE_GATE; a method meets it only at 100%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from surrogatemodeling.registry import METHODS
from surrogatemodeling.report.plots import GRID_POINTS, curves_on_grid

RANKED = ("nrmse", "ncrps")
COVERAGE_GATE = 0.90


def _auc(runs: pd.DataFrame, col: str, grid: np.ndarray) -> float:
    median = np.nanmedian(curves_on_grid(runs, col, grid), axis=0)
    return float(10 ** np.mean(np.log10(median)))


def auc_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows (problem, output, method) with AUC-NRMSE, final NRMSE and final coverage."""
    outputs = sorted({c.split("/", 1)[1] for c in df.columns if c.startswith("nrmse/")})
    rows = []
    for problem, sub in df.groupby("problem"):
        # Shared grid: from the first cost every method has scored to the smallest final cost.
        start = sub.groupby(["method", "seed"])["cost"].min().max()
        stop = sub.groupby(["method", "seed"])["cost"].max().min()
        grid = np.geomspace(start, stop, GRID_POINTS)
        for output in outputs:
            col = f"nrmse/{output}"
            if sub[col].isna().all():
                continue
            for method, runs in sub.groupby("method"):
                final = runs.sort_values("cost").groupby("seed").tail(1)
                row = {
                    "problem": problem,
                    "output": output,
                    "method": method,
                    "final_nrmse": float(final[col].median()),
                    "final_coverage": float(final[f"coverage/{output}"].median()),
                    "seeds": runs["seed"].nunique(),
                }
                for metric in RANKED:
                    mcol = f"{metric}/{output}"
                    has = mcol in runs and runs[mcol].notna().all()
                    row[f"auc_{metric}"] = _auc(runs, mcol, grid) if has else np.nan
                rows.append(row)
    table = pd.DataFrame(rows)
    for metric in RANKED:
        table[f"rank_{metric}"] = table.groupby(["problem", "output"])[f"auc_{metric}"].rank(method="average")
    table["rank"] = table["rank_nrmse"]  # back-compat
    return table


def _method_order(methods) -> list[str]:
    return [m for m in METHODS if m in set(methods)] + sorted(set(methods) - set(METHODS))


def summary_markdown(table: pd.DataFrame) -> str:
    order = _method_order(table["method"].unique())
    key = table["problem"] + ":" + table["output"]
    pivot = lambda col: table.assign(key=key).pivot(index="method", columns="key", values=col).loc[order]
    has_ncrps = table["auc_ncrps"].notna().any()

    def fmt(frame: pd.DataFrame, digits: int = 3) -> str:
        return frame.map(lambda v: "—" if pd.isna(v) else f"{v:.{digits}f}").to_markdown()

    gate = table.groupby("method")["final_coverage"].apply(lambda c: np.nan if c.isna().all() else float((c >= COVERAGE_GATE).mean()))
    summary = pd.DataFrame({"rank (AUC-NRMSE)": table.groupby("method")["rank_nrmse"].mean()})
    if has_ncrps:
        summary["rank (AUC-NCRPS)"] = table.groupby("method")["rank_ncrps"].mean()
    summary[f"coverage ≥ {COVERAGE_GATE:.2f}"] = gate.map(lambda v: "—" if pd.isna(v) else f"{100 * v:.0f}%")
    summary = summary.loc[order].sort_values(summary.columns[1] if has_ncrps else summary.columns[0])

    lines = [
        "# Ranking",
        "",
        "Mean rank across (problem, output); lower is better. AUC-NRMSE ranks accuracy; AUC-NCRPS ranks probabilistic",
        f"quality (accuracy *and* honest error bars). The gate column is the share of (problem, output) pairs whose final",
        f"coverage is ≥ {COVERAGE_GATE:.2f}; a method passes the calibration gate only at 100%. See docs/EvaluationCriteria.md.",
        "",
        summary.round(2).to_markdown(),
        "",
        "## AUC-NRMSE (geometric-mean NRMSE over the shared cost range)",
        "",
        fmt(pivot("auc_nrmse")),
        "",
    ]
    if has_ncrps:
        lines += ["## AUC-NCRPS (geometric-mean normalized CRPS over the shared cost range)", "", fmt(pivot("auc_ncrps")), ""]
    lines += [
        "## Final NRMSE (seed median at full budget)",
        "",
        fmt(pivot("final_nrmse")),
        "",
        "## Final 95% coverage (seed median; — = no variance reported)",
        "",
        fmt(pivot("final_coverage"), 2),
        "",
    ]
    return "\n".join(lines)
