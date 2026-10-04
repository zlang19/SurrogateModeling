"""Cross-problem ranking by normalized area under the error-vs-cost curve.

For each (problem, output, method) the seed-median NRMSE curve is put on a shared log-cost
grid and its mean log10 NRMSE is taken: the area under the log-log curve divided by the
log-cost span. That number is reported as a geometric-mean NRMSE ("AUC-NRMSE"). Methods
are ranked per (problem, output) and the ranks averaged; ties share the mean rank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from surrogatemodeling.registry import METHODS
from surrogatemodeling.report.plots import GRID_POINTS, curves_on_grid


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
                median = np.nanmedian(curves_on_grid(runs, col, grid), axis=0)
                final = runs.sort_values("cost").groupby("seed").tail(1)
                rows.append(
                    {
                        "problem": problem,
                        "output": output,
                        "method": method,
                        "auc_nrmse": float(10 ** np.mean(np.log10(median))),
                        "final_nrmse": float(final[col].median()),
                        "final_coverage": float(final[f"coverage/{output}"].median()),
                        "seeds": runs["seed"].nunique(),
                    }
                )
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby(["problem", "output"])["auc_nrmse"].rank(method="average")
    return table


def _method_order(methods) -> list[str]:
    return [m for m in METHODS if m in set(methods)] + sorted(set(methods) - set(METHODS))


def summary_markdown(table: pd.DataFrame) -> str:
    order = _method_order(table["method"].unique())
    key = table["problem"] + ":" + table["output"]
    auc = table.assign(key=key).pivot(index="method", columns="key", values="auc_nrmse").loc[order]
    final = table.assign(key=key).pivot(index="method", columns="key", values="final_nrmse").loc[order]
    cover = table.assign(key=key).pivot(index="method", columns="key", values="final_coverage").loc[order]
    mean_rank = table.groupby("method")["rank"].mean().loc[order]

    def fmt(frame: pd.DataFrame, digits: int = 3) -> str:
        return frame.map(lambda v: "—" if pd.isna(v) else f"{v:.{digits}f}").to_markdown()

    ranked = mean_rank.sort_values()
    lines = [
        "# Ranking",
        "",
        "Mean rank across (problem, output) by AUC-NRMSE (lower is better):",
        "",
        pd.DataFrame({"mean rank": ranked.round(2)}).to_markdown(),
        "",
        "## AUC-NRMSE (geometric-mean NRMSE over the shared cost range)",
        "",
        fmt(auc),
        "",
        "## Final NRMSE (seed median at full budget)",
        "",
        fmt(final),
        "",
        "## Final 95% coverage (seed median; — = no variance reported)",
        "",
        fmt(cover, 2),
        "",
    ]
    return "\n".join(lines)
