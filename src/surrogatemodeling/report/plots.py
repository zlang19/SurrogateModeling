"""Error-vs-cost curves: median and IQR across seeds, one figure per (problem, output)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surrogatemodeling.registry import METHODS

# Categorical palette, assigned by the method's fixed registry position (never by rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
GRID_POINTS = 60


def method_color(method: str) -> str:
    names = list(METHODS)
    return PALETTE[names.index(method) % len(PALETTE)] if method in names else MUTED


def curves_on_grid(df: pd.DataFrame, metric: str, grid: np.ndarray, x: str = "cost", extend: bool = True) -> np.ndarray:
    """(n_seeds, len(grid)) step-interpolated metric: value at x=c is the latest row with x <= c.

    With extend=False, grid points past a seed's last row are NaN (for runs still in progress).
    """
    out = []
    for _, run in df.groupby("seed"):
        run = run.sort_values(x)
        xs = run[x].to_numpy()
        idx = np.searchsorted(xs, grid, side="right") - 1
        vals = np.where(idx >= 0, run[metric].to_numpy()[np.clip(idx, 0, None)], np.nan)
        if not extend:
            vals = np.where(grid > xs[-1] * (1 + 1e-9), np.nan, vals)
        out.append(vals)
    return np.array(out)


def plot_error_vs_cost(df: pd.DataFrame, problem: str, output: str, path: Path, metric: str = "nrmse") -> None:
    sub = df[df["problem"] == problem]
    col = f"{metric}/{output}"
    grid = np.geomspace(sub["cost"].min(), sub["cost"].max(), GRID_POINTS)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    methods = [m for m in METHODS if m in set(sub["method"])] + sorted(set(sub["method"]) - set(METHODS))
    for method in methods:
        runs = sub[sub["method"] == method]
        Y = curves_on_grid(runs, col, grid)
        q25, q50, q75 = np.nanpercentile(Y, [25, 50, 75], axis=0)
        color = method_color(method)
        n_seeds = runs["seed"].nunique()
        ax.fill_between(grid, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, q50, color=color, linewidth=2, label=f"{method} (n={n_seeds})")
        if len(methods) <= 4:
            last = np.flatnonzero(~np.isnan(q50))[-1]
            ax.annotate(method, (grid[last], q50[last]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative cost (high-fidelity run equivalents)", color=TEXT)
    ax.set_ylabel(f"{metric.upper()} on test set", color=TEXT)
    ax.set_title(f"{problem}: {output}", color=TEXT, loc="left", fontsize=11)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, which="both")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, title="median, IQR band", title_fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def report(results_dir: Path) -> list[Path]:
    """Read every run Parquet under results_dir/runs; write NRMSE plots and ranking.md."""
    from surrogatemodeling.report.ranking import auc_table, summary_markdown

    df = pd.concat([pd.read_parquet(p) for p in sorted((results_dir / "runs").glob("*.parquet"))])
    outputs = sorted({c.split("/", 1)[1] for c in df.columns if c.startswith("nrmse/")})
    written = []
    for problem in sorted(df["problem"].unique()):
        for output in outputs:
            if df.loc[df["problem"] == problem, f"nrmse/{output}"].notna().any():
                path = results_dir / "plots" / f"{problem}__{output}__nrmse.png"
                plot_error_vs_cost(df, problem, output, path)
                written.append(path)
    table = auc_table(df)
    table.to_csv(results_dir / "ranking.csv", index=False)
    ranking = results_dir / "ranking.md"
    ranking.write_text(summary_markdown(table))
    written += [results_dir / "ranking.csv", ranking]
    return written
