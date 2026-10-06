"""Pareto frontier: AUC-NCRPS vs run time, for every NCRPS-era method on one problem.

    uv run python studies/pareto.py [problem]       # default toymc_axial -> results/all_ncrps/

Each point is a method: x = median wall-clock run time over its seeds, y = AUC-NCRPS (seed-median
curve on a shared log-cost grid, as in the ranking report). The frontier connects methods that no
other method beats on both. Run times were measured under different machine loads, so they are
comparable to within roughly a factor of 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surrogatemodeling.dashboard.data import Store  # noqa: E402
from surrogatemodeling.report.plots import GRID_POINTS, curves_on_grid  # noqa: E402

OUT = Path("results/all_ncrps")
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
# Families (fixed palette order), so 15+ methods stay readable.
FAMILIES = [
    ("High-fidelity baseline (RBF / Matérn)", "#2a78d6", lambda m: m in ("sobol_gp", "sobol_gp_matern")),
    ("SAAS (fully Bayesian GP)", "#eb6834", lambda m: m == "sobol_saas"),
    ("#7s and one-change variants", "#1baf7a", lambda m: m.startswith("adaptive_iv_mf_cks") and "warp" not in m and "safe" not in m),
    ("Converged-source menu (safe)", "#d6336c", lambda m: "safe" in m),
    ("Full input warping", "#eda100", lambda m: "warp" in m and "dwarp" not in m),
    ("Delayed top-8 warping", "#4a3aa7", lambda m: "dwarp" in m),
]


def short(m: str) -> str:
    return {"sobol_gp": "baseline", "sobol_gp_matern": "baseline (Matérn)", "sobol_saas": "SAAS", "adaptive_iv_mf_cks": "#7s"}.get(m, m.replace("adaptive_iv_mf_cks_", ""))


def family(m: str) -> tuple[str, str]:
    for name, color, test in FAMILIES:
        if test(m):
            return name, color
    return "Other", MUTED


def pareto(points: pd.DataFrame) -> pd.DataFrame:
    """Rows not dominated on (runtime, auc): no other point is <= on both and < on one."""
    keep = []
    for i, p in points.iterrows():
        dominated = ((points["runtime"] <= p["runtime"]) & (points["auc"] <= p["auc"]) &
                     ((points["runtime"] < p["runtime"]) | (points["auc"] < p["auc"]))).any()
        if not dominated:
            keep.append(i)
    return points.loc[keep].sort_values("runtime")


def table(problem: str) -> pd.DataFrame:
    snap = Store(Path("results")).snapshot_all()
    rows, runs = snap.rows, snap.runs
    sub = rows[rows["problem"] == problem]
    done = runs[(runs["problem"] == problem) & (runs["state"] == "done")]
    sub = sub[sub.set_index(["method", "seed"]).index.isin(done.set_index(["method", "seed"]).index)]
    start = sub.groupby(["method", "seed"])["cost"].min().max()
    stop = sub.groupby(["method", "seed"])["cost"].max().min()
    grid = np.geomspace(start, stop, GRID_POINTS)
    outputs = [c.split("/", 1)[1] for c in sub.columns if c.startswith("ncrps/") and sub[c].notna().any()]
    out = []
    for method, r in sub.groupby("method"):
        runtime = r.sort_values("cost").groupby("seed")["elapsed"].last()
        for o in outputs:
            med = np.nanmedian(curves_on_grid(r, f"ncrps/{o}", grid), axis=0)
            out.append({"method": method, "output": o, "runtime": float(runtime.median()) / 60,
                        "runtime_lo": float(runtime.min()) / 60, "runtime_hi": float(runtime.max()) / 60,
                        "auc": float(10 ** np.mean(np.log10(med))), "seeds": r["seed"].nunique()})
    return pd.DataFrame(out)


def plot(problem: str = "toymc_axial") -> Path:
    t = table(problem)
    outputs = list(dict.fromkeys(t["output"]))
    ncol = 2
    fig, axes = plt.subplots(int(np.ceil(len(outputs) / ncol)), ncol, figsize=(13, 5.2 * np.ceil(len(outputs) / ncol)), dpi=140, squeeze=False)
    for ax, o in zip(axes.flat, outputs):
        pts = t[t["output"] == o].reset_index(drop=True)
        front = pareto(pts)
        ax.step(front["runtime"], front["auc"], where="post", color=MUTED, linewidth=1.5, linestyle="--", zorder=1)
        # Frontier labels alternate above/below so near-coincident points stay readable.
        offsets = {m: (6, 5 if i % 2 == 0 else -11) for i, m in enumerate(front["method"])}
        for _, p in pts.iterrows():
            _, color = family(p["method"])
            on_front = p["method"] in offsets
            ax.plot([p["runtime_lo"], p["runtime_hi"]], [p["auc"], p["auc"]], color=color, linewidth=1, alpha=0.5, zorder=2)
            ax.scatter(p["runtime"], p["auc"], s=70 if on_front else 40, color=color, edgecolor="white", linewidth=1.5, zorder=3)
            if on_front:
                ax.annotate(short(p["method"]), (p["runtime"], p["auc"]), xytext=offsets[p["method"]], textcoords="offset points", fontsize=8.5, color=TEXT)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{problem} · {o}", loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel("Wall-clock run time per run (min; median, bar = min–max over seeds)", color=TEXT, fontsize=9)
        ax.set_ylabel("AUC-NCRPS (lower is better)", color=TEXT, fontsize=9)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, which="both", labelsize=8)
    for ax in axes.flat[len(outputs):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markeredgecolor="white", markersize=8, label=n)
               for n, c, _ in FAMILIES if any(family(m)[0] == n for m in t["method"])]
    handles.append(plt.Line2D([], [], color=MUTED, linestyle="--", label="Pareto frontier (labeled points)"))
    fig.suptitle(f"Accuracy-and-calibration vs compute: all NCRPS-era methods · {problem}", y=0.995, fontsize=12, color=TEXT)
    fig.text(0.5, 0.968, "Run times were measured under different machine loads; compare to within ~2×.", ha="center", fontsize=9, color=MUTED)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(handles), frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pareto_{problem}.png"
    fig.savefig(path)
    t.to_csv(OUT / f"pareto_{problem}.csv", index=False)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(plot(sys.argv[1] if len(sys.argv) > 1 else "toymc_axial"))
