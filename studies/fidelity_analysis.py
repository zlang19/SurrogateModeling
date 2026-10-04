"""Analyze the OpenMC fidelity grid -> results/openmc_fidelity/{report.md, summary.csv, *.png}.

Per (input point, config), over the replicates:
- cost: median CPU seconds per run (child process, so start-up and cross-section loading count)
- spread: replicate standard deviation of each output (the real noise)
- under-reporting: spread / mean reported sigma (1 = honest; >1 = sigma too small)
- bias: replicate mean minus the reference run, in units of the HF replicate spread, with
  its standard error, so |z| > 3 means detectable at that replicate count
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUTS = ["k_eff", "axial_offset", "axial_peaking", "capture_to_fission"]
REF_PARTICLE_MULT = 10  # reference runs use 10x HF particles (grid_spec.json ref_mult)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # reference categorical slots, fixed order
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def summarize(grid: pd.DataFrame) -> pd.DataFrame:
    ref = grid[grid["kind"] == "reference"].set_index("point")
    reps = grid[grid["kind"] == "grid"]
    hf_spread = reps[reps["config"] == "hf"].groupby("point")[OUTPUTS].std(ddof=1)
    rows = []
    for (point, config), g in reps.groupby(["point", "config"]):
        fid = json.loads(g["fidelity"].iloc[0])
        row = {"point": point, "config": config, **{f"knob/{k}": v for k, v in fid.items()},
               "n": len(g), "cpu_median": g["cpu_total"].median()}
        for o in OUTPUTS:
            spread = g[o].std(ddof=1)
            row[f"spread/{o}"] = spread
            row[f"underreport/{o}"] = spread / g[f"sigma/{o}"].mean()
            bias = g[o].mean() - ref.loc[point, o]
            # The reference's *reported* sigma is under-reported like every other run (4x for
            # axial offset), so use its real noise: HF replicate spread / sqrt(10x particles).
            ref_se = hf_spread.loc[point, o] / np.sqrt(REF_PARTICLE_MULT)
            se = np.sqrt(spread**2 / len(g) + ref_se**2)
            row[f"bias/{o}"] = bias
            row[f"bias_hf_sigma/{o}"] = bias / hf_spread.loc[point, o]
            row[f"bias_z/{o}"] = bias / se
        rows.append(row)
    return pd.DataFrame(rows)


def by_config(summary: pd.DataFrame) -> pd.DataFrame:
    """Average over the input points (bias: mean of |bias| in HF-sigma units, signed z kept per point)."""
    agg = {"cpu_median": "mean", "n": "sum", **{c: "first" for c in summary.columns if c.startswith("knob/")}}
    for o in OUTPUTS:
        agg[f"underreport/{o}"] = "mean"
        agg[f"bias_hf_sigma/{o}"] = lambda v: np.mean(np.abs(v))
        agg[f"bias_z/{o}"] = lambda v: np.max(np.abs(v))
    out = summary.groupby("config").agg(agg)
    hf_cost = out.loc["hf", "cpu_median"]
    out.insert(0, "cost_rel", out["cpu_median"] / hf_cost)
    return out.sort_values("cost_rel")


def cost_fit(grid: pd.DataFrame) -> dict:
    g = grid[grid["kind"].isin(["grid", "reference"])]
    pb = g["knob/particles"] * (g["knob/inactive"] + g["knob/active"])
    A = np.column_stack([np.ones(len(g)), pb])
    (overhead, rate), *_ = np.linalg.lstsq(A, g["cpu_total"].to_numpy(), rcond=None)
    return {"overhead_s": float(overhead), "s_per_particle_batch": float(rate)}


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_title(title, color=TEXT, loc="left", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def sweep_plot(cfg: pd.DataFrame, knob: str, metric: str, ylabel: str, title: str, path: Path, ref_line: float | None) -> None:
    rows = cfg[cfg.index.str.startswith(f"{knob}=") | (cfg.index == "hf")].sort_values(f"knob/{knob}")
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    if ref_line is not None:
        ax.axhline(ref_line, color=MUTED, linewidth=1, linestyle=":")
    for o, c in zip(OUTPUTS, COLORS):
        ax.plot(rows[f"knob/{knob}"], rows[f"{metric}/{o}"], color=c, linewidth=2, marker="o", markersize=5, label=o)
    ax.set_xscale("log")
    _style(ax, knob.replace("_", " "), ylabel, title)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analyze(out_dir: Path) -> None:
    grid = pd.read_parquet(out_dir / "grid.parquet")
    summary = summarize(grid)
    summary.to_csv(out_dir / "summary.csv", index=False)
    cfg = by_config(summary)
    fit = cost_fit(grid)

    sweep_plot(cfg, "active", "underreport", "real spread ÷ reported σ", "σ under-reporting vs active cycles",
               out_dir / "underreport_vs_active.png", 1.0)
    sweep_plot(cfg, "inactive", "bias_hf_sigma", "|bias| in HF-noise units", "Bias vs inactive cycles",
               out_dir / "bias_vs_inactive.png", None)
    sweep_plot(cfg, "particles", "bias_hf_sigma", "|bias| in HF-noise units", "Bias vs particles per cycle",
               out_dir / "bias_vs_particles.png", None)

    cols = ["cost_rel", "cpu_median", "knob/particles", "knob/inactive", "knob/active"]
    lines = [
        "# OpenMC fidelity characterization",
        "",
        f"{len(grid)} runs; cost model: **{fit['overhead_s']:.1f} s fixed + {fit['s_per_particle_batch']:.3g} s per particle-batch** "
        "(CPU, one thread, under 12-way load). Bias is against one long reference run per point "
        "(10× particles, 2× inactive); `bias_hf_sigma` is the mean over the 4 points of |bias| in units of the HF replicate spread; "
        "`bias_z` is the largest |bias| / its standard error over the points (> 3 ≈ detectable).",
        "",
        "## Cost",
        "",
        cfg[cols].round(3).to_markdown(),
        "",
        "## σ under-reporting (real spread ÷ mean reported σ, mean over points; 1 = honest)",
        "",
        cfg[[f"underreport/{o}" for o in OUTPUTS]].round(2).rename(columns=lambda c: c.split("/")[1]).to_markdown(),
        "",
        "## Bias (mean |bias| in HF-noise units)",
        "",
        cfg[[f"bias_hf_sigma/{o}" for o in OUTPUTS]].round(2).rename(columns=lambda c: c.split("/")[1]).to_markdown(),
        "",
        "## Bias detectability (max |z| over points)",
        "",
        cfg[[f"bias_z/{o}" for o in OUTPUTS]].round(1).rename(columns=lambda c: c.split("/")[1]).to_markdown(),
        "",
        "Plots: `underreport_vs_active.png`, `bias_vs_inactive.png`, `bias_vs_particles.png`.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))
    (out_dir / "cost_fit.json").write_text(json.dumps(fit, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    analyze(Path("results/openmc_fidelity"))
