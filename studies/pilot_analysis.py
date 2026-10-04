"""Read the OpenMC pilot: cost model and source-convergence length.

    uv run python studies/pilot_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/openmc_fidelity")
WINDOW = 20


def cost_fit(df: pd.DataFrame) -> dict:
    """CPU = overhead + inactive_rate * N*I + active_rate * N*A (least squares)."""
    n, i, a = df["knob/particles"], df["knob/inactive"], df["knob/active"]
    A = np.column_stack([np.ones(len(df)), n * i, n * a])
    coef, *_ = np.linalg.lstsq(A, df["cpu_total"].to_numpy(), rcond=None)
    return {"overhead_s": coef[0], "inactive_s_per_particle": coef[1], "active_s_per_particle": coef[2]}


def settle_batch(series: np.ndarray, tail: int = 150) -> int:
    """First batch after which the WINDOW-batch moving average stays within 2 standard
    errors (of a WINDOW-batch mean, from the tail's batch scatter) of the tail mean."""
    tail_vals = series[-tail:]
    target, band = tail_vals.mean(), 2 * tail_vals.std(ddof=1) / np.sqrt(WINDOW)
    ma = np.convolve(series, np.ones(WINDOW) / WINDOW, mode="valid")
    outside = np.flatnonzero(np.abs(ma - target) > band)
    return int(outside[-1] + WINDOW) if len(outside) else WINDOW


def main() -> None:
    df = pd.read_parquet(OUT / "pilot.parquet")
    report = {}

    conv = df[df["kind"] == "convergence"]
    # Every batch of a convergence run is active, so they also pin down the active rate.
    fit = cost_fit(pd.concat([df[df["kind"] == "cost"], conv]))
    report["cost"] = fit
    print("Cost model (CPU s):", {k: f"{v:.3g}" for k, v in fit.items()})

    rows = []
    for r in conv.itertuples():
        ent = np.array(json.loads(r.entropy))
        ao = np.array(json.loads(r.ao_batches))
        rows.append({"point": r.point, "particles": json.loads(r.fidelity)["particles"],
                     "seed": r.seed, "settle_entropy": settle_batch(ent), "settle_ao": settle_batch(ao),
                     "ao_first20": ao[:20].mean(), "ao_tail": ao[-150:].mean()})
    conv_table = pd.DataFrame(rows).sort_values(["point", "particles", "seed"])
    print("\nSource convergence (batches until the 20-batch moving average settles):")
    print(conv_table.round(3).to_string(index=False))
    report["convergence"] = conv_table.to_dict("records")
    report["settle_max"] = int(conv_table[["settle_entropy", "settle_ao"]].max().max())
    (OUT / "pilot_summary.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
