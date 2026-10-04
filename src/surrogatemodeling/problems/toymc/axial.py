"""`toymc_axial`: a fast 1D twin of the OpenMC pin column (problems/openmc/model.py).

Same structure and outputs as the OpenMC problem, in the toy's 2-group physics:

    | bottom reflector | 20 fuel slabs over 200 cm (3 enrichment zones) | top reflector |

with vacuum beyond the reflectors, coolant density falling with height, a control rod
entering from the top, and Gd homogenized into the fuel. The 2 m height makes the slab
loosely coupled, so its slowest source mode is the top/bottom tilt — the mode that makes
axial offset converge slowly and its batch sigma under-reported, as in the OpenMC column.

Outputs (as model.py): k_eff, axial_offset, axial_peaking (max/mean over the 20 slabs),
capture_to_fission. Sigmas are MCNP-style batch statistics over active generations
(ratios: ratio of sums with delta-method errors), so they are optimistic for the same
reason MCNP's are.
"""

from __future__ import annotations

import numpy as np

from surrogatemodeling.problems.toymc.physics import material_xs

FUEL_HEIGHT = 200.0
N_SLABS = 20
SLAB = FUEL_HEIGHT / N_SLABS
ZONES = ((0, 7), (7, 13), (13, 20))  # bottom / middle / top enrichment zones, as model.py
ROD_THERMAL_ABS = 0.03  # homogenized B4C in one of nine lattice cells (1/cm, thermal)
ROD_FAST_ABS = 0.002
GD_PER_WT = 0.04  # toy "gd" absorber strength per wt% Gd2O3


def region_xs(p: dict[str, float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-region 2-group cross sections and region edges (cm) for the axial column."""
    vf = (8 / 9) * np.pi * p["pellet_radius"] ** 2 / p["pitch"] ** 2  # fuel volume fraction
    base = {**p, "fuel_vf": vf, "clad_abs": p["clad_abs"] * p["clad_thickness"] / 0.06, "mod_temp": 580.0}
    rod_tip = FUEL_HEIGHT - p["rod_insertion"]
    mats = [material_xs({**base, "refl_density": p["mod_density_in"]}, None)]
    rod = []
    for k in range(N_SLABS):
        rho = p["mod_density_in"] - p["density_drop"] * (k + 0.5) / N_SLABS
        zone = next(z for z, (a, b) in enumerate(ZONES) if a <= k < b)
        enr = (p["enr_bottom"], p["enr_middle"], p["enr_top"])[zone]
        mats.append(material_xs({**base, "mod_density": rho}, enr, GD_PER_WT * p["gd_wt"]))
        rod.append(np.clip(((k + 1) * SLAB - rod_tip) / SLAB, 0.0, 1.0))  # rodded fraction of the slab
    mats.append(material_xs({**base, "refl_density": p["mod_density_in"] - p["density_drop"]}, None))

    arr = lambda i: np.array([m[i] for m in mats], dtype=float)
    total, absorb, nu_fission, fission, down = arr(0), arr(1), arr(2), arr(3), arr(4)
    rod = np.array([0.0, *rod, 0.0])
    extra = np.column_stack([ROD_FAST_ABS * rod, ROD_THERMAL_ABS * rod])
    absorb, total = absorb + extra, total + extra
    edges = np.concatenate([[0.0], p["refl_bottom"] + SLAB * np.arange(N_SLABS + 1), [p["refl_bottom"] + FUEL_HEIGHT + p["refl_top"]]])
    xs = {"total": total, "absorb": absorb, "nu_fission": nu_fission, "fission": fission, "downscatter": down}
    return xs, edges


def _ratio(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    r = a.sum() / b.sum()
    return float(r), float(np.std(a - r * b, ddof=1) / (np.sqrt(len(a)) * b.mean()))


def run_axial(p: dict[str, float], particles: int, inactive: int, active: int, rng: np.random.Generator, record: bool = False) -> dict:
    """KCODE-style power iteration on the axial column. Returns outputs and batch-stat sigmas;
    with record=True also per-generation k and axial offset (inactive generations included)."""
    xs, edges = region_xs(p)
    n_reg = len(edges) - 1
    width = edges[-1]
    majorant = xs["total"].max(axis=0)
    p_down = xs["downscatter"] / (xs["total"][:, 0] - xs["absorb"][:, 0])
    fuel_lo, fuel_hi = edges[1], edges[-2]

    source = rng.uniform(fuel_lo, fuel_hi, particles)
    gens = []  # per generation: k, 20 slab fission scores, capture, fission
    for _ in range(inactive + active):
        x = source.copy()
        g = np.zeros(particles, dtype=np.int64)
        mu = rng.uniform(-1, 1, particles)
        site_x, site_w = [], []
        k_score, slab_fis = 0.0, np.zeros(N_SLABS)
        capture = fission = 0.0
        while x.size:
            x = x + mu * (-np.log(rng.random(x.size)) / majorant[g])
            inside = (x > 0) & (x < width)
            x, g, mu = x[inside], g[inside], mu[inside]
            reg = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_reg - 1)
            sig_t = xs["total"][reg, g]
            real = rng.random(x.size) * majorant[g] < sig_t
            u = rng.random(x.size) * sig_t
            absorbed = real & (u < xs["absorb"][reg, g])
            scattered = real & ~absorbed
            if absorbed.any():
                ra, ga = reg[absorbed], g[absorbed]
                nu_w = xs["nu_fission"][ra, ga] / xs["absorb"][ra, ga]
                f_w = xs["fission"][ra, ga] / xs["absorb"][ra, ga]
                k_score += nu_w.sum()
                fuel = (ra >= 1) & (ra <= N_SLABS)
                slab_fis += np.bincount(ra[fuel] - 1, weights=f_w[fuel], minlength=N_SLABS)
                fission += f_w[fuel].sum()
                capture += (1 - f_w[fuel]).sum()
                site_x.append(x[absorbed])
                site_w.append(nu_w)
            if scattered.any():
                down = scattered & (g == 0) & (rng.random(x.size) < p_down[reg])
                g = np.where(down, 1, g)
                mu = np.where(scattered, rng.uniform(-1, 1, x.size), mu)
            keep = ~absorbed
            x, g, mu = x[keep], g[keep], mu[keep]
        sites, weights = np.concatenate(site_x), np.concatenate(site_w)
        source = sites[rng.choice(sites.size, particles, p=weights / weights.sum())]
        gens.append(np.concatenate([[k_score / particles], slab_fis, [capture, fission]]))

    G = np.array(gens)
    A = G[inactive:]
    P = A[:, 1 : 1 + N_SLABS]
    half = N_SLABS // 2
    top, bottom = P[:, half:].sum(axis=1), P[:, :half].sum(axis=1)
    k = A[:, 0]
    ao = _ratio(top - bottom, top + bottom)
    peak_bin = int(np.argmax(P.sum(axis=0)))
    peak = _ratio(P[:, peak_bin] * N_SLABS, P.sum(axis=1))
    cf = _ratio(A[:, -2], A[:, -1])
    out = {
        "outputs": np.array([k.mean(), ao[0], peak[0], cf[0]]),
        "sigma": np.array([k.std(ddof=1) / np.sqrt(len(k)), ao[1], peak[1], cf[1]]),
    }
    if record:
        Pall = G[:, 1 : 1 + N_SLABS]
        t, b = Pall[:, half:].sum(axis=1), Pall[:, :half].sum(axis=1)
        out["k_generation"] = G[:, 0]
        out["ao_generation"] = (t - b) / (t + b)
    return out
