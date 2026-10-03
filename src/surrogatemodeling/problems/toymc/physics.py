"""Two-group k-eigenvalue Monte Carlo on a symmetric 1D slab.

Geometry (x from 0 to W, vacuum outside):

    | reflector | outer core | inner core | outer core | reflector |

Cross sections are made up but parametric, chosen so the sensitivities look like a
light-water lattice: enrichment, boron and moderator density dominate k; the zone
enrichment split drives the power ratio; Doppler and nuclear-data multipliers are small.

Transport is analog with Woodcock (delta) tracking and isotropic scattering, vectorized
over the particles of a generation. Power iteration runs `inactive` generations to
converge the source, then scores `active` generations; estimates and their standard
errors come from the active-generation batch statistics, as in an MCNP KCODE run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Material indices
REFL, OUTER, INNER = 0, 1, 2
# Reference conditions the parametric cross sections are normalized to
_VF0, _ENR0, _RHO0, _TF0, _TM0 = 0.35, 3.5, 0.72, 900.0, 580.0
_WATER_FRAC0 = 1 - _VF0


@dataclass(frozen=True)
class XS:
    """Macroscopic two-group cross sections (1/cm), shape (n_materials, 2) unless noted."""

    total: np.ndarray
    absorb: np.ndarray
    nu_fission: np.ndarray
    fission: np.ndarray
    downscatter: np.ndarray  # (n_materials,) fast -> thermal


def material_xs(p: dict[str, float], enrichment: float | None, gd: float = 0.0) -> tuple[float, ...]:
    """(total, absorb, nu_fission, fission) per group plus downscatter, for one material.

    enrichment=None means pure reflector water.
    """
    is_fuel = enrichment is not None
    vf = p["fuel_vf"] if is_fuel else 0.0
    rho = p["refl_density"] if not is_fuel else p["mod_density"]
    water = rho * (1 - vf) / (_RHO0 * _WATER_FRAC0)  # water number density, relative to reference core
    fuel = vf / _VF0
    enr = (enrichment or 0.0) / _ENR0
    doppler = 1 + 0.15 * (np.sqrt(p["fuel_temp"] / _TF0) - 1)
    thermal = np.sqrt(_TM0 / p["mod_temp"])  # 1/v-like thermal cross-section shift
    boron = 1.2e-5 * p["boron_ppm"] * water * p["boron_abs"]

    # Fast group
    downscatter = 0.020 * water * p["h_scat"]
    fast_scatter = 0.20 * (0.4 * fuel * p["o_scat"] + 0.6 * water * p["h_scat"])
    f_fis1 = 0.0024 * fuel * (0.5 + 0.5 * enr) * p["u238_fastfis"]
    a1 = 0.0090 * fuel * doppler * p["u238_cap"] + f_fis1 + 0.0004 * water + 0.0005 * fuel * p["clad_abs"]
    nf1 = 2.5 * f_fis1 * p["nu_fast"]

    # Thermal group
    f_fis2 = 0.060 * fuel * enr * thermal * p["u235_fis"]
    a2 = (
        f_fis2
        + 0.17 * 0.060 * fuel * enr * thermal * p["u235_cap"]
        + 0.012 * fuel * thermal * p["u238_cap"]
        + 0.010 * water * thermal * p["h_cap"]
        + boron * thermal
        + 0.05 * gd * thermal
        + 0.002 * fuel * p["clad_abs"]
    )
    nf2 = 2.43 * f_fis2 * p["nu_thermal"]
    thermal_scatter = 1.5 * water * p["h_scat"] + 0.3 * fuel * p["o_scat"]

    t1 = a1 + downscatter + fast_scatter
    t2 = a2 + thermal_scatter
    return (t1, t2), (a1, a2), (nf1, nf2), (f_fis1, f_fis2), downscatter


def build_xs(p: dict[str, float]) -> XS:
    mats = [material_xs(p, None), material_xs(p, p["enr_outer"]), material_xs(p, p["enr_inner"], p["gd_inner"])]
    arr = lambda i: np.array([m[i] for m in mats], dtype=float)
    return XS(total=arr(0), absorb=arr(1), nu_fission=arr(2), fission=arr(3), downscatter=arr(4))


def boundaries(p: dict[str, float]) -> np.ndarray:
    """Region edges [0, r, r+o, r+o+i, r+2o+i, W] for the five slab regions."""
    r, core = p["refl_thickness"], p["core_width"]
    inner = p["inner_fraction"] * core
    outer = (core - inner) / 2
    return np.cumsum([0.0, r, outer, inner, outer, r])


_REGION_MATERIAL = np.array([REFL, OUTER, INNER, OUTER, REFL])


@dataclass(frozen=True)
class Estimate:
    mean: np.ndarray  # (3,) k_eff, power_ratio (inner/outer power density), capture_to_fission
    std_err: np.ndarray  # (3,)


def run_kcode(p: dict[str, float], n_particles: int, rng: np.random.Generator, inactive: int = 15, active: int = 20) -> Estimate:
    xs = build_xs(p)
    edges = boundaries(p)
    width = edges[-1]
    majorant = xs.total.max(axis=0)  # (2,)
    # Scattering is the remainder after absorption; probability a fast scatter is a downscatter.
    p_down = xs.downscatter / (xs.total[:, 0] - xs.absorb[:, 0])

    inner_len, outer_len = edges[3] - edges[2], (edges[2] - edges[1]) * 2
    source = rng.uniform(edges[1], edges[4], n_particles)  # flat over the core
    scores = []
    for gen in range(inactive + active):
        x = source.copy()
        g = np.zeros(n_particles, dtype=np.int64)
        mu = rng.uniform(-1, 1, n_particles)
        site_x, site_w = [], []
        k_score = fis_inner = fis_outer = capture_core = fission_core = 0.0

        while x.size:
            s = -np.log(rng.random(x.size)) / majorant[g]
            x = x + mu * s
            inside = (x > 0) & (x < width)
            x, g, mu = x[inside], g[inside], mu[inside]  # leaked particles are dropped

            mat = _REGION_MATERIAL[np.searchsorted(edges, x, side="right") - 1]
            sig_t = xs.total[mat, g]
            real = rng.random(x.size) * majorant[g] < sig_t
            u = rng.random(x.size) * sig_t
            absorbed = real & (u < xs.absorb[mat, g])
            scattered = real & ~absorbed

            # Absorption: analog scores, and a fission site weighted by nu*Sigma_f/Sigma_a.
            if absorbed.any():
                ma, ga, xa = mat[absorbed], g[absorbed], x[absorbed]
                nu_w = xs.nu_fission[ma, ga] / xs.absorb[ma, ga]
                f_w = xs.fission[ma, ga] / xs.absorb[ma, ga]
                k_score += nu_w.sum()
                fis_inner += f_w[ma == INNER].sum()
                fis_outer += f_w[ma == OUTER].sum()
                in_core = ma != REFL
                fission_core += f_w[in_core].sum()
                capture_core += (1 - f_w[in_core]).sum()
                site_x.append(xa)
                site_w.append(nu_w)

            # Scattering: isotropic; fast neutrons may downscatter to thermal.
            if scattered.any():
                down = scattered & (g == 0) & (rng.random(x.size) < p_down[mat])
                g = np.where(down, 1, g)
                mu = np.where(scattered, rng.uniform(-1, 1, x.size), mu)

            keep = ~absorbed
            x, g, mu = x[keep], g[keep], mu[keep]

        k_gen = k_score / n_particles
        sites, weights = np.concatenate(site_x), np.concatenate(site_w)
        source = sites[rng.choice(sites.size, n_particles, p=weights / weights.sum())]
        if gen >= inactive:
            scores.append((k_gen, fis_inner / inner_len, fis_outer / outer_len, capture_core, fission_core))

    k, inner, outer, capture, fission = np.array(scores).T
    stats = [(k.mean(), k.std(ddof=1) / np.sqrt(active)), _ratio(inner, outer), _ratio(capture, fission)]
    return Estimate(mean=np.array([m for m, _ in stats]), std_err=np.array([se for _, se in stats]))


def _ratio(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Ratio of generation totals, with a delta-method standard error.

    Averaging per-generation ratios instead would carry an O(1/N) bias, making low
    fidelity systematically biased rather than just noisier.
    """
    r = a.sum() / b.sum()
    se = np.std(a - r * b, ddof=1) / (np.sqrt(len(a)) * b.mean())
    return float(r), float(se)
