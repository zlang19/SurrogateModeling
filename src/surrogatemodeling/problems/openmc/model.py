"""OpenMC model of a full-height 3x3 pin column, run INSIDE the openmc conda env.

This file is executed as `<openmc-env python> model.py <run_dir>` by problem.py and must
stay compatible with that env (Python 3.10, OpenMC 0.15); it imports nothing from this
project. It reads `<run_dir>/input.json`:

    {"params": {...14 inputs...}, "fidelity": {"particles": N, "inactive": I, "active": A},
     "seed": S, "threads": 1, "record_batches": false}

and writes `<run_dir>/output.json` with outputs, batch-statistics sigmas, CPU/wall time
and convergence diagnostics.

Geometry: a 3x3 lattice (8 fuel pins, one Gd-bearing corner pin, centre guide tube),
reflective radially, 200 cm of fuel in three enrichment zones, water reflectors above and
below, vacuum beyond. Coolant density falls linearly with height over 20 axial slabs; a
B4C control rod enters the guide tube from the top. The 2 m height gives the slow axial
source convergence of a full core at a fraction of the cost per history.

Sigmas follow MCNP practice: per-batch tally values -> mean and standard error over the
active batches (ratios: ratio of sums with a delta-method error). Generation-to-generation
correlation therefore makes them optimistic, exactly as in a production KCODE run.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import openmc
import openmc.lib

FUEL_HEIGHT = 200.0
N_SLABS = 20  # axial slabs (coolant density steps) == axial power bins
SLAB = FUEL_HEIGHT / N_SLABS
ZONES = ((0, 7), (7, 13), (13, 20))  # slab ranges of bottom / middle / top enrichment zones
GAP = 0.008
GT_INNER, GT_OUTER, ROD_RADIUS = 0.561, 0.602, 0.433
COOLANT_T, CLAD_T = 575.0, 600.0


def uo2(name, enrichment, temperature, gd_wt=0.0):
    m = openmc.Material(name=name, temperature=temperature)
    if gd_wt > 0:
        # Gd2O3 mixed into UO2 by weight
        uo = openmc.Material()
        uo.add_element("U", 1.0, enrichment=enrichment)
        uo.add_element("O", 2.0)
        uo.set_density("g/cm3", 10.4)
        gd = openmc.Material()
        gd.add_element("Gd", 2.0)
        gd.add_element("O", 3.0)
        gd.set_density("g/cm3", 7.4)
        m = openmc.Material.mix_materials([uo, gd], [1 - gd_wt / 100, gd_wt / 100], "wo", name=name)
        m.temperature = temperature
        return m
    m.add_element("U", 1.0, enrichment=enrichment)
    m.add_element("O", 2.0)
    m.set_density("g/cm3", 10.4)
    return m


def water(name, density, boron_ppm):
    m = openmc.Material(name=name, temperature=COOLANT_T)
    m.add_element("H", 2.0)
    m.add_element("O", 1.0)
    if boron_ppm > 0:
        # ppm by mass of the solution -> boron atoms per H2O molecule
        w_b = boron_ppm * 1e-6
        m.add_element("B", (w_b / (1 - w_b)) * (2 * 1.008 + 15.999) / 10.81, "ao")
    m.set_density("g/cm3", density)
    m.add_s_alpha_beta("c_H_in_H2O")
    return m


def build(p, fid, seed, threads):
    fuel_r = p["pellet_radius"]
    clad_ir = fuel_r + GAP
    clad_or = clad_ir + p["clad_thickness"]
    pitch = p["pitch"]

    zirc = openmc.Material(name="zirc", temperature=CLAD_T)
    zirc.add_element("Zr", 1.0)
    zirc.set_density("g/cm3", 6.55)
    b4c = openmc.Material(name="b4c", temperature=COOLANT_T)
    b4c.add_element("B", 4.0)
    b4c.add_element("C", 1.0)
    b4c.set_density("g/cm3", 1.76)

    enr = (p["enr_bottom"], p["enr_middle"], p["enr_top"])
    fuels = [uo2(f"fuel{z}", enr[z], p["fuel_temp"]) for z in range(3)]
    gd_fuels = [uo2(f"gdfuel{z}", enr[z], p["fuel_temp"], p["gd_wt"]) for z in range(3)]
    # coolant density at slab centres, falling linearly from inlet to outlet
    centres = (np.arange(N_SLABS) + 0.5) / N_SLABS
    coolants = [water(f"cool{k}", p["mod_density_in"] - p["density_drop"] * c, p["boron_ppm"]) for k, c in enumerate(centres)]
    refl_bot = water("refl_bot", p["mod_density_in"], p["boron_ppm"])
    refl_top = water("refl_top", p["mod_density_in"] - p["density_drop"], p["boron_ppm"])
    materials = openmc.Materials([zirc, b4c, refl_bot, refl_top, *fuels, *gd_fuels, *coolants])

    c_fuel, c_ci, c_co = (openmc.ZCylinder(r=r) for r in (fuel_r, clad_ir, clad_or))
    c_gti, c_gto, c_rod = (openmc.ZCylinder(r=r) for r in (GT_INNER, GT_OUTER, ROD_RADIUS))
    rod_tip = FUEL_HEIGHT - p["rod_insertion"]  # z of the rod tip, measured from the fuel bottom
    tip_plane = openmc.ZPlane(z0=rod_tip)

    def fuel_pin(fuel, cool):
        return openmc.Universe(cells=[
            openmc.Cell(fill=fuel, region=-c_fuel),
            openmc.Cell(region=+c_fuel & -c_ci),  # gap: void
            openmc.Cell(fill=zirc, region=+c_ci & -c_co),
            openmc.Cell(fill=cool, region=+c_co),
        ])

    def guide_tube(cool, k):
        lo, hi = k * SLAB, (k + 1) * SLAB
        if rod_tip >= hi:  # rod above this slab
            inner = [openmc.Cell(fill=cool, region=-c_gti)]
        elif rod_tip <= lo:  # fully rodded slab
            inner = [openmc.Cell(fill=b4c, region=-c_rod), openmc.Cell(fill=cool, region=+c_rod & -c_gti)]
        else:  # rod tip inside this slab
            inner = [
                openmc.Cell(fill=b4c, region=-c_rod & +tip_plane),
                openmc.Cell(fill=cool, region=-c_rod & -tip_plane),
                openmc.Cell(fill=cool, region=+c_rod & -c_gti),
            ]
        return openmc.Universe(cells=inner + [
            openmc.Cell(fill=zirc, region=+c_gti & -c_gto),
            openmc.Cell(fill=cool, region=+c_gto),
        ])

    half = 1.5 * pitch
    x0, x1 = openmc.XPlane(-half, boundary_type="reflective"), openmc.XPlane(half, boundary_type="reflective")
    y0, y1 = openmc.YPlane(-half, boundary_type="reflective"), openmc.YPlane(half, boundary_type="reflective")
    radial = +x0 & -x1 & +y0 & -y1
    z_bot = openmc.ZPlane(-p["refl_bottom"], boundary_type="vacuum")
    z_top = openmc.ZPlane(FUEL_HEIGHT + p["refl_top"], boundary_type="vacuum")
    slab_planes = [openmc.ZPlane(k * SLAB) for k in range(N_SLABS + 1)]

    cells = [
        openmc.Cell(fill=refl_bot, region=radial & +z_bot & -slab_planes[0]),
        openmc.Cell(fill=refl_top, region=radial & +slab_planes[-1] & -z_top),
    ]
    for k in range(N_SLABS):
        zone = next(z for z, (a, b) in enumerate(ZONES) if a <= k < b)
        f, g = fuel_pin(fuels[zone], coolants[k]), fuel_pin(gd_fuels[zone], coolants[k])
        lat = openmc.RectLattice()
        lat.pitch = (pitch, pitch)
        lat.lower_left = (-half, -half)
        lat.universes = [[g, f, f], [f, guide_tube(coolants[k], k), f], [f, f, f]]
        cells.append(openmc.Cell(fill=lat, region=radial & +slab_planes[k] & -slab_planes[k + 1]))
    geometry = openmc.Geometry(openmc.Universe(cells=cells))

    settings = openmc.Settings()
    settings.particles = int(fid["particles"])
    settings.inactive = int(fid["inactive"])
    settings.batches = int(fid["inactive"]) + int(fid["active"])
    settings.seed = int(seed)
    settings.temperature = {"method": "interpolation", "tolerance": 400}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box((-half, -half, 0.0), (half, half, FUEL_HEIGHT)),
        constraints={"fissionable": True},
    )
    ent = openmc.RegularMesh()
    ent.lower_left, ent.upper_right, ent.dimension = (-half, -half, 0.0), (half, half, FUEL_HEIGHT), (1, 1, 40)
    settings.entropy_mesh = ent
    settings.output = {"tallies": False, "summary": False}

    mesh = openmc.RegularMesh()
    mesh.lower_left, mesh.upper_right, mesh.dimension = (-half, -half, 0.0), (half, half, FUEL_HEIGHT), (1, 1, N_SLABS)
    t_power = openmc.Tally(name="axial")
    t_power.filters = [openmc.MeshFilter(mesh)]
    t_power.scores = ["fission"]
    t_fuel = openmc.Tally(name="fuel")
    t_fuel.filters = [openmc.MaterialFilter([*fuels, *gd_fuels])]
    t_fuel.scores = ["absorption", "fission"]
    tallies = openmc.Tallies([t_power, t_fuel])
    return openmc.model.Model(geometry, materials, settings, tallies), t_power.id, t_fuel.id


def ratio(a, b):
    """Ratio of sums over batches with a delta-method standard error."""
    r = a.sum() / b.sum()
    se = np.std(a - r * b, ddof=1) / (np.sqrt(len(a)) * b.mean())
    return float(r), float(se)


def main(run_dir):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "input.json").read_text())
    p, fid = cfg["params"], cfg["fidelity"]
    t0_cpu, t0_wall = time.process_time(), time.time()
    os.chdir(run_dir)
    model, axial_id, fuel_id = build(p, fid, cfg.get("seed", 1), cfg.get("threads", 1))
    model.export_to_model_xml()

    openmc.lib.init(["-s", str(cfg.get("threads", 1))], output=False)
    try:
        openmc.lib.simulation_init()
        t_axial, t_fuel = openmc.lib.tallies[axial_id], openmc.lib.tallies[fuel_id]
        n_batches = int(fid["inactive"]) + int(fid["active"])
        prev_axial, prev_fuel = 0.0, 0.0
        axial_batches, fuel_batches = [], []
        for b in range(n_batches):
            openmc.lib.next_batch()
            if b >= int(fid["inactive"]):
                cur_axial = t_axial.results[:, 0, 1].copy()  # running sum over active batches
                cur_fuel = t_fuel.results[:, :, 1].copy()
                axial_batches.append(cur_axial - prev_axial)
                fuel_batches.append(cur_fuel - prev_fuel)
                prev_axial, prev_fuel = cur_axial, cur_fuel
        k_mean, k_sd = openmc.lib.keff()
        openmc.lib.simulation_finalize()
    finally:
        openmc.lib.finalize()

    P = np.array(axial_batches)  # (active, 20)
    F = np.array(fuel_batches)  # (active, n_fuel_materials, 2): absorption, fission
    half_bins = N_SLABS // 2
    top, bottom = P[:, half_bins:].sum(axis=1), P[:, :half_bins].sum(axis=1)
    ao = ratio(top - bottom, top + bottom)
    peak_bin = int(np.argmax(P.sum(axis=0)))
    peak = ratio(P[:, peak_bin] * N_SLABS, P.sum(axis=1))
    absorption, fission = F[:, :, 0].sum(axis=1), F[:, :, 1].sum(axis=1)
    cf = ratio(absorption - fission, fission)

    out = {
        "outputs": {"k_eff": float(k_mean), "axial_offset": ao[0], "axial_peaking": peak[0], "capture_to_fission": cf[0]},
        "sigma": {"k_eff": float(k_sd), "axial_offset": ao[1], "axial_peaking": peak[1], "capture_to_fission": cf[1]},
        "cpu_time": time.process_time() - t0_cpu,
        "wall_time": time.time() - t0_wall,
        "peak_bin": peak_bin,
    }
    if cfg.get("record_batches"):
        sp_path = sorted(run_dir.glob("statepoint.*.h5"))[-1]
        with openmc.StatePoint(sp_path) as sp:
            out["k_generation"] = [float(k) for k in sp.k_generation]
            out["entropy"] = [float(e) for e in sp.entropy]
        out["axial_power"] = (P.sum(axis=0) / P.sum()).tolist()
        tb, bb = P[:, half_bins:].sum(axis=1), P[:, :half_bins].sum(axis=1)
        out["ao_batches"] = ((tb - bb) / (tb + bb)).tolist()
    (run_dir / "output.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
