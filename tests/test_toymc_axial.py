import numpy as np
import pytest

from surrogatemodeling.problems.toymc.axial import run_axial
from surrogatemodeling.problems.toymc.axial_problem import DIST, ToyMCAxialProblem

NOMINAL = {m.name: float(m.rv.mean()) for m in DIST.marginals}


def test_menu_costs_are_normalized_and_sorted():
    spec = ToyMCAxialProblem().spec
    costs = [f.cost for f in spec.fidelities]
    assert spec.cost(spec.hf) == pytest.approx(1.0)
    assert costs == sorted(costs)
    assert spec.dim == 25 and spec.n_outputs == 4
    # Cutting active cycles alone still pays for the inactive ones (OpenMC's cost shape).
    active = next(f for f in spec.fidelities if f.name == "active=0.125")
    assert active.cost > 0.4


def test_rod_insertion_pushes_power_down():
    out = lambda rod: run_axial({**NOMINAL, "rod_insertion": rod}, 400, 20, 40, np.random.default_rng(0))["outputs"]
    assert out(90.0)[1] < out(0.0)[1] - 0.3  # axial offset falls with rod depth
    assert out(90.0)[0] < out(0.0)[0]  # and k-eff with it


def test_axial_offset_sigma_is_under_reported():
    """The calibration's headline: batch sigma for axial offset is several times too small."""
    reps = [run_axial(NOMINAL, 300, 20, 40, np.random.default_rng(s)) for s in range(24)]
    ao = np.array([r["outputs"][1] for r in reps])
    reported = np.mean([r["sigma"][1] for r in reps])
    assert ao.std(ddof=1) / reported > 2.0
