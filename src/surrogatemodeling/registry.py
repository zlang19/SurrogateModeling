"""Name → factory lookup for experiment configs. Order of METHODS fixes plot colors."""

from __future__ import annotations

from collections.abc import Callable

from surrogatemodeling.core.protocols import Method, Problem
from surrogatemodeling.methods.gp_fixed import SobolGP
from surrogatemodeling.problems.analytic import borehole, morris, otl, padded, wing_weight
from surrogatemodeling.problems.toymc.problem import ToyMCProblem

PROBLEMS: dict[str, Callable[[], Problem]] = {
    "borehole": borehole,
    "otl": otl,
    "wing_weight": wing_weight,
    "morris": morris,
    "borehole_d30": lambda: padded(borehole(), 30),
    "wing_weight_d40": lambda: padded(wing_weight(), 40),
    "toymc": ToyMCProblem,
}

METHODS: dict[str, Callable[[], Method]] = {
    "sobol_gp": SobolGP,
}
