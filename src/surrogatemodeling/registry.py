"""Name → factory lookup for experiment configs. Order of METHODS fixes plot colors."""

from __future__ import annotations

from collections.abc import Callable

from surrogatemodeling.core.protocols import Method, Problem
from surrogatemodeling.methods.adaptive import AdaptiveGP, ScreenedGP
from surrogatemodeling.methods.gp_fixed import SobolGP
from surrogatemodeling.methods.pce import SobolPCE
from surrogatemodeling.methods.saas import SobolSAAS
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
    "sobol_pce": SobolPCE,
    "sobol_saas": SobolSAAS,
    "adaptive_iv": lambda: AdaptiveGP(score="iv"),
    "adaptive_epig": lambda: AdaptiveGP(score="epig"),
    "adaptive_iv_mf": lambda: AdaptiveGP(score="iv", cost_aware=True),
    "screen_gp": ScreenedGP,
    # #5 plus a learned homoscedastic noise term on top of the reported σ (guards against
    # under-reported tally errors, which otherwise get interpolated as signal).
    "adaptive_iv_mf_xn": lambda: AdaptiveGP(score="iv", cost_aware=True, extra_noise=True),
}
