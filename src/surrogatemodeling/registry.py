"""Name → factory lookup for experiment configs. Order of METHODS fixes plot colors."""

from __future__ import annotations

from collections.abc import Callable

from surrogatemodeling.core.protocols import Method, Problem
from surrogatemodeling.methods.adaptive import AdaptiveGP, ScreenedGP
from surrogatemodeling.methods.cokriging import CoKrigingAdaptive
from surrogatemodeling.methods.gp_fixed import SobolGP
from surrogatemodeling.methods.pce import SobolPCE
from surrogatemodeling.methods.saas import SobolSAAS
from surrogatemodeling.problems.analytic import borehole, morris, otl, padded, wing_weight
from surrogatemodeling.problems.toymc.axial_problem import ToyMCAxialProblem
from surrogatemodeling.problems.toymc.problem import ToyMCProblem

PROBLEMS: dict[str, Callable[[], Problem]] = {
    "borehole": borehole,
    "otl": otl,
    "wing_weight": wing_weight,
    "morris": morris,
    "borehole_d30": lambda: padded(borehole(), 30),
    "wing_weight_d40": lambda: padded(wing_weight(), 40),
    "toymc": ToyMCProblem,
    # 1D twin of the OpenMC column, with the calibrated 12-config cycle/particle menu.
    "toymc_axial": ToyMCAxialProblem,
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
    # #5b with a per-config bias GP (co-kriging), for menus whose cheap configs are biased.
    "adaptive_iv_mf_ck": CoKrigingAdaptive,
    # #7 plus conformal-style LOO calibration of the predictive variance (coverage).
    "adaptive_iv_mf_ck_cal": lambda: CoKrigingAdaptive(calibrate=True),
    # #7 with a learned per-config multiplicative scale on reported sigma^2 (instead of
    # additive extra noise), for tally errors that are under-reported by a factor.
    "adaptive_iv_mf_cks": lambda: CoKrigingAdaptive(noise_scale=True),
    # Calibration screening variants of #7s (CostAwarePlan.md, calibration follow-up): one change each.
    "adaptive_iv_mf_cks_pq": lambda: CoKrigingAdaptive(noise_scale=True, prequential=True),
    "adaptive_iv_mf_cks_matern": lambda: CoKrigingAdaptive(noise_scale=True, kernel="matern"),
    "adaptive_iv_mf_cks_pooled": lambda: CoKrigingAdaptive(noise_scale=True, pooled_scale=True),
    "adaptive_iv_mf_cks_log": lambda: CoKrigingAdaptive(noise_scale=True, log_outputs=True),
    "adaptive_iv_mf_cks_wt": lambda: CoKrigingAdaptive(noise_scale=True, weighted=True),
    "adaptive_iv_mf_cks_ens": lambda: CoKrigingAdaptive(noise_scale=True, ensemble=3),
    "adaptive_iv_mf_cks_warp": lambda: CoKrigingAdaptive(noise_scale=True, warp=True),
}
