"""Name → factory lookup for experiment configs. Order of METHODS fixes plot colors."""

from __future__ import annotations

from collections.abc import Callable

from surrogatemodeling.core.protocols import Method, Problem
from surrogatemodeling.methods.adaptive import AdaptiveGP, ScreenedGP
from surrogatemodeling.methods.cokriging import CoKrigingAdaptive
from surrogatemodeling.methods.gp_fixed import SobolGP
from surrogatemodeling.methods.pce import SobolPCE
from surrogatemodeling.methods.saas import SobolSAAS
from surrogatemodeling.methods.saas_mf import SaasCostAware
from surrogatemodeling.problems.analytic import borehole, morris, otl, padded, wing_weight
from surrogatemodeling.problems.openmc.problem import OpenMCProblem
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
    # The OpenMC column itself (subprocess, ~6 CPU-min per HF run): final validation only.
    "openmc": OpenMCProblem,
}

METHODS: dict[str, Callable[[], Method]] = {
    "sobol_gp": SobolGP,
    "sobol_gp_matern": lambda: SobolGP(kernel="matern"),
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
    # Tier 1.5 combinations of the screening winners (factorial over pq, matern, warp; + pooled).
    "adaptive_iv_mf_cks_pq_matern": lambda: CoKrigingAdaptive(noise_scale=True, prequential=True, kernel="matern"),
    "adaptive_iv_mf_cks_pq_warp": lambda: CoKrigingAdaptive(noise_scale=True, prequential=True, warp=True),
    "adaptive_iv_mf_cks_matern_warp": lambda: CoKrigingAdaptive(noise_scale=True, kernel="matern", warp=True),
    "adaptive_iv_mf_cks_pq_matern_warp": lambda: CoKrigingAdaptive(noise_scale=True, prequential=True, kernel="matern", warp=True),
    "adaptive_iv_mf_cks_pq_matern_warp_pooled": lambda: CoKrigingAdaptive(
        noise_scale=True, prequential=True, kernel="matern", warp=True, pooled_scale=True
    ),
    # pq + matern + delayed warp: warping off until 200 points, then only the 8 most relevant inputs.
    "adaptive_iv_mf_cks_pq_matern_dwarp": lambda: CoKrigingAdaptive(
        noise_scale=True, prequential=True, kernel="matern", warp=True, warp_after=200, warp_top_k=8
    ),
    # Pre-validation screening (2026-10-06): converged-source menu, and a 50% HF seed.
    "adaptive_iv_mf_cks_pq_matern_safe": lambda: CoKrigingAdaptive(noise_scale=True, prequential=True, kernel="matern", safe_menu=True),
    # OpenMC validation round 2 (2026-10-06): 2x2 on the safe menu. "conv" pins only inactive cycles
    # (active cycles may be cut: cheaper, still converged); "pooled" learns one sigma scale per output.
    "adaptive_iv_mf_cks_pq_matern_safe_pooled": lambda: CoKrigingAdaptive(noise_scale=True, prequential=True, kernel="matern", safe_menu=True, pooled_scale=True),
    "adaptive_iv_mf_cks_pq_matern_conv": lambda: CoKrigingAdaptive(noise_scale=True, prequential=True, kernel="matern", safe_menu="inactive"),
    "adaptive_iv_mf_cks_pq_matern_conv_pooled": lambda: CoKrigingAdaptive(noise_scale=True, prequential=True, kernel="matern", safe_menu="inactive", pooled_scale=True),
    "adaptive_iv_mf_cks_pq_matern_seed50": lambda: CoKrigingAdaptive(noise_scale=True, prequential=True, kernel="matern", seed_fraction=0.5),
    # #8: pq_matern's cost-aware sampling, SAAS surrogate (subsample NUTS, fidelity knobs as inputs).
    "adaptive_iv_mf_saas": SaasCostAware,
    # #8 without knob inputs: all fidelities one function, noise corrected by learned config scales.
    "adaptive_iv_mf_saas_noknob": lambda: SaasCostAware(knobs=False),
}
