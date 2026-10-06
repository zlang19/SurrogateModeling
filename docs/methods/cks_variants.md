# #7s calibration variants (screening)

Each variant is [#7s `adaptive_iv_mf_cks`](adaptive_iv_mf_cks.md) with **exactly one change**, so its effect can be isolated. They come from the calibration follow-up in [CostAwarePlan.md](../CostAwarePlan.md#calibration-follow-up-decided-2026-10-04-starts-after-costaware-finishes), which responds to every cost-aware method being badly overconfident on `toymc_axial` (axial-offset coverage about 0.5).

Code: options of `CoKrigingAdaptive` / `CoKrigingGPs` in [cokriging.py](../../src/surrogatemodeling/methods/cokriging.py), plus the shared support in [gp_common.py](../../src/surrogatemodeling/methods/gp_common.py).

| Registry name | Change | Targets |
|---|---|---|
| `adaptive_iv_mf_cks_pq` | **Next-batch (prequential) calibration.** Each new batch is predicted at its own fidelity config *before* the model trains on it. The residuals, compared with the latent variance plus the model's own noise estimate, are kept for the latest 300 points. The predictive sd is then scaled by the smallest factor that makes 95% of them fall inside the interval. Error-bar scaling only; the acquisition is unchanged | Overconfidence from any cause, measured out of sample, unlike #7c's in-sample LOO |
| `adaptive_iv_mf_cks_matern` | **Matérn-5/2** kernel for f, instead of RBF | RBF's over-smoothness, which makes it confident between points |
| `adaptive_iv_mf_cks_pooled` | **One σ scale per output**, shared by all configs, instead of one per (output, config) | Poorly identified scales at rarely used configs; OpenMC showed under-reporting depends on the output, not the knobs |
| `adaptive_iv_mf_cks_log` | Outputs that are positive in the first fit's data are **modeled on the log scale** (delta-method variances), with lognormal mean and variance mapped back | Skewed positive outputs such as peaking and borehole flow, which get asymmetric error bars |
| `adaptive_iv_mf_cks_wt` | **Error-weighted acquisition.** Each output's variance reduction is weighted by its estimated latent error, the root of the mean of (residual² − noise variance) from the same out-of-sample residuals, normalized by its spread | Equal weights let easy outputs dominate sampling while axial offset lost to the baseline |
| `adaptive_iv_mf_cks_ens` | **Hyperparameter ensemble of 3 GPs.** Extra members start each full refit from randomly perturbed lengthscales. Predictions are the mixture (mean of means; variance = mean of variances + variance of means); member 0 drives the acquisition | Plug-in hyperparameters treated as certain, the leading suspect for coverage decaying with data. A cheap partial stand-in for SAAS. About 3× the fit cost |
| `adaptive_iv_mf_cks_warp` | **Learned input warping**: a Kumaraswamy CDF per input (botorch `Warp`), learned with the other hyperparameters | Non-stationary responses, e.g. near the rod tip and zone boundaries |

## Diagnostics added
- `prequential_scale`: per output, the sd multiplier. 1 means already calibrated; 2 means real errors are twice the claimed ones.
- `output_weights`: per output, the acquisition weight (mean 1).

## Combining variants (Tier 1.5)
If several variants win on their own, their combinations are tested with a small factorial design rather than stacked blindly:
- **`pq` composes cleanly.** It rescales error bars from held-out residuals on top of any model.
- **`matern` and `warp` may interact.** Both target kernel misspecification, and warping adds ~50 hyperparameters for 25 inputs.
- **`ens` is added last.** At ~3× the fit cost, it goes onto the best combination only if coverage is still short.

For {pq, matern, warp}, only 4 new arms are needed beyond screening (`pq+matern`, `pq+warp`, `matern+warp`, all three). Arms are compared paired by seed. See [CostAwarePlan.md](../CostAwarePlan.md#calibration-follow-up-decided-2026-10-04-starts-after-costaware-finishes).

## Pre-validation results (2026-10-06, 3 seeds, paired by seed)

| `toymc_axial` final NRMSE (coverage) | `pq_matern_safe` | `pq_matern` | `pq_matern_seed50` | `sobol_gp_matern` |
|---|---|---|---|---|
| k-eff | **0.048** (0.98) | 0.059 (0.96) | 0.101 (0.78) | 0.145 (0.92) |
| axial offset | **0.144** (0.97) | 0.287 (0.72) | 0.275 (0.90) | 0.188 (0.93) |
| axial peaking | **0.283** (0.97) | 0.370 (0.86) | 0.365 (0.92) | 0.387 (0.92) |
| capture/fission | **0.052** (0.97) | 0.066 (0.94) | 0.079 (0.93) | 0.173 (0.90) |
| run time | 25 min | 125 min | 41 min | 6 min |

`safe` keeps only configs with high-fidelity cycle counts (particles 0.1/0.25/0.5/1×). On Borehole-30D, where only histories vary, it is identical to `pq_matern`. See [ValidationRecommendation.md](../ValidationRecommendation.md).

## Combination results (2026-10-05, 3 seeds, paired by seed against the base #7s)

| Arm | AUC-NRMSE | AUC-NCRPS | Final NRMSE | Coverage, worst output | Axial offset: final NRMSE / coverage |
|---|---|---|---|---|---|
| base (#7s) | 1.000 | 1.000 | 1.000 | 0.51 | 0.295 / 0.51 |
| matern | 1.005 | 0.999 | 0.973 | 0.67 | 0.287 / 0.67 |
| pq_matern | 1.005 | 1.000 | 0.973 | 0.72 | 0.287 / 0.72 |
| matern_warp | 1.091 | 1.211 | **0.789** | 0.84 | **0.177** / 0.89 |
| **pq_matern_warp** | 1.091 | 1.277 | **0.789** | **0.96** | **0.177 / 0.96** |
| pq_matern_warp_pooled | 1.115 | 1.264 | 0.915 | 0.93 | 0.186 / 0.95 |

(Ratios vs the base, median over problem × output × seed; < 1 is better.)

**Factorial main effects** (on vs off, averaged over the other two factors):
- **`matern`:** final NRMSE ×0.90, coverage +0.14.
- **`warp`:** final NRMSE ×0.89, coverage +0.05, but AUC-NRMSE ×1.11 and AUC-NCRPS ×1.26 (bad early).
- **`pq`:** accuracy unchanged by design, coverage +0.08.

**Strong `matern` × `warp` interaction:** either alone cuts final error about 3%; together, 21%. Matérn's rougher kernel seems to stop the warp from overfitting.

**Over the budget** (axial offset, `toymc_axial`):

| Cost | base | pq_matern | pq_matern_warp |
|---|---|---|---|
| 25 | NRMSE 0.44, NLL −0.3 | 0.45 | **0.73**, NLL 1.7 (matern_warp without pq: NLL 247) |
| 50 | 0.32 | 0.31 | **0.25**, coverage 0.95 |
| 100 | 0.295, coverage 0.51 | 0.287, coverage 0.72 | **0.177**, coverage **0.96** |

**What the numbers say:**
- **Warping hurts with little data and wins with enough:** it overfits below cost ≈ 30, and is best on every output by cost 50. AUC penalizes the early phase, which is why the AUC ratios disagree with the final numbers.
- **`pq_matern_warp` is the first arm to pass the coverage gate on every output** (0.96–0.97). Its final axial offset (0.177) beats the high-fidelity baseline's 0.216 from `costaware`, and its final peaking is 0.295 vs the base's 0.373.
- `pooled` adds nothing on top.
`configs/experiments/calib_screen.toml`: the base and all 7 variants × `toymc_axial` and Borehole-30D × 3 seeds, all in one 8-worker pool. They're scored by AUC-NRMSE, AUC-NCRPS and final coverage against the 0.90 gate ([EvaluationCriteria.md](../EvaluationCriteria.md)). Winners go to the combination stage (above), then the best one or two combinations go to a 10-seed confirmation run.

## Screening results (2026-10-05, 3 seeds, paired by seed against the base #7s)

| Variant | AUC-NRMSE ratio | AUC-NCRPS ratio (share improved) | Final coverage, median (axial offset) | Verdict |
|---|---|---|---|---|
| `matern` | 1.005 | 0.999 | **0.92** (**0.67**); base 0.77 (0.51) | **Big calibration win** at unchanged accuracy: k-eff 0.96, capture/fission 0.94, Borehole 0.96 |
| `pq` | 1.000 | **0.993 (93%)** | 0.74 (0.58) | Consistent small NCRPS win, but it **shrank** error bars where residuals are noise-dominated (Borehole 0.83 → 0.77). Now clamped to widen only |
| `warp` | 1.12 | 1.26 (0%) | 0.72 (0.64) | Worse early (overfits small data), but **final axial-offset NRMSE 0.200**, the first method to beat the baseline (0.216) on axial offset |
| `pooled` | 0.997 | 1.003 | 0.79 (**0.63**) | Mild axial-offset win (final 0.281) |
| `log` | 0.987 | 0.982 | 0.74 (0.56) | Marginal |
| `ens` | 0.994 | 0.994 | 0.79 (0.50) | No gain at 3× cost: dropped |
| `wt` | 1.000 | 1.000 | 0.75 (0.54) | No effect: dropped |

**Warping's NLL blow-up:** the `warp` arms (alone and in combinations) reach NLL of 600–30,000 early in runs, while their NRMSE stays reasonable. Most likely, test points outside the training box (normal-tailed inputs) get clamped by botorch's `Warp`, so the GP's variance collapses there while the error doesn't. That's extreme overconfidence outside the box. Watch it in the combination analysis, and consider warping on a padded box (or clamping only the variance) if warping is kept.

**Carried to the combination stage** (`calib_combo.toml`, results in `results/calib_screen`): `pq+matern`, `pq+warp`, `matern+warp`, `pq+matern+warp`, and `pq+matern+warp+pooled`. These use the clamped `pq` (scale ≥ 1); the screening `pq` arm used the unclamped version.

**Sanity check** (Borehole, honest σ, 1 seed): every variant ran. `_pq` found a scale of 1.00, i.e. nothing to fix, and `_wt` matched the base exactly, as it must with a single output.
