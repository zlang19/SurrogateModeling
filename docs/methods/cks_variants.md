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

## Screening experiment
`configs/experiments/calib_screen.toml`: the base and all 7 variants × `toymc_axial` and Borehole-30D × 3 seeds, all in one 8-worker pool. They're scored by AUC-NRMSE, AUC-NCRPS and final coverage against the 0.90 gate ([EvaluationCriteria.md](../EvaluationCriteria.md)). The winners go to a 10-seed confirmation run.

**Sanity check** (Borehole, honest σ, 1 seed): every variant ran. `_pq` found a scale of 1.00, i.e. nothing to fix, and `_wt` matched the base exactly, as it must with a single output.
