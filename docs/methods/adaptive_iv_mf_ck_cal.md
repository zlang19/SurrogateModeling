# #7c `adaptive_iv_mf_ck_cal`: #7 with conformal-style LOO calibration

| | |
|---|---|
| Design, fidelity, model, acquisition | Identical to [#7](adaptive_iv_mf_ck.md) |
| Uncertainty | Yes, rescaled by leave-one-out calibration |
| Code | [gp_common.py](../../src/surrogatemodeling/methods/gp_common.py) (`IndependentGPs._loo_scale`), [cokriging.py](../../src/surrogatemodeling/methods/cokriging.py) |

## How it works
1. **Standardized LOO residuals:** after each fit, a GP gives each training point's leave-one-out residual in closed form, already standardized by that prediction's sd (noise included):

   ```
   z_i = [K⁻¹ (y − m)]_i / sqrt([K⁻¹]_ii)
   ```

2. **Scale:** the predictive sd is multiplied by s = quantile₀.₉₅(|z|) / 1.96, clamped to [0.5, 5]. This makes 95% of the held-out residuals fall inside the nominal 95% interval.
3. **What stays the same:** the mean, and therefore the error, is unchanged. Only the error bars move, and the acquisition still uses the unscaled model.

This is split-conformal calibration using LOO residuals in place of a held-out set. It costs one extra Cholesky factorization per fit. The plan's "conformalized ensembles" candidate is covered as a *calibration wrapper*. Ensembles of neural networks are deferred with the other neural methods.

## Why it's here
The surrogate feeds UQ, so coverage matters as well as error. Coverage has generally run a little low (0.84–0.95), and badly low wherever σ is under-reported.

## Diagnostics
Same as #7, plus `calibration_scale` per output (1 = already calibrated).

## Results
- **Biased Borehole test:** the scale came out at 1.02–1.05, because #7's error bars were already honest there. Coverage was 0.93–0.97, with accuracy identical to #7.
- **`costaware` (5 seeds, `toymc_axial`):** accuracy identical to #7, as designed, but coverage barely moved (axial offset 0.486 → 0.500, k-eff 0.647 → 0.662). In-sample LOO residuals share the fitted hyperparameters' optimism, so they don't see the overconfidence. It's being replaced by next-batch (prequential) calibration, which uses each batch's out-of-sample errors (see [CostAwarePlan.md](../CostAwarePlan.md#calibration-follow-up-decided-2026-10-04-starts-after-costaware-finishes)).
