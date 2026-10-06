# `adaptive_iv_mf_cks_pq_matern`: #7s + prequential calibration + Matérn

The best Tier 1.5 combination by AUC, and the full-menu ablation in the OpenMC validation ([ValidationRecommendation.md](../ValidationRecommendation.md)).

| | |
|---|---|
| Design | 25% high-fidelity Sobol seed → greedy cost-aware batches |
| Fidelity | Chooses from the **full** menu (particles, inactive and active cycles) |
| Model | Co-kriging (#7) with learned per-config σ scales (#7s); Matérn-5/2 kernels |
| Uncertainty | Yes. Latent variance, scaled by prequential calibration (≥ 1) |
| Code | `CoKrigingAdaptive(noise_scale=True, prequential=True, kernel="matern")` in [cokriging.py](../../src/surrogatemodeling/methods/cokriging.py) |

## How it works
It combines two of the screening variants ([cks_variants.md](cks_variants.md)) on top of [#7s](adaptive_iv_mf_cks.md):
- **Matérn-5/2 kernel.** It's less smooth than RBF, so it's less confident between points. This was the biggest single calibration win in screening (coverage 0.77 → 0.92).
- **Prequential calibration (`pq`).** Each new batch is predicted before the model trains on it. The predictive sd is then scaled by the smallest factor in [1, 10] that makes 95% of the latest 300 out-of-sample residuals fall inside the interval. The scale is clamped at ≥ 1, so it only widens error bars.

## Results (3 seeds, budget 100)
- **Strong on the smooth outputs:** k-eff 0.059 and capture/fission 0.066 final NRMSE on `toymc_axial` vs 0.142 / 0.177 for the baseline. Coverage 0.94–0.96.
- **Fails on axial offset:** 0.287 vs the baseline's 0.179, with coverage 0.72. Cheap configs with few inactive cycles carry a source-convergence bias the model can't learn.
- **Run time:** 125 min per run on `toymc_axial` (about 1,600 evaluations).

[`adaptive_iv_mf_cks_pq_matern_safe`](adaptive_iv_mf_cks_pq_matern_safe.md) removes the cycle-cutting configs and fixes both the accuracy and the coverage.

## Role in the validation
An optional ablation. If the same axial-offset failure appears on OpenMC, it confirms that the safe menu wins for the reason we think, not by chance on the toy.
