# Surrogate modeling methods

Every method implements the same ask/tell protocol ([protocols.py](../../src/surrogatemodeling/core/protocols.py)):

```python
setup(spec, budget, rng)                    # spec: input distribution, outputs, fidelity menu (v2)
ask(n, budget_remaining) -> (X, fidelity)   # up to n points + a fidelity config index each
tell(X, fidelity, y, sigma)                 # noisy outputs and their reported standard errors
predict(X) -> Prediction(mean, var | None)  # var is the predictive variance of the noise-free output
diagnostics() -> dict                       # optional; logged every batch, shown in the dashboard
```

The runner asks for batches of 5 and stops when the budget (in high-fidelity run equivalents) is spent. After each batch it scores `predict` on a cached test set drawn from the input distribution.

## The methods

| # | Registry name | Design | Model | Fidelity | Doc |
|---|---|---|---|---|---|
| 1 | `sobol_gp` | Fixed Sobol | GP, ARD, fixed per-point noise | High only | [sobol_gp.md](sobol_gp.md) |
| 2 | `sobol_pce` | Fixed Sobol | Sparse polynomial chaos (hybrid LARS) | High only | [sobol_pce.md](sobol_pce.md) |
| 3 | `sobol_saas` | Fixed Sobol | Fully Bayesian SAAS GP (NUTS) | High only | [sobol_saas.md](sobol_saas.md) |
| 4 | `adaptive_iv` | Sobol seed → adaptive | GP + integrated-variance reduction | High only | [adaptive_iv.md](adaptive_iv.md) |
| 4b | `adaptive_epig` | Sobol seed → adaptive | GP + EPIG | High only | [adaptive_epig.md](adaptive_epig.md) |
| 5 | `adaptive_iv_mf` | Sobol seed → adaptive | GP + integrated variance per unit cost | Chooses from ladder | [adaptive_iv_mf.md](adaptive_iv_mf.md) |
| 5b | `adaptive_iv_mf_xn` | Sobol seed → adaptive | #5 + learned extra noise | Chooses from ladder | [adaptive_iv_mf_xn.md](adaptive_iv_mf_xn.md) |
| 6 | `screen_gp` | Sobol seed → adaptive | ARD screening → GP on active inputs | High only | [screen_gp.md](screen_gp.md) |
| 7 | `adaptive_iv_mf_ck` | Sobol seed → adaptive | Co-kriging: per-config bias GP + bias-aware joint acquisition | Chooses from menu | [adaptive_iv_mf_ck.md](adaptive_iv_mf_ck.md) |
| 7c | `adaptive_iv_mf_ck_cal` | Sobol seed → adaptive | #7 + conformal-style LOO calibration | Chooses from menu | [adaptive_iv_mf_ck_cal.md](adaptive_iv_mf_ck_cal.md) |
| 7s | `adaptive_iv_mf_cks` | Sobol seed → adaptive | #7 with learned per-config σ scales (multiplicative) | Chooses from menu | [adaptive_iv_mf_cks.md](adaptive_iv_mf_cks.md) |
| 8 | `adaptive_iv_mf_saas`, `_saas_noknob` | Sobol seed → adaptive | `pq_matern`'s cost-aware sampling with a SAAS surrogate (subsample NUTS; fidelity knobs as inputs, or none) | Chooses from menu | [adaptive_iv_mf_saas.md](adaptive_iv_mf_saas.md) |
| 1m | `sobol_gp_matern` | Fixed Sobol | #1 with a Matérn-5/2 kernel | High only | [sobol_gp_matern.md](sobol_gp_matern.md) |
| 7s-pm | `adaptive_iv_mf_cks_pq_matern` | Sobol seed → adaptive | #7s + prequential calibration + Matérn | Chooses from menu | [adaptive_iv_mf_cks_pq_matern.md](adaptive_iv_mf_cks_pq_matern.md) |
| 7s-safe | `adaptive_iv_mf_cks_pq_matern_safe` | Sobol seed → adaptive | `pq_matern` on a converged-source menu (only particles vary) | Chooses from menu (HF cycles only) | [adaptive_iv_mf_cks_pq_matern_safe.md](adaptive_iv_mf_cks_pq_matern_safe.md) |
| 7s-x | `adaptive_iv_mf_cks_{pq,matern,pooled,log,wt,ens,warp}` | Sobol seed → adaptive | #7s with one calibration change each (screening) | Chooses from menu | [cks_variants.md](cks_variants.md) |

The registry order in [registry.py](../../src/surrogatemodeling/registry.py) also fixes each method's color in plots and the dashboard.

## Shared machinery

### GP core
Code: [gp_common.py](../../src/surrogatemodeling/methods/gp_common.py).

All GP methods except SAAS use `IndependentGPs`:
- **Per output:** one botorch `SingleTaskGP` for each output.
- **Inputs:** mapped to the unit training box.
- **Outputs:** standardized inside the class.
- **Noise:** fixed per point from the reported σ², so a 1/16-fidelity point is automatically trusted 16× less. With `extra_noise=True`, a homoscedastic noise term is learned on top.
- **Refits:** hyperparameters are re-optimized only when the data has grown by 20% since the last full fit (`refit_growth`); in between, the previous hyperparameters are reused (kernel, mean, and any trainable noise model). The baseline `sobol_gp` re-optimizes every batch.
- **Exact solves:** every solve uses exact Cholesky. gpytorch's default switches to CG/Lanczos above 800 points, and its caches made cost-aware runs (about 2,000 cheap points) grow to 6–7 GB.

### Hybrid design and greedy batches
Code: [adaptive.py](../../src/surrogatemodeling/methods/adaptive.py), [acquisition.py](../../src/surrogatemodeling/methods/acquisition.py).

The adaptive methods share one loop:
1. **Seed:** the first 25% of the budget is a high-fidelity scrambled Sobol design.
2. **Per adaptive batch:**
   - draw 1024 fresh candidates, half uniform over the training box and half from the input distribution (clipped to the box);
   - compute the GP's joint posterior covariance over those candidates plus 512 fixed **reference points** sampled from the input distribution.
3. **Greedy batch selection:** pick the best (candidate, fidelity) by the acquisition score, then apply the exact **rank-one covariance update** for observing it, and repeat. A GP's posterior variance doesn't depend on the observed value, so this needs no fantasies and no refits.
4. **Expected noise:** the high-fidelity noise is estimated as the median of `var × fidelity` over the data, using the fact that MC variance scales as 1/fidelity.

Because the reference points are samples of the input distribution, both acquisition scores aim at the benchmark's own objective: pdf-weighted predictive error.

## Results so far

From the `full` experiment: 7 problems, 10 seeds, a budget of 100 high-fidelity-equivalent runs. SAAS is not run yet.

**Final NRMSE** (seed median, at the full budget):

| Method | borehole | borehole_d30 | morris | otl | toymc k_eff | toymc power_ratio | toymc capture/fission | wing_weight | wing_weight_d40 |
|---|---|---|---|---|---|---|---|---|---|
| sobol_gp | 0.029 | 0.091 | 0.451 | 0.041 | 0.099 | 0.155 | 0.121 | 0.065 | 0.104 |
| sobol_pce | 0.048 | 0.129 | 0.594 | 0.049 | 0.139 | 0.203 | 0.185 | 0.071 | 0.153 |
| adaptive_iv | 0.028 | 0.093 | 0.459 | 0.033 | 0.101 | 0.147 | 0.111 | 0.054 | 0.106 |
| adaptive_epig | 0.031 | 0.099 | 0.486 | 0.033 | 0.105 | 0.138 | 0.110 | 0.053 | 0.107 |
| adaptive_iv_mf | 0.019 | 0.047 | 0.197 | 0.031 | 0.068 | 0.272 | 0.067 | 0.051 | 0.083 |
| **adaptive_iv_mf_xn** | 0.020 | **0.045** | **0.190** | 0.034 | **0.062** | **0.105** | **0.062** | **0.049** | 0.084 |
| screen_gp | 0.024 | 0.090 | 0.471 | 0.035 | 0.137 | 0.130 | 0.146 | 0.051 | 0.107 |

**Mean rank** by area under the error-vs-cost curve (lower is better):

| Rank | Method | Mean rank |
|---|---|---|
| 1 | `adaptive_iv_mf_xn` | 1.33 |
| 2 | `adaptive_iv_mf` | 2.22 |
| 3 | `adaptive_iv` | 3.67 |
| 4 | `screen_gp` | 3.78 |
| 5 | `adaptive_epig` | 4.78 |
| 6 | `sobol_gp` | 5.22 |
| 7 | `sobol_pce` | 7.00 |

`uv run sm report results/full` regenerates the full tables in `results/full/ranking.md`, including coverage and the AUC metric. What each table measures is in [../EvaluationCriteria.md](../EvaluationCriteria.md). What these numbers mean is in [../Learnings.md](../Learnings.md).
