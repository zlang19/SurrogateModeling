# #5b `adaptive_iv_mf_xn`: cost-aware IV with learned extra noise (current best)

| | |
|---|---|
| Design | Hybrid: 25% high-fidelity Sobol seed, then adaptive greedy batches |
| Fidelity | Chooses per point from the ladder (default [1/16, 1/4, 1]) |
| Model | Independent GPs: fixed per-point noise from σ² **plus a learned homoscedastic noise term** |
| Uncertainty | Yes (latent variance; the extra term is treated as observation noise) |
| Code | [adaptive.py](../../src/surrogatemodeling/methods/adaptive.py) (`cost_aware=True, extra_noise=True`), [registry.py](../../src/surrogatemodeling/registry.py) |

## How it works
This is #5 with one change: the GP likelihood is `FixedNoiseGaussianLikelihood(σ², learn_additional_noise=True)`.
- **Fitting:** the extra noise variance is learned by marginal likelihood, so the GP can find out from the data that the reported σ is too small.
- **Acquisition:** the learned term is added to every option's expected noise τ², so the cost trade-off accounts for it too.

## Why it was added
After the first full run, #5 plateaued on the toy MC power ratio, whose σ is under-reported. MCNP tally errors have the same weakness in loosely coupled cores, so taking σ at face value is a real risk for the target application. See [Learnings](../Learnings.md#the-reported-σ-cant-be-taken-at-face-value).

## Diagnostics
Same as #4, plus `extra_noise_sd` per output.

## Results (10 seeds)

| Output | Baseline | #5 | #5b |
|---|---|---|---|
| toymc power_ratio | 0.155 | 0.272 | **0.105** (best of all methods) |
| toymc k_eff | 0.099 | 0.068 | **0.062** |
| toymc capture/fission | 0.121 | 0.067 | **0.062** |
| borehole_d30 | 0.091 | 0.047 | **0.045** |
| morris | 0.451 | 0.197 | **0.190** |

- Rank 1 (mean rank 1.33). It matches or beats #5 on 8 of 9 outputs; OTL is within noise.
- Like #5, it still always chooses the lowest fidelity. Whether that holds with a realistic MCNP per-run overhead is an open question (see [Learnings](../Learnings.md#open-questions)).

## When to use
It's the current recommendation for the MCNP use case, pending the overhead-sensitivity study.
