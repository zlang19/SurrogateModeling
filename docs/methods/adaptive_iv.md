# #4 `adaptive_iv`: GP + integrated-variance reduction

| | |
|---|---|
| Design | Hybrid: 25% Sobol seed, then adaptive greedy batches |
| Fidelity | High only |
| Model | Independent GPs with fixed per-point noise ([gp_common.py](../../src/surrogatemodeling/methods/gp_common.py)) |
| Uncertainty | Yes |
| Code | [adaptive.py](../../src/surrogatemodeling/methods/adaptive.py), [acquisition.py](../../src/surrogatemodeling/methods/acquisition.py) |

## How it works
This method uses the hybrid loop described in [the overview](README.md#hybrid-design-and-greedy-batches). Each candidate *c* is scored, for each output, by how much observing it would shrink the **mean posterior variance over the reference points** (samples of the input distribution):

```
IV(c) = mean_r k(r, c)^2 / (k(c, c) + τ²)
```

- k is the latent posterior covariance in standardized units, and τ² is the expected observation noise.
- Scores are summed over outputs.
- The batch is chosen greedily, with an exact rank-one covariance update after each pick.

## Why it's here
It's the acquisition that most directly targets the benchmark's own metric: pdf-weighted squared error is approximately the integrated posterior variance under the input distribution.

## Diagnostics
GP diagnostics, plus `batch`: the phase (seed / adaptive), `fidelities`, `best_score`, and `hf_noise_sd`.

## Results
- A small but consistent gain over the fixed-design baseline (rank 3 vs 6). For example, OTL 0.033 vs 0.041 and wing weight 0.054 vs 0.065.
- No gain in high dimension (Borehole-30D 0.093 vs 0.091). There the GP's uncertainty map is too poor to steer sampling.

## When to use
When runs must be high fidelity and the dimension is moderate. If fidelity can be chosen, the cost-aware variants (#5 and #5b) are much better.
