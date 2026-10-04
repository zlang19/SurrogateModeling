# #4b `adaptive_epig`: GP + EPIG

| | |
|---|---|
| Design | Hybrid: 25% Sobol seed, then adaptive greedy batches |
| Fidelity | High only |
| Model | Independent GPs with fixed per-point noise |
| Uncertainty | Yes |
| Code | [adaptive.py](../../src/surrogatemodeling/methods/adaptive.py), [acquisition.py](../../src/surrogatemodeling/methods/acquisition.py) (`epig_score`) |

## How it works
This is the same loop as #4 with a different score: **expected predictive information gain** (Bickford Smith et al., 2023). It's the mutual information between a noisy observation at candidate *c* and a noisy high-fidelity prediction y\* at each reference point, averaged over the reference set. For Gaussians:

```
ρ²(c, r) = k(r, c)² / ((k(r, r) + τ*²) (k(c, c) + τ²))
EPIG(c)  = mean_r −½ log(1 − ρ²)
```

The target noise τ\*² is the high-fidelity noise, which is the paper's formulation. Batches are greedy with the same rank-one update.

## Design note
Targeting the *latent* f\* (τ\*² = 0) instead makes EPIG depend only on correlations. In the low-noise limit it then ignores how much variance is left to remove, and on a near-noiseless 2D test it levels off around 1% NRMSE, where IV reaches 0.5%. Targeting the noisy y\*, as in the paper, is the default. It doesn't help in that near-noiseless test, but it behaves well on the real problems.

## Diagnostics
Same as #4.

## Results
- Rank 5. Essentially tied with #4 on most outputs (OTL 0.033, wing weight 0.053).
- Slightly worse on Morris (0.486 vs 0.459).
- Slightly better on the toy MC power ratio (0.138 vs 0.147).

## When to use
It's interchangeable with #4 here. EPIG would matter more if the target distribution differed strongly from where sampling is allowed, e.g. a narrow operating region inside a wide training box.
