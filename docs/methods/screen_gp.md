# #6 `screen_gp`: ARD screening → GP on the active inputs

| | |
|---|---|
| Design | Hybrid: 25% Sobol seed, then adaptive IV batches (as #4) |
| Fidelity | High only |
| Model | Learned-extra-noise GP on a screened subset of inputs |
| Uncertainty | Yes. The variance includes the learned extra noise, which stands in for the dropped inputs |
| Code | [adaptive.py](../../src/surrogatemodeling/methods/adaptive.py) (`ScreenedGP`) |

## How it works
1. **Seed phase:** an ordinary full-dimension GP is used until the Sobol seed is complete. A handful of points can't rank 20+ inputs.
2. **Screening:**
   - fit a full-dimension extra-noise GP;
   - rank inputs by ARD relevance 1/ℓ², taking the maximum share across outputs;
   - keep inputs until **99%** of the relevance is covered, with a minimum of 2 and a maximum of 12.
3. **Model:** an extra-noise GP on the active inputs only. The learned extra noise absorbs what the dropped inputs contribute.
4. **Re-screening:** happens whenever the data has doubled since the last screen.
5. **Prediction:** the variance includes the extra-noise term. The dropped inputs' effect is part of the true function, so leaving it out made coverage far too low.

## Diagnostics
GP diagnostics (lengthscales for active inputs only), `extra_noise_sd`, `active_inputs`, and `batch`.

## Results
- Rank 4. It's good where a few inputs dominate: Borehole 0.024 (best among high-fidelity-only methods), toy MC power ratio 0.130.
- Worse on toy MC k-eff (0.137 vs 0.099): it drops weak inputs whose combined effect matters at this accuracy.
- Coverage is uneven: Borehole-30D 0.74, wing-weight-40D 0.82.

## History
Two bugs were found and fixed (see [Learnings](../Learnings.md#methods)):
- it screened on 5 points during the seed phase;
- its predictive variance left out the dropped inputs.

## When to use
When you believe strongly that a few inputs dominate and want an interpretable active set. For accuracy alone, #5b is better.
