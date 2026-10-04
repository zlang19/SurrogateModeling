# #2 `sobol_pce`: sparse polynomial chaos (hybrid LARS)

| | |
|---|---|
| Design | Fixed Sobol (same as #1) |
| Fidelity | High only |
| Model | Sparse Legendre polynomial expansion per output |
| Uncertainty | No (`var=None`; coverage/NLL reported as "—") |
| Code | [pce.py](../../src/surrogatemodeling/methods/pce.py) |

## How it works
1. **Basis:**
   - inputs are mapped from the training box to [-1, 1];
   - candidate terms are orthonormal Legendre products with total degree ≤ p and at most **pairwise** interactions;
   - p ranges up to 5 when d ≤ 10, otherwise up to 3, which keeps a 40D basis around 2,500 terms.
2. **Hybrid LARS** (Blatman & Sudret 2011), for each output and each candidate degree:
   - the LAR path orders the terms;
   - every prefix of that order is refit by ordinary least squares;
   - each prefix is scored by **corrected leave-one-out error**, LOO × n/(n−P).
3. The (degree, prefix) with the lowest corrected LOO wins.

## Why it's here
PCE is the classic UQ surrogate. Its coefficients give Sobol sensitivity indices for free, and it handles low effective dimension when the important terms are few.

## Diagnostics
Per output: `degree`, `n_terms`, `inputs_used`.

## Results
- Last on every output (mean rank 7.0).
- On smooth low-dimensional problems it's close to the GP (unpadded wing weight 0.071 vs 0.065).
- In high dimensions with noise, LARS picks up false terms in inert inputs: wing-weight-40D gets 0.153 vs 0.104.

## History
The first version used cross-validated LASSO to choose terms, and it overfit badly. Switching to true hybrid LARS (full path plus corrected LOO) fixed that. See [Learnings](../Learnings.md#methods).

## When to use
For sensitivity analysis on low-dimensional, low-noise problems, or as a cheap second opinion. Not as the primary surrogate for this use case.
