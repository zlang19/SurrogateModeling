# #5 `adaptive_iv_mf`: cost-aware integrated variance over (x, fidelity)

| | |
|---|---|
| Design | Hybrid: 25% high-fidelity Sobol seed, then adaptive greedy batches |
| Fidelity | Chooses per point from the problem's ladder (default [1/16, 1/4, 1]) |
| Model | Independent GPs with fixed per-point noise; low-fidelity points get σ² ∝ 1/fidelity |
| Uncertainty | Yes |
| Code | [adaptive.py](../../src/surrogatemodeling/methods/adaptive.py) (`cost_aware=True`) |

## How it works
In this test bed, "fidelity" is the Monte Carlo **history count**: same physics, unbiased, just noisier. So multi-fidelity here doesn't need co-kriging. It's a noise-aware model plus a **cost-aware acquisition**:
- every (candidate, fidelity) option is scored by IV reduction (see #4), with τ² = τ²_HF / fidelity;
- that score is divided by `cost(fidelity) = overhead + (1 − overhead)·fidelity`, where the overhead defaults to 1%;
- the greedy batch picks the best option and only considers options that fit the remaining budget.

## Why it's here
It answers the practical MCNP question: for a fixed compute budget, many cheap noisy runs or a few precise ones?

## Diagnostics
Same as #4. `batch.fidelities` shows what it chose.

## Results
- Rank 2. Beats the baseline by 1.5–2.3× on most outputs: Morris 0.197 vs 0.451, Borehole-30D 0.047 vs 0.091, toy MC k-eff 0.068 vs 0.099.
- **It always picked the lowest fidelity (1/16)**, using about 1,000 runs for 100 high-fidelity equivalents.
- **It fails where σ is under-reported.** On the toy MC power ratio, whose batch-statistics σ is about 2× too small, it plateaus at 0.27, the worst of any method. The GP trusts about 1,000 noisy points whose error bars are too narrow and interpolates their noise. #5b fixes this.

## When to use
Use #5b instead. This version is kept as the control that shows why learned noise matters.
