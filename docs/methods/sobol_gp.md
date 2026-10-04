# #1 `sobol_gp`: GP on a fixed Sobol design (baseline)

| | |
|---|---|
| Design | Fixed: the whole budget as one scrambled Sobol design, queued at setup |
| Fidelity | High only |
| Model | Independent GP per output, ARD RBF kernel, fixed per-point noise from σ² |
| Uncertainty | Yes (latent posterior variance) |
| Code | [gp_fixed.py](../../src/surrogatemodeling/methods/gp_fixed.py), [gp_common.py](../../src/surrogatemodeling/methods/gp_common.py) |

## How it works
- **Design:** at `setup`, `floor(budget / cost(1))` Sobol points over the training box are generated and handed out 5 at a time.
- **Fitting:** after each batch the GPs are refit, with full hyperparameter optimization every time (`refit_growth=1.0`).
- **Inputs and outputs:** inputs are mapped to the unit box; outputs are standardized per output.

## Why it's here
It's the reference every other method is measured against: the most common default for expensive simulators. Because the design is fixed, any method that beats it is choosing better points or using a better model, and the fixed-design siblings (#2, #3) isolate the model effect.

## Diagnostics
`reoptimized`, `fit_time`, and `lengthscales` (per output, per input, in unit-box units; short means relevant).

## Results
- It's the middle of the pack, ranked 6th of 7 by AUC.
- It's strong on low-dimensional smooth problems (Borehole 0.029).
- It suffers from padded inputs: Borehole-30D gets 0.091, about 3× worse, although only 0.5% of the variance comes from the extra inputs.
- Coverage is mostly 0.84–0.96. The worst is the toy MC power ratio at 0.84, because its reported σ is too small and the GP trusts it.

## When to use
As a baseline, or when the budget is small, the dimension is low (< 10 inputs), and simulator noise is reported accurately.
