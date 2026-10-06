# `sobol_gp_matern`: the baseline with a Matérn kernel

| | |
|---|---|
| Design | Fixed: the whole budget as one scrambled Sobol design (identical to `sobol_gp` for the same seed) |
| Fidelity | High only |
| Model | Independent GP per output, ARD **Matérn-5/2** kernel (botorch's dimension-scaled prior), fixed per-point noise from σ² |
| Uncertainty | Yes (latent posterior variance) |
| Code | `SobolGP(kernel="matern")` → `_MaternGPs` in [gp_fixed.py](../../src/surrogatemodeling/methods/gp_fixed.py) |

## How it works
It's [`sobol_gp`](sobol_gp.md) with only the kernel changed. Because the design and the noise seeds come from the run seed alone, it evaluates exactly the same points as `sobol_gp`. On OpenMC it reuses `sobol_gp`'s cached simulations and costs only model time.

## Why it's here
Matérn was the calibration winner for the cost-aware methods. This arm checks whether the plain baseline benefits the same way, so the validation compares the candidate against the fairest high-fidelity reference.

## Results (pre-validation screen, 2026-10-06, 3 seeds, budget 100)
- **Accuracy:** the same as `sobol_gp` (`toymc_axial` final NRMSE 0.145 / 0.188 / 0.387 / 0.173 vs 0.142 / 0.179 / 0.407 / 0.177).
- **Calibration:** much better. Coverage is 0.90–0.96 vs 0.81–0.90, and it passes the gate on 80% of outputs vs 0%.
- **Run time:** 6 min per run.

## When to use
Whenever you'd use `sobol_gp`. It costs nothing extra and its error bars are more honest.
