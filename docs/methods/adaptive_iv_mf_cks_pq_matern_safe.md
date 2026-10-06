# `adaptive_iv_mf_cks_pq_matern_safe`: cost-aware co-kriging on a converged-source menu

The lead candidate for the OpenMC validation ([ValidationRecommendation.md](../ValidationRecommendation.md)).

| | |
|---|---|
| Design | 25% high-fidelity Sobol seed → greedy cost-aware batches |
| Fidelity | Chooses from the menu, **restricted to configs with high-fidelity inactive and active cycles**: only particles per cycle vary |
| Model | Co-kriging (#7): GP for f plus a per-config bias GP; Matérn-5/2 kernels; learned per-config σ scales (#7s) |
| Uncertainty | Yes. Latent variance, scaled by prequential calibration (≥ 1) |
| Code | `CoKrigingAdaptive(noise_scale=True, prequential=True, kernel="matern", safe_menu=True)` in [cokriging.py](../../src/surrogatemodeling/methods/cokriging.py) |

## How it works
It is [`adaptive_iv_mf_cks_pq_matern`](adaptive_iv_mf_cks_pq_matern.md) with one change, in `setup`:

```python
keep = [i for i, f in enumerate(spec.fidelities)
        if all(f.knobs.get(k, hf[k]) == hf[k] for k in ("inactive", "active") if k in hf)]
```

Configs that cut inactive or active cycles are removed from the menu. Problems without cycle knobs, such as Borehole-30D (histories only), are unaffected.

| Problem | Menu kept |
|---|---|
| `toymc_axial` | particles 0.1×, 0.25×, 0.5×, 1× of HF (2000 × (70 + 140)) |
| `openmc` | particles 1,000 / 2,500 / 5,000 / 10,000, all at 100 inactive + 200 active (cost 0.13 / 0.27 / 0.52 / 1 of HF) |

Everything else is shared with `pq_matern`:
- **Bias model:** co-kriging with a per-config bias GP ([adaptive_iv_mf_ck.md](adaptive_iv_mf_ck.md)).
- **Noise:** learned per-config σ scales ([adaptive_iv_mf_cks.md](adaptive_iv_mf_cks.md)).
- **Kernel:** Matérn-5/2.
- **Error bars:** prequential calibration, clamped so it only widens them ([cks_variants.md](cks_variants.md)).

## Why it's here
Cutting inactive cycles leaves the fission source unconverged. That biases the slowest axial mode, which drives axial offset and peaking. The bias depends on the inputs in a way the bias GP can't learn from a few points, and the σ reported by short runs doesn't include it. With the full menu, cost-aware methods bought cheap runs whose bias cost more than their noise reduction saved, so they lost to the plain baseline on axial offset. Cutting only particles keeps every run's source converged: cheap runs are noisier but nearly unbiased. Real MCNP users do the same thing, never shortening the source convergence.

## Results (pre-validation screen, 2026-10-06, 3 seeds, `toymc_axial`, budget 100)

| Output | Final NRMSE | vs `pq_matern` | vs `sobol_gp` | Final coverage |
|---|---|---|---|---|
| k-eff | **0.048** | 0.059 | 0.142 | 0.98 |
| axial offset | **0.144** | 0.287 | 0.179 | 0.97 |
| axial peaking | **0.283** | 0.370 | 0.407 | 0.97 |
| capture/fission | **0.052** | 0.066 | 0.177 | 0.97 |

- **Wins everywhere:** it won 12 of 12 seed × output pairs against `pq_matern`.
- **Best whole-budget score:** best AUC-NCRPS on all four outputs, and the only non-baseline method on the AUC-NCRPS-vs-run-time Pareto frontier.
- **First to pass the gate:** it is the first cost-aware method to pass the coverage gate (≥ 0.90) on every output.
- **Faster:** 25 min per run vs 125 for `pq_matern`. It reaches the budget with ~620 evaluations instead of ~1,600, so the GP fits are smaller.

## Diagnostics
The same as #7s and `pq`: `noise_scales`, `prequential_scale`, `lengthscales`, `fidelity_counts`, `fit_time`.

## Caveats
- **Peaking may differ in OpenMC.** In the OpenMC fidelity study, 1,000 particles per cycle biased peaking upward. That setting is in the safe menu.
- **The menu is a modeling choice**, made using knowledge from the fidelity study. The method doesn't learn which knobs are safe.

## When to use
Monte Carlo eigenvalue problems with spatial outputs, whenever cheap runs can be made by cutting histories per cycle instead of cycles.
