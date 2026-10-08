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

## OpenMC validation (round 1, 3 seeds)
It passed: final NRMSE 0.138 / 0.078 / 0.166 / 0.120 (k-eff / axial offset / peaking / capture-fission), vs the baseline's 0.264 / 0.153 / 0.306 / 0.241. It won all 12 seed × output pairs and passed the gate (0.91–0.97). The full menu was 10–20% better on k-eff and capture/fission. See [OpenMCValidation.md](../OpenMCValidation.md).

## Variants (OpenMC round 2)
- `safe_menu="inactive"` (`_conv`): pin only inactive cycles, so active cycles can be cut. It keeps sources converged and adds cheaper configs (1,000 particles × (100 + 25), cost 0.071 on OpenMC).
- `pooled_scale=True` (`_pooled`): one σ scale per output instead of per (output, config), because per-config scales were unstable on OpenMC.
- Registry: `adaptive_iv_mf_cks_pq_matern_conv`, `_conv_pooled`, `_safe_pooled`.
- **Result:** `safe_pooled` is the new leader on OpenMC and the toy (k-eff 0.111, axial offset 0.079, peaking 0.148, capture/fission 0.110, vs 0.138 / 0.078 / 0.166 / 0.120 for `safe`; 75 vs 109 min). `conv` ruins peaking, because peaking is a maximum over noisy bins and shorter runs bias it upward. See [OpenMCValidation.md](../OpenMCValidation.md#round-2-openmc_validation_r2-2026-10-06-fixing-the-two-gaps).

## Noise-scale floor (screen, 2026-10-06, 3 seeds, `toymc_axial` + Borehole-30D)
On OpenMC, `safe_pooled`'s k-eff and capture/fission σ scales sat at the 0.2 floor on every seed. Three arms test whether that floor was hiding something: `_floor05`, `_floor001` (effectively free), and `_prior` (floor 0.01 + log-normal prior, median 1).

| `toymc_axial` final NRMSE, seeds 0/1/2 | `safe_pooled` | `_floor05` | `_floor001` | `_prior` |
|---|---|---|---|---|
| k-eff | 0.045 / 0.050 / 0.047 | 0.049 / 0.049 / 0.046 | 0.046 / 0.047 / 0.051 | 0.045 / 0.047 / 0.049 |
| axial offset | **0.125 / 0.132 / 0.119** | 0.150 / 0.137 / 0.131 | 0.142 / 0.140 / 0.148 | 0.143 / 0.143 / 0.132 |
| axial peaking | 0.243 / 0.256 / 0.263 | 0.245 / 0.295 / 0.260 | 0.306 / 0.243 / 0.256 | 0.247 / 0.241 / 0.275 |
| capture/fission | 0.049 / 0.054 / 0.054 | 0.051 / 0.050 / 0.051 | 0.052 / 0.051 / 0.053 | 0.049 / 0.049 / 0.054 |
| Borehole-30D (worst-seed coverage) | 0.041 / 0.052 / 0.042 (0.94) | 0.048 / 0.050 / 0.050 (0.89) | 0.045 / 0.045 / 0.050 (0.89) | 0.051 / 0.045 / 0.067 (**0.82**) |

**Result: keep the 0.2 floor.**
- **Accuracy:** no arm improves any output. All are slightly worse on axial offset (3/3 seeds each). The changed scales on the other outputs change which points are picked, because point selection is joint across outputs.
- **The scale isn't identified from below:** with a floor of 0.01, the fit runs straight to it (seed 0: toy peaking and capture/fission at 0.01, Borehole flow at 0.01). The GP is absorbing noise as signal; the reported σ isn't really too large. The 0.2 floor is acting as a useful regularizer.
- **Calibration:** the prior keeps the scales near 1 (0.37–1.0) but costs coverage on Borehole (0.82).

## Diagnostics
The same as #7s and `pq`: `noise_scales`, `prequential_scale`, `lengthscales`, `fidelity_counts`, `fit_time`.

## Caveats
- **Peaking in OpenMC (resolved).** The fidelity study found that 1,000 particles per cycle biases peaking upward, and the safe menu keeps that setting. In the validation, peaking was still `safe`'s best output relative to the baseline (0.166 vs 0.306).
- **The menu is a modeling choice**, made using knowledge from the fidelity study. The method doesn't learn which knobs are safe.

## When to use
Monte Carlo eigenvalue problems with spatial outputs, whenever cheap runs can be made by cutting histories per cycle instead of cycles.
