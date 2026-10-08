# Test backlog

A ranked list of things still to test. The first item is what to run next.

**Maintenance:**
- **Ranking:** re-rank when results come in.
- **Completed items:** move them to [Done](#done) with a one-line result and a link to where the result is written up.
- **Additions:** add new ideas with a *why*, a rough *cost*, and *what would change our mind*.

The current leader, **confirmed at 10 seeds**, is **`adaptive_iv_mf_cks_pq_matern_safe_pooled`** ([OpenMCValidation.md](OpenMCValidation.md)): co-kriging (Matérn, prequential calibration, one σ scale per output), buying only particle-reduced runs. Scoring follows [EvaluationCriteria.md](EvaluationCriteria.md).

Cost is on this 12-core machine. OpenMC runs are ~10 CPU-h of simulation each, at 2 concurrent runs (~2 h wall-clock time per run). `toymc_axial` runs take 20–40 min each.

## Ranked

| # | Test | Why | Cost | Decides |
|---|---|---|---|---|
| 1 | **Budget sensitivity**: budgets 30 and 50 (high-fidelity equivalents) | Real campaigns may stop early, and cost-aware methods have been weak early in the budget | Toy: ~2 h; OpenMC: proportional to budget | Whether to recommend a minimum budget, or a baseline below it |
| 2 | **Seed-design size**: high-fidelity seed fraction 10% and 15% (now 25%) | 50% hurt (k-eff 0.101 vs 0.059 on the toy); cheap runs carry most of the information | Toy screen ~2 h | Default `seed_fraction` |
| 3 | **Prequential calibration settings on the safe menu**: window (300), target (95%), range [1, 10] | Tuned on the old full menu; the safe menu's residuals differ | Toy screen ~2 h | Calibration defaults |
| 4 | **Batch size 10–20** | Real MCNP campaigns run many jobs in parallel; batch 5 may overstate the benefit of adapting | Toy ~2 h; OpenMC ~10 h | Whether the method holds up at cluster parallelism |
| 5 | **Output-aware cost weighting**: one model, but the acquisition discounts configs whose learned bias for an output is large | `conv` showed global outputs gain 3–14% from short runs while peaking loses ~50%. A middle ground that needs no second model | ~half a day of code + toy screen | Whether one model can get both gains |
| 6 | **`safe_pooled` + cold-start input warping** | Warping gave the best final accuracy on the old menu but took 6 h. With ~620 points instead of ~1,100, it may take 1.5–2 h | Toy screen ~6 h | Worth it only if final accuracy matters more than AUC |
| 7 | **SAAS surrogate on the safe menu** (#8 revisited) | SAAS was the best-calibrated high-fidelity-only method. #8 ran out of memory at ~2,000 points; the safe menu needs ~620 | ~half a day to re-check memory + toy screen | High risk, possibly a calibration win |
| 8 | **Fix deep-ensemble calibration on multi-fidelity data**: calibrate only on full-fidelity residuals (or residuals with the noise term removed), keep the scale ≥ 1, and stabilize the learned noise multiplier (e.g. a prior around the GP's pooled value) | The cost-aware deep ensemble's coverage collapsed to 0.07–0.50, and its noise multipliers ranged 0.01–77. Even fixed, its accuracy trails the GP by 10–110%, so this matters mainly to unblock #9 and #12. Most recent residuals come from noisy cheap runs that the noise term already covers, so the calibration scale shrinks to its 0.2 minimum, which is wrong at full fidelity. Any neural variant needs this fixed | ~1 h of code; re-scoring reuses cached designs (toy ~1 h; OpenMC from cache) | Whether neural surrogates can pass the coverage gate at all |
| 9 | **Upstream-downstream (stacked multi-fidelity) networks**: an upstream network learns the cheap-run response, and a downstream network maps (x, upstream prediction) → full fidelity. Ensembled, for uncertainty | The neural version of co-kriging. Expected to gain little on the safe menu (cheap runs are nearly unbiased, so the correction is ~identity) and to overfit its 25–100 full-fidelity points; most useful on the biased full menu. Nearly free to test: reuse the existing `safe_pooled` and full-menu designs (cached, OpenMC included) | ~half a day of code; runs ~2–3 h, mostly cached | Whether a learned fidelity correction beats the GP bias model. Needs #8 first |
| 10 | **Per-output menus**: separate models; global outputs may use short runs, axial outputs use the safe menu | Gets the 3–14% global-output gain without the peaking bias | ~1 day of code; doubles model cost | Probably not worth the complexity unless #5 fails |
| 11 | **A second OpenMC problem**: different geometry, or more inputs (e.g. 25) | Check the method isn't tuned to this one column | ~1 day to build + truth set (~27 CPU-h) + validation (~10 h) | Generality of the recommendation |
| 12 | **Field outputs with a neural surrogate**: predict the full axial power profile (e.g. 50 bins) instead of 4 scalars | Where networks should beat per-output GPs: one model shares structure across many correlated outputs | ~1–2 days (new outputs on the toy and OpenMC + truth sets) | Whether to move beyond scalar outputs |
| 13 | **MF-SNPE** (multi-fidelity sequential neural posterior estimation) | Built for *inverse* problems (which inputs are consistent with an observation), not forward surrogates. Its reusable idea, pretrain on cheap runs then fine-tune on full fidelity, is what the deep ensemble already does (see #8). Relevant only if the project takes on an inverse problem, e.g. inferring parameters from measured k-eff or axial offset, where our GP could serve as its cheap simulator | Days | Only if an inverse-UQ use case is added |

## In progress
| Test | Status |
|---|---|
| — | Nothing running |

## Known issues to keep an eye on
- **Warp-arm tests:** `test_fits_smooth_2d_function` fails for the warp arms (`_warp`, `_pq_matern_warp`, `_pq_matern_warp_pooled`, `_pq_matern_dwarp`), with the round-2 changes. Without them, `_pq_matern_warp_pooled` and `_pq_matern_dwarp` fail too, so the failures predate round 2. It needs a looser tolerance or a fix if warping comes back (#6).
- **OpenMC run time:** wall-clock time isn't comparable across methods that share cached simulations. Use the simulation budget, not run time, for OpenMC Pareto views.

## Done

| Test | Result | Write-up |
|---|---|---|
| Converged-source menu (`safe`), Matérn baseline, 50% seed | `safe` won 12/12 on the toy; the 50% seed was worse | [ValidationRecommendation.md](ValidationRecommendation.md) |
| OpenMC validation round 1 | `safe` beat the baseline 12/12 and passed the gate | [OpenMCValidation.md](OpenMCValidation.md) |
| OpenMC round 2: pooled scales and an inactive-only menu | `safe_pooled` is the new leader; cutting active cycles biases peaking | [OpenMCValidation.md](OpenMCValidation.md#round-2-openmc_validation_r2-2026-10-06-fixing-the-two-gaps) |
| SAAS + cost-aware (#8) | Cancelled after out-of-memory kills (see #7) | [adaptive_iv_mf_saas.md](methods/adaptive_iv_mf_saas.md) |
| Delayed top-8 warping | Failed on `toymc_axial` (k-eff 0.121 vs 0.059) | [cks_variants.md](methods/cks_variants.md) |
| **10-seed confirmation** (OpenMC + toy) | `safe_pooled` beat `sobol_gp` and `sobol_gp_matern` on every seed, output and criterion on OpenMC (10/10, p = 0.002) and passed the gate on all seeds; ~half the baseline's error | [OpenMCValidation.md](OpenMCValidation.md#confirmation-openmc_confirmation--toy_confirmation-2026-10-07-10-seeds-confirmed) |
| Noise-scale floor (0.05, 0.01, prior at median 1) on `safe_pooled` | No gain; all slightly worse on axial offset, and the prior failed the gate on Borehole (0.82). With the floor at 0.01 the scale runs to it, so the GP absorbs noise. Keep 0.2 | [adaptive_iv_mf_cks_pq_matern_safe.md](methods/adaptive_iv_mf_cks_pq_matern_safe.md#noise-scale-floor-screen-2026-10-06-3-seeds-toymc_axial--borehole-30d) |
| Deep ensembles (`sobol_de`, `adaptive_iv_mf_de_safe`) | GP wins at equal data everywhere (OpenMC: network 20–120% worse AUC-NCRPS); the cost-aware network's coverage collapses (0.07–0.50), see #8 | [deep_ensembles.md](methods/deep_ensembles.md) |
| Calibration screen and combinations (Tier 1, Tier 1.5) | Matérn + prequential calibration kept; ensembles and weighting dropped | [cks_variants.md](methods/cks_variants.md) |
