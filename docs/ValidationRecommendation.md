# Methods report and OpenMC validation recommendation (2026-10-06)

This report covers every method tested so far, ranks each by whether to carry it into the OpenMC validation (plan stage 5), and gives the evidence. The scoring rules are in [EvaluationCriteria.md](EvaluationCriteria.md): AUC-NRMSE and AUC-NCRPS rank the methods, and final coverage must be ≥ 0.90 on every output (the gate).

## Recommendation

Validate **`adaptive_iv_mf_cks_pq_matern_safe`** against the high-fidelity baselines on the OpenMC column:

| Queue | Arm | Role | Extra OpenMC compute |
|---|---|---|---|
| 1 | [`adaptive_iv_mf_cks_pq_matern_safe`](methods/adaptive_iv_mf_cks_pq_matern_safe.md) | Candidate | ~10 CPU-h per seed |
| 2 | [`sobol_gp`](methods/sobol_gp.md) | Reference baseline (the success bar is defined against it) | ~10 CPU-h per seed |
| 3 | [`sobol_gp_matern`](methods/sobol_gp_matern.md) | Baseline with honest error bars | none (reuses `sobol_gp`'s cached runs) |
| 4 | [`sobol_saas`](methods/sobol_saas.md) | Best-calibrated high-fidelity method | none for simulations; ~20 min of model time per run |
| 5 | [`adaptive_iv_mf_cks_pq_matern`](methods/adaptive_iv_mf_cks_pq_matern.md) (full menu) | *Optional* ablation: does the reason `safe` wins carry over to OpenMC? | ~10 CPU-h per seed |

With 12 cores, each candidate-plus-baseline seed pair is ~20 CPU-h, about 2 h of wall-clock time. **3 seeds take ~6 h; 5 seeds take ~10 h** (about the one-day cap). The ablation adds ~1 h per seed. The 100-point reference test set is already built and cached.

**Status: rounds 1 and 2 complete. Round 1 passed; round 2's `safe_pooled` is the new leader** — see [OpenMCValidation.md](OpenMCValidation.md). It ran as experiment `openmc_validation` (3 seeds, all five arms, queued in the order above; config `configs/experiments/openmc_validation.toml`). Two runs at a time, each evaluating 5 OpenMC points in parallel (~10 of 12 cores). Expected total: ~10–12 h.

## Headline finding: keep cycles at high fidelity, cut only particles

`safe` is #7s + prequential calibration + Matérn (`pq_matern`) with one change: its fidelity menu keeps only configs whose **inactive and active cycle counts equal high fidelity**. Cheap runs come only from fewer particles per cycle (on `toymc_axial`: 0.1×, 0.25×, 0.5× and HF particles).

- Cutting inactive cycles leaves the fission source unconverged. That biases the slowest axial mode, which is exactly what axial offset and peaking measure. The co-kriging bias model can't fully learn that bias. So the cost-aware methods were trading a little noise for a lot of bias on the spatial outputs, which is why they had lost to the plain baseline on axial offset all along.
- With the safe menu, the cost-aware method **beats the baseline on axial offset for the first time over the whole budget**, not only at the end, and it **passes the coverage gate on every output**.
- It is also **~5× faster than `pq_matern`** (25 vs 125 min per run): it reaches the same budget with ~620 evaluations instead of ~1,500–1,700, so every GP fit is smaller.

On `toymc_axial`, `safe` is the only method on the AUC-NCRPS-vs-run-time Pareto frontier beyond the baselines, for all four outputs (`results/all_ncrps/pareto_toymc_axial.png`).

### Paired by seed (3 seeds, `toymc_axial`, final values at budget 100)

| Output | `safe` | `pq_matern` | `pq_matern_warp` | `sobol_gp` | `sobol_saas` |
|---|---|---|---|---|---|
| k-eff NRMSE | **0.048** | 0.059 | 0.052 | 0.142 | 0.124 |
| axial offset NRMSE | **0.144** | 0.287 | 0.177 | 0.179 | 0.177 |
| axial peaking NRMSE | **0.283** | 0.370 | 0.295 | 0.407 | 0.382 |
| capture/fission NRMSE | **0.052** | 0.066 | 0.060 | 0.177 | 0.158 |
| coverage (worst output) | **0.97** | 0.72 | 0.96 | 0.81 | 0.89 |
| run time | 25 min | 125 min | 367 min | 7 min | 20 min |

`safe` beats `pq_matern`'s final NRMSE on **every seed for every output** (12 of 12 pairs), and is the lowest of all methods on 10 of 12. The exceptions are axial offset seed 0 (baselines 0.171 vs 0.179) and peaking seed 1 (warp 0.300 vs 0.302). Seed-level coverage is 0.93–0.99.

On AUC (the whole budget), `safe` has the best AUC-NCRPS on all four `toymc_axial` outputs and the best AUC-NRMSE on axial offset (0.426 vs the next best 0.507) and peaking (0.622 vs 0.669). On k-eff and capture/fission its AUC-NRMSE is within 1–3% of the best (`log`, 0.296 / 0.307 vs 0.299 / 0.315), which is inside the run-to-run noise.

**Against the success bar:** on the MCNP-like problem, `safe` is the first method to pass the coverage gate *and* lead (or tie within noise) on both AUCs on every output.

## The other two new arms

- **`sobol_gp_matern` (the baseline with a Matérn kernel):** same accuracy as `sobol_gp`, but coverage 0.90–0.96 instead of 0.81–0.90 (80% of outputs pass the gate vs 0%). It is the fair high-fidelity reference; keep it as a free extra baseline.
- **`seed50` (a 50% high-fidelity seed design):** worse than `safe` and `pq_matern` everywhere (k-eff 0.101 vs 0.059), and it fails the gate on k-eff (0.78). Spending half the budget up front on precise runs leaves too little for the cheap runs that make the cost-aware methods win. Dropped.

## All methods, ranked for OpenMC validation

Tiers: **Run** = in the validation; **Free** = shares cached simulations, so adds only model time; **Optional** = worth it if compute allows; **No** = not carried forward. Run times are median wall-clock time per `toymc_axial` run under similar load.

| # | Method | Tier | Why |
|---|---|---|---|
| 1 | [`adaptive_iv_mf_cks_pq_matern_safe`](methods/adaptive_iv_mf_cks_pq_matern_safe.md) | **Run** | Best or tied-best on every MCNP-like output, passes the gate, 25 min |
| 2 | [`sobol_gp`](methods/sobol_gp.md) | **Run** | Defined reference baseline; 7 min; fails the gate (0.81–0.90) |
| 3 | [`sobol_gp_matern`](methods/sobol_gp_matern.md) | **Free** | Same accuracy as the baseline, honest error bars; 6 min |
| 4 | [`sobol_saas`](methods/sobol_saas.md) | **Free** | Best high-fidelity-only method: beats `sobol_gp` on every output, coverage 0.89–0.95; 20 min |
| 5 | [`adaptive_iv_mf_cks_pq_matern`](methods/adaptive_iv_mf_cks_pq_matern.md) | **Optional** | Ablation for the safe-menu mechanism; fails the gate on axial offset (0.72); 125 min |
| 6 | `adaptive_iv_mf_cks_pq_matern_warp` | No | Passes the gate with good final accuracy, but `safe` beats it on both AUCs and every final value at 1/15 the run time |
| 7 | `adaptive_iv_mf_cks_pq_matern_warp_pooled` | No | Like #6, slightly worse |
| 8 | `adaptive_iv_mf_cks_matern` | No | Superseded by `pq_matern` (same model plus clamped calibration) |
| 9 | `adaptive_iv_mf_cks_log` | No | Best AUC-NRMSE on k-eff/capture by a hair, but coverage 0.56–0.93 |
| 10 | `adaptive_iv_mf_cks_pq` | No | Calibration alone doesn't fix axial coverage (0.58) |
| 11 | `adaptive_iv_mf_cks_pooled` | No | Coverage 0.63–0.81 |
| 12 | `adaptive_iv_mf_cks` (#7s) | No | Base of the family; superseded |
| 13 | `adaptive_iv_mf_cks_pq_matern_seed50` | No | See above |
| 14 | `adaptive_iv_mf_cks_matern_warp` | No | Fails the gate on Borehole (0.84); slow |
| 15 | `adaptive_iv_mf_cks_pq_warp` | No | Poor early, slow, fails the gate |
| 16 | `adaptive_iv_mf_cks_warp` | No | As above; coverage 0.64–0.78 |
| 17 | `adaptive_iv_mf_cks_ens` | No | No gain over #7s; slowest non-warp arm (344 min) |
| 18 | `adaptive_iv_mf_cks_wt` | No | Slightly worse than #7s |
| 19 | `adaptive_iv_mf_cks_pq_matern_dwarp` | No | Failed on `toymc_axial` (k-eff 0.121 vs 0.059; coverage 0.54) |
| 20 | `adaptive_iv_mf_ck` | No | Superseded by #7s (no σ scales; passes the gate on 12% of outputs) |
| 21 | `adaptive_iv_mf_ck_cal` | No | Superseded by #7s |
| 22 | `adaptive_iv_mf_xn` | No | Best of the first-generation multi-fidelity methods; superseded by co-kriging |
| 23 | `adaptive_iv_mf` | No | Assumes cheap runs are unbiased; loses badly on axial offset |
| 24 | `adaptive_iv` | No | Single-fidelity adaptive; ≈ `sobol_gp` (small gain on analytic problems only) |
| 25 | `adaptive_epig` | No | ≈ `adaptive_iv`, slower |
| 26 | `screen_gp` | No | Mixed results; no gain on the Monte Carlo problems |
| 27 | `sobol_pce` | No | Worst on every problem in the `full` experiment |
| 28 | `adaptive_iv_mf_saas` (#8) | No | Cancelled after two out-of-memory kills |
| 29 | `adaptive_iv_mf_saas_noknob` | No | Cancelled with #8 |

Methods 20–27 were scored in the earlier `full` and `costaware` experiments (see [Learnings](Learnings.md)); methods 1–19 were scored in the same `calib_screen` experiment, paired by seed. Full tables: `results/calib_screen/ranking.md`, `results/costaware/ranking.md`, `results/full/ranking.md`.

## Caveats

- **3 seeds only.** `safe` won 12 of 12 paired comparisons with large margins, so the result is unlikely to be luck, but the planned 10-seed confirmation (stage 3) hasn't been run. OpenMC with 3–5 seeds partly serves as that confirmation.
- **Run-to-run noise floor.** On Borehole-30D, `safe` and `pq_matern` have identical menus (only histories vary) yet differ by up to 0.008 final NRMSE. Differences smaller than that are noise.
- **Particle-count bias in OpenMC.** The fidelity study found that 1,000 particles per cycle biases **peaking upward** in OpenMC (the maximum over noisy bins). The safe menu keeps 1,000 particles (0.1× of HF), so peaking is the output most likely to behave differently from the toy. The optional full-menu ablation and the per-output results will show it.
- **The menu was chosen by hand, using knowledge from the fidelity study.** That is legitimate for a real MCNP user (source convergence is a known requirement), but it is a modeling choice, not something the method learned.
