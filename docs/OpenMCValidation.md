# OpenMC validation

Plan stage 5: the methods chosen in [ValidationRecommendation.md](ValidationRecommendation.md), tested on the real OpenMC column. The column is a 2 m 3×3 pin column with 14 inputs and 4 outputs. The fidelity menu has 12 configs over particles and inactive and active cycles; high fidelity is 10,000 particles × (100 inactive + 200 active cycles), ≈ 6 CPU-min per run. The test set is 100 reference runs at 2× particles and 2× inactive cycles. The budget is 100 high-fidelity-equivalent runs, in batches of 5. Scoring follows [EvaluationCriteria.md](EvaluationCriteria.md).

Regenerate the tables with `uv run python studies/validation_tables.py openmc_validation openmc` and `uv run sm report results/openmc_validation`.

## Round 1 (`openmc_validation`, 2026-10-06, 3 seeds): **passed**

All 15 runs finished, taking 9.7 h with 2 concurrent runs × 5 parallel OpenMC processes.

**Verdict:** [`adaptive_iv_mf_cks_pq_matern_safe`](methods/adaptive_iv_mf_cks_pq_matern_safe.md) meets the success bar on OpenMC.
- **Accuracy:** it beats the `sobol_gp` baseline on **every seed for every output** (12/12), on both AUC-NRMSE and AUC-NCRPS.
- **Calibration:** it passes the coverage gate on every output (0.91–0.97).
- **Reproduces the toy:** the toy-problem result carried over to the real code.

### Seed-median results at the full budget

| Output | Metric | `safe` | `pq_matern` (full menu) | `sobol_gp` | `sobol_gp_matern` | `sobol_saas` |
|---|---|---|---|---|---|---|
| k-eff | final NRMSE | 0.138 | **0.121** | 0.264 | 0.255 | 0.208 |
| | coverage | 0.95 | 0.95 | 0.86 | 0.93 | 0.90 |
| | AUC-NCRPS | 0.211 | **0.196** | 0.295 | 0.291 | 0.271 |
| axial offset | final NRMSE | **0.078** | 0.144 | 0.153 | 0.115 | 0.116 |
| | coverage | 0.94 | **0.82** ✗ | 0.91 | 0.96 | 0.97 |
| | AUC-NCRPS | **0.148** | 0.181 | 0.201 | 0.194 | 0.217 |
| axial peaking | final NRMSE | **0.166** | 0.227 | 0.306 | 0.278 | 0.292 |
| | coverage | 0.97 | 0.86 ✗ | 0.86 ✗ | 0.92 | 0.92 |
| | AUC-NCRPS | **0.260** | 0.282 | 0.335 | 0.332 | 0.345 |
| capture/fission | final NRMSE | 0.120 | **0.099** | 0.241 | 0.229 | 0.204 |
| | coverage | 0.91 | 0.95 | 0.88 ✗ | 0.92 | 0.94 |
| | AUC-NCRPS | 0.196 | **0.182** | 0.270 | 0.264 | 0.260 |
| | **gate (≥ 0.90 on all)** | **pass** | fail | fail | pass | pass |

Ranking report (mean rank over the 4 outputs, AUC-NRMSE / AUC-NCRPS):

| Method | AUC-NRMSE rank | AUC-NCRPS rank | Gate |
|---|---|---|---|
| `safe` | 1.5 | 1.5 | 100% |
| `pq_matern` | 1.5 | 1.5 | 50% |
| `sobol_gp_matern` | 3.5 | 3.5 | 100% |
| `sobol_saas` | 4.0 | 4.0 | 100% |
| `sobol_gp` | 4.5 | 4.5 | 25% |

### Final NRMSE per seed

| Output | `safe` s0 / s1 / s2 | `pq_matern` s0 / s1 / s2 | `sobol_gp` s0 / s1 / s2 |
|---|---|---|---|
| k-eff | 0.156 / 0.138 / 0.095 | 0.093 / 0.134 / 0.121 | 0.292 / 0.254 / 0.264 |
| axial offset | 0.091 / 0.078 / 0.067 | 0.144 / 0.136 / 0.168 | 0.139 / 0.153 / 0.155 |
| axial peaking | 0.187 / 0.166 / 0.153 | 0.220 / 0.315 / 0.227 | 0.298 / 0.306 / 0.306 |
| capture/fission | 0.130 / 0.120 / 0.097 | 0.085 / 0.110 / 0.099 | 0.243 / 0.229 / 0.241 |

### What we learned

1. **The cost-aware advantage is larger on OpenMC than on the toy.** `safe`'s final NRMSE is about half the baseline's on every output: k-eff 0.138 vs 0.264, axial offset 0.078 vs 0.153.
2. **The source-convergence explanation holds on the real code.** The full-menu ablation spent almost its whole budget on `joint_low` (1,000 particles, 10 inactive, 25 active; cost 0.042): ~1,825 evaluations. Its learned σ scale for that config on axial offset is about 22× (seeds 0–2: 22.7, 24.7, 22.0). That is the unconverged-source bias, absorbed as noise. Axial offset is then no better than the baseline (0.144 vs 0.153) and fails the gate (0.82). `safe` bought only `particles=1000` (cost 0.127): 614 evaluations, all with converged sources.
3. **The full menu wins on the smooth outputs, unlike on the toy.** On k-eff and capture/fission, the 3× more points from `joint_low` beat `safe` by 10–20% at the median. That's unlike the toy, where `safe` won everywhere. Per seed it is closer: k-eff splits 2–1, and capture/fission is within 0.01 on seed 2. Global outputs don't care about source shape, so cheap unconverged runs are nearly free information for them.
4. **Per-config σ scales are unstable on OpenMC.** Most fitted scales sit at the 0.2 lower bound, and a few blow up: 31× on `safe` seed 1 axial offset, and 1,383× on the full menu's seed 2 high-fidelity axial offset, which effectively discards its own high-fidelity data. The OpenMC fidelity study had already shown that under-reporting depends on the output, not the config, which is what the pooled (one scale per output) variant assumes.
5. **A Matérn kernel makes the high-fidelity baseline well calibrated on OpenMC too.** Coverage rises from 0.86–0.91 to 0.92–0.96 at the same accuracy, so `sobol_gp_matern` passes the gate. SAAS is the most accurate high-fidelity-only method on k-eff and capture/fission, but not on the axial outputs. Both are far behind the cost-aware methods.
6. **Wall-clock time on OpenMC is dominated by simulation** (~85 min per run for `sobol_gp`, ~110 min for `safe`). Methods that share the Sobol design reused cached simulations, so their wall-clock time isn't comparable. Every method spends the same simulation budget, so the run-time Pareto view (`results/all_ncrps/pareto_openmc.png`) adds little here.

## Round 2 (`openmc_validation_r2`, 2026-10-06): fixing the two gaps

Findings 3 and 4 suggest two targeted changes to `safe`, tested as a 2×2 design with round 1's `safe` as the base. Seeds and the high-fidelity seed design are shared, so arms are paired by seed with round 1.

| Queue | Arm | Change from `safe` | Hypothesis |
|---|---|---|---|
| 1 | `adaptive_iv_mf_cks_pq_matern_conv` | Pin only **inactive** cycles, so active cycles can be cut too. Adds `low_particles_low_active` (1,000 particles × (100 + 25), cost 0.071) and the `active=` configs | ~1.8× more points with converged sources: the smooth outputs gain (finding 3) without the axial-offset bias (finding 2) |
| 2 | `adaptive_iv_mf_cks_pq_matern_conv_pooled` | `conv` + **one σ scale per output** | Both fixes together |
| 3 | `adaptive_iv_mf_cks_pq_matern_safe_pooled` | `safe` + one σ scale per output | Stable noise model (finding 4) on its own |

The same three arms are also screened on `toymc_axial` (`configs/experiments/conv_screen.toml`, writing into `calib_screen`) as a cheap cross-check. It took ~6 h on OpenMC; all 18 runs finished.

### Result: **`safe_pooled` is the new leader**; cutting active cycles hurts peaking

Seed-median final NRMSE (worst-seed coverage) on OpenMC, paired by seed with round 1. Regenerate with `uv run python studies/validation_tables.py openmc_validation,openmc_validation_r2 openmc`.

| Output | `safe` | **`safe_pooled`** | `conv` | `conv_pooled` | full menu | `sobol_gp` |
|---|---|---|---|---|---|---|
| k-eff | 0.138 (0.93) | 0.111 (0.92) | **0.108** (0.91) | 0.125 (0.93) | 0.121 (0.85) | 0.264 (0.86) |
| axial offset | **0.078** (0.94) | 0.079 (0.96) | 0.082 (0.94) | 0.080 (0.95) | 0.144 (0.80) | 0.153 (0.89) |
| axial peaking | 0.166 (0.96) | **0.148** (0.97) | 0.221 (0.93) | 0.187 (0.91) | 0.227 (0.81) | 0.306 (0.72) |
| capture/fission | 0.120 (0.91) | 0.110 (0.90) | **0.095** (0.94) | 0.110 (0.93) | 0.099 (0.91) | 0.241 (0.86) |
| AUC-NCRPS, k / offset / peaking / c-f | .211 / .148 / .260 / .196 | .203 / **.137** / **.248** / .193 | **.196** / .155 / .275 / .191 | .200 / .149 / .265 / .192 | **.196** / .181 / .282 / **.182** | .295 / .201 / .335 / .270 |
| run time / evaluations | 109 min / 614 | 75 min / 614 | 94 min / 1,085 | 66 min / 1,085 | 90 min / 1,825 | 85 min / 100 |

Mean AUC-NCRPS rank among the five cost-aware arms: `safe_pooled` 2.5, `conv` 2.9, `conv_pooled` 3.0, full menu 3.1, `safe` 3.5.

**The toy cross-check agrees** (`toymc_axial`, final NRMSE, seeds 0 / 1 / 2):

| Output | `safe` | `safe_pooled` | `conv` | `conv_pooled` |
|---|---|---|---|---|
| k-eff | 0.048 / 0.048 / 0.047 | 0.045 / 0.050 / 0.047 | 0.060 / 0.063 / 0.059 | 0.059 / 0.059 / 0.063 |
| axial offset | 0.179 / 0.144 / 0.137 | **0.125 / 0.132 / 0.119** | 0.256 / 0.191 / 0.227 | 0.262 / 0.171 / 0.154 |
| axial peaking | 0.257 / 0.302 / 0.283 | **0.243 / 0.256 / 0.263** | 0.556 / 0.605 / 0.592 | 0.535 / 0.548 / 0.595 |
| capture/fission | 0.053 / 0.052 / 0.050 | 0.049 / 0.054 / 0.054 | 0.063 / 0.066 / 0.061 | 0.063 / 0.063 / 0.067 |

### What we learned

1. **Pooled σ scales help the safe menu everywhere, and make it faster.**
   - **OpenMC:** `safe_pooled` beats `safe` on peaking on 3/3 seeds and has the best AUC-NCRPS on both axial outputs. Its k-eff and capture/fission medians improve by about 10–20%, but seeds split 2–1. It passes the gate (worst seed 0.90–0.97).
   - **Toy:** it beats `safe` on axial offset 3/3 (0.119–0.132 vs 0.137–0.179) and on peaking 3/3.
   - **Speed:** 30% faster (75 vs 109 min), because there are fewer noise parameters to fit.
2. **The pooled scales are stable and match the OpenMC fidelity study.** Fitted variance multipliers, seeds 0 / 1 / 2:
   - axial offset 34 / 36 / 32 (σ × ~5.8);
   - peaking 5.7 / 6.0 / 4.5 (σ × ~2.3);
   - k-eff and capture/fission at or near the 0.2 lower bound.

   The study measured under-reporting of ~4× and ~2× for the axial outputs and ~1× for the global ones. Compare round 1's per-config scales, which ranged from 0.2 to 1,383.
3. **Never cut active cycles either: peaking is a maximum, and noise biases it upward.**
   - `conv` buys `low_particles_low_active` (1,000 particles × 25 active cycles), which gives the best k-eff and capture/fission on OpenMC.
   - But it ruins peaking: 0.221 vs 0.148 on OpenMC, and 0.55–0.61 vs 0.24–0.26 on the toy.
   - The cause is that peaking is the maximum over noisy axial bins, so a noisier tally reads high. The fidelity study saw the same upward bias. That is a bias, not noise, so the bias GP and the σ scales can't fully absorb it.
   - The original safe menu (inactive *and* active cycles pinned) is the right one.
4. **The trade-off left is small.** Relative to `safe_pooled`, cheaper short runs (`conv`) gain 3% on k-eff and 14% on capture/fission on OpenMC, but lose 4% on axial offset and ~50% on peaking (over 2× on the toy). A single menu can't have both; per-output models with separate menus could, at the cost of more complexity and separate sampling.

### Current recommendation
**`adaptive_iv_mf_cks_pq_matern_safe_pooled`** (registry name), meaning:
- #7s co-kriging with the Matérn kernel and prequential calibration;
- one σ scale per output;
- buying only particle-reduced configs, with inactive and active cycles at full fidelity.

On OpenMC it roughly halves the baseline's error on every output (k-eff 0.111 vs 0.264, axial offset 0.079 vs 0.153, peaking 0.148 vs 0.306, capture/fission 0.110 vs 0.241). It passes the coverage gate, and runs in about the same wall-clock time as the baseline.

## Confirmation (`openmc_confirmation` + `toy_confirmation`, 2026-10-07, 10 seeds): **confirmed**

Plan stage 3 (backlog #1): `adaptive_iv_mf_cks_pq_matern_safe_pooled` vs `sobol_gp` and `sobol_gp_matern`, 10 seeds each, on OpenMC and on `toymc_axial`. All 60 runs finished; they took ~12 h. Seeds 0–2 on OpenMC came from the evaluation cache.

**Verdict:** `safe_pooled` beats both baselines on **every seed, every output, and every criterion on OpenMC** (final NRMSE, final NCRPS, AUC-NCRPS: 10/10 each, sign test p = 0.002). It **passes the coverage gate on every seed and output** (worst per-seed coverage 0.90–0.96). On the toy it is the same, except toy peaking's final NRMSE and NCRPS, which win 9/10 (p = 0.02); peaking's AUC-NCRPS still wins 10/10.

### OpenMC (10 seeds)

| Output | | `safe_pooled` | `sobol_gp` | `sobol_gp_matern` |
|---|---|---|---|---|
| k-eff | final NRMSE, median [range] | **0.117** [0.088–0.157] | 0.284 [0.243–0.347] | 0.266 [0.230–0.316] |
| | AUC-NRMSE / AUC-NCRPS | **0.441 / 0.208** | 0.586 / 0.295 | 0.588 / 0.294 |
| | coverage median (worst seed); seeds ≥ 0.90 | 0.96 (0.90); **10/10** | 0.86 (0.78); 2/10 | 0.92 (0.84); 8/10 |
| axial offset | final NRMSE | **0.067** [0.052–0.089] | 0.148 [0.085–0.168] | 0.108 [0.077–0.151] |
| | AUC-NRMSE / AUC-NCRPS | **0.275 / 0.144** | 0.366 / 0.196 | 0.355 / 0.190 |
| | coverage; seeds ≥ 0.90 | 0.96 (0.92); **10/10** | 0.89 (0.82); 4/10 | 0.95 (0.90); 10/10 |
| axial peaking | final NRMSE | **0.151** [0.140–0.181] | 0.296 [0.240–0.330] | 0.273 [0.214–0.290] |
| | AUC-NRMSE / AUC-NCRPS | **0.461 / 0.253** | 0.594 / 0.328 | 0.574 / 0.314 |
| | coverage; seeds ≥ 0.90 | 0.97 (0.94); **10/10** | 0.88 (0.72); 3/10 | 0.92 (0.86); 9/10 |
| capture/fission | final NRMSE | **0.107** [0.087–0.134] | 0.252 [0.229–0.294] | 0.248 [0.216–0.283] |
| | AUC-NRMSE / AUC-NCRPS | **0.405 / 0.200** | 0.540 / 0.281 | 0.540 / 0.279 |
| | coverage; seeds ≥ 0.90 | 0.95 (0.90); **10/10** | 0.86 (0.81); 2/10 | 0.91 (0.87); 8/10 |

Median per-seed final NRMSE ratio, `safe_pooled` ÷ baseline: **0.41–0.51 vs `sobol_gp`** and 0.43–0.60 vs `sobol_gp_matern`. That is about half the error at the same simulation budget. Median run time: 109 min vs 86 for `sobol_gp`; both are dominated by OpenMC.

### `toymc_axial` (10 seeds)

| Output | `safe_pooled` final NRMSE | `sobol_gp` | `sobol_gp_matern` | Wins vs each baseline (final / AUC-NCRPS) | `safe_pooled` coverage; seeds ≥ 0.90 |
|---|---|---|---|---|---|
| k-eff | **0.048** | 0.139 | 0.139 | 10/10 / 10/10 | 0.97; 10/10 |
| axial offset | **0.125** | 0.216 | 0.207 | 10/10 / 10/10 | 0.96; 10/10 |
| axial peaking | **0.274** | 0.389 | 0.374 | 9/10 / 10/10 | 0.97; 10/10 |
| capture/fission | **0.052** | 0.163 | 0.160 | 10/10 / 10/10 | 0.97; 10/10 |

### What it settles
1. **The success bar is met with statistical weight.** `safe_pooled` beats the reference baseline on AUC-NRMSE and AUC-NCRPS for every output on both MCNP-like problems, on all 10 seeds. It also passes the coverage gate on all 10 seeds for every output, while `sobol_gp` passes on only 1–4 of 10 seeds per output.
2. **It also beats the better-calibrated `sobol_gp_matern`** everywhere. The gain is from the design and the multi-fidelity model, not only from the Matérn kernel.
3. **The OpenMC advantage is larger than the toy's on the axial outputs** (axial offset ratio 0.50 vs 0.59), so the toy was, if anything, conservative.
4. **Low variance:** `safe_pooled`'s worst OpenMC seed (k-eff 0.157) is still better than the baseline's best (0.243) on k-eff, peaking and capture/fission. Axial offset is the exception (0.089 vs `sobol_gp`'s best seed 0.085).
