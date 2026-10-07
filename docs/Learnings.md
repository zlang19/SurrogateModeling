# Learnings

What the test bed has taught us so far, as of 2026-10-04. Each entry gives the evidence and what it means for the MCNP application. Numbers are seed medians from the `full` experiment (10 seeds, a budget of 100 high-fidelity-equivalent runs) unless stated otherwise. Method details are in [methods/](methods/README.md).

## Headline results

### Cheap, noisy runs beat expensive precise ones by a wide margin
- **Evidence:** the cost-aware methods (#5, #5b) beat the high-fidelity-only baseline by 1.5–2.4× on most outputs: Morris 0.45 → 0.19, Borehole-30D 0.091 → 0.045, toy MC k-eff 0.099 → 0.062. They took the top two places in the ranking.
- **For MCNP:** spending the same compute on many short runs (fewer histories each) across the input space is worth far more than a few well-converged runs, *as long as the noise is modeled correctly* (next entry).

### The reported σ can't be taken at face value
- **Evidence:** the toy MC power ratio's batch-statistics σ is about 2× too small, because successive generations are correlated in a loosely coupled core. MCNP shows the same effect for local tallies.
  - #5 put about 1,000 low-fidelity points into a GP that trusted those σ values, and it plateaued at 0.27 NRMSE, worse than every high-fidelity method.
  - Adding a learned extra noise term (#5b) fixed it, giving 0.105, the best of all methods, with no loss elsewhere.
- **For MCNP:** never feed tally relative errors into a surrogate as exact noise. Let the model learn extra noise on top. The more low-precision runs a strategy uses, the more this matters.

### The cost-aware methods always chose the lowest fidelity
- **Evidence:** every adaptive pick by #5 and #5b was the cheapest option, 1/16 of the high-fidelity histories.
- **Caveat:** that answer depends on the 1% fixed cost per run and on 1/16 being the cheapest option offered. Real MCNP runs have larger fixed costs (geometry and cross-section loading, inactive cycles), which would favor more histories per run. This is the top open question.

### Extra, mostly irrelevant inputs are expensive
- **Evidence:** padding Borehole from 8 to 30 inputs, half of them weak (0.5% of the variance in total) and half inert, tripled the baseline error (0.029 → 0.091). Adaptive sampling (#4, #4b) didn't help there, screening only roughly matched the baseline, and only the cost-aware methods halved it (0.045).
- **For MCNP:** with tens of parameters and 50–100 runs, the number of points matters more than where they go. More, cheaper points is the lever that works.

### Adaptive point selection helps a little; the model and the noise matter more
- **Evidence:** integrated variance (#4) and EPIG (#4b) beat the fixed Sobol design only modestly (e.g. OTL 0.033 vs 0.041) and not at all in 30–40 dimensions. They are essentially tied with each other.

## Fidelity in a real Monte Carlo code (OpenMC study, 2026-10-04)

From `studies/fidelity_characterization.py`: 1,204 OpenMC runs of a 2 m-tall 3×3 pin column, which shares a full core's slow axial source convergence. Full tables are in `results/openmc_fidelity/report.md`.

- **Source convergence takes about 50 batches** from a flat start, and high fidelity uses 100 inactive. Matching *convergence* to production MCNP mattered more than matching the raw history count: 10⁵ × 500 would be about an hour per run here.
- **Cost:** about 11 s fixed (3% of a high-fidelity run), plus cost proportional to particles × (inactive + active).
  - Cutting active cycles still pays for every inactive cycle: 1/8 of the active cycles costs 37% of high fidelity.
  - Cutting particles scales almost proportionally, so it's the cheap way to get fewer histories.
  - The 1% overhead the analytic problems assumed was far too optimistic.
- **σ under-reporting depends on the output, not the knobs:**

  | Output | Real spread ÷ reported σ |
  |---|---|
  | axial offset | about 4× |
  | axial peaking | about 2× |
  | k-eff | about 1.15× |
  | capture-to-fission | about 1.0× |

  It's the same at every particle and cycle setting. Spatial outputs tied to the slowest source mode are the dangerous ones.
- **Bias is real but concentrated:**
  - Combined low settings (1,000 particles, 10 inactive, 25 active) are biased by 3–5 high-fidelity σ.
  - Ten inactive cycles alone, or few particles, bias **peaking upward**: the maximum over noisy bins.
  - Otherwise bias stays within about 1 high-fidelity σ.
- **Measure the reference's error by its real noise, not its reported σ.** The first analysis flagged high fidelity itself as "biased" (z = 4.4 on axial offset), because the reference run's σ was under-reported too.
- **The toy MC needed an axial redesign to mimic this.** Axial offset is an *antisymmetric* top/bottom tilt driven by the slowest source mode, and the symmetric slab's inner/outer ratio never sees it. `toymc_axial`, a 1D twin of the column, reproduces OpenMC's patterns: axial offset under-reported about 5×, peaking about 2×, bias only at low settings. It's somewhat harsher than OpenMC.

## Bias-aware multi-fidelity

- **Learned extra noise can't absorb a smooth bias.** On a Borehole variant whose cheap configs carry a smooth bias, #5b reached only 0.13 NRMSE, with **0.2 coverage**: confidently wrong.
- **Co-kriging needs a bias-aware acquisition as well as a bias-aware model.**
  - Adding each config's bias variance to the acquisition as independent noise improved the model (0.09–0.10, coverage 0.77), but it still bought only the cheapest config.
  - Bias is *correlated*: many cheap runs near each other share one bias. Giving the acquisition a joint covariance over every (candidate, config) pair fixed this: **0.035–0.041 NRMSE, coverage 0.92–0.96**, with a genuine mix of high-fidelity and cheap runs.
- **Bias at rarely used configs is poorly identified.** Its learned size there drifts large (about 5× the truth on the test problem), which makes the method conservative about those configs. That's safe, but it could waste options.

## The `costaware` experiment (2026-10-04)

Baseline, #5b, #7 and #7s × `toymc_axial`, `toymc`, Borehole-30D × 10 seeds, 120 runs. Mean rank by AUC-NRMSE: **#7s 1.62, #7 1.62, #5b 3.0, baseline 3.75.**

**Final NRMSE (10-seed median):**

| Problem / output | Baseline | #5b | #7 | #7s |
|---|---|---|---|---|
| borehole_d30 | 0.091 | 0.048 | 0.047 | 0.047 |
| toymc k_eff | 0.099 | 0.064 | 0.063 | 0.064 |
| toymc power_ratio | 0.155 | 0.108 | **0.097** | 0.118 |
| toymc_axial k_eff | 0.139 | 0.130 | 0.085 | **0.061** |
| toymc_axial capture/fission | 0.163 | 0.120 | 0.094 | **0.067** |
| toymc_axial axial_offset | **0.216** | 0.318 | 0.311 | 0.270 |
| toymc_axial axial_peaking | 0.389 | 0.396 | 0.395 | **0.361** |

- **On unbiased menus (`toymc`, Borehole-30D) the new methods tie #5b.** All are about 2× better than the baseline, and all pick the cheapest config 100% of the time. When cheap runs are unbiased, the fidelity choice really is trivial.
- **On the realistic biased menu (`toymc_axial`), modeling noise and bias pays off for the well-behaved outputs.** For k-eff and capture/fission, #7s is about **2×** better than #5b and the baseline.
- **#7s is the only method that diversifies:** 13% of its adaptive picks are not the cheapest config, and it uses about 1,540 runs vs about 1,870.
- **Axial offset is still unsolved.** Every cost-aware method loses to the high-fidelity baseline (0.27–0.32 vs 0.22).
  - #7s learned a σ variance scale of **11** for the cheapest config's axial offset, where the toy's truth is about 34, so it corrected only about a third of the under-reporting.
  - For peaking it learned a scale below 1 and attributed the error to bias instead.
- **Calibration fails on `toymc_axial` for all cost-aware methods:**

  | Output | Baseline coverage | Cost-aware coverage |
  |---|---|---|
  | axial offset | 0.87 | 0.46–0.53 |
  | peaking | 0.89 | 0.58–0.68 |
  | k-eff | 0.87 | 0.45 (#5b) – 0.80 (#7s) |

  Axial-offset NLL is 5.5–10 vs −0.8. These models are **confidently wrong** where it matters most for a reactor, and no cost-aware method passes the planned 0.90 coverage gate there. #7s comes closest.
- **Extra runs (5 seeds, `toymc_axial`):**
  - **#5, trusting reported σ, is the worst** cost-aware method on the realistic menu: axial offset 0.443, peaking 0.537. It's worse than the baseline on 3 of 4 outputs.
  - **#7c's LOO calibration barely moves coverage** (axial offset 0.486 → 0.500). In-sample leave-one-out residuals share the fitted model's optimism, so this calls for out-of-sample (next-batch) calibration instead.
- **Against the plan's success bar** (beat #5b on the MCNP-like problems, with coverage at least as good): #7 and #7s meet it. #7s wins 7 of 8 outputs, losing only the toy MC power ratio, with better coverage on `toymc_axial`. But "as good as #5b" turns out to be a weak bar for calibration, and the calibration follow-up is now the priority.

## SAAS and the Pareto view (2026-10-05)

- **SAAS beats the plain GP at the same high-fidelity budget on every output and is much better calibrated.** Same 3 seeds:
  - final NRMSE: Borehole-30D 0.074 vs 0.086; toy MC power ratio 0.136 vs 0.162; `toymc_axial` k-eff 0.124 vs 0.142, peaking 0.382 vs 0.407;
  - coverage 0.89–0.97 vs 0.81–0.93.

  Averaging over hyperparameters fixes much of the overconfidence that plug-in GPs show. It costs 5–20 min per run vs 0.2–7 min for the baseline.
- **It can't match the cost-aware methods on smooth outputs** (k-eff, capture/fission: 0.12–0.16 vs 0.05–0.07), which get ~15× more (cheap) data points.
- **On axial offset, high-fidelity sampling (plain GP or SAAS) remains as good as anything over the whole budget.** On AUC-NCRPS, the 7-minute baseline is the *only* Pareto-optimal method for axial offset (`results/all_ncrps/pareto_toymc_axial.png`). Cost-aware methods only catch up late (full warp: 0.177 final). For a budget that may end early, the plain approach is the safe choice for the hardest output.
- **Pareto summary (AUC-NCRPS vs run time, `toymc_axial`):**
  - k-eff / capture/fission: frontier = baseline → SAAS → `pooled` / `matern` / `log` (~1.5–2 h).
  - peaking: baseline → `pooled` → `pq_matern`.
  - **Every warping arm is dominated** over the whole budget (slow, and poor early), even though full warp has the best *final* numbers.
- **Delayed warping failed on the hard problem.** It stalled k-eff at 0.12 (vs 0.06) after switching on, most likely the warm-start trap again. Warping looks powerful only when fitted cold, with enough data, and at full cost.

## Methods

- **PCE needs proper hybrid LARS.** The first version, which chose terms by cross-validated LASSO, overfit and picked inert inputs. Scoring the full LAR path by corrected leave-one-out error fixed it. Even so, PCE ranks last here: it's competitive only on smooth, low-dimensional, low-noise problems.
- **EPIG's target matters.** With a latent (noise-free) target, EPIG scores only correlation and ignores how much variance remains; it levels off near 1% on a near-noiseless test where IV reaches 0.5%. We use the paper's noisy-target formulation.
- **Screening must wait for data.** The first `screen_gp` re-screened after every batch, starting at **5 points** in 8 dimensions, and permanently dropped an input carrying 3% of the variance (0.17 NRMSE). It now screens only once the seed is complete, and keeps inputs up to 99% of relevance, not 95%.
- **Dropped inputs belong in the error bars.** After screening, the learned extra noise stands in for the dropped inputs, which are part of the true function. Leaving it out of the predictive variance gave 0.5–0.6 coverage on some problems; including it raised Borehole from 0.51 to 0.97.
- **Coverage is generally a little low.** Most GP coverage is 0.84–0.95 against a nominal 0.95. The worst cases are where σ is under-reported or inputs were dropped.

## Building a toy Monte Carlo problem

- **Averaging per-generation ratios biases low fidelity.** The power ratio drifted from 1.40 to 1.33 as histories went from 100 to 3200 per generation, because E[a/b] ≠ E[a]/E[b]. Ratio-of-sums with a delta-method σ removed the drift. This matters doubly here: multi-fidelity methods assume low fidelity is unbiased.
- **A peaking output should be a power *density* ratio.** Inner/outer total power mostly measured zone volume (range 0.4–5.3).
- **Check noise scaling with replicates, not single runs.** A 16-replicate test gave a misleading 1.14 spread ratio for 4× histories (2 expected). With 96 replicates, std × √N was flat.

## Engineering and operations

- **Long runs must be memory-capped systemd services.** The full benchmark ran the 30 GB machine out of memory twice. Both times the kernel killed **VS Code** rather than the benchmark, and the run died with it. Background shells die with the Claude session, and `setsid` didn't escape VS Code's process group either. What works is `systemd-run --user ... -p MemoryHigh=20G -p MemoryMax=22G -p OOMScoreAdjust=500` (see the [README](../README.md#running-long-experiments)).
- **JAX recompiles for every new data size.** SAAS reached about 6 GB per run from cached compiled NUTS kernels. Calling `jax.clear_caches()` once per fit brought the peak to about 3 GB. Calling it once per *output* made runs slower, because outputs share a compile.
- **Memory-heavy methods need a concurrency limit.** `[limits] sobol_saas = 3` gives such a method its own process pool. Even at 3 GB each, SAAS runs take about 20 min, roughly 4 h for 70 runs on 6 workers, so they're split into their own config (`full_saas.toml`).
- **gpytorch's iterative solvers made long GP runs grow to 6–7 GB.** Above 800 points, gpytorch switches from Cholesky to CG/Lanczos. Cost-aware runs on `toymc_axial` reach about 2,000 points, grew to 6–7 GB each, and systemd-oomd killed the experiment. That was the memory cap working as intended: VS Code survived. Forcing exact Cholesky (`max_cholesky_size`) kept a run flat at about 1 GB at n = 875 (2 GB before) and was slightly faster. Glibc allocator tuning made no difference, so it wasn't fragmentation.
- **Warm-starting full refits is a big, mostly free speed-up, except for unstable parameterizations.** Starting each full refit from the previous optimum made #7s 2.8× faster with identical results. But a model with input warping from the very first (25-point) fit got stuck in a bad optimum (NRMSE 0.36 vs 0.05). Path-dependent fitting needs the early fits to be sane, so delay complex model components until there's enough data.
- **Count the optimizer's work, not just the parameters.** Warping doubled the hyperparameters but made each fit about 10× slower, because the optimizer needed 10× more steps.
- **Warm starts must include every trainable piece.** Hyperparameters copied between full refits initially left out a new noise-scale module. It silently reset each batch, so learned σ scales read 0.9 instead of 25.
- **Test fixtures can trigger expensive builds.** A test that builds every registered problem's test set tried to build `toymc_axial`'s 2-hour truth set inside pytest's temporary cache.
- **Use spawn, not fork, for workers.** torch autograd state doesn't survive `fork`.
- **Pin each worker to one thread** (torch and XLA flags). Ten workers each starting their own thread pool oversubscribe 12 cores.
- **Standardize with ddof=1** to match botorch's input check. Otherwise every early fit emits a warning.
- **Write results atomically** (`.tmp` then rename) and **skip runs that already finished**, so any interruption costs only the runs in flight.

## Converged-source menu (2026-10-06)

- **Never cut cycles; cut only particles.** Restricting the cost-aware menu to configs with high-fidelity inactive and active cycles (`pq_matern_safe`) turned the worst output into the best one. Axial offset final NRMSE fell from 0.287 to 0.144 (baseline 0.179), and coverage rose from 0.72 to 0.97. It won all 12 seed × output pairs against the full menu. Unconverged sources bias the slow axial mode in a way the bias GP can't learn, so the extra cheap points were costing more in bias than they gave in noise reduction.
- **Fewer points is also faster:** ~620 evaluations instead of ~1,600 at the same budget, so 25 min per run instead of 125.
- **A Matérn kernel fixes most of the baseline's overconfidence** (`sobol_gp_matern`: coverage 0.90–0.96 vs 0.81–0.90) at no cost in accuracy.
- **A bigger high-fidelity seed design hurts** (50%: k-eff 0.101 vs 0.059). The budget is better spent on cheap runs.
- Full report and validation ranking: [ValidationRecommendation.md](ValidationRecommendation.md).

## OpenMC validation, round 1 (2026-10-06)

- **The safe menu passes on the real code:** about half the baseline's final NRMSE on every output, 12/12 seeds, gate passed (0.91–0.97). See [OpenMCValidation.md](OpenMCValidation.md).
- **The source-convergence bias is real in OpenMC.** The full menu spent its budget on `joint_low` (10 inactive cycles), learned a ~22× σ scale for it on axial offset, and failed the gate (0.82).
- **But global outputs (k-eff, capture/fission) like the unconverged cheap runs.** The full menu beat `safe` by 10–20% on them. Whether a cheap run is safe depends on the output.
- **Per-config σ scales are unstable on OpenMC** (stuck at the lower bound, or blowing up to 1,383×). Round 2 tests pooled scales and a menu that pins only inactive cycles.

## OpenMC validation, round 2 (2026-10-06)

- **`safe_pooled` is the leader on OpenMC and the toy:** the safe menu plus one σ scale per output. It improves `safe` on most outputs, passes the gate, and is 30% faster. The pooled scales are stable across seeds and match the fidelity study (axial offset σ × ~5.8, peaking × ~2.3).
- **Keep active cycles at full fidelity too.** Peaking is a maximum over noisy bins, so short runs bias it upward. Cutting active cycles (`conv`) helped k-eff and capture/fission slightly, but ruined peaking (OpenMC 0.221 vs 0.148; toy 0.55–0.61 vs 0.24–0.26).
- **k-eff and capture/fission scales hit the 0.2 lower bound** on every seed: the model treats the reported σ as too large for the global outputs. This is worth investigating (bound, prior, or a real over-report).

## Open questions

1. **Axial offset under realistic fidelity.** Cost-aware methods still lose to high-fidelity-only sampling on axial offset, and are badly overconfident there. Is the fix better noise learning (a pooled scale per output), weighting the acquisition toward the weakest output, or just calibration? (The calibration follow-up in CostAwarePlan.md.)
2. **SAAS:** does it close the padded-input gap at high fidelity? Its 70 runs are on hold, pending a decision on when to run them.
3. **Multiplicative vs additive noise correction:** σ under-reporting is roughly a multiplicative factor, but #5b learns an *additive* term. A learned σ scale might be better on outputs whose noise varies strongly with fidelity.
4. **Coverage:** can calibration be improved (e.g. by learning the noise model) without losing accuracy?
