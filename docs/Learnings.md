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
- **Warm starts must include every trainable piece.** Hyperparameters copied between full refits initially left out a new noise-scale module. It silently reset each batch, so learned σ scales read 0.9 instead of 25.
- **Test fixtures can trigger expensive builds.** A test that builds every registered problem's test set tried to build `toymc_axial`'s 2-hour truth set inside pytest's temporary cache.
- **Use spawn, not fork, for workers.** torch autograd state doesn't survive `fork`.
- **Pin each worker to one thread** (torch and XLA flags). Ten workers each starting their own thread pool oversubscribe 12 cores.
- **Standardize with ddof=1** to match botorch's input check. Otherwise every early fit emits a warning.
- **Write results atomically** (`.tmp` then rename) and **skip runs that already finished**, so any interruption costs only the runs in flight.

## Open questions

1. **Fidelity choice under realistic overhead:** with OpenMC's measured cost shape and the 12-config cycle/particle menu (`toymc_axial`), which configs do #5, #5b and #7 actually choose, and does bias-aware co-kriging beat #5b there? (The `costaware` experiment.)
2. **SAAS:** does it close the padded-input gap at high fidelity? Its 70 runs are on hold, pending a decision on when to run them.
3. **Multiplicative vs additive noise correction:** σ under-reporting is roughly a multiplicative factor, but #5b learns an *additive* term. A learned σ scale might be better on outputs whose noise varies strongly with fidelity.
4. **Coverage:** can calibration be improved (e.g. by learning the noise model) without losing accuracy?
