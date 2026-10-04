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
- **Use spawn, not fork, for workers.** torch autograd state doesn't survive `fork`.
- **Pin each worker to one thread** (torch and XLA flags). Ten workers each starting their own thread pool oversubscribe 12 cores.
- **Standardize with ddof=1** to match botorch's input check. Otherwise every early fit emits a warning.
- **Write results atomically** (`.tmp` then rename) and **skip runs that already finished**, so any interruption costs only the runs in flight.

## Open questions

1. **Fidelity choice under realistic overhead:** does "always the lowest fidelity" survive a 10–20% fixed cost per run and a finer ladder? (Follow-up (b).)
2. **SAAS:** does it close the padded-input gap at high fidelity? Its 70 runs are on hold, pending a decision on when to run them.
3. **Multiplicative vs additive noise correction:** σ under-reporting is roughly a multiplicative factor, but #5b learns an *additive* term. A learned σ scale might be better on outputs whose noise varies strongly with fidelity.
4. **Coverage:** can calibration be improved (e.g. by learning the noise model) without losing accuracy?
