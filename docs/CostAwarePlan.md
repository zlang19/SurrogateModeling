# Cost Aware Plan

## Goal
Use the learnings to find and implement new modeling methods that are cost aware and multi-fidelity.

## Decisions

### Success bar
- New methods must beat #5b (`adaptive_iv_mf_xn`) on the **MCNP-like problems** (toy MC, padded high-dimensional problems), with coverage at least as good as #5b's.
- Surrogate compute is reported but not ranked, with a soft cap of about 1 minute per refit.
- Outputs stay scalar.

### Fidelity model (interface v2)
- **Knobs:** history count, particles per cycle, inactive cycles, active cycles. No physics, geometry or material fidelities.
- **Low fidelity is biased**, not just noisy:
  - unconverged source with too few inactive cycles;
  - population bias with few particles per cycle;
  - generation correlation that makes the reported σ too small.
- **Menu interface:**
  - `ProblemSpec.fidelities` is a list of `FidelityConfig(name, knobs, cost)`, with one high-fidelity config at cost 1;
  - `ask()` and `tell()` use config indices, and methods read the knob values from the spec;
  - the analytic [1/16, 1/4, 1] ladder becomes a 3-entry menu with a single `histories` knob;
  - recorded in PLAN.md as a versioned break of the frozen interface.
- **Existing #5/#5b:** expected noise per config is the median reported σ² among points run at that config, falling back to history-ratio scaling before a config has been tried. Bias isn't modeled; that is the new methods' job.

### OpenMC problem
- **Purpose:** a harder problem that matches the real MCNP case: a 3D quarter core about 2 m tall, slow axial source convergence, 10⁵ particles × 500 cycles.
- **Execution:**
  - OpenMC 0.15 from the existing `openmc-env` conda env, run as a subprocess, with ENDF/B-VIII.1 data (`OPENMC_CROSS_SECTIONS` set for the subprocess only);
  - one thread per run, 12 in parallel, inside the memory-capped systemd service;
  - cost measured in CPU time.
- **Model:** a **3×3 pin column, 2 m tall**, reflective radially, with top and bottom water reflectors. It keeps the real core's slow axial convergence at a fraction of the cost per history.
- **Inputs (14):**
  - fuel: bottom / middle / top zone enrichment, fuel temperature, pellet radius;
  - coolant: inlet moderator density, axial density drop, soluble boron;
  - lattice: pitch, cladding thickness;
  - axial: control-rod insertion from the top (B4C in the centre guide tube), top and bottom reflector thickness;
  - absorber: Gd₂O₃ in one corner pin.
- **Outputs (4):** k-eff, axial offset, axial peaking factor (max/mean of 20 bins; realistically biased upward at low histories), capture-to-fission ratio.
- **Roles:** (1) fidelity characterization, (2) toy MC calibration, (3) final validation of the top 1–2 methods. It is not a regular benchmark problem.
- **Code:**
  - `problems/openmc/model.py`: builds and reads the model; runs only inside the conda env;
  - `problems/openmc/problem.py`: the subprocess wrapper behind the `Problem` protocol;
  - `studies/fidelity_characterization.py`.

### Characterization study (≤ ~10 h, staged; re-evaluate afterwards)
1. **Pilot (~30 min):** time per particle-cycle, and how many inactive cycles source convergence needs (Shannon entropy and axial tallies).
2. **High fidelity:** chosen to be as converged as the MCNP production runs. Matching convergence matters more than matching the raw history count.
3. **Grid:**
   - 4 input points;
   - one-at-a-time sweeps of particles per cycle, inactive cycles and active cycles around high fidelity, plus joint low settings;
   - replicates;
   - one long reference run per point.

   Levels and replicates are trimmed to fit about 10 h.
4. **Measures:**
   - cost vs knobs (giving the fixed overhead);
   - relative σ vs histories;
   - actual spread ÷ reported σ vs active cycles;
   - bias vs the reference as a function of inactive cycles and particles per cycle.

### Toy MC calibration
Match OpenMC's **shapes**, not its absolute numbers:
- the overhead share of high-fidelity cost;
- under-reporting vs active cycles;
- bias vs inactive cycles, relative to each output's spread;
- whether population bias appears.

The knobs are the slab's height and coupling and its default fidelity menu. A high-fidelity run stays at a few seconds.

## Order of work
1. ✅ OpenMC model + pilot. The source settles in ~50 batches. HF is 10⁴ particles × (100 inactive + 200 active), about 6 CPU-min per run.
2. ✅ Characterization study: 1,204 runs, 5.8 h, `results/openmc_fidelity/report.md`. Findings are in [Learnings](Learnings.md#fidelity-in-a-real-monte-carlo-code-openmc-study-2026-10-04).
3. ✅ Interface v2 (fidelity menu), plus `toymc_axial`: a 1D twin of the OpenMC column calibrated to its patterns, with the same 12-config menu and the same cost shape. The original symmetric toy couldn't show axial-offset effects.
4. ✅ Follow-up (b) + new methods, the `costaware` experiment (120 runs). #7s ≈ #7 > #5b > baseline by AUC-NRMSE. On `toymc_axial`, #7s is 2× better than #5b on k-eff and capture/fission, but **every cost-aware method loses to the baseline on axial offset and fails the coverage gate there**. Details are in [Learnings](Learnings.md#the-costaware-experiment-2026-10-04). Extra runs: #5 is the worst on the realistic menu, and #7c's LOO calibration barely moves coverage (0.486 → 0.500 on axial offset).
5. ⏳ Triage and implement. Already built from the "In" rows:
   - **#7 `adaptive_iv_mf_ck`:** co-kriging with a bias-aware joint acquisition;
   - **#7c `adaptive_iv_mf_ck_cal`:** #7 plus closed-form LOO conformal calibration.

   The gated rows wait for the `costaware` results.

## Candidates (from the original considerations, with status)
| Candidate | Status | Why |
|---|---|---|
| Multi-Fidelity Fourier Neural Operators (MF-FNOs) | **Later** | Built for field outputs; ours are scalar. Revisit with the neural methods |
| Co-Kriging DeepONets | **Later** | Same as above. The co-kriging *GP* is in scope instead (below) |
| Co-kriging / autoregressive multi-fidelity GP | **In** | Low fidelity is now biased, so modeling the bias between configs is worthwhile |
| Variational sparse inducing-point MF GPs (VMFGPs) | **Conditional** | Only if data grows past a few thousand points (cheap configs at realistic overhead) |
| Heteroscedastic joint cost-target models | **Conditional** | Only if the characterization shows cost varies strongly with the inputs |
| Conformalized ensembles | **In, as a calibration wrapper** | Targets coverage; members can be GPs |
| Dynamic multi-armed bandits (over fidelity arms) | **Gated** | Only if step 4 shows a non-trivial fidelity choice |
| Non-myopic / multi-step lookahead (rollout, DP) | **Gated** | Same as above |
| Budget-aware entropy search | **Gated** | Same as above |
| Information-theoretic multi-fidelity active learning | **Gated** | Same as above; a cost-aware EPIG would be the natural first version |

## Later: neural methods
Test neural methods (MF-FNO, DeepONet, neural-network ensembles) once the GP-based work settles. They would likely use the GPU (RTX 3060, 12 GB) behind a per-method concurrency limit (`[limits]`), since one GPU can't serve 10 workers.

## Calibration follow-up (decided 2026-10-04; starts after `costaware` finishes)

**Problem:** every GP method drifts overconfident as data grows. In `full`, median coverage went 0.95 → 0.92 → 0.88–0.90 at cost 25 → 50 → 100, so real errors at full budget are about 1.2–1.27× the claimed error bars. The NRMSE leaders (cost-aware methods, padded problems) drift most. Likely causes:
- plug-in hyperparameters (no hyperparameter uncertainty);
- variance shrinking as σ²/n with many cheap points while structured error doesn't;
- trusting reported σ;
- the smoothness of the RBF kernel;
- stale hyperparameters between refits.

**Criteria (add, don't replace):**
- Add **normalized CRPS** (CRPS ÷ the output's spread) as a per-batch metric, and **AUC-NCRPS** as a second ranked criterion next to AUC-NRMSE. It's a proper scoring rule, comparable across outputs, robust to single outliers, and reduces to MAE for methods without variance.
- Add a **calibration gate** to the success bar: final coverage ≥ 0.90 on every output.
- NLL and max error stay diagnostics.

**Models:**
1. **Next-batch (prequential) calibration** as an option for every GP method. Predict each incoming batch *before* training on it, keep a running set of those out-of-sample standardized residuals, and scale the error bars so they cover 95%. It replaces in-sample LOO (#7c), which shares the hyperparameters' optimism.
2. **A Matérn-5/2 kernel** variant.
3. **Hyperparameter uncertainty** (SAAS, on hold; or a small hyperparameter ensemble), if 1 and 2 aren't enough.

**More model variants (added 2026-10-04).** All target the same accuracy-vs-calibration trade-off, so they're tested in the same experiment.

*Tier 1: cheap, and motivated by results so far*
| Variant | Evidence behind it |
|---|---|
| **Pooled σ scale**: one learned scale per output, shared across configs (a variant of #7s) | OpenMC showed under-reporting is a property of the *output* (axial offset ~4× at every setting), not of the knobs. Per-config scales are poorly identified for rarely used configs |
| **Log-transform skewed positive outputs** (peaking, borehole flow, power ratio) | These are skewed and bounded below. A log-scale GP usually fits better and gives asymmetric, better-calibrated error bars |
| **Weighting outputs in the acquisition**: weight each output's variance reduction by its current estimated error | Equal weights let k-eff and capture/fission dominate on `toymc_axial`, while axial offset lost to the baseline |
| **Hyperparameter ensemble**: a few GPs from fit restarts or posterior samples, combined as a mixture | Plug-in hyperparameters are the leading suspect for the coverage drift. A cheap partial substitute for SAAS (on hold) |
| **Input warping**: learned monotone per-input transforms (botorch `Warp`) | Rod insertion and zone boundaries make some responses non-stationary |

*Tier 1.5: combinations of the Tier 1 winners (added 2026-10-04)*

If several variants help on their own in screening, test their combinations with a small **factorial design** rather than stacking everything:

- **`pq`** (next-batch calibration) is a layer on top of any model, since it only rescales the error bars using held-out residuals. It should combine cleanly; if it wins alone, every final candidate includes it.
- **`matern` and `warp`** both target kernel misspecification (roughness vs non-stationarity). They may help each other or be redundant, and warping adds ~2 hyperparameters per input (~50 for 25 inputs), which risks overfitting.
- **`pooled`, `log`, `wt`** join the factorial only if they win on their own.
- **`ens`** costs ~3× per fit, so it's added last, to the best combination, and only if calibration is still short of the gate.

For winners {pq, matern, warp}: screening already covers 4 of the 2³ = 8 arms (base and each one alone), so only **4 new arms** are needed: `pq+matern`, `pq+warp`, `matern+warp` and `pq+matern+warp`. At 3 seeds on `toymc_axial` and Borehole-30D that's 24 runs, about 3 h. It measures whether the effects add up or interact.

**Reading noisy 3-seed results:** all arms share seeds, so they share initial designs and simulator noise. Compare arms **paired by seed**, and call an effect real only if it's consistent across both problems and across outputs, especially axial offset.

*Tier 2: later, if Tier 1 leaves gaps*
- **HIPE-style seed phase:** choose early points to pin down hyperparameters.
- **Multi-output GP** across related outputs.
- **Screening combined with cost-aware sampling.**

*Tier 3: still gated*
- Bandits, lookahead, cost-aware EPIG: the gate is partly met, since co-kriging now mixes fidelities, but these wait until results show the fidelity *choice*, rather than the noise model, is what limits accuracy.
- SAAS + cost-aware: blocked by the SAAS hold.
- Neural ensembles: deferred with the other neural methods.

**Experiment (three stages):**
1. ✅ **Screening** (`calib_screen.toml`; results in [cks_variants.md](methods/cks_variants.md#screening-results-2026-10-05-3-seeds-paired-by-seed-against-the-base-7s)). **Matérn is the calibration winner** (coverage 0.77 → 0.92). `pq` improves NCRPS consistently. `warp` is the first to beat the baseline on axial offset at full budget (0.200), though it's worse early. `ens` and `wt` are dropped. each calibration option and each Tier 1 variant on the leading cost-aware method (#7s), with **3 seeds** on `toymc_axial` and Borehole-30D. About 4–5 h with the 8-worker pool.
2. ✅ **Combinations** (Tier 1.5; results in [cks_variants.md](methods/cks_variants.md#combination-results-2026-10-05-3-seeds-paired-by-seed-against-the-base-7s)). **`pq_matern_warp` passes the coverage gate on every output (0.96) and has the best final accuracy** (axial offset 0.177 vs 0.295), but warping is poor below cost ≈ 30, so its AUC is worse. Strong Matérn × warp synergy. After review, two trials followed (2026-10-05):
  - **Delayed top-8 warp:** fast (~93 min vs ~6 h) but **fails on `toymc_axial`** (k-eff 0.121 vs 0.059).
  - **SAAS trial:** better than the plain GP everywhere at the same budget, and well calibrated.

  See [Learnings](Learnings.md#saas-and-the-pareto-view-2026-10-05).
  - **SAAS + cost-aware (#8):** built, but its trial was **cancelled** after two out-of-memory kills, with memory growing ~1 GB/min per run from an unidentified cause ([adaptive_iv_mf_saas.md](methods/adaptive_iv_mf_saas.md)).

  - **Pre-validation screen (2026-10-06, `prevalidation_screen.toml`):** three arms. **The converged-source menu (`pq_matern_safe`) wins outright:** best final NRMSE on every `toymc_axial` output (axial offset 0.144 vs baseline 0.179), passes the gate everywhere (0.93–0.99), and takes 25 min per run vs 125. `sobol_gp_matern` is a better-calibrated baseline; the 50% seed (`seed50`) is worse. Report and validation recommendation: [ValidationRecommendation.md](ValidationRecommendation.md).

  Arms were: `pq+matern`, `pq+warp`, `matern+warp`, `pq+matern+warp`, `pq+matern+warp+pooled`, with 3 seeds, compared paired by seed.
3. **Confirmation:** the best one or two combinations against the base, at **10 seeds** on the MCNP-like problems.

All stages are scored by AUC-NRMSE, AUC-NCRPS and final coverage (gate ≥ 0.90), with NLL and max error as diagnostics.

**Then (decided 2026-10-05):**
4. **SAAS trial:** `sobol_saas` × `toymc_axial`, `toymc`, Borehole-30D × 3 seeds, against the baseline and the confirmed best method. The full SAAS set stays on hold.
5. **OpenMC validation** (plan role 3): the baseline and the best method on the OpenMC column, sized to about a day of compute:
   - a reduced reference test set;
   - batches evaluated in parallel;
   - a shared evaluation cache for the common seed design.
