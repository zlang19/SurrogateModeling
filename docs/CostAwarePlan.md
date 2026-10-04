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
4. ⏳ Follow-up (b): #5 and #5b on the realistic menu, run as the `costaware` experiment alongside #7 and #7c.
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
