# PLAN

## Goal
Build a test bed for surrogate modeling methods against complex simulation problems.

- **Primary:** pick the best surrogate approach for the target code (MCNP).
- **Secondary:** keep the harness general enough to serve as a reusable benchmark.

## Context
- Optimization criterion: the most accurate cheap replacement for the true simulator, for the least simulation cost.
- Target simulator: MCNP (Monte Carlo neutronics).
  - Inputs: tens of continuous parameters across all domains (enrichment, geometry, materials, ...). A few dominate; most are weak or inert.
  - Outputs: a few scalars.
  - Cost: hours per run; can be run in parallel in synchronous batches of ~5.
  - Noise: Monte Carlo; each tally reports its own relative error.
  - Fidelity: particle/history count. It changes precision only, not physics, so low fidelity is unbiased but noisier.
- The surrogate will be used for UQ over the input distributions.
- MCNP is **not** in the test bed for now. That decision is deferred. The interface must not preclude adding it.

## Decisions

### Scope
- Sampling strategy belongs to the method. Fixed designs queue their points, adaptive methods decide at ask time, and hybrids return a fixed seed design and then switch to adaptive.
- Fixed-design baselines are always included, so that model quality can be separated from point selection.
- Multi-fidelity = a noise-aware (heteroscedastic) model plus cost-aware acquisition of (x, fidelity). Co-kriging/AR1 are excluded from v1 because there is no inter-fidelity bias to model.
- Batches are synchronous: `ask(5)` → wait for all → `tell`.
- Multi-output: `tell` receives all outputs. Each method decides whether to model them jointly or independently.

### Budget and evaluation
- Budget: 50–100 high-fidelity-equivalent runs, where cost(fidelity=1) = 1.
- 10 seeds per (method, problem). Plots show median with IQR bands.
- Error is evaluated after every batch on a cached test set of 2000 points drawn from the input distribution, so the error is pdf-weighted.
- Truth is exact for analytic problems and 100× high-fidelity N for the toy MC.
- **Ranking metric:** NRMSE per output (pdf-weighted). Cross-problem ranking uses normalized area under the error-vs-cost curve.
- **Secondary (reported, never ranked):** max error, 95% coverage, NLL, for methods that return variance.

### Input distributions
- Independent marginals (normal, uniform, truncated normal, ...) behind a `Distribution` object with `sample(n, rng)` and `bounds()`, so correlations can be added later.
- Training domain is a declared box that covers most of the probability mass. Methods may sample anywhere in it.

### Fidelity and noise
- `fidelity` is a float in (0, 1]. Each problem interprets it (for MC: a fraction of high-fidelity histories) and supplies the cost estimate.
- Problems may declare a recommended fidelity ladder; methods may ignore it.
- Noise model is configurable per problem. Default is relative: σ = r·|f(x)|/√(N/N₀).
- The reported σ is itself a noisy estimate, as in MCNP.
- Analytic defaults: 1% relative noise plus an absolute floor of 0.1% of each output's std, so that zero-crossing outputs (Morris) never get noise-free points.
- Cost includes a fixed per-run overhead: cost(f) = overhead + (1 − overhead)·f, with a 1% default. Without it, near-zero fidelity would be almost free.
- Padded problems: half the padding inputs are weak (linear or quadratic, together 0.5% of output variance), half inert. A model that ignores the weak ones is floored at NRMSE ≈ 0.07.

### Stand-in problems
- **Analytic suite:** Borehole (8D), wing weight (10D), OTL (6D), Morris (20D), plus versions padded to 20–40D with weak/inert inputs. All are wrapped with synthetic MC noise and cost ∝ N.
- **Toy MC neutronics code:**
  - 2-group k-eigenvalue power iteration (KCODE-like), on a 1D multi-region slab, vectorized in numpy.
  - Parametric made-up cross sections with realistic sensitivities (enrichment, Doppler-like fuel temperature, moderator density, boron, ...).
  - About 20–30 inputs, roughly 5 of them strong.
  - Outputs: k-eff, a leakage/peaking-type ratio, and one reaction-rate ratio.
  - Fidelity = histories per batch × active batches. A high-fidelity run takes a few seconds.
- Toy MC as built:
  - High fidelity = 1000 histories/generation, 15 inactive + 20 active generations, about 0.3 s per run.
  - σ comes from active-generation batch statistics, as in MCNP. For k-eff and capture-to-fission it matches the replicate spread.
  - For the power ratio, the reported σ is about 2× too small (generation-to-generation correlation in a loosely coupled core). Kept on purpose as realistic MCNP-like behaviour. Grouping several generations per batch would fix it if needed.
  - Ratio outputs are computed as ratio-of-sums over active generations, with delta-method σ. Averaging per-generation ratios biased low fidelity.
  - The power ratio is inner/outer power *density*, not total power.
  - Truth: 100× HF histories, about 4 s per point. 2000 points are cached in resumable chunks.
- OpenMC is an optional later addition.

### Methods (v1)
| # | Method | Design |
|---|---|---|
| 1 | GP (ARD, per-point fixed noise) | Fixed Sobol |
| 2 | Sparse PCE (Legendre/Hermite basis + LassoLars) | Fixed Sobol |
| 3 | SAAS-GP (BoTorch, fully Bayesian) | Fixed Sobol |
| 4 | GP + pdf-weighted integrated-variance acquisition | Hybrid |
| 4b | GP + EPIG acquisition (targets sampled from input dist) | Hybrid |
| 5 | #4 with cost-aware (x, fidelity) selection | Hybrid |
| 6 | Screening (Morris or early ARD) → GP on active inputs | Hybrid |

- Excluded from v1: neural nets, ensembles, RBF/kriging duplicates.
- As built:
  - GPs: one independent GP per output on unit-box inputs, fixed per-point noise from σ. Adaptive methods re-optimize hyperparameters only when the data has grown by 20% since the last fit.
  - PCE: true hybrid LARS. Every prefix of the LAR path is refit by OLS and scored by corrected LOO. Degree ≤ 5 when d ≤ 10, else ≤ 3; interactions ≤ 2.
  - SAAS: botorch 0.18 runs it on JAX/NumPyro, on CPU, not the GPU. One thread per worker process. NUTS settings 256 warmup / 128 samples / thinning 16.
  - Hybrids:
    - Seed: 25% of the budget as a high-fidelity Sobol design.
    - Batches: chosen greedily from 1024 fresh candidates (half uniform over the box, half from the input distribution), scored against 512 fixed reference samples of the input distribution, with an exact rank-one covariance update between picks.
    - Noise: expected high-fidelity noise is the median of var·fidelity over the data.
  - EPIG targets a noisy high-fidelity observation y\*, as in the paper. With latent targets it ignores remaining variance in the low-noise limit.
  - #5 chooses from the problem's fidelity ladder [1/16, 1/4, 1] by score per unit cost.
  - Screening:
    - Until the seed is complete: full-dimension GP.
    - Then: ARD screening keeps inputs up to 99% of relevance (max 12), and re-screens when the data doubles.
    - Noise: a learned extra noise term absorbs the dropped inputs and is added to the predictive variance.
- Optional later: SAAS + cost-aware acquisition, co-kriging as a naive multi-fidelity baseline.

## Interface (frozen)
```python
@dataclass
class ProblemSpec:            # what a Method may see; never the truth function
    name: str
    dist: Distribution        # independent marginals: sample(n, rng), bounds() -> box
    output_names: list[str]
    cost: Callable[[float], float]      # fidelity in (0,1] -> cost; cost(1.0) == 1
    fidelity_ladder: list[float] | None # optional hint, methods may ignore

class Problem(Protocol):
    spec: ProblemSpec
    def evaluate(self, X, fidelity, rng) -> Observation  # y (n,m), sigma (n,m); sigma itself noisy
    def test_set(self) -> tuple[X, Y_true]               # cached, 2000 pts from dist

class Method(Protocol):
    def setup(self, spec: ProblemSpec, budget: float, rng) -> None
    def ask(self, n: int, budget_remaining: float) -> tuple[X, fidelity]  # may return < n
    def tell(self, X, fidelity, y, sigma) -> None
    def predict(self, X) -> Prediction   # mean (n,m), var (n,m) | None
```
- Runner loop: `ask(5, remaining)` → `evaluate` → `tell` → `predict(test)` → log a row. It stops when the budget is exhausted or `ask` returns nothing.
- A batch whose cost exceeds the remaining budget is an **error**. The runner never truncates it silently.
- The noise `rng` is derived from (seed, batch index), so every run is exactly reproducible.

## Layout
```
src/surrogatemodeling/
  core/      protocols.py, distributions.py, runner.py, metrics.py
  problems/  analytic.py (+ noise wrapper, dimension padding), toymc/
  methods/   gp_fixed.py, pce.py, saas.py, adaptive.py, screening.py
  report/    plots.py, ranking.py
configs/     experiments/*.toml   (methods × problems × seeds × budget)
results/     (gitignored) parquet per run + cached test sets
tests/
```
- CLI: `uv run sm run configs/experiments/full.toml` and `uv run sm report results/`.
- Results: one Parquet file per (method, problem, seed), with one row per batch: cost spent, error per output, coverage, NLL, fit time.
- Reports: static matplotlib PNGs (log-log error vs cumulative cost, median + IQR), plus a rank table.
- Execution: a local process pool over (method, problem, seed) on 12 cores, one thread each. SAAS runs on CPU (see Methods, As built).

## Dependencies
numpy, scipy, torch, gpytorch, botorch, scikit-learn, SALib, pandas, pyarrow, matplotlib, pytest. Python 3.13 via uv. Check each package's 3.13 compatibility when adding it.

## Testing
- **Unit tests:**
  - The noise wrapper's empirical σ matches the declared σ.
  - Toy MC k-eff converges, and its spread scales as 1/√N.
  - Each method reaches < 2% NRMSE on a smooth 2D function given ample data (a sanity bar; EPIG levels off near 1% in the near-noiseless limit).
  - Runner budget accounting is exact.
- **Smoke test:** `smoke.toml` (2 problems × 2 methods × 2 seeds, tiny budget) runs end to end in under 1 minute, as a pytest.

## Steps (thin vertical slice first)
1. ✅ Core protocols, distributions, runner and metrics, plus one analytic problem, the Sobol-GP baseline and one plot. Proves the whole pipeline end to end.
2. ✅ Full analytic suite: noise wrapper, padded dimensions, multi-fidelity cost model.
3. ✅ Toy MC problem with cached truth.
4. ✅ Remaining methods: PCE → adaptive GP (IV, EPIG) → cost-aware adaptive → screening → SAAS.
5. Ranking and report, then the full run (7 methods × ~6 problems × 10 seeds ≈ 420 runs).

## Future Notes
* Documentation of the problems and modeling methods
* EPIG is in v1 as method #4b
* HIPE with EPIG: HIPE replaces the Sobol seed phase of the hybrids (#4c); evaluate once the baselines work
* Actual OpenMC reactor models and cross sections and uncertainties
* Live dashboard
