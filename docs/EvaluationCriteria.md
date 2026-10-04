# Evaluation criteria

How methods are scored, and how to read the numbers.
- **[Part 1](#part-1-the-four-metrics)** covers the **four metrics** computed after every batch of every run: NRMSE, coverage, NLL and max error.
- **[Part 2](#part-2-the-ranking-report)** covers the **three tables** in each `ranking.md`, which summarize those metrics across seeds and budget.
- **[Part 3](#part-3-diagnosing-a-model-from-all-four)** shows how to diagnose a model by reading the metrics together.

Code: [metrics.py](../src/surrogatemodeling/core/metrics.py), [ranking.py](../src/surrogatemodeling/report/ranking.py).

## The test set behind every metric

Every metric compares the method's predictions with **truth** on a fixed, cached **test set** for each problem:
- **Points:** 2,000 points drawn from the problem's **input distribution**, not uniformly over the training box. A plain average over these points is therefore an average weighted by how likely each input is, which is what matters when the surrogate is used for UQ.
- **Truth:** noise-free for the analytic problems. For Monte Carlo problems it's a high-precision reference run (toy MC: 100× high-fidelity histories; `toymc_axial`: 10× particles and 2× inactive cycles).

After every batch, the runner asks the method to `predict` all 2,000 points and records the four metrics **for each output**. They appear:
- in each run's Parquet file, as `nrmse/<output>`, `coverage/<output>`, `nll/<output>` and `max_error/<output>`;
- in the dashboard's **Metric** selector;
- in the plots and ranking tables (NRMSE and coverage).

**Notation:** for test point *i* and one output, the truth is yᵢ, the method's prediction is μᵢ, and its predictive variance is σᵢ², the variance of the noise-free output, i.e. the method's own uncertainty about f(xᵢ). s_y is the standard deviation of the truth across the test set.

---

## Part 1: the four metrics

### NRMSE: typical accuracy (*the primary metric*)

```
NRMSE = sqrt( mean_i (μ_i − y_i)² ) / s_y
```

**What it measures:** the typical size of the prediction error, relative to how much the output naturally varies over the input distribution.

**Scale and reference points:**

| NRMSE | Meaning |
|---|---|
| ≥ 1.0 | No better than always predicting the average. The model has learned nothing useful (common after the first batch) |
| 0.3 | Captures the main trends; errors are 30% of the output's natural spread |
| 0.1 | Good: errors are a tenth of the natural spread |
| ≤ 0.03 | Very good; usually limited by simulator noise or the truth's own precision |

Because it's normalized, NRMSE is **comparable across outputs and problems**: 0.1 on k-eff and 0.1 on axial offset are equally good *relative to how much each varies*. Lower is better.

**How to interpret it for a model:**
- **Falling steadily with cost** (a straight line on the log-log plots): the method is still learning. More budget would keep helping.
- **Flattening out** (a plateau): something besides data limits the model. Common causes:
  - the simulator's noise floor;
  - a model that can't represent the function (e.g. PCE degree too low);
  - **trusting wrong noise information**, as when #5 plateaued at 0.27 on the toy MC power ratio because it believed under-reported σ.
- **Higher than the baseline on one output only:** that output has a specific difficulty, e.g. axial offset with its 4× under-reported σ.

**What it doesn't tell you:** whether errors are spread evenly or concentrated (see max error), or whether the model *knows* how wrong it is (see coverage and NLL).

### Coverage: are the error bars honest?

```
coverage = fraction of test points with |μ_i − y_i| ≤ 1.96 · σ_i
```

**What it measures:** how often the truth falls inside the method's own nominal 95% interval. It's only defined for methods that report variance; PCE shows "—".

**Scale and reference points:**

| Coverage | Meaning for the model |
|---|---|
| ≈ 0.95 | **Well calibrated**: its error bars can be taken at face value |
| 0.85–0.93 | Mildly **overconfident**: real errors are a bit larger than claimed. Typical of GPs here |
| < 0.8 | Clearly overconfident. Its uncertainty estimates **understate** risk and shouldn't feed UQ uncorrected |
| ≪ 0.5 | **Confidently wrong**, e.g. #5b at 0.2 on the biased-Borehole test. The model is sure of a biased answer |
| > 0.98 | **Underconfident**: error bars wider than needed. Safe, but less informative than they could be |

**How to interpret it for a model:**
- Low coverage usually means the model's **noise or bias model is wrong**:
  - it trusted under-reported σ;
  - it treated a smooth bias as independent noise;
  - it dropped inputs without accounting for them (the early `screen_gp` bug).
- Coverage is about **honesty, not accuracy**. A model with NRMSE 0.5 and coverage 0.95 is bad but honest. One with NRMSE 0.05 and coverage 0.6 is good but overconfident, and the second is the riskier one for UQ.
- It can be **gamed**: inflating every σᵢ raises coverage while making the model less useful. That's why it's reported but never used for ranking, and why NLL is useful alongside it.

### NLL: honesty and sharpness in one number

```
NLL = mean_i [ ½·log(2π σ_i²) + ½·(μ_i − y_i)² / σ_i² ]
```

**What it measures:** the Gaussian negative log-likelihood of the truth under the method's predictive distribution, averaged over the test set. Lower is better, and it can be negative. Each term has a job:
- **½·log(2πσᵢ²)** rewards **sharpness**: narrower error bars lower it.
- **½·(μᵢ − yᵢ)²/σᵢ²** punishes **overconfidence**: errors that are large *relative to the claimed σᵢ* raise it **quadratically**.

So NLL is lowest when the method is both accurate *and* reports error bars that match its real errors. It can't be gamed the way coverage can:
- inflating σ raises the first term;
- shrinking σ explodes the second.

**Scale and reference points:**
- **NLL is in the units of the output**, so it's **only comparable between methods on the same (problem, output)**. Never compare NLL across outputs or problems.
- **Compare differences:** a method whose NLL is lower by Δ assigns the truth e^Δ times more probability density per test point on average. Δ = 0.5 is a clear improvement; Δ = 2 is a large one.
- **A perfectly calibrated reference:** if the errors truly were Gaussian with sd σ, the expected NLL would be ½·log(2πσ²) + ½. A method well above that for its typical σ is miscalibrated.

**How to interpret it for a model:**
- **NLL very large or rising** while NRMSE is fine: a few points have errors far outside their claimed σ, i.e. **severe local overconfidence**. Check coverage, which will usually also be low.
- **NLL better than a rival despite similar NRMSE:** the better-NLL method's uncertainty is more useful. It knows *where* it's uncertain.
- **Coverage ≈ 0.95 but NLL worse than a rival:** both are honest, but the rival is **sharper** (narrower intervals for the same honesty), so the rival is better.
- Not available for methods without variance (PCE).

### Max error: the worst case

```
max_error = max_i |μ_i − y_i| / s_y
```

**What it measures:** the single largest prediction error over the 2,000 test points, normalized like NRMSE.

**Scale and reference points:**
- It's always ≥ NRMSE. For smooth errors spread fairly evenly, **max error ≈ 3–5 × NRMSE** is typical.
- **max error ≫ 5 × NRMSE** means the errors are **concentrated**: the model is good almost everywhere and badly wrong somewhere.

**How to interpret it for a model:**
- **High max error with low NRMSE** points to a localized failure:
  - an unsampled corner of the input space;
  - a sharp feature, e.g. the rod tip crossing a peaking bin;
  - extrapolation outside the training box, since test points come from the full input distribution and a few fall outside the box for normal inputs.
- This matters for **safety-type uses** of the surrogate, e.g. a peaking-factor limit, where the worst case counts more than the average.
- It's a **noisy** statistic: one test point decides it, so look at it across seeds before concluding anything.
- It isn't used for ranking. Space-filling designs (Sobol) tend to have lower max error than adaptive designs that concentrate points where the pdf is high.

---

## Part 2: the ranking report

`uv run sm report results/<experiment>` writes `ranking.md` and `ranking.csv`. Each table has one row per method and one column per (problem, output). Every run produces an error-vs-cost curve, with cost in high-fidelity-run equivalents, for each of its seeds (usually 10). The report condenses those curves into three criteria:

| Criterion | Built from | Question it answers | Ranked? |
|---|---|---|---|
| [AUC-NRMSE](#1-auc-nrmse) | NRMSE over the whole budget | How accurate *across the budget*? | **Yes**: the mean rank |
| [Final NRMSE](#2-final-nrmse) | NRMSE at the end | How accurate *when the budget runs out*? | No |
| [Final 95% coverage](#3-final-95-coverage) | Coverage at the end | Are the error bars *honest*? | No |

NLL and max error aren't in the report tables. Use the run files or the dashboard's Metric selector for them.

### 1. AUC-NRMSE

**Definition**, for each (problem, output, method):
1. **Grid:** put every seed's NRMSE curve on a shared, **log-spaced cost grid** (60 points). Between batches the curve holds its last value.
2. **Range:** the grid runs from the first cost at which *every* run has a score to the smallest final cost of any run, so all methods are compared over the same span.
3. **Median:** take the seed median at each grid point.
4. **Average:** `AUC-NRMSE = 10^mean(log10 median NRMSE)`. That's the area under the log-log curve divided by the width of the log-cost range, reported as a **geometric-mean NRMSE**, so it reads on the same scale as NRMSE.

**Ranking:** within each (problem, output), methods are ranked by AUC-NRMSE (1 = best; ties share the average rank). The **mean rank** at the top of `ranking.md` averages those ranks over all (problem, output) pairs.

**Why it's the primary criterion:** the real budget for an MCNP campaign isn't known in advance, so a method that gets good early is worth more than one that only catches up at the end.
- **Log-spaced cost:** every factor of cost counts equally, so going from 5 to 10 runs counts as much as going from 50 to 100.
- **Geometric mean:** stops one bad stretch from dominating, and treats halving the error from 0.2 to 0.1 the same as from 0.02 to 0.01.

**Example (`full`, Borehole-30D):** baseline `sobol_gp` scores 0.349; cost-aware `adaptive_iv_mf_xn` scores 0.248. That's about 30% lower on average over the whole budget, not just at the end.

**Caveats:**
- AUC values are only comparable **within one report**, because the shared cost range depends on which methods and seeds are in it.
- A problem with 3 outputs (`toymc`) carries 3× the weight of a single-output problem in the mean rank.

### 2. Final NRMSE

**Definition:** for each seed, the NRMSE in the run's **last row** (when the budget is spent), then the **median over seeds**.

**Why it's reported:** it answers "if I spend the whole budget, how good is the surrogate?", and it shows the end state that AUC blends with the early part of the curve.

**Example (`full`, toy MC power ratio):**

| Method | Final NRMSE |
|---|---|
| `adaptive_iv_mf` | 0.272 |
| `adaptive_iv_mf_xn` | 0.105 |
| `sobol_gp` (baseline) | 0.155 |

This is where #5's under-reported-σ failure was most visible.

**Caveats:**
- It ignores how the method got there. Two methods with the same final NRMSE can have very different curves; use AUC or the plots for that.
- It's a median over seeds. The IQR bands in `results/<experiment>/plots/` show the spread.

### 3. Final 95% coverage

**Definition:** the coverage of each seed's final model, then the **median over seeds**. "—" means the method reports no variance.

**Why it's reported:** the surrogate feeds UQ. Accurate predictions with dishonest error bars understate uncertainty in everything built on them. Read it with the coverage scale in [Part 1](#coverage-are-the-error-bars-honest).

**Example:**
- *Biased-Borehole test:* `adaptive_iv_mf_xn` had **0.2** (confidently wrong); co-kriging reached 0.92–0.96.
- *`full`:* coverage ranged 0.74–0.97, so most GP methods run slightly overconfident.

**Caveat:** it's reported but deliberately not ranked, because inflating error bars raises coverage while hurting usefulness. Check NLL to tell honest-and-sharp from honest-but-vague.

---

## Part 3: diagnosing a model from all four

| NRMSE | Coverage | NLL (vs rivals) | Max error | What it means for the model |
|---|---|---|---|---|
| Low | ≈ 0.95 | Lowest | ≈ 3–5× NRMSE | **Healthy:** accurate, honest, sharp, errors evenly spread. Trust it for UQ |
| Low | ≪ 0.95 | High | — | **Overconfident:** good predictions, error bars too narrow. Fix the noise/bias model (learned noise, σ scales, co-kriging) or calibrate (#7c) before using it for UQ |
| Low | ≈ 0.95 | Higher than a rival | — | **Honest but vague:** error bars wider than necessary; a sharper rival is better |
| Low | ≈ 0.95 | — | ≫ 5× NRMSE | **Localized failure:** good on average, badly wrong in a corner (unsampled region, extrapolation, sharp feature) |
| High, plateaued | ≪ 0.95 | High | — | **Trusting bad information:** under-reported σ or unmodeled bias being fit as signal (#5 on the toy MC power ratio) |
| High, falling | ≈ 0.95 | — | — | **Honest and still learning:** more budget will help |
| High, plateaued | ≈ 0.95 | — | — | **Model limit:** honest, but can't represent the function or is at the noise floor; try a more flexible model or more precise runs |

The success bar for new methods in [CostAwarePlan.md](CostAwarePlan.md) uses these together: **beat #5b on the MCNP-like problems by AUC/final NRMSE, with coverage at least as good.** NLL breaks ties between equally covered methods, and max error flags anything unsafe for worst-case use.
