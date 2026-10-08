# Deep ensembles: `sobol_de` and `adaptive_iv_mf_de_safe`

The first neural surrogates (backlog item 12, [TestBacklog.md](../TestBacklog.md)).

| | `sobol_de` | `adaptive_iv_mf_de_safe` |
|---|---|---|
| Design | The baseline's high-fidelity Sobol design (same points and noise as `sobol_gp`) | The leader's: `safe_pooled`'s co-kriging GP still chooses every point and fidelity |
| Fidelity | High only | Safe menu (particles only); knobs enter the network as input features (HF = 0) |
| Model | Deep ensemble of 5 MLPs | Same |
| Compare with | `sobol_gp`, `sobol_gp_matern`, `sobol_saas` | `adaptive_iv_mf_cks_pq_matern_safe_pooled` |
| Code | [neural.py](../../src/surrogatemodeling/methods/neural.py) | [neural.py](../../src/surrogatemodeling/methods/neural.py) |

Both arms keep the design fixed to an existing method's, so any difference comes from the surrogate alone. On OpenMC they reuse cached simulations, which makes them nearly free to validate.

## Model
- **Network:** 5 two-hidden-layer SiLU MLPs (width 32), inputs → all outputs jointly. They are trained as one batched network: the weights carry a member axis, so 5 members cost about as much as 1.
- **Diversity:** each member has its own random initialization and mini-batch order (Lakshminarayanan et al., 2017).
- **Loss:** Gaussian NLL with a per-point noise variance `a_o · σ_i² + b_o`, using the reported MC σ. `a_o` (a pooled σ scale, as in `safe_pooled`) and `b_o` (extra noise) are learned per output. Weight decay is scaled by 1/n.
- **Prediction:** the mean of the members' means is the noise-free output; the spread of the members' means is the (epistemic) variance.
- **Calibration:** prequential, as in `pq`. Each new batch is predicted before training on it, and the sd is scaled so 95% of the **last 25** residuals are covered. Unlike the GP version, the scale **may also narrow** (range [0.2, 10]). An ensemble's spread has no natural scale, and at the tuned weight decay it is too wide: coverage ~1.0.
- **Training:** full retrain (1,500 Adam steps) when the data has grown 1.5× since the last one; otherwise a warm continuation (300 steps).

## Tuning (Borehole-30D, seed 0 only; `toymc_axial` and OpenMC held out)

| Setting | Final NRMSE |
|---|---|
| First guess (width 64, weight decay 1e-3, lr 3e-3) | 0.222 |
| Strong weight decay (1.0), width 32, lr 1e-2 | **0.108** |
| `sobol_gp`, same seed | 0.075 |

Strong regularization was the main lever. With 100 points in 30 dimensions, the network otherwise overfits.

**Calibration window:** with the GP's 300-point window, the scale stayed pinned at 10× (NCRPS 0.12), because early residuals from bad models dominated. With 25 points it fell to 0.06–0.08.

**Known limitation:** the strong regularization underfits small, low-dimensional problems (2-D, 60 points: NRMSE 0.127 vs < 0.02 for GPs). The method test uses a looser bar for neural arms.

## Results (2026-10-06, 3 seeds; screen in `calib_screen`, OpenMC in `openmc_validation_nn`)

**Verdict: the GP wins on accuracy everywhere, at equal training data. The cost-aware network also fails calibration badly.** Not carried forward as-is.

Final NRMSE, seed median (coverage in parentheses):

| Problem · output | `sobol_gp` | **`sobol_de`** | `safe_pooled` (GP) | **`de_safe`** |
|---|---|---|---|---|
| OpenMC k-eff | 0.264 (0.86) | 0.293 (0.96) | 0.111 (0.96) | 0.141 (**0.18**) |
| OpenMC axial offset | 0.153 (0.91) | 0.232 (0.92) | 0.079 (0.97) | 0.101 (**0.14**) |
| OpenMC peaking | 0.306 (0.86) | 0.383 (0.96) | 0.148 (0.98) | 0.261 (**0.30**) |
| OpenMC capture/fission | 0.241 (0.88) | 0.292 (0.98) | 0.110 (0.95) | 0.143 (**0.13**) |
| toy k-eff | 0.142 (0.84) | 0.175 (0.93) | 0.047 (0.97) | 0.061 (0.16) |
| toy axial offset | 0.179 (0.90) | 0.380 (0.94) | 0.125 (0.96) | 0.226 (0.08) |
| toy peaking | 0.407 (0.87) | 0.621 (0.77) | 0.256 (0.96) | 0.380 (0.07) |
| toy capture/fission | 0.177 (0.81) | 0.193 (0.96) | 0.054 (0.97) | 0.072 (0.14) |
| Borehole-30D | 0.086 (0.87) | 0.109 (0.96) | 0.042 (0.97) | 0.063 (0.26) |

On OpenMC AUC-NCRPS, the network is worse than the GP on the same points by 20–120% (`sobol_de` vs `sobol_gp`: 0.355–0.534 vs 0.201–0.335; `de_safe` vs `safe_pooled`: 0.280–0.440 vs 0.137–0.248).

**What we learned**
1. **At 100–620 points in 14–30 dimensions, a GP is the better learner.** The network loses by 10–110% in final NRMSE on identical data, and by more on the axial outputs, which have the sharpest features. That matches the usual picture: networks pay off with thousands of points or highly structured or field outputs, not here.
2. **The cost-aware data still helps the network a lot.** `de_safe` beats `sobol_gp` on every OpenMC output (k-eff 0.141 vs 0.264), so the safe-menu design is valuable whatever surrogate is used.
3. **`sobol_de` is calibrated, but `de_safe` is not, for two reasons:**
   - Its recent calibration residuals come mostly from noisy cheap runs, which the noise term alone covers, so the scale drops to its 0.2 minimum (seen on most outputs and seeds). At full fidelity, where there's no noise term, the error bars are then far too narrow.
   - Its learned noise multipliers are erratic (0.01–77 across seeds and outputs, vs the GP's stable ~34 for axial offset), so it partly explains its own misfit as noise.

   The fix is backlog #8: calibrate on full-fidelity residuals (or with the noise term removed) and keep the scale ≥ 1.
