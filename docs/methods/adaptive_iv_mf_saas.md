# #8 `adaptive_iv_mf_saas`: cost-aware sampling with a SAAS surrogate

| | |
|---|---|
| Design, fidelity, acquisition | As `pq_matern` (#7s + prequential calibration + Matérn): unchanged |
| Model (predictions) | SAAS fully Bayesian GP, hyperparameters sampled by NUTS on a subsample, each sample conditioned on all the data |
| Uncertainty | Yes (moment-matched mixture over 8 hyperparameter samples) |
| Code | [saas_mf.py](../../src/surrogatemodeling/methods/saas_mf.py) |

## Why
In the SAAS trial, high-fidelity SAAS beat the plain GP on every output at the same budget and was much better calibrated (coverage 0.89–0.97), but it had far less data than the cost-aware methods. This combines SAAS's calibration with cost-aware data.

## How it works
- **Acquisition:** the cost-aware co-kriging model of `pq_matern` chooses (point, fidelity) as before, so only the *surrogate being evaluated* changes.
- **Subsample NUTS:** NUTS cost grows ~n³, and these runs reach ~2,000 points. Hyperparameters are sampled on ≤ 250 points (every high-fidelity point plus a random draw of the rest), refit when the data grows 1.5×. Settings: 128 warmup, 64 samples, thinning 8.
- **Full-data conditioning:** each hyperparameter sample's GP is conditioned on all the data (botorch's MCMC samples are captured and reloaded into a full-data model).
- **Noise:** the reported σ² × the per-config scale learned by the co-kriging model, since SAAS can only take fixed noise and batch σ is under-reported.
- **Fidelity:**
  - `adaptive_iv_mf_saas` adds one input per varying knob (log distance from HF, scaled to [0, 1]) and predicts at HF. SAAS's product kernel only partly shares information across fidelities unless it learns the knob dimension is irrelevant.
  - `adaptive_iv_mf_saas_noknob` treats every fidelity as the same function and relies on the corrected noise.

## First check (Borehole, budget 40, seed 0)

| Method | Time | NRMSE | Coverage | NCRPS |
|---|---|---|---|---|
| `pq_matern` | 19 s | 0.076 | 1.00 | 0.043 |
| `_saas_noknob` | 233 s | 0.081 | 0.82 | 0.042 |
| `_saas` | 238 s | 0.092 | 0.94 | 0.048 |

These are close on an unbiased, honest-σ problem.

## Trial status: cancelled (2026-10-06)
The 3-seed trial (`saas_mf_trial.toml`) was **killed twice by systemd-oomd** and then cancelled at the user's request.
- **First attempt, 4 workers:** the unit reached 20.3 GB after ~20 min (runs at ~40% of budget).
- **Second attempt, 3 workers, chunked prediction:** memory still rose ~1 GB per minute per run (5.6 → 20.6 GB in 12 min).

Chunking the test-set prediction didn't stop the growth, so the main cause is elsewhere and hasn't been identified. Candidates are the per-batch rebuild of the full-data SAAS model (a new pyro model each batch) or memory retained across NUTS refits. Profile a single run before trying again. No results were produced.
