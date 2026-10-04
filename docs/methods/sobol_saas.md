# #3 `sobol_saas`: fully Bayesian SAAS GP

| | |
|---|---|
| Design | Fixed Sobol (same as #1) |
| Fidelity | High only |
| Model | SAAS GP per output: half-Cauchy shrinkage prior on inverse lengthscales, hyperparameters sampled by NUTS |
| Uncertainty | Yes (moment-matched mixture over the MCMC samples) |
| Code | [saas.py](../../src/surrogatemodeling/methods/saas.py) |

## How it works
- **Model:** botorch's `SaasFullyBayesianSingleTaskGP` with fixed per-point noise from σ².
- **Prior:** the sparse axis-aligned subspace prior (Eriksson & Jankowiak, 2021) pushes irrelevant inputs' inverse lengthscales toward zero. That suits many-inputs, few-points problems like ours.
- **Sampling:** NUTS settings are 256 warmup / 128 samples / thinning 16.
- **Prediction:** the moment-matched mixture over the samples.

## Status: not yet benchmarked
The SAAS runs in `full` are **on hold until the user says to run them**.
- Each run takes roughly 20 minutes and about 3 GB.
- All 70 runs take roughly 4 hours with 6 workers, using `configs/experiments/full_saas.toml`.
- A single-seed check on Borehole-30D gave NRMSE 0.087 at 100 points, against 0.091 for `sobol_gp`.

## Operational notes
- **Backend:** botorch 0.18 runs fully Bayesian models on **JAX/NumPyro**, installed via `botorch[fully_bayesian]`. It runs on CPU: no CUDA build of jaxlib is installed.
- **Memory:** every fit has a new data size, so JAX compiles a new NUTS kernel each time. Without `jax.clear_caches()` after each fit, the compiled kernels piled up to about 6 GB per run and helped crash VS Code twice (out of memory). With the cache cleared, the peak is about 3 GB.
- **Concurrency:** cap it with `[limits] sobol_saas = N` in an experiment config. Run it as a memory-capped systemd service (see the README).

## Diagnostics
`nuts_time`, and `median_inverse_lengthscales` (per output, per input; large means relevant).

## When to use
When there are many inputs, few points, and compute isn't the constraint. It's 10–100× slower than an ordinary GP.
