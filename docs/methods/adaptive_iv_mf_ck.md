# #7 `adaptive_iv_mf_ck`: co-kriging with a bias-aware cost-aware acquisition

| | |
|---|---|
| Design | Hybrid: 25% high-fidelity Sobol seed, then adaptive greedy batches |
| Fidelity | Chooses per point from the problem's whole fidelity menu |
| Model | Per output: f(x) plus a per-config bias GP b_c(x), learned extra noise, fixed per-point noise from σ² |
| Uncertainty | Yes (latent HF variance) |
| Code | [cokriging.py](../../src/surrogatemodeling/methods/cokriging.py) |

## Why it exists
Once cycles are fidelity knobs, cheap configs can be **biased**, not just noisy:
- an unconverged source with too few inactive cycles;
- population bias with few particles;
- the max-of-noisy-bins bias in peaking factors.

#5b treats everything as noise, which can't absorb a smooth bias. On a biased test problem it was confidently wrong (coverage 0.2). See [Learnings](../Learnings.md#bias-aware-multi-fidelity).

## Model
For each output:

```
y_c(x) = f(x) + b_c(x) + ε,   b_HF ≡ 0
f   ~ GP(0, ARD-RBF)                        (botorch's dimension-scaled prior)
b_c ~ GP(0, s_c · RBF_iso(x, x')), independent across configs
```

- Each config c learns its own bias variance s_c; high fidelity's is pinned to 0.
- The bias processes share one isotropic lengthscale. Bias from unconverged sources is smooth, and few parameters suit noisy data.
- The model inputs carry the config index as a last column, and the kernel is `AdditiveKernel(f_kernel, ConfigBiasKernel)`.
- Predictions are of f, which is the high-fidelity output.

## Acquisition
- **Joint covariance:** the GP's covariance is taken over the 512 reference points *at high fidelity* (i.e. f) plus every **(candidate, config) row**. The number of candidates is about 1024 / number of configs, which keeps the matrix size steady.
- **Scoring:** each row is scored by integrated-variance reduction per unit cost, as in #5. The batch is chosen greedily with exact rank-one updates over the whole matrix.
- **Why the joint matrix matters:** because each row carries its config's bias process, the updates capture that **bias is correlated**. After a few cheap runs in a region, more cheap runs there no longer reduce uncertainty in f, because they share the same bias.
  - An earlier version added s_c to the noise as if it were independent. It kept buying only the cheapest config, and its error was about 2.5× higher on the biased test (0.09–0.10 vs 0.035–0.041).

## Diagnostics
Same as #5b, plus `bias_sd`: the learned bias sd per output and per config, in output units.

## Known limitation
A config that is rarely used has a poorly identified bias variance, which can drift large: about 5× the truth on the test problem. The method then avoids that config. That's conservative, but it may leave a useful option untried.

## Results
- **Biased Borehole test (2 seeds, budget 60):** NRMSE 0.035–0.041 vs 0.125–0.134 for #5b; coverage 0.92–0.96 vs 0.19–0.23.
- The full 10-seed comparison on the MCNP-like problems is in the `costaware` experiment (see [README](README.md#results-so-far)).
