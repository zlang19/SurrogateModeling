# #7s `adaptive_iv_mf_cks`: co-kriging with learned per-config σ scales

| | |
|---|---|
| Design, fidelity, acquisition | As [#7](adaptive_iv_mf_ck.md) |
| Model | #7's per-config bias GP, but the noise is **s_c × reported σ²**, with one learned scale s_c per fidelity config (per output), instead of additive extra noise |
| Uncertainty | Yes (latent HF variance) |
| Code | [cokriging.py](../../src/surrogatemodeling/methods/cokriging.py) (`ConfigScaledNoise`, `noise_scale=True`) |

## Why it exists
The OpenMC study showed that batch-statistics σ is under-reported by a roughly **constant factor per output**: about 4× for axial offset, so about 16× in variance. That factor can also differ between fidelity configs.
- An additive extra-noise term (#5b, #7) is the wrong shape for that. It over-inflates precise points and under-inflates noisy ones.
- On `toymc_axial`, every method that trusted reported σ (or corrected it additively) bought only the cheapest config and lost badly on axial offset to the high-fidelity baseline.

## How it works
- **Likelihood:** a custom noise model, `ConfigScaledNoise`. Each training point's noise is `s_c · σ²_reported`, where c is the point's fidelity config and s_c ≥ 0.2 is learned by marginal likelihood along with the kernel hyperparameters. The fixed per-point variances are stored outside the state dict, so warm-started hyperparameters stay valid as the data grows.
- **Acquisition:** the expected noise of each config is `s_c · median σ²` at that config. Under-reported configs are therefore valued at their *real* information content.

## Diagnostics
As #7, plus `noise_scale`: the learned multiplier per output and config. A value of 25 means "the reported variance is 25× too small".

## Pitfall found while building it
The adaptive methods rebuild their GPs every batch and copy the previous hyperparameters between full refits. The copy initially didn't include the new noise module, so the scales silently reset to their initial value between full refits: learned scales of 0.9 everywhere, and coverage of 0.24–0.34. Any trainable noise model is now part of the warm start.

## Results
- **Borehole test, cheapest config's σ under-reported 5×** (2 seeds, budget 60): it learned scales of 25.9 and 29.5 (truth 25), with NRMSE 0.024–0.027 and coverage 0.86–0.89. It used 318–615 runs, i.e. a mix of fidelities.
- **`costaware` (10 seeds):** mean rank 1.62 (tied with #7). On `toymc_axial`: k-eff **0.061** (#5b 0.130), capture/fission **0.067** (0.120), peaking 0.361 (best), axial offset 0.270 (baseline 0.216). It's the only method that mixes fidelities, with 13% of adaptive picks above the cheapest config. The learned σ scale for axial offset at the cheapest config is 11, about a third of the truth, so axial-offset coverage is still 0.53.
