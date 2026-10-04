"""Greedy batch acquisition on a GP's joint posterior covariance.

Works on K, the latent posterior covariance over [reference points; candidates] in
standardized output units. Reference points are samples of the input distribution, so
both scores target the pdf-weighted error the benchmark ranks on:

- "iv":   reduction of integrated posterior variance over the reference set
- "epig": expected predictive information gain about a high-fidelity observation y* at the
          reference points (Bickford Smith et al., 2023), averaged over the reference set

Observing candidate c with noise τ² changes the posterior exactly by a rank-one update,
which is independent of the observed value — so a batch is chosen greedily without
fantasies or refits.
"""

from __future__ import annotations

import numpy as np


def iv_score(K: np.ndarray, n_ref: int, tau2: np.ndarray, target_tau2: float = 0.0) -> np.ndarray:
    """(C,) reduction in mean posterior variance over the reference set."""
    k_rc = K[:n_ref, n_ref:]
    k_cc = np.diag(K)[n_ref:]
    return (k_rc**2).mean(axis=0) / (k_cc + tau2)


def epig_score(K: np.ndarray, n_ref: int, tau2: np.ndarray, target_tau2: float = 0.0) -> np.ndarray:
    """(C,) mean mutual information between a noisy candidate observation and each y*(x_ref).

    target_tau2 is the noise on the target y*. Without it (latent f* targets) the score
    depends only on correlations, not on how much variance is left to remove.
    """
    k_rc = K[:n_ref, n_ref:]
    diag = np.diag(K)
    k_rr, k_cc = diag[:n_ref] + target_tau2, diag[n_ref:]
    rho2 = k_rc**2 / (np.maximum(k_rr, 1e-300)[:, None] * (k_cc + tau2)[None, :])
    return -0.5 * np.log1p(-np.clip(rho2, 0, 1 - 1e-12)).mean(axis=0)


SCORES = {"iv": iv_score, "epig": epig_score}


def rank_one_update(K: np.ndarray, idx: int, tau2: float) -> np.ndarray:
    """Posterior covariance after observing point `idx` with noise variance tau2."""
    v = K[:, idx]
    return K - np.outer(v, v) / (K[idx, idx] + tau2)


def greedy_batch(
    covs: list[np.ndarray],
    n_ref: int,
    tau2: np.ndarray,
    costs: np.ndarray,
    k: int,
    budget: float,
    score: str = "iv",
    cost_aware: bool = False,
    target_tau2: np.ndarray | None = None,
    scores_out: list[float] | None = None,
) -> list[tuple[int, int]]:
    """Choose up to k (candidate, fidelity-option) pairs.

    covs:  per output, (R + C, R + C) latent covariance (standardized units)
    tau2:  (m, F, C) observation noise variance per output, fidelity option and candidate
    costs: (F,) cost of each fidelity option
    target_tau2: (m,) noise on EPIG's targets y*; ignored by "iv"
    scores_out: if given, the winning score of each pick is appended to it
    """
    fn = SCORES[score]
    K = [c.copy() for c in covs]
    target = np.zeros(len(K)) if target_tau2 is None else target_tau2
    chosen = []
    for _ in range(k):
        affordable = costs <= budget + 1e-9
        if not affordable.any():
            break
        total = sum(np.stack([fn(K[o], n_ref, tau2[o, f], target[o]) for f in range(len(costs))]) for o in range(len(K)))
        if cost_aware:
            total = total / costs[:, None]
        total[~affordable] = -np.inf
        f, c = np.unravel_index(np.argmax(total), total.shape)
        if scores_out is not None:
            scores_out.append(float(total[f, c]))
        for o in range(len(K)):
            K[o] = rank_one_update(K[o], n_ref + c, tau2[o, f, c])
        chosen.append((int(c), int(f)))
        budget -= costs[f]
    return chosen
