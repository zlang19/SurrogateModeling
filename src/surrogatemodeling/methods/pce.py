"""Method #2: sparse polynomial chaos (hybrid LARS) on a fixed Sobol design.

Basis: orthonormal Legendre polynomials of the inputs mapped from the training box to
[-1, 1], total degree <= p, interaction order <= 2. p goes to 3 (5 when d <= 10), which
keeps the 40D basis at ~2.5k terms.
Per output and per candidate degree, LARS orders the terms; every prefix of that order is
refit by OLS and scored by its corrected leave-one-out error, and the best (degree, prefix)
wins (hybrid LARS, Blatman & Sudret 2011). No predictive variance is reported.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from numpy.polynomial import legendre
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path

from surrogatemodeling.core.protocols import Prediction
from surrogatemodeling.methods.gp_fixed import FixedDesignMethod

DEGREES = (1, 2, 3)
LOW_DIM_DEGREES = (1, 2, 3, 4, 5)  # affordable when d is small
LOW_DIM = 10
MAX_INTERACTION = 2


def multi_indices(d: int, p: int) -> list[tuple[tuple[int, int], ...]]:
    """Non-constant terms as ((dim, degree), ...) with total degree <= p, <= 2 dims each."""
    terms = [((i, k),) for i in range(d) for k in range(1, p + 1)]
    for i, j in itertools.combinations(range(d), MAX_INTERACTION):
        terms += [((i, a), (j, b)) for a in range(1, p) for b in range(1, p - a + 1)]
    return terms


def design_matrix(Z: np.ndarray, terms, p: int) -> np.ndarray:
    """Z in [-1, 1]^(n, d) -> (n, len(terms)) orthonormal Legendre features."""
    # univariate[k][:, i] = sqrt(2k+1) P_k(z_i)
    univariate = [np.sqrt(2 * k + 1) * legendre.legval(Z, np.eye(p + 1)[k]) for k in range(p + 1)]
    cols = [np.prod([univariate[k][:, i] for i, k in term], axis=0) for term in terms]
    return np.column_stack(cols)


class SparsePCE:
    """One output: LARS-selected Legendre expansion with LOO degree selection."""

    def fit(self, Z: np.ndarray, y: np.ndarray) -> None:
        n, d = Z.shape
        best = (self._ols_loo(np.empty((n, 0)), y)[1], 1, [], np.array([y.mean()]))  # constant model
        for p in LOW_DIM_DEGREES if d <= LOW_DIM else DEGREES:
            terms = multi_indices(d, p)
            Phi = design_matrix(Z, terms, p)
            for active in self._lars_prefixes(Phi, y):
                coef, loo = self._ols_loo(Phi[:, active], y)
                if loo < best[0]:
                    best = (loo, p, [terms[a] for a in active], coef)
        _, self.p, self.terms, self.coef = best

    @staticmethod
    def _lars_prefixes(Phi: np.ndarray, y: np.ndarray):
        """Active sets along the LAR path, smallest first, capped so OLS stays determined."""
        max_terms = len(y) - 3
        if max_terms < 1:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            _, order, _ = lars_path(Phi, y - y.mean(), method="lar", max_iter=max_terms)
        for k in range(1, min(len(order), max_terms) + 1):
            yield np.asarray(order[:k])

    @staticmethod
    def _ols_loo(Phi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Least squares with intercept; returns coef (intercept first) and corrected LOO error.

        The n/(n - P) factor penalizes LOO's optimism when P approaches n.
        """
        n = len(y)
        A = np.column_stack([np.ones(n), Phi])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        H_diag = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
        loo = np.mean((resid / np.maximum(1 - H_diag, 1e-6)) ** 2) * n / max(n - A.shape[1], 1)
        return coef, float(loo)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if not self.terms:
            return np.full(len(Z), self.coef[0])
        return self.coef[0] + design_matrix(Z, self.terms, self.p) @ self.coef[1:]


class SobolPCE(FixedDesignMethod):
    def _z(self, X: np.ndarray) -> np.ndarray:
        return 2 * self.spec.dist.to_unit(X) - 1

    def predict(self, X: np.ndarray) -> Prediction:
        if self.stale:
            Z = self._z(self.data.X)
            self.loc = self.data.y.mean(axis=0)
            self.scale = np.where(self.data.y.std(axis=0) > 0, self.data.y.std(axis=0), 1.0)
            self.models = []
            for j in range(self.spec.n_outputs):
                pce = SparsePCE()
                pce.fit(Z, (self.data.y[:, j] - self.loc[j]) / self.scale[j])
                self.models.append(pce)
            self.stale = False
        Z = self._z(X)
        return Prediction(mean=np.column_stack([m.predict(Z) * s + l for m, s, l in zip(self.models, self.scale, self.loc)]))
