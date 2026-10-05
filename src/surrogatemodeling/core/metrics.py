"""Error metrics against noise-free truth on a test set drawn from the input distribution.

Because test points are samples of the input pdf, plain averages are pdf-weighted.
All functions return one value per output, shape (m,).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from surrogatemodeling.core.protocols import Prediction


def nrmse(pred: Prediction, Y: np.ndarray) -> np.ndarray:
    """RMSE normalized by the standard deviation of the truth (1.0 == predicting the mean)."""
    return np.sqrt(np.mean((pred.mean - Y) ** 2, axis=0)) / np.std(Y, axis=0)


def max_error(pred: Prediction, Y: np.ndarray) -> np.ndarray:
    """Max absolute error, normalized by the standard deviation of the truth."""
    return np.max(np.abs(pred.mean - Y), axis=0) / np.std(Y, axis=0)


def coverage(pred: Prediction, Y: np.ndarray, level: float = 0.95) -> np.ndarray:
    """Fraction of truths inside the central `level` predictive interval; NaN without variance."""
    if pred.var is None:
        return np.full(Y.shape[1], np.nan)
    z = stats.norm.ppf(0.5 + level / 2)
    return np.mean(np.abs(pred.mean - Y) <= z * np.sqrt(pred.var), axis=0)


def nll(pred: Prediction, Y: np.ndarray) -> np.ndarray:
    """Mean Gaussian negative log-likelihood of the truth; NaN without variance."""
    if pred.var is None:
        return np.full(Y.shape[1], np.nan)
    var = np.maximum(pred.var, 1e-300)
    return np.mean(0.5 * (np.log(2 * np.pi * var) + (pred.mean - Y) ** 2 / var), axis=0)


def ncrps(pred: Prediction, Y: np.ndarray) -> np.ndarray:
    """Mean continuous ranked probability score, normalized by the truth's std.

    A proper scoring rule: rewards predictions that are accurate *and* honestly uncertain,
    can't be improved by inflating or shrinking the error bars, and (unlike NLL) is
    comparable across outputs and robust to a single overconfident point. Gaussian closed
    form; without a variance it reduces to the mean absolute error.
    """
    err = Y - pred.mean
    if pred.var is None:
        crps = np.abs(err)
    else:
        sd = np.sqrt(np.maximum(pred.var, 1e-300))
        z = err / sd
        crps = sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps, axis=0) / np.std(Y, axis=0)


def all_metrics(pred: Prediction, Y: np.ndarray, output_names: list[str]) -> dict[str, float]:
    """Flat {'<metric>/<output>': value} dict for one evaluation."""
    out = {}
    for metric in (nrmse, max_error, coverage, nll, ncrps):
        for name, v in zip(output_names, metric(pred, Y), strict=True):
            out[f"{metric.__name__}/{name}"] = float(v)
    return out
