"""The ask → evaluate → tell → score loop for one (problem, method, seed)."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from surrogatemodeling.core.live import RunLog
from surrogatemodeling.core.metrics import all_metrics
from surrogatemodeling.core.protocols import Method, Problem

# Independent random streams derived from the run seed, so a method's choices never
# perturb the simulator noise and every run is exactly reproducible.
_METHOD_STREAM, _NOISE_STREAM = 0, 1
_COST_TOL = 1e-9


class BudgetExceededError(RuntimeError):
    pass


def method_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, _METHOD_STREAM])


def noise_rng(seed: int, batch: int) -> np.random.Generator:
    return np.random.default_rng([seed, _NOISE_STREAM, batch])


def _diagnostics(method: Method) -> str | None:
    """JSON of the method's optional diagnostics() for the batch just finished."""
    fn = getattr(method, "diagnostics", None)
    return json.dumps(fn(), default=float) if fn is not None else None


def run(
    problem: Problem, method: Method, seed: int, budget: float, batch_size: int = 5, log: RunLog | None = None
) -> pd.DataFrame:
    """Run until the budget is spent or the method stops asking. One row per batch.

    With `log`, each row is also appended to the run's live log as it is produced.
    """
    spec = problem.spec
    start = time.perf_counter()
    X_test, Y_test = problem.test_set()
    method.setup(spec, budget, method_rng(seed))

    rows = []
    spent, n_evals, batch = 0.0, 0, 0
    while budget - spent > _COST_TOL:
        t0 = time.perf_counter()
        X, fid = method.ask(batch_size, budget - spent)
        ask_time = time.perf_counter() - t0
        if len(X) == 0:
            break
        if len(X) > batch_size:
            raise ValueError(f"ask({batch_size}) returned {len(X)} points")
        fid = np.asarray(fid)
        if not np.issubdtype(fid.dtype, np.integer) or np.any(fid < 0) or np.any(fid >= len(spec.fidelities)):
            raise ValueError(f"fidelity must be config indices in [0, {len(spec.fidelities)}), got {fid}")
        cost = float(sum(spec.cost(f) for f in fid))
        if cost > budget - spent + _COST_TOL:
            raise BudgetExceededError(f"batch {batch} costs {cost:.6g} but only {budget - spent:.6g} remains")

        t0 = time.perf_counter()
        obs = problem.evaluate(X, fid, noise_rng(seed, batch))
        sim_time = time.perf_counter() - t0
        method.tell(X, fid, obs.y, obs.sigma)
        spent += cost
        n_evals += len(X)

        t0 = time.perf_counter()
        pred = method.predict(X_test)
        predict_time = time.perf_counter() - t0

        row = {
            "batch": batch,
            "n_evals": n_evals,
            "cost": spent,
            "elapsed": time.perf_counter() - start,
            "ask_time": ask_time,
            "sim_time": sim_time,
            "predict_time": predict_time,
            **all_metrics(pred, Y_test, spec.output_names),
            "diagnostics": _diagnostics(method),
        }
        rows.append(row)
        if log is not None:
            log.batch(row)
        batch += 1
    return pd.DataFrame(rows)
