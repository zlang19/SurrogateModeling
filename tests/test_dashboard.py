import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from surrogatemodeling.core import live
from surrogatemodeling.core.live import RunLog, write_manifest
from surrogatemodeling.dashboard import data as D
from surrogatemodeling.dashboard.server import serve
from surrogatemodeling.report.plots import curves_on_grid


def dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def batch_row(batch, cost, elapsed, err):
    return {"batch": batch, "n_evals": 5 * (batch + 1), "cost": cost, "elapsed": elapsed, "nrmse/y": err,
            "coverage/y": 0.9, "diagnostics": json.dumps({"batch": batch})}


def write_done(exp, problem, method, seed, costs, errs, elapsed=None):
    rows = [batch_row(i, c, (elapsed or costs)[i], e) for i, (c, e) in enumerate(zip(costs, errs))]
    df = pd.DataFrame(rows).assign(problem=problem, method=method, seed=seed)
    (exp / "runs").mkdir(parents=True, exist_ok=True)
    df.to_parquet(exp / "runs" / f"{live.run_id(problem, method, seed)}.parquet")


def write_status(exp, rid, state, pid, heartbeat_age=0.0, started=None, cost=5.0, budget=20.0):
    now = time.time()
    (exp / "live").mkdir(parents=True, exist_ok=True)
    (exp / "live" / f"{rid}.status.json").write_text(json.dumps({
        "run": rid, "state": state, "pid": pid, "invocation": None, "started": started or now - 10,
        "ended": None, "heartbeat": now - heartbeat_age, "budget": budget, "cost": cost, "batches": 1, "error": None,
    }))


# --- live logging ----------------------------------------------------------------------------


def test_runlog_lifecycle_done_removes_jsonl(tmp_path):
    log = RunLog(tmp_path, "p__m__s0", budget=10.0)
    log.batch(batch_row(0, 5.0, 1.0, 0.5))
    lines = log.jsonl.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["cost"] == 5.0
    st = json.loads(log.status_path.read_text())
    assert st["state"] == "running" and st["cost"] == 5.0 and st["pid"] == os.getpid()
    log.finish("done")
    assert not log.jsonl.exists()
    assert json.loads(log.status_path.read_text())["state"] == "done"


def test_runlog_failed_keeps_partial_log(tmp_path):
    log = RunLog(tmp_path, "p__m__s0", budget=10.0)
    log.batch(batch_row(0, 5.0, 1.0, 0.5))
    log.finish("failed", "boom")
    assert log.jsonl.exists()
    st = json.loads(log.status_path.read_text())
    assert st["state"] == "failed" and st["error"] == "boom"


def test_heartbeat_thread_refreshes_status(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "HEARTBEAT_S", 0.05)
    log = RunLog(tmp_path, "p__m__s0", budget=10.0)
    first = json.loads(log.status_path.read_text())["heartbeat"]
    time.sleep(0.3)
    assert json.loads(log.status_path.read_text())["heartbeat"] > first
    log.finish("done")


# --- run states ------------------------------------------------------------------------------


def test_run_states(tmp_path):
    exp = tmp_path / "exp"
    write_done(exp, "p", "sobol_gp", 0, [5, 10], [0.5, 0.2])
    write_status(exp, "p__sobol_gp__s1", "running", os.getpid())
    write_status(exp, "p__sobol_gp__s2", "running", dead_pid())
    write_status(exp, "p__sobol_gp__s3", "running", os.getpid(), heartbeat_age=3 * D.STALE_S)
    write_status(exp, "p__sobol_gp__s4", "failed", dead_pid())
    write_status(exp, "p__sobol_gp__s7", "failed", dead_pid(), started=time.time() - 100)
    live_cfg = {"budget": 20}
    write_manifest(exp / "live", tmp_path / "a.toml", live_cfg, [("p", "sobol_gp", 5), ("p", "sobol_gp", 7)], workers=2)
    m = next((exp / "live" / "manifests").glob("*.json"))
    m.rename(m.with_name("live.json"))
    dead = json.loads((exp / "live" / "manifests" / "live.json").read_text()) | {"pid": dead_pid(), "runs": [["p", "sobol_gp", 6]]}
    (exp / "live" / "manifests" / "dead.json").write_text(json.dumps(dead))

    states = D.Store(tmp_path).snapshot("exp").runs.set_index("seed")["state"].to_dict()
    assert states == {
        0: "done", 1: "running", 2: "stale", 3: "stale", 4: "failed",
        5: "queued", 6: "not_scheduled",
        7: "queued",  # failed earlier, re-listed by a newer live invocation
    }


# --- ETA -------------------------------------------------------------------------------------


def test_run_eta_follows_reference_shape():
    frac = np.linspace(0, 1, 11)
    quadratic = [(frac, 100 * frac**2, 100.0)]  # run time grows like cost^2 (GP fits)
    # Halfway in cost after 25 s: the reference spent 25 of 100 s there, so 75 s remain.
    assert D.run_eta(25.0, 0.5, quadratic) == pytest.approx(75.0)
    assert D.run_eta(10.0, 0.5, []) == pytest.approx(10.0)  # no references: linear
    assert D.run_eta(10.0, 0.0, quadratic) == pytest.approx(90.0)  # not started: typical duration
    assert D.run_eta(0.0, 0.0, []) is None


def test_experiment_eta_uses_method_durations(tmp_path):
    exp = tmp_path / "exp"
    for seed in range(2):
        write_done(exp, "p", "sobol_gp", seed, [10, 20], [0.5, 0.2], elapsed=[30.0, 60.0])
    write_manifest(exp / "live", tmp_path / "a.toml", {"budget": 20}, [("p", "sobol_gp", s) for s in (2, 3, 4)], workers=3)
    runs, summary = D.add_etas(D.Store(tmp_path).snapshot("exp"))
    assert summary["eta"] == pytest.approx(60.0) and not summary["eta_partial"]


# --- curves ----------------------------------------------------------------------------------


def test_curves_match_static_report(tmp_path):
    exp = tmp_path / "exp"
    rng = np.random.default_rng(0)
    for seed in range(5):
        write_done(exp, "p", "sobol_gp", seed, [5, 10, 15, 20], np.sort(rng.random(4))[::-1])
    snap = D.Store(tmp_path).snapshot("exp")
    c = D.curves(snap, "p", "y")["methods"][0]
    expected = np.nanmedian(curves_on_grid(snap.rows, "nrmse/y", np.asarray(c["x"])), axis=0)
    np.testing.assert_allclose(c["median"], expected)


# --- server ----------------------------------------------------------------------------------


def test_server_endpoints_return_json(tmp_path):
    exp = tmp_path / "exp"
    write_done(exp, "p", "sobol_gp", 0, [5, 10], [0.5, 0.2])
    server = serve(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        def get(path):
            with urllib.request.urlopen(base + path) as r:
                return r.status, r.read()

        assert get("/")[0] == 200
        assert get("/static/plotly-basic.min.js")[0] == 200
        for path in ("/api/experiments", "/api/experiment?name=exp", "/api/curves?name=exp&problem=p&output=y&seeds=1",
                     "/api/curves?name=exp&problem=p&output=y&x=elapsed", "/api/run?name=exp&run=p__sobol_gp__s0"):
            status, body = get(path)
            assert status == 200
            # Browsers reject NaN/Infinity tokens, which Python's json would happily accept.
            json.loads(body, parse_constant=lambda c: pytest.fail(f"{path} returned {c}"))
        with pytest.raises(urllib.error.HTTPError) as err:
            get("/api/experiment?name=../etc")
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
