"""Live run logging for the dashboard.

Each run keeps two files under `results/<exp>/live/`:

- `<run>.jsonl`        one JSON line per batch (the Parquet row fields)
- `<run>.status.json`  state, pid, timing, progress and a heartbeat refreshed every
                       HEARTBEAT_S seconds by a background thread

On success the `.jsonl` is deleted (the Parquet file holds the same rows) and the status
file is kept; a crashed run keeps its partial `.jsonl`. Each `sm run` invocation also
writes a manifest of the runs it plans, so the dashboard can show queued runs.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

HEARTBEAT_S = 30.0


def run_id(problem: str, method: str, seed: int) -> str:
    return f"{problem}__{method}__s{seed}"


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


class RunLog:
    def __init__(self, live_dir: Path, rid: str, budget: float, invocation: str | None = None):
        live_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = live_dir / f"{rid}.jsonl"
        self.status_path = live_dir / f"{rid}.status.json"
        self._lock = threading.Lock()
        now = time.time()
        self.status = {
            "run": rid,
            "state": "running",
            "pid": os.getpid(),
            "invocation": invocation,
            "started": now,
            "ended": None,
            "heartbeat": now,
            "budget": budget,
            "cost": 0.0,
            "batches": 0,
            "error": None,
        }
        self.jsonl.write_text("")  # a re-run replaces any partial log from an earlier attempt
        self._write_status()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def _write_status(self) -> None:
        with self._lock:
            self.status["heartbeat"] = time.time()
            _write_json_atomic(self.status_path, self.status)

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            self._write_status()

    def batch(self, row: dict) -> None:
        with self.jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.status["cost"] = row["cost"]
        self.status["batches"] = row["batch"] + 1
        self._write_status()

    def finish(self, state: str = "done", error: str | None = None) -> None:
        self._stop.set()
        self._thread.join()
        self.status.update(state=state, ended=time.time(), error=error)
        self._write_status()
        if state == "done":
            self.jsonl.unlink(missing_ok=True)


def write_manifest(live_dir: Path, config_path: Path, cfg: dict, runs: list[tuple[str, str, int]], workers: int) -> Path:
    """Record one `sm run` invocation's plan; returns the manifest path (its name is the invocation id)."""
    started = time.time()
    path = live_dir / "manifests" / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(started))}__{config_path.stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        path,
        {
            "pid": os.getpid(),
            "started": started,
            "config": str(config_path),
            "workers": workers,
            "limits": cfg.get("limits", {}),
            "budget": float(cfg["budget"]),
            "runs": [list(r) for r in runs],
        },
    )
    return path
