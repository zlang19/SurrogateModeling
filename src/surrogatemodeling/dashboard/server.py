"""Read-only dashboard server: stdlib HTTP, one static page, four JSON endpoints.

GET /api/experiments                         experiment list
GET /api/experiment?name=                    services, manifests, progress, ETA, run table
GET /api/curves?name=&problem=&output=&metric=&x=&seeds=&methods=
GET /api/run?name=&run=                      one run's rows and diagnostics
"""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from surrogatemodeling.dashboard import data as D

STATIC = Path(__file__).parent / "static"
STATIC_FILES = {"index.html": "text/html; charset=utf-8", "plotly-basic.min.js": "text/javascript"}
SNAPSHOT_TTL_S = 2.0  # one page refresh fires several requests; share a single read


class Dashboard:
    def __init__(self, results: Path):
        self.store = D.Store(results)
        self._lock = threading.Lock()
        self._snaps: dict[str, tuple[float, D.Snapshot]] = {}

    def snapshot(self, name: str) -> D.Snapshot:
        with self._lock:
            hit = self._snaps.get(name)
            if hit and time.monotonic() - hit[0] < SNAPSHOT_TTL_S:
                return hit[1]
            snap = self.store.snapshot_all() if name == D.ALL else self.store.snapshot(name)
            self._snaps[name] = (time.monotonic(), snap)
            return snap

    def experiments(self) -> list[dict]:
        with self._lock:
            return self.store.experiments()

    def experiment(self, name: str) -> dict:
        snap = self.snapshot(name)
        runs, summary = D.add_etas(snap)
        counts = runs["state"].value_counts().to_dict() if len(runs) else {}
        methods = D.method_order(runs["method"].unique()) if len(runs) else []
        return {
            "name": name,
            "now": snap.now,
            "services": D.systemd_services(),
            "manifests": [
                {k: m.get(k) for k in ("id", "alive", "workers", "started", "config", "budget")} | {"n_runs": len(m["runs"])}
                for m in snap.manifests
            ],
            "counts": counts,
            "total": len(runs),
            **summary,
            "problems": D.problem_outputs(snap.rows),
            "methods": [{"name": m, "index": D.method_index(m)} for m in methods],
            "runs": runs.drop(columns=["started"], errors="ignore").to_dict("records"),
        }


def make_handler(app: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet: polling would flood the terminal
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(D.jsonable(obj)).encode(), "application/json")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, (STATIC / "index.html").read_bytes(), STATIC_FILES["index.html"])
                elif url.path.startswith("/static/") and url.path.removeprefix("/static/") in STATIC_FILES:
                    name = url.path.removeprefix("/static/")
                    self._send(200, (STATIC / name).read_bytes(), STATIC_FILES[name])
                elif url.path == "/api/experiments":
                    self._json(app.experiments())
                elif url.path == "/api/experiment":
                    self._json(app.experiment(q.get("name", "")))
                elif url.path == "/api/curves":
                    snap = app.snapshot(q.get("name", ""))
                    methods = [m for m in q.get("methods", "").split(",") if m] or None
                    self._json(
                        D.curves(
                            snap, q.get("problem", ""), q.get("output", ""), q.get("metric", "nrmse"),
                            q.get("x", "cost"), q.get("seeds") == "1", methods,
                        )
                    )
                elif url.path == "/api/run":
                    self._json(D.run_detail(app.snapshot(q.get("name", "")), q.get("run", "")))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(results: Path, host: str = "127.0.0.1", port: int = 8050) -> ThreadingHTTPServer:
    """Create the server (not yet serving); call serve_forever() on the result."""
    return ThreadingHTTPServer((host, port), make_handler(Dashboard(results)))
