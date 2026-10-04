"""`sm run <config.toml>`, `sm report <results_dir>` and `sm dashboard`."""

from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _single_threaded() -> None:
    """One process per core: keep torch and JAX from spawning their own thread pools."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    import torch

    torch.set_num_threads(1)


def _run_one(
    problem: str, method: str, seed: int, budget: float, batch_size: int, out: Path, invocation: str | None = None
) -> Path:
    _single_threaded()

    from surrogatemodeling.core.live import RunLog, run_id
    from surrogatemodeling.core.runner import run
    from surrogatemodeling.registry import METHODS, PROBLEMS

    log = RunLog(out.parent.parent / "live", run_id(problem, method, seed), budget, invocation)
    try:
        df = run(PROBLEMS[problem](), METHODS[method](), seed, budget, batch_size, log=log)
        df.insert(0, "seed", seed)
        df.insert(0, "method", method)
        df.insert(0, "problem", problem)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.rename(out)
    except BaseException as e:
        log.finish("failed", repr(e))
        raise
    log.finish("done")
    return out


def cmd_run(config_path: Path, results_root: Path, force: bool) -> None:
    from surrogatemodeling.problems.base import cache_dir
    from surrogatemodeling.registry import METHODS, PROBLEMS

    cfg = tomllib.loads(config_path.read_text())
    unknown = [p for p in cfg["problems"] if p not in PROBLEMS] + [m for m in cfg["methods"] if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown problems/methods in config: {unknown}")

    out_dir = results_root / cfg["name"]
    # Build test sets once in the parent so workers only read the cache.
    for p in cfg["problems"]:
        PROBLEMS[p]().test_set()
    print(f"test sets cached under {cache_dir()}")

    jobs = []
    for problem, method, seed in itertools.product(cfg["problems"], cfg["methods"], range(cfg["seeds"])):
        out = out_dir / "runs" / f"{problem}__{method}__s{seed}.parquet"
        if out.exists() and not force:
            continue
        jobs.append((problem, method, seed, float(cfg["budget"]), int(cfg.get("batch_size", 5)), out))
    print(f"{len(jobs)} runs to do ({cfg['name']})", flush=True)

    workers = int(cfg.get("workers", os.cpu_count() or 1))
    from surrogatemodeling.core.live import write_manifest

    manifest = write_manifest(out_dir / "live", config_path, cfg, [j[:3] for j in jobs], workers)
    jobs = [(*job, manifest.stem) for job in jobs]
    # Optional per-method concurrency caps, e.g. limits = { sobol_saas = 3 } for memory-heavy
    # methods. Capped methods get their own pools; everything else shares the remainder.
    limits = {m: int(n) for m, n in cfg.get("limits", {}).items() if any(j[1] == m for j in jobs)}
    pools = {m: n for m, n in limits.items()}
    pools[None] = max(1, workers - sum(limits.values()))
    failures = 0
    ctx = multiprocessing.get_context("spawn")  # spawn, not fork: torch autograd state does not survive fork
    executors = {key: ProcessPoolExecutor(max_workers=n, mp_context=ctx) for key, n in pools.items()}
    try:
        futures = {}
        for job in jobs:
            pool = executors[job[1] if job[1] in limits else None]
            futures[pool.submit(_run_one, *job)] = job
        for i, fut in enumerate(as_completed(futures), 1):
            problem, method, seed = futures[fut][:3]
            try:
                fut.result()
                print(f"[{i}/{len(jobs)}] done {problem} / {method} / seed {seed}", flush=True)
            except Exception as e:  # report and keep going; the run can be retried later
                failures += 1
                print(f"[{i}/{len(jobs)}] FAILED {problem} / {method} / seed {seed}: {e!r}", flush=True)
    finally:
        for ex in executors.values():
            ex.shutdown()
    if failures:
        raise SystemExit(f"{failures} run(s) failed")


def cmd_report(results_dir: Path) -> None:
    from surrogatemodeling.report.plots import report

    for path in report(results_dir):
        print(f"wrote {path}")


def cmd_dashboard(results: Path, host: str, port: int) -> None:
    from surrogatemodeling.dashboard.server import serve

    server = serve(results, host, port)
    print(f"dashboard on http://{host}:{port}  (results: {results.resolve()})  Ctrl-C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="sm", description="Surrogate modeling test bed")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run an experiment config")
    p_run.add_argument("config", type=Path)
    p_run.add_argument("--results", type=Path, default=Path("results"))
    p_run.add_argument("--force", action="store_true", help="re-run runs whose output already exists")
    p_rep = sub.add_parser("report", help="plot results of an experiment")
    p_rep.add_argument("results_dir", type=Path)
    p_dash = sub.add_parser("dashboard", help="live read-only dashboard of experiments under --results")
    p_dash.add_argument("--results", type=Path, default=Path("results"))
    p_dash.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other devices on the LAN")
    p_dash.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    if args.cmd == "run":
        cmd_run(args.config, args.results, args.force)
    elif args.cmd == "dashboard":
        cmd_dashboard(args.results, args.host, args.port)
    else:
        cmd_report(args.results_dir)


if __name__ == "__main__":
    main()
