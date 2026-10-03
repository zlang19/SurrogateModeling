"""`sm run <config.toml>` and `sm report <results_dir>`."""

from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _run_one(problem: str, method: str, seed: int, budget: float, batch_size: int, out: Path) -> Path:
    import torch

    torch.set_num_threads(1)  # one process per core; avoid oversubscription

    from surrogatemodeling.core.runner import run
    from surrogatemodeling.registry import METHODS, PROBLEMS

    df = run(PROBLEMS[problem](), METHODS[method](), seed, budget, batch_size)
    df.insert(0, "seed", seed)
    df.insert(0, "method", method)
    df.insert(0, "problem", problem)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.rename(out)
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
    print(f"{len(jobs)} runs to do ({cfg['name']})")

    workers = int(cfg.get("workers", os.cpu_count() or 1))
    failures = 0
    # spawn, not fork: torch autograd state does not survive fork.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_run_one, *job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            problem, method, seed = futures[fut][:3]
            try:
                fut.result()
                print(f"[{i}/{len(jobs)}] done {problem} / {method} / seed {seed}")
            except Exception as e:  # report and keep going; the run can be retried later
                failures += 1
                print(f"[{i}/{len(jobs)}] FAILED {problem} / {method} / seed {seed}: {e!r}")
    if failures:
        raise SystemExit(f"{failures} run(s) failed")


def cmd_report(results_dir: Path) -> None:
    from surrogatemodeling.report.plots import report

    for path in report(results_dir):
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sm", description="Surrogate modeling test bed")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run an experiment config")
    p_run.add_argument("config", type=Path)
    p_run.add_argument("--results", type=Path, default=Path("results"))
    p_run.add_argument("--force", action="store_true", help="re-run runs whose output already exists")
    p_rep = sub.add_parser("report", help="plot results of an experiment")
    p_rep.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    if args.cmd == "run":
        cmd_run(args.config, args.results, args.force)
    else:
        cmd_report(args.results_dir)


if __name__ == "__main__":
    main()
