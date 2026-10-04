# Surrogate Modeling Test Bed

A test bed for comparing surrogate modeling methods on noisy, expensive simulators. The goal is the **most accurate cheap replacement for the simulator, for the least simulation cost**. The target application is **MCNP** (Monte Carlo neutronics):
- tens of continuous inputs,
- a few scalar outputs,
- hours per run,
- Monte Carlo noise, with particle count as the fidelity knob.

Every method and simulator speaks the same **ask/tell** protocol. The runner spends a budget (in high-fidelity run equivalents) batch by batch, and scores each method's predictions against noise-free truth on a test set drawn from the input distribution. That makes the error pdf-weighted, which is what matters for UQ.

## Contents
- [Quick start](#quick-start)
- [Problems](#problems)
- [Methods](#methods)
- [Results so far](#results-so-far)
- [Running long experiments](#running-long-experiments)
- [Experiment configs](#experiment-configs)
- [Project layout](#project-layout)
- [Documentation](#documentation)

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                                               # install
uv run pytest                                         # full test suite (~3 min)
uv run sm run configs/experiments/smoke.toml          # tiny end-to-end run (<1 min)
uv run sm report results/smoke                        # plots + ranking.md
uv run sm dashboard                                   # live dashboard → http://localhost:8050
```

| Command | What it does |
|---|---|
| `sm run <config.toml> [--results results] [--force]` | Runs every (problem, method, seed) in the config across a process pool. It writes one Parquet file per run to `results/<name>/runs/` and live logs to `results/<name>/live/`. Runs that already finished are skipped, so an interrupted experiment resumes where it stopped. |
| `sm report <results/name>` | Writes error-vs-cost plots (median + IQR across seeds) to `plots/`, plus `ranking.md` and `ranking.csv`. |
| `sm dashboard [--results] [--port 8050] [--host 127.0.0.1]` | Read-only live dashboard: progress, ETAs, live curves, run table, per-run diagnostics. See [docs/DashboardPlan.md](docs/DashboardPlan.md). |

## Problems

| Name | Inputs | Outputs | Notes |
|---|---|---|---|
| `borehole` | 8 | flow | Analytic benchmark, UQ-literature input distributions |
| `otl` | 6 | vm | OTL circuit |
| `wing_weight` | 10 | weight | A few inputs dominate |
| `morris` | 20 | y | 10 strong inputs with high-order interactions |
| `borehole_d30` | 30 | flow | Borehole + 22 padding inputs (half weak, half inert) |
| `wing_weight_d40` | 40 | weight | Wing weight + 30 padding inputs |
| `toymc` | 23 | k_eff, power_ratio, capture_to_fission | **Toy MC neutronics:** 2-group k-eigenvalue Monte Carlo on a 1D slab, with real MC noise and batch-statistics σ |

Analytic problems get MCNP-like synthetic noise: σ ∝ 1/√fidelity, with a reported σ that is itself noisy. Cost is `overhead + (1 − overhead)·fidelity`, with a 1% default overhead. The fidelity ladder is [1/16, 1/4, 1]. Details are in [docs/PLAN.md](docs/PLAN.md).

## Methods

| # | Name | Summary |
|---|---|---|
| 1 | [`sobol_gp`](docs/methods/sobol_gp.md) | GP on a fixed Sobol design (baseline) |
| 2 | [`sobol_pce`](docs/methods/sobol_pce.md) | Sparse polynomial chaos, hybrid LARS |
| 3 | [`sobol_saas`](docs/methods/sobol_saas.md) | Fully Bayesian SAAS GP (not yet benchmarked) |
| 4 | [`adaptive_iv`](docs/methods/adaptive_iv.md) | GP + integrated-variance reduction |
| 4b | [`adaptive_epig`](docs/methods/adaptive_epig.md) | GP + EPIG |
| 5 | [`adaptive_iv_mf`](docs/methods/adaptive_iv_mf.md) | Cost-aware IV: chooses fidelity per point |
| 5b | [`adaptive_iv_mf_xn`](docs/methods/adaptive_iv_mf_xn.md) | #5 + learned extra noise (**current best**) |
| 6 | [`screen_gp`](docs/methods/screen_gp.md) | ARD screening → GP on active inputs |

The shared machinery (GP core, greedy batch acquisition) and a full results table are in [docs/methods/README.md](docs/methods/README.md).

## Results so far

Mean rank across 9 (problem, output) pairs, by area under the error-vs-cost curve (10 seeds, budget 100; lower is better). SAAS isn't run yet.

| Rank | Method | Mean rank |
|---|---|---|
| 1 | #5b `adaptive_iv_mf_xn` | **1.33** |
| 2 | #5 `adaptive_iv_mf` | 2.22 |
| 3 | #4 `adaptive_iv` | 3.67 |
| 4 | #6 `screen_gp` | 3.78 |
| 5 | #4b `adaptive_epig` | 4.78 |
| 6 | #1 `sobol_gp` (baseline) | 5.22 |
| 7 | #2 `sobol_pce` | 7.00 |

| Takeaway | Evidence |
|---|---|
| Many cheap, noisy runs beat a few precise ones | 1.5–2.4× lower error than the baseline (e.g. Morris 0.45 → 0.19) |
| Don't trust reported σ | Trusting under-reported σ gave 0.27 on the toy MC power ratio; learning extra noise gave 0.105 |
| Extra, mostly irrelevant inputs hurt a lot | 22 padding inputs (0.5% of variance) tripled baseline error |

See [docs/Learnings.md](docs/Learnings.md) for the full list, the caveats, and the open questions.

## Running long experiments

A full benchmark can need more than 20 GB and many hours. **Always launch long runs as a memory-capped systemd user service.**

Background shells die with the session. Worse, an uncapped run once used up memory and the kernel killed VS Code instead of the benchmark, twice.

```bash
systemd-run --user --unit=sm-full --collect --working-directory="$PWD" \
  -p MemoryHigh=20G -p MemoryMax=22G -p OOMScoreAdjust=500 \
  -p StandardOutput=append:"$PWD/results/full_run.log" -p StandardError=append:"$PWD/results/full_run.log" \
  "$(command -v uv)" run sm run configs/experiments/full_fast.toml

systemctl --user status sm-full          # state, memory
grep -c "] done" results/full_run.log    # progress (or use the dashboard)
systemctl --user stop sm-full            # stop; re-running the command resumes
```

Memory-heavy methods can be capped per config:

```toml
[limits]
sobol_saas = 3   # at most 3 SAAS runs at a time (~3 GB each)
```

## Experiment configs

| Config | Contents |
|---|---|
| `smoke.toml` | 2 problems × 2 methods × 2 seeds, tiny budget; also run as a test |
| `full.toml` | Every problem × every method × 10 seeds, **including SAAS** |
| `full_fast.toml` | `full` without SAAS (writes to the same `results/full`) |
| `full_saas.toml` | SAAS only, into `results/full`. **On hold:** don't run without the owner's go-ahead |
| `step1–3.toml` | Historical baseline runs from the build-up |
| `dashdemo.toml` | Small live run for trying the dashboard |

## Project layout

```
src/surrogatemodeling/
  core/        protocols.py (ask/tell contract), distributions.py, runner.py, metrics.py, live.py (live logging)
  problems/    analytic.py (+ noise wrapper, padding), base.py, toymc/ (physics.py, problem.py)
  methods/     gp_common.py, gp_fixed.py, pce.py, saas.py, adaptive.py, acquisition.py
  report/      plots.py, ranking.py
  dashboard/   data.py, server.py, static/ (index.html, bundled Plotly)
  registry.py  name → factory for problems and methods (order fixes plot colors)
  cli.py       `sm` command
configs/experiments/   experiment TOMLs
docs/                  plans, method docs, learnings
results/               (gitignored) run Parquet files, live logs, cached test sets, plots
tests/                 pytest suite
```

To add a method or problem, implement the protocol in `core/protocols.py`, register it in `registry.py`, and add it to a config. The method tests pick up every registered method automatically.

## Documentation

| Doc | What it covers |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | Design decisions: scope, budget and metrics, fidelity and noise, problems, the frozen interface, build steps |
| [docs/methods/](docs/methods/README.md) | One page per method, the shared GP and acquisition machinery, and results tables |
| [docs/Learnings.md](docs/Learnings.md) | Findings so far: results, method pitfalls, toy MC lessons, operations, open questions |
| [docs/DashboardPlan.md](docs/DashboardPlan.md) | The live dashboard: logging format, run states, ETAs, page design |
