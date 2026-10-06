# Dashboard Plan

## Goal
A simple dashboard that live reports the cost vs accuracy plots and the progress of the current running samples.

## Details
* Add standardized logging in core code or models
* Dashboard is simple and inspired by TensorBoard
* Live plots to see the accuracy improve over time
* Live information on any current runs and ETAs to completion
* Run with a simple command

## Decisions

### Scope
- A read-only local web app, custom-built (TensorBoard can't do medians across seeds on a float cost axis, or queued runs and ETAs).
- `uv run sm dashboard [--results results] [--port 8050] [--host 127.0.0.1]`. It binds to localhost by default; `--host 0.0.0.0` opens it to the LAN.
- It runs independently of `sm run`, and only reads files.
- It shows every experiment under `results/`, with live ones highlighted.
- It displays the state of the `sm-*` systemd services but never starts or stops runs.

### Logging (written by the runner, so every method gets it)
- Per run, under `results/<exp>/live/`:
  - `<run>.jsonl`: one line per batch, with the Parquet row fields plus method diagnostics.
  - `<run>.status.json`: state (running / done / failed), pid, invocation, start and end times, cost spent / budget, batch count, error, and `heartbeat`.
- Heartbeat: a background thread rewrites it every 30 s, so slow batches don't look dead.
- A run is **stale** (killed) when its pid is gone or its heartbeat is more than 2 min old.
- New columns, in the live log and in Parquet:
  - `elapsed`: wall seconds since the run started;
  - `sim_time`: simulator seconds in the batch;
  - `diagnostics`: JSON.
- On completion the `.jsonl` is deleted, because the Parquet file now holds everything, and the status file is kept. A crashed run's partial `.jsonl` is kept.
- Each `sm run` invocation writes `live/manifests/<start>__<config>.json` with its pid, workers, limits, budget and planned runs.
  - A run with no status or Parquet file counts as **queued** if some live invocation lists it, and **not scheduled** otherwise.
- Older runs, without live files or `elapsed`, show as done on the cost axis only. Nothing is backfilled.

### Method diagnostics (optional `diagnostics() -> dict` hook, recorded each batch)
| Method | Diagnostics |
|---|---|
| GP methods | lengthscales per output, learned extra noise, fit time, whether hyperparameters were re-optimized |
| Adaptive (IV / EPIG / cost-aware) | the batch's fidelities, best acquisition score, estimated HF noise |
| screen_gp | active inputs (names) |
| PCE | degree, number of terms, inputs used |
| SAAS | median inverse lengthscales across MCMC samples, NUTS time |

### ETA
- **Per run:**
  - Uses completed runs of the same (problem, method) that recorded `elapsed`.
  - Scales the current elapsed time by each reference run's total time over its time to reach the same cost fraction, then takes the median.
  - Falls back to a linear extrapolation from elapsed time when there are no references.
- **Per experiment:**
  - Estimates each pending run as the median duration of its (problem, method), falling back to the method alone, then to all runs.
  - Running runs use their per-run ETA.
  - The total is max(longest single remaining run, total remaining work / live workers).

### Page (one page, polling every 5 s)
1. **Header:** experiment selector, `sm-*` service states, a progress bar (done / running / queued / failed / stale) and the experiment ETA.
2. **Controls (one row):** metric (NRMSE, coverage, NLL, max error), show individual seeds, x-axis (cost or wall time) and a method filter.
3. **Plot grid:** one panel per (problem, output), median + IQR across seeds, colors fixed by registry order.
4. **Runs table:** problem, method, seed, state, cost/budget, batches, elapsed, ETA, heartbeat age. Sortable and filterable, with failed and stale runs flagged.
5. **Run detail:** click a row to see that run's curves and diagnostics.
- Light/dark theme follows the OS setting.

### Implementation
- `src/surrogatemodeling/dashboard/`:
  - `data.py`: reading, state, aggregation and ETA;
  - `server.py`: stdlib `ThreadingHTTPServer`;
  - `static/`: `index.html`, plus Plotly bundled in the repo so it works offline.
- Seed aggregation runs on the server using the report's `curves_on_grid`. Parquet reads are cached by mtime.

### Testing
- **Unit tests:**
  - live files, including partial logs from crashed runs;
  - stale detection;
  - ETA from synthetic histories;
  - aggregation matches the static report.
- **Server smoke test:** every endpoint returns valid JSON.
- **No browser automation:** the page is checked by eye during a smoke run.

## Steps
1. ✅ Runner logging: live files, heartbeat, manifests, `elapsed` / `sim_time` / `diagnostics` columns.
2. ✅ Method `diagnostics()` hooks.
3. ✅ `dashboard/data.py`: states, curves, ETA.
4. ✅ Server + page.
5. ✅ Tests (`tests/test_dashboard.py`), then the page checked by eye in light and dark mode on `configs/experiments/dashdemo.toml`.

## As built (small additions found while checking the page by eye)
- While seeds are still running, the median is drawn only where at least half the seeds have data, and the IQR band only where at least two do. Otherwise the curve's tail is just the one or two seeds furthest along.
- Log axes use explicit 1-2-5 ticks. Plotly's toolbar is hidden; drag still zooms and double-click resets.
- Deep links: `?exp=<name>` selects an experiment, and `#run=<run id>` opens that run's detail drawer.
- Run detail colors the outputs with the first palette slots, since those series are outputs, not methods.
- **Robust y-axes (2026-10-05).** Extreme values in a few runs, e.g. NLL in the thousands from the warping variants, had flattened whole panels.
  - Log-scale metrics use the 1st–99th percentile of the medians and IQR bands.
  - Linear metrics (NLL, coverage) use a Tukey fence over the *later half* of the cost range only, so early transients are clipped rather than setting the axis.
  - Individual seed lines never set the range, and a "some values outside range" note marks clipped panels.
  - The static report plots use the same percentile rule.
  - `?metric=<name>` opens the page on a given metric.
- **"All (since NCRPS)" virtual experiment (2026-10-05).** It's the first entry in the selector, with id `_all`.
  - It merges every run from experiments that recorded NCRPS, so methods from different trials share panels, e.g. SAAS next to the cost-aware arms.
  - Run ids become `<experiment>::<run>`, and the run table gains an Experiment column.
  - Where a (problem, method, seed) appears in several experiments, the newest wins. Runs with only pre-NCRPS data are excluded.
  - With more than 8 methods, palette colors repeat. Use the legend chips to hide methods when comparing.
