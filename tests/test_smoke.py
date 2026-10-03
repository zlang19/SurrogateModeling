from pathlib import Path

import pandas as pd

from surrogatemodeling.cli import cmd_report, cmd_run

SMOKE = Path(__file__).parents[1] / "configs" / "experiments" / "smoke.toml"


def test_smoke_pipeline(tmp_path):
    cmd_run(SMOKE, tmp_path, force=False)
    runs = sorted((tmp_path / "smoke" / "runs").glob("*.parquet"))
    assert runs
    df = pd.concat(pd.read_parquet(p) for p in runs)
    assert df.groupby(["problem", "method", "seed"])["cost"].max().eq(10).all()
    cmd_report(tmp_path / "smoke")
    assert list((tmp_path / "smoke" / "plots").glob("*.png"))
