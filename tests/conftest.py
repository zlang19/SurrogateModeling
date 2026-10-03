import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Keep cached test sets out of the repo's results/ during tests."""
    monkeypatch.setenv("SM_CACHE_DIR", str(tmp_path / "cache"))
