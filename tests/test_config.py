from __future__ import annotations

from pathlib import Path

import pytest

from storm_outage_forecast.config import ROOT, ConfigError, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.years.train == list(range(2015, 2023))
    assert cfg.years.test == [2024, 2025]
    assert cfg.paths.gold == ROOT / "data" / "gold" / "outages.duckdb"
    assert cfg.risk.quantiles == (0.1, 0.5, 0.9)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"years": {"extra": 1}}, "unknown keys"),
        ({"risk": {"num_leaves": "63"}}, "must be an integer"),
        ({"runtime": {"threads": True}}, "must be an integer"),
        ({"years": {"train_end": 2023}}, "first <= train_end < valid"),
        ({"years": {"valid": 2024, "test_start": 2025}}, "right after train_end"),
        ({"risk": {"quantiles": [0.2, 0.5]}}, "three increasing values"),
        ({"risk": {"quantiles": [0.1, 0.6, 0.9]}}, "0.5 in the middle"),
        ({"download": {"ghcn_stations": "http://example.com/x"}}, "https url"),
        (
            {"download": {"ghcn_by_year": "https://example.com/by_year.csv.gz"}},
            "{year} placeholder",
        ),
        ({"download": {"eia861_first": 2013}}, "2015 <= eia861_first"),
        ({"outages": {"significant_share": 1.5}}, "must be in (0, 1)"),
        ({"outages": {"event_min_peak": 10}}, "event_min_peak"),
        ({"weather": {"neighbors": 0}}, "neighbors"),
        ({"staging": {"restore_rate": 0}}, "restore_rate"),
        ({"surprise": {"a": 1}}, "unknown sections"),
        ({"download": {"eia861": []}}, "non-empty list"),
    ],
)
def test_invalid_values_are_rejected(overrides, message):
    with pytest.raises(ConfigError) as err:
        load_config(overrides=overrides)
    assert message in str(err.value)


def test_missing_key_is_reported(tmp_path):
    text = (ROOT / "config.toml").read_text(encoding="utf-8").replace("seed = 7\n", "")
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="missing keys: seed"):
        load_config(path)


def test_relative_paths_follow_the_config_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.paths.raw == tmp_path / "data" / "raw"


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("does/not/exist.toml"))


def test_integers_are_accepted_for_floats():
    cfg = load_config(overrides={"runtime": {"duckdb_memory_gb": 2}})
    assert cfg.runtime.duckdb_memory_gb == 2.0
