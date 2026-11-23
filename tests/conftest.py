from __future__ import annotations

from pathlib import Path

import pytest

from storm_outage_forecast import pipeline, synthetic
from storm_outage_forecast.config import load_config

FOLDERS = {
    "raw": "raw",
    "silver": "silver",
    "gold": "gold/outages.duckdb",
    "models": "models",
    "reports": "reports",
    "bundle": "bundle",
    "tmp": "tmp",
}


def tiny_overrides(root: Path) -> dict:
    return {
        "paths": {key: str(root / value) for key, value in FOLDERS.items()},
        "years": {
            "first": 2019,
            "last": 2023,
            "train_end": 2021,
            "valid": 2022,
            "test_start": 2023,
        },
        "runtime": {"threads": 2, "duckdb_memory_gb": 1.0},
        "download": {"eia861_first": 2019, "eia861_last": 2023, "retries": 1, "timeout_s": 5},
        "risk": {
            "min_child_samples": 20,
            "shap_rows": 2000,
            "logistic_rows": 5000,
            "max_rounds": 300,
            "early_stopping": 30,
        },
        "survival": {"min_events": 10},
        "staging": {"mutual_aid_crews": 20, "top_days": 5, "samples": 100},
        "app": {"top_days": 8},
    }


def make_cfg(root: Path, **extra):
    overrides = tiny_overrides(root)
    for section, values in extra.items():
        overrides.setdefault(section, {}).update(values)
    return load_config(overrides=overrides)


@pytest.fixture(scope="session")
def replica(tmp_path_factory):
    root = tmp_path_factory.mktemp("replica")
    cfg = make_cfg(root)
    synthetic.generate(cfg, "tiny", log=lambda message: None)
    return cfg


@pytest.fixture(scope="session")
def built(replica):
    metrics = pipeline.run(replica, log=lambda message: None)
    return replica, metrics
