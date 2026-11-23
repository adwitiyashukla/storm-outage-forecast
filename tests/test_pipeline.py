from __future__ import annotations

import numpy as np
import pytest

from storm_outage_forecast import bundle, pipeline
from storm_outage_forecast.__main__ import main, report
from storm_outage_forecast.config import ROOT
from storm_outage_forecast.db import connect


def test_metrics_sections(built):
    cfg, metrics = built
    for key in (
        "data",
        "quality",
        "reliability",
        "risk",
        "restoration",
        "staging",
        "timing_s",
        "environment",
        "peak_memory_mb",
    ):
        assert key in metrics
    assert metrics["data_kind"] == "synthetic"
    assert not [q for q in metrics["quality"] if q["status"] == "fail"]
    assert metrics["reliability"]["spearman"] > 0.5
    shares = metrics["staging"]["share_of_perfect_savings"]
    assert shares["forecast"] <= 1.0 + 1e-9
    hours = metrics["staging"]["customer_hours"]
    assert hours["perfect"] <= min(hours["forecast"], hours["climatology"], hours["proportional"])
    assert hours["none"] >= max(hours["forecast"], hours["proportional"])
    assert "lightgbm" in report(metrics)


def test_feature_table_rules(built):
    cfg, _ = built
    con = connect(cfg, read_only=True)
    rows = con.execute("""
        SELECT m.unit_id, m.day, m.prcp_3d, m.sig, m.peak_out, m.customers, m.split, m.year,
               m.prev_share
        FROM model_rows m WHERE m.unit_id = (SELECT min(unit_id) FROM units) ORDER BY m.day
    """).df()
    weather = con.execute("""
        SELECT w.day, w.prcp FROM weather_day w
        WHERE w.unit_id = (SELECT min(unit_id) FROM units) ORDER BY w.day
    """).df()
    con.close()
    merged = rows.merge(weather, on="day", how="left")
    manual = merged["prcp"].shift(1).rolling(3, min_periods=1).sum()
    ok = merged["prcp_3d"].notna() & merged["prcp"].notna()
    assert np.allclose(merged.loc[ok, "prcp_3d"].iloc[5:200], manual[ok].iloc[5:200])
    floor = np.maximum(
        cfg.outages.significant_floor, cfg.outages.significant_share * rows["customers"]
    )
    assert (rows["sig"] == (rows["peak_out"] >= floor)).all()
    assert (rows.loc[rows["year"] <= cfg.years.train_end, "split"] == "train").all()
    assert (rows.loc[rows["year"] == cfg.years.valid, "split"] == "valid").all()
    expected_prev = np.minimum(rows["peak_out"].shift(1) / rows["customers"], 1.0)
    assert np.allclose(rows["prev_share"].iloc[1:], expected_prev.iloc[1:])


def test_bundle_and_space(built, tmp_path):
    cfg, _ = built
    for name in ("app.duckdb", "counties.json", "meta.json"):
        assert (cfg.paths.bundle / name).exists()
    space = bundle.build_space(cfg, tmp_path / "space", log=lambda m: None)
    for name in (
        "Dockerfile",
        "app.py",
        "requirements.txt",
        "README.md",
        ".gitattributes",
        "bundle/app.duckdb",
        ".streamlit/config.toml",
    ):
        assert (space / name).exists()
    readme = (space / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\ntitle: Storm Outage Forecast")
    assert "sdk: docker" in readme


def test_build_resumes_from_a_stage(built):
    cfg, metrics = built
    before = cfg.paths.gold.stat().st_mtime_ns
    again = pipeline.build(cfg, log=lambda message: None, start="storms")
    assert cfg.paths.gold.stat().st_mtime_ns != before
    assert again["timing_s"]["outages"] == metrics["timing_s"]["outages"]
    assert again["data"]["events"] == metrics["data"]["events"]
    assert again["data"]["county_days"] == metrics["data"]["county_days"]
    with pytest.raises(ValueError, match="unknown stage"):
        pipeline.build(cfg, start="models")


def test_cli(built, tmp_path, capsys):
    cfg, _ = built
    text = (ROOT / "config.toml").read_text(encoding="utf-8")
    for key in ("raw", "silver", "models", "reports", "bundle", "tmp"):
        text = text.replace(
            f'{key} = "data/{key}"', f'{key} = "{getattr(cfg.paths, key).as_posix()}"'
        )
    text = text.replace(
        'gold = "data/gold/outages.duckdb"', f'gold = "{cfg.paths.gold.as_posix()}"'
    )
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    assert main(["--config", str(path), "report"]) == 0
    assert "restoration c-index" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--config", str(path), "build", "--from", "nowhere"])
    assert main(["--config", str(path), "download", "--only", "geo"]) == 1
    assert "synthetic replica" in capsys.readouterr().err
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("seed = 7", "seed = -1"), encoding="utf-8")
    assert main(["--config", str(bad), "report"]) == 2
    with pytest.raises(SystemExit):
        main(["--config", str(path), "fly"])
