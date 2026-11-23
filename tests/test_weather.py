from __future__ import annotations

import gzip

import pandas as pd
import pytest

from storm_outage_forecast import weather
from storm_outage_forecast.db import connect

from .conftest import make_cfg


def station_line(sid: str, lat: float, lon: float, state: str) -> str:
    return f"{sid:<11} {lat:>8.4f} {lon:>9.4f} {100.0:>6.1f} {state:<2} {'NAME':<30}\n"


def test_read_stations_keeps_conus_first_order_only(tmp_path):
    path = tmp_path / "ghcnd-stations.txt"
    path.write_text(
        station_line("USW00000001", 40.0, -83.0, "OH")
        + station_line("USW00000002", 61.2, -149.9, "AK")
        + station_line("USC00000003", 40.0, -83.0, "OH")
        + station_line("USW00000004", 18.4, -66.0, "PR")
        + station_line("CA000000005", 45.0, -75.0, "")
    )
    stations = weather.read_stations(path)
    assert stations["station"].tolist() == ["USW00000001"]


def test_read_inventory_filters_years(tmp_path):
    path = tmp_path / "ghcnd-inventory.txt"
    path.write_text(
        "USW00000001  40.0000  -83.0000 WSF5 1996 2026\n"
        "USW00000002  40.0000  -83.0000 WSF5 1950 1990\n"
        "USC00000003  40.0000  -83.0000 PRCP 1950 2026\n"
    )
    inventory = weather.read_inventory(path, 2015, 2025)
    assert inventory.to_dict(orient="records") == [{"station": "USW00000001", "element": "WSF5"}]


@pytest.fixture
def mapped(tmp_path):
    cfg = make_cfg(tmp_path, weather={"neighbors": 3, "radius_km": 200.0})
    units = pd.DataFrame({"unit_id": ["A", "B"], "lat": [40.0, 30.0], "lon": [-83.0, -100.0]})
    stations = pd.DataFrame(
        {
            "station": ["USW00000001", "USW00000002", "USW00000003"],
            "lat": [40.0 + 10 / 111.195, 40.0 + 50 / 111.195, 40.0 + 300 / 111.195],
            "lon": [-83.0, -83.0, -83.0],
            "state": ["OH", "OH", "MI"],
            "name": ["A", "B", "C"],
        }
    )
    inventory = pd.DataFrame(
        {
            "station": ["USW00000001", "USW00000002", "USW00000003", "USW00000001", "USW00000002"],
            "element": ["WSF5", "WSF5", "WSF5", "PRCP", "PRCP"],
        }
    )
    return cfg, units, stations, weather.station_map(units, stations, inventory, cfg)


def test_station_map_respects_radius_and_weights(mapped):
    _, _, _, mapping = mapped
    wind = mapping[mapping["grp"] == "wind"].sort_values("rank")
    assert wind["unit_id"].tolist() == ["A", "A"]
    assert wind["dist_km"].round(1).tolist() == [10.0, 50.0]
    assert wind["weight"].tolist() == pytest.approx([1 / 10.0**2, 1 / 50.0**2], rel=1e-3)
    assert "B" not in set(mapping["unit_id"])


def test_idw_daily_features(mapped, tmp_path):
    cfg, units, stations, mapping = mapped
    rows = [
        ("USW00000001", "20200105", "WSF5", "100", "", "", "W", ""),
        ("USW00000002", "20200105", "WSF5", "200", "", "", "W", ""),
        ("USW00000002", "20200105", "WSF5", "900", "", "X", "W", ""),
        ("USW00000001", "20200106", "WSF5", "100", "", "", "W", ""),
        ("USW00000002", "20200106", "PRCP", "50", "", "", "W", ""),
        ("USW00000002", "20200106", "WT06", "1", "", "", "W", ""),
        ("USW00000001", "20200106", "PRCP", "-9999", "", "", "W", ""),
        ("USC00000009", "20200105", "WSF5", "999", "", "", "W", ""),
        ("USW00000001", "20200105", "TAVG", "10", "", "", "W", ""),
    ]
    folder = cfg.paths.raw / "ghcn"
    folder.mkdir(parents=True)
    with gzip.open(folder / "2020.csv.gz", "wt") as handle:
        for r in rows:
            handle.write(",".join(r) + "\n")
    con = connect(cfg)
    con.register("s", stations)
    con.execute("CREATE TABLE ghcn_stations AS SELECT * FROM s")
    con.register("m", mapping)
    con.execute("CREATE TABLE station_map AS SELECT * FROM m")
    con.execute("""
        CREATE TABLE weather_day (
            unit_id VARCHAR, day DATE, gust DOUBLE, gust2 DOUBLE, wind DOUBLE, gust_max DOUBLE,
            n_gust BIGINT, prcp DOUBLE, snow DOUBLE, snwd DOUBLE, tmax DOUBLE, tmin DOUBLE,
            ice INTEGER, thunder INTEGER)
    """)
    weather.silver_year(cfg, con, 2020)
    silver = con.execute(f"SELECT * FROM read_parquet('{cfg.paths.silver}/ghcn/2020.parquet')").df()
    assert len(silver) == 5
    weather.gold_year(cfg, con, 2020)
    out = con.execute("SELECT * FROM weather_day ORDER BY day").df()
    w1, w2 = 1 / 100, 1 / 2500
    assert out.loc[0, "gust"] == pytest.approx((10 * w1 + 20 * w2) / (w1 + w2))
    assert out.loc[0, "gust_max"] == 20.0
    assert out.loc[0, "n_gust"] == 2
    assert out.loc[1, "gust"] == pytest.approx(10.0)
    assert out.loc[1, "prcp"] == pytest.approx(5.0)
    assert out.loc[1, "ice"] == 1
    con.close()
