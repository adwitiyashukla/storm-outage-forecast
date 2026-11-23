from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import Config
from .db import record_check, sql_path
from .geo import chord_km, unit_vectors

ELEMENTS = (
    "PRCP",
    "SNOW",
    "SNWD",
    "TMAX",
    "TMIN",
    "AWND",
    "WSF2",
    "WSF5",
    "WT03",
    "WT06",
    "WT15",
    "WT17",
)
GROUPS = {"wind": ("WSF5", "WSF2"), "all": ("PRCP", "TMAX")}
CONUS = (24.0, 50.0, -125.5, -66.5)
OUTSIDE = {"AK", "HI", "PR", "VI", "GU", "AS", "MP"}
EARTH_KM = 6371.0


def read_stations(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="latin-1") as handle:
        for line in handle:
            if not line.startswith("USW") or len(line) < 40:
                continue
            try:
                lat, lon = float(line[12:20]), float(line[21:30])
            except ValueError:
                continue
            rows.append((line[0:11].strip(), lat, lon, line[38:40].strip(), line[41:71].strip()))
    frame = pd.DataFrame(rows, columns=["station", "lat", "lon", "state", "name"])
    inside = (
        frame["lat"].between(CONUS[0], CONUS[1])
        & frame["lon"].between(CONUS[2], CONUS[3])
        & ~frame["state"].isin(OUTSIDE)
    )
    return frame[inside].drop_duplicates("station").reset_index(drop=True)


def read_inventory(path: Path, first: int, last: int) -> pd.DataFrame:
    rows = []
    with open(path, encoding="latin-1") as handle:
        for line in handle:
            if not line.startswith("USW"):
                continue
            try:
                start, end = int(line[36:40]), int(line[41:45])
            except ValueError:
                continue
            if end >= first and start <= last:
                rows.append((line[0:11].strip(), line[31:35].strip()))
    return pd.DataFrame(rows, columns=["station", "element"]).drop_duplicates()


def station_map(
    units: pd.DataFrame, stations: pd.DataFrame, inventory: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    w = cfg.weather
    frames = []
    for group, elements in GROUPS.items():
        ids = set(inventory.loc[inventory["element"].isin(elements), "station"])
        pool = stations[stations["station"].isin(ids)].reset_index(drop=True)
        if pool.empty:
            continue
        k = min(w.neighbors, len(pool))
        tree = cKDTree(unit_vectors(pool["lat"], pool["lon"]))
        dist, idx = tree.query(
            unit_vectors(units["lat"], units["lon"]),
            k=k,
            distance_upper_bound=chord_km(w.radius_km) / EARTH_KM,
        )
        dist = np.asarray(dist).reshape(len(units), k)
        idx = np.asarray(idx).reshape(len(units), k)
        ok = idx < len(pool)
        rows, ranks = np.nonzero(ok)
        km = 2 * EARTH_KM * np.arcsin(np.clip(dist[rows, ranks] / 2, 0, 1))
        frames.append(
            pd.DataFrame(
                {
                    "unit_id": units["unit_id"].to_numpy()[rows],
                    "station": pool["station"].to_numpy()[idx[rows, ranks]],
                    "grp": group,
                    "rank": ranks + 1,
                    "dist_km": km,
                    "weight": 1.0 / np.maximum(km, 1.0) ** w.idw_power,
                }
            )
        )
    if not frames:
        raise ValueError("no weather stations with wind or precipitation in the inventory")
    return pd.concat(frames, ignore_index=True)


def silver_year(cfg: Config, con: duckdb.DuckDBPyConnection, year: int) -> None:
    src = cfg.paths.raw / "ghcn" / f"{year}.csv.gz"
    if not src.exists():
        raise FileNotFoundError(f"{src} is missing, run the download step first")
    dest = cfg.paths.silver / "ghcn" / f"{year}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "{'station': 'VARCHAR', 'date': 'VARCHAR', 'element': 'VARCHAR', 'value': 'VARCHAR', "
        "'mflag': 'VARCHAR', 'qflag': 'VARCHAR', 'sflag': 'VARCHAR', 'obstime': 'VARCHAR'}"
    )
    elements = ", ".join(f"'{e}'" for e in ELEMENTS)
    con.execute(f"""
        COPY (
            SELECT g.station, CAST(strptime(g.date, '%Y%m%d') AS DATE) AS day, g.element,
                   TRY_CAST(g.value AS INTEGER) AS value
            FROM read_csv('{sql_path(src)}', header = false, columns = {columns},
                          auto_detect = false, compression = 'gzip', quote = '', escape = '')
                 AS g
            SEMI JOIN ghcn_stations s ON s.station = g.station
            WHERE g.element IN ({elements}) AND coalesce(g.qflag, '') = ''
              AND TRY_CAST(g.value AS INTEGER) IS NOT NULL AND TRY_CAST(g.value AS INTEGER) > -9999
        ) TO '{sql_path(dest)}' (FORMAT parquet, COMPRESSION zstd)
    """)


def gold_year(cfg: Config, con: duckdb.DuckDBPyConnection, year: int) -> None:
    src = sql_path(cfg.paths.silver / "ghcn" / f"{year}.parquet")
    wind = "grp = 'wind'"
    rest = "grp = 'all'"
    con.execute(f"""
        INSERT INTO weather_day
        WITH wide AS (
            SELECT station, day,
                   max(value) FILTER (WHERE element = 'WSF5') / 10.0 AS wsf5,
                   max(value) FILTER (WHERE element = 'WSF2') / 10.0 AS wsf2,
                   max(value) FILTER (WHERE element = 'AWND') / 10.0 AS awnd,
                   max(value) FILTER (WHERE element = 'PRCP') / 10.0 AS prcp,
                   max(value) FILTER (WHERE element = 'SNOW') / 1.0 AS snow,
                   max(value) FILTER (WHERE element = 'SNWD') / 1.0 AS snwd,
                   max(value) FILTER (WHERE element = 'TMAX') / 10.0 AS tmax,
                   max(value) FILTER (WHERE element = 'TMIN') / 10.0 AS tmin,
                   max(CASE WHEN element IN ('WT06', 'WT15', 'WT17') THEN 1 ELSE 0 END) AS ice,
                   max(CASE WHEN element = 'WT03' THEN 1 ELSE 0 END) AS thunder
            FROM read_parquet('{src}') GROUP BY station, day
        ),
        j AS (SELECT m.unit_id, m.grp, m.weight, w.* FROM wide w JOIN station_map m USING (station))
        SELECT unit_id, day,
               sum(weight * wsf5) FILTER (WHERE {wind})
                   / sum(weight) FILTER (WHERE {wind} AND wsf5 IS NOT NULL) AS gust,
               sum(weight * wsf2) FILTER (WHERE {wind})
                   / sum(weight) FILTER (WHERE {wind} AND wsf2 IS NOT NULL) AS gust2,
               sum(weight * awnd) FILTER (WHERE {wind})
                   / sum(weight) FILTER (WHERE {wind} AND awnd IS NOT NULL) AS wind,
               max(wsf5) FILTER (WHERE {wind}) AS gust_max,
               count(wsf5) FILTER (WHERE {wind}) AS n_gust,
               sum(weight * prcp) FILTER (WHERE {rest})
                   / sum(weight) FILTER (WHERE {rest} AND prcp IS NOT NULL) AS prcp,
               sum(weight * snow) FILTER (WHERE {rest})
                   / sum(weight) FILTER (WHERE {rest} AND snow IS NOT NULL) AS snow,
               sum(weight * snwd) FILTER (WHERE {rest})
                   / sum(weight) FILTER (WHERE {rest} AND snwd IS NOT NULL) AS snwd,
               sum(weight * tmax) FILTER (WHERE {rest})
                   / sum(weight) FILTER (WHERE {rest} AND tmax IS NOT NULL) AS tmax,
               sum(weight * tmin) FILTER (WHERE {rest})
                   / sum(weight) FILTER (WHERE {rest} AND tmin IS NOT NULL) AS tmin,
               max(ice) FILTER (WHERE {rest}) AS ice,
               max(thunder) FILTER (WHERE {rest}) AS thunder
        FROM j GROUP BY unit_id, day
    """)


def build(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> None:
    raw = cfg.paths.raw / "ghcn"
    stations = read_stations(raw / "ghcnd-stations.txt")
    inventory = read_inventory(raw / "ghcnd-inventory.txt", cfg.years.first, cfg.years.last)
    units = con.execute("SELECT unit_id, lat, lon FROM units").df()
    mapping = station_map(units, stations, inventory, cfg)
    con.register("stations_df", stations)
    con.execute("CREATE OR REPLACE TABLE ghcn_stations AS SELECT * FROM stations_df")
    con.register("map_df", mapping)
    con.execute("CREATE OR REPLACE TABLE station_map AS SELECT * FROM map_df")
    con.unregister("stations_df")
    con.unregister("map_df")
    con.execute("""
        CREATE OR REPLACE TABLE weather_day (
            unit_id VARCHAR, day DATE, gust DOUBLE, gust2 DOUBLE, wind DOUBLE, gust_max DOUBLE,
            n_gust BIGINT, prcp DOUBLE, snow DOUBLE, snwd DOUBLE, tmax DOUBLE, tmin DOUBLE,
            ice INTEGER, thunder INTEGER)
    """)
    for year in cfg.years.all:
        log(f"weather {year}: reading station file")
        silver_year(cfg, con, year)
        gold_year(cfg, con, year)
    wind_units = mapping.loc[mapping["grp"] == "wind", "unit_id"].nunique()
    share = wind_units / max(len(units), 1)
    record_check(
        con,
        "weather",
        "units_with_wind_station",
        "warn" if share < 0.95 else "pass",
        share,
        f"within {cfg.weather.radius_km:.0f} km",
    )
    used = mapping.loc[mapping["grp"] == "wind", "station"].nunique()
    record_check(con, "weather", "wind_stations_used", "pass" if used else "fail", used)
    stats = con.execute("""
        SELECT max(gust_max), avg(CASE WHEN gust IS NULL THEN 0 ELSE 1 END), min(tmin), max(tmax)
        FROM weather_day
    """).fetchone()
    record_check(
        con, "weather", "max_gust_ms", "fail" if (stats[0] or 0) > 120 else "pass", stats[0] or 0
    )
    record_check(
        con,
        "weather",
        "unit_days_with_gust",
        "warn" if (stats[1] or 0) < 0.9 else "pass",
        stats[1] or 0,
    )
    bad_temp = (stats[2] or 0) < -65 or (stats[3] or 0) > 60
    record_check(
        con,
        "weather",
        "temperature_range",
        "fail" if bad_temp else "pass",
        stats[3] or 0,
        f"min {stats[2]} max {stats[3]}",
    )
