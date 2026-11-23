from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from .config import Config
from .db import record_check
from .refdata import states

CAUSES = {
    "tropical": [
        "Hurricane",
        "Hurricane (Typhoon)",
        "Tropical Storm",
        "Tropical Depression",
        "Storm Surge/Tide",
    ],
    "ice": ["Ice Storm", "Sleet"],
    "winter": ["Winter Storm", "Heavy Snow", "Blizzard", "Winter Weather", "Lake-Effect Snow"],
    "convective": ["Thunderstorm Wind", "Tornado", "Hail", "Lightning", "Funnel Cloud"],
    "wind": ["High Wind", "Strong Wind"],
    "flood": ["Flood", "Flash Flood", "Heavy Rain", "Coastal Flood"],
    "wildfire": ["Wildfire"],
    "heat": ["Heat", "Excessive Heat"],
    "cold": ["Cold/Wind Chill", "Extreme Cold/Wind Chill", "Frost/Freeze"],
}
PRIORITY = [
    "tropical",
    "ice",
    "winter",
    "convective",
    "wind",
    "flood",
    "wildfire",
    "heat",
    "cold",
    "other",
]
TYPE_CAUSE = {t.lower(): cause for cause, types in CAUSES.items() for t in types}
ZONE_OFFSETS = {
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "AKS": -9,
    "AKD": -8,
    "HST": -10,
    "AST": -4,
    "SST": -11,
    "GST": 10,
}
COLUMNS = [
    "BEGIN_YEARMONTH",
    "BEGIN_DAY",
    "BEGIN_TIME",
    "END_YEARMONTH",
    "END_DAY",
    "END_TIME",
    "EPISODE_ID",
    "EVENT_ID",
    "STATE_FIPS",
    "EVENT_TYPE",
    "CZ_TYPE",
    "CZ_FIPS",
    "CZ_TIMEZONE",
    "MAGNITUDE",
    "MAGNITUDE_TYPE",
]


def latest_files(folder: Path, years: list[int]) -> dict[int, Path]:
    found: dict[int, tuple[str, Path]] = {}
    for path in folder.glob("StormEvents_details-ftp_v1.0_d*_c*.csv.gz"):
        match = re.search(r"_d(\d{4})_c(\d{8})\.csv\.gz$", path.name)
        if not match:
            continue
        year, stamp = int(match.group(1)), match.group(2)
        if year in years and (year not in found or stamp > found[year][0]):
            found[year] = (stamp, path)
    missing = sorted(set(years) - set(found))
    if missing:
        raise FileNotFoundError(f"no Storm Events details file for {missing} in {folder}")
    return {year: found[year][1] for year in sorted(found)}


def tz_offsets(labels: pd.Series) -> pd.Series:
    text = labels.fillna("").str.strip().str.upper()
    numeric = pd.to_numeric(text.str.extract(r"([+-]?\d{1,2})$")[0], errors="coerce")
    named = text.str[:3].map(ZONE_OFFSETS)
    return numeric.fillna(named)


def local_times(yearmonth: pd.Series, day: pd.Series, hhmm: pd.Series) -> pd.Series:
    ym = pd.to_numeric(yearmonth, errors="coerce")
    base = pd.to_datetime(
        pd.DataFrame(
            {"year": ym // 100, "month": ym % 100, "day": pd.to_numeric(day, errors="coerce")}
        ),
        errors="coerce",
    )
    clock = pd.to_numeric(hhmm, errors="coerce").fillna(0)
    return base + pd.to_timedelta(clock // 100, unit="h") + pd.to_timedelta(clock % 100, unit="m")


def parse(frame: pd.DataFrame) -> pd.DataFrame:
    offsets = tz_offsets(frame["CZ_TIMEZONE"])
    begin = local_times(frame["BEGIN_YEARMONTH"], frame["BEGIN_DAY"], frame["BEGIN_TIME"])
    end = local_times(frame["END_YEARMONTH"], frame["END_DAY"], frame["END_TIME"])
    shift = pd.to_timedelta(offsets, unit="h")
    begin_utc = begin - shift
    end_utc = (end - shift).where(end >= begin, begin_utc)
    end_utc = end_utc.clip(upper=begin_utc + pd.Timedelta(days=5))
    kind = frame["EVENT_TYPE"].fillna("").str.strip()
    cause = kind.str.lower().map(TYPE_CAUSE).fillna("other")
    out = pd.DataFrame(
        {
            "event_id": pd.to_numeric(frame["EVENT_ID"], errors="coerce"),
            "episode_id": pd.to_numeric(frame["EPISODE_ID"], errors="coerce"),
            "event_type": kind,
            "cause": cause,
            "priority": cause.map({c: i for i, c in enumerate(PRIORITY)}),
            "cz_type": frame["CZ_TYPE"].fillna("").str.strip().str.upper(),
            "state_fips": pd.to_numeric(frame["STATE_FIPS"], errors="coerce"),
            "cz_fips": pd.to_numeric(frame["CZ_FIPS"], errors="coerce"),
            "begin_utc": begin_utc,
            "end_utc": end_utc,
            "magnitude": pd.to_numeric(frame["MAGNITUDE"], errors="coerce"),
            "magnitude_type": frame["MAGNITUDE_TYPE"].fillna("").str.strip(),
            "tz_named": offsets.notna()
            & frame["CZ_TIMEZONE"].fillna("").str.strip().str[-1:].str.isalpha(),
        }
    )
    return out


def read_all(cfg: Config) -> pd.DataFrame:
    frames = []
    for year, path in latest_files(cfg.paths.raw / "storm_events", cfg.years.all).items():
        raw = pd.read_csv(path, dtype=str, usecols=COLUMNS, encoding="latin-1", compression="gzip")
        frames.append(parse(raw).assign(file_year=year))
    return pd.concat(frames, ignore_index=True)


def map_units(reports: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    info = states()
    abbr = dict(zip(info["state_fips"].astype(int), info["abbr"], strict=True))
    reports = reports.dropna(subset=["begin_utc", "state_fips", "cz_fips", "event_id"]).copy()
    reports["fips"] = reports["state_fips"].astype(int).astype(str).str.zfill(2) + reports[
        "cz_fips"
    ].astype(int).astype(str).str.zfill(3)
    reports["state_zone"] = reports["state_fips"].astype(int).map(abbr).fillna("") + reports[
        "cz_fips"
    ].astype(int).astype(str).str.zfill(3)
    fips_unit = con.execute("SELECT fips, unit_id FROM fips_unit").df()
    zone_unit = con.execute("SELECT state_zone, unit_id FROM zone_unit").df()
    counties = reports[reports["cz_type"] == "C"].merge(fips_unit, on="fips", how="inner")
    zones = reports[reports["cz_type"] == "Z"].merge(zone_unit, on="state_zone", how="inner")
    mapped = pd.concat([counties, zones], ignore_index=True)
    return mapped.drop(columns=["fips", "state_zone"])


def build(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> None:
    log("storms: reading NOAA Storm Events")
    reports = read_all(cfg)
    land = reports[reports["cz_type"].isin(["C", "Z"])]
    mapped = map_units(reports, con)
    mapped_ids = set(mapped["event_id"])
    unmapped = 1 - land["event_id"].isin(mapped_ids).mean() if len(land) else 0.0
    record_check(con, "storms", "reports", "pass" if len(reports) else "fail", len(reports))
    record_check(
        con,
        "storms",
        "unmapped_land_share",
        "warn" if unmapped > 0.05 else "pass",
        unmapped,
        "county or zone reports that match no CONUS unit",
    )
    named = float(reports["tz_named"].mean()) if len(reports) else 0.0
    record_check(con, "storms", "timezone_name_fallback_share", "pass", named)
    bad_tz = float(reports["begin_utc"].isna().mean()) if len(reports) else 0.0
    record_check(con, "storms", "unparsed_time_share", "warn" if bad_tz > 0.01 else "pass", bad_tz)
    mapped = mapped.drop(columns=["tz_named"])
    con.register("reports_df", mapped)
    con.execute("CREATE OR REPLACE TABLE storm_reports AS SELECT * FROM reports_df")
    con.unregister("reports_df")
    labels = ", ".join(f"({i}, '{c}')" for i, c in enumerate(PRIORITY))
    con.execute(f"""
        CREATE OR REPLACE TABLE unit_cause_day AS
        WITH spans AS (
            SELECT r.unit_id, r.priority,
                   CAST(r.begin_utc + to_hours(u.offset_hours) AS DATE) AS d0,
                   CAST(r.end_utc + to_hours(u.offset_hours) AS DATE) AS d1
            FROM storm_reports r JOIN units u USING (unit_id)
        ),
        days AS (
            SELECT unit_id, priority,
                   CAST(unnest(generate_series(d0, d1, INTERVAL 1 DAY)) AS DATE) AS day
            FROM spans
        ),
        names (priority, cause) AS (VALUES {labels})
        SELECT d.unit_id, d.day, n.cause
        FROM (SELECT unit_id, day, min(priority) AS priority FROM days GROUP BY ALL) d
        JOIN names n USING (priority)
    """)
    s = cfg.storms
    con.execute(f"""
        UPDATE events SET cause = c.cause
        FROM (
            SELECT e.event_id, arg_min(r.cause, r.priority) AS cause
            FROM events e JOIN storm_reports r
              ON r.unit_id = e.unit_id
             AND r.cause <> 'other'
             AND r.begin_utc <= e.peak_ts + INTERVAL '{int(s.lookahead_hours * 60)} minutes'
             AND r.end_utc >= e.start_ts - INTERVAL '{int(s.lookback_hours * 60)} minutes'
            GROUP BY e.event_id
        ) c
        WHERE events.event_id = c.event_id
    """)
    share = con.execute(
        "SELECT avg(CASE WHEN cause = 'none' THEN 0 ELSE 1 END) FROM events"
    ).fetchone()[0]
    record_check(con, "storms", "events_with_cause", "pass", share or 0.0)
