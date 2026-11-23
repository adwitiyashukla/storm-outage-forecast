from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from storm_outage_forecast import storms
from storm_outage_forecast.db import connect

from .conftest import make_cfg


def test_timezone_offsets():
    labels = pd.Series(["EST-5", "CST-6", "CSC-6", "GST10", "PST", "AKST-9", "", None, "XYZ"])
    got = storms.tz_offsets(labels).tolist()
    assert got[:6] == [-5, -6, -6, 10, -8, -9]
    assert all(np.isnan(v) for v in got[6:])


def test_local_times():
    stamps = storms.local_times(
        pd.Series(["202003", "202012"]), pd.Series(["1", "31"]), pd.Series(["45", "2330"])
    )
    assert stamps.tolist() == [pd.Timestamp("2020-03-01 00:45"), pd.Timestamp("2020-12-31 23:30")]


def frame(**changes) -> pd.DataFrame:
    base = {
        "BEGIN_YEARMONTH": "202007",
        "BEGIN_DAY": "4",
        "BEGIN_TIME": "1800",
        "END_YEARMONTH": "202007",
        "END_DAY": "4",
        "END_TIME": "1700",
        "EPISODE_ID": "1",
        "EVENT_ID": "10",
        "STATE_FIPS": "39",
        "EVENT_TYPE": "Thunderstorm Wind",
        "CZ_TYPE": "C",
        "CZ_FIPS": "1",
        "CZ_TIMEZONE": "EST-5",
        "MAGNITUDE": "55",
        "MAGNITUDE_TYPE": "MG",
    }
    base.update(changes)
    return pd.DataFrame([base])


def test_parse_converts_to_utc_and_fixes_end():
    out = storms.parse(frame()).iloc[0]
    assert out["begin_utc"] == pd.Timestamp("2020-07-04 23:00")
    assert out["end_utc"] == out["begin_utc"]
    assert out["cause"] == "convective"
    assert out["priority"] == storms.PRIORITY.index("convective")


@pytest.mark.parametrize(
    ("kind", "cause"),
    [
        ("Hurricane (Typhoon)", "tropical"),
        ("Ice Storm", "ice"),
        ("Blizzard", "winter"),
        ("High Wind", "wind"),
        ("Flash Flood", "flood"),
        ("Dense Fog", "other"),
        ("Wildfire", "wildfire"),
    ],
)
def test_cause_groups(kind, cause):
    assert storms.parse(frame(EVENT_TYPE=kind)).iloc[0]["cause"] == cause


def test_long_events_are_capped():
    out = storms.parse(frame(END_YEARMONTH="202009", END_DAY="1")).iloc[0]
    assert out["end_utc"] - out["begin_utc"] == pd.Timedelta(days=5)


def test_latest_file_per_year(tmp_path):
    for name in (
        "StormEvents_details-ftp_v1.0_d2020_c20240101.csv.gz",
        "StormEvents_details-ftp_v1.0_d2020_c20260101.csv.gz",
        "StormEvents_details-ftp_v1.0_d2021_c20250101.csv.gz",
    ):
        (tmp_path / name).write_bytes(b"")
    files = storms.latest_files(tmp_path, [2020, 2021])
    assert files[2020].name.endswith("c20260101.csv.gz")
    with pytest.raises(FileNotFoundError, match="2022"):
        storms.latest_files(tmp_path, [2020, 2022])


def test_county_and_zone_reports_map_to_units(tmp_path):
    cfg = make_cfg(tmp_path)
    con = connect(cfg)
    con.execute(
        "CREATE TABLE fips_unit AS SELECT * FROM "
        "(VALUES ('39001', '39001'), ('09001', '09000')) t(fips, unit_id)"
    )
    con.execute(
        "CREATE TABLE zone_unit AS SELECT * FROM "
        "(VALUES ('OH005', '39001'), ('OH005', '39003')) t(state_zone, unit_id)"
    )
    reports = pd.concat(
        [
            storms.parse(frame()),
            storms.parse(frame(EVENT_ID="11", CZ_TYPE="Z", CZ_FIPS="5", EVENT_TYPE="High Wind")),
            storms.parse(frame(EVENT_ID="12", STATE_FIPS="9", CZ_FIPS="1")),
            storms.parse(frame(EVENT_ID="13", CZ_TYPE="M", CZ_FIPS="42")),
            storms.parse(frame(EVENT_ID="14", STATE_FIPS="2", CZ_FIPS="20")),
        ],
        ignore_index=True,
    )
    mapped = storms.map_units(reports, con)
    pairs = sorted(zip(mapped["event_id"], mapped["unit_id"], strict=True))
    assert pairs == [(10, "39001"), (11, "39001"), (11, "39003"), (12, "09000")]
    con.close()
