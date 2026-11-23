from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from storm_outage_forecast import outages
from storm_outage_forecast.db import connect
from storm_outage_forecast.units import coverage_year, read_coverage

from .conftest import make_cfg


def build_db(cfg):
    con = connect(cfg)
    units = pd.DataFrame(
        {
            "unit_id": ["39001", "39003", "09000"],
            "offset_hours": [-5, -5, -5],
            "customers": [10000, 5000, 20000],
            "abbr": ["OH", "OH", "CT"],
            "density": [100.0, 50.0, 400.0],
            "lat": [40.0, 40.1, 41.5],
            "lon": [-83.0, -83.1, -72.7],
        }
    )
    codes = pd.DataFrame(
        {
            "fips": ["39001", "39003", "09001", "09110"],
            "unit_id": ["39001", "39003", "09000", "09000"],
        }
    )
    con.register("u", units)
    con.execute("CREATE TABLE units AS SELECT * FROM u")
    con.register("c", codes)
    con.execute("CREATE TABLE fips_unit AS SELECT * FROM c")
    con.execute(
        "CREATE TABLE state_coverage (year INTEGER, abbr VARCHAR, max_pct DOUBLE, min_pct DOUBLE)"
    )
    return con


def write_csv(cfg, lines: list[bytes]) -> None:
    folder = cfg.paths.raw / "eaglei"
    folder.mkdir(parents=True, exist_ok=True)
    header = b"fips_code,county,state,sum,run_start_time\n"
    (folder / "eaglei_outages_2020.csv").write_bytes(header + b"".join(lines))


def row(fips: str, out, stamp: datetime | str, county: bytes = b"Adams") -> bytes:
    text = stamp if isinstance(stamp, str) else stamp.strftime("%Y-%m-%d %H:%M:%S")
    return (
        fips.encode() + b"," + county + b",Ohio," + str(out).encode() + b"," + text.encode() + b"\n"
    )


@pytest.fixture
def scenario(tmp_path):
    cfg = make_cfg(tmp_path, outages={"min_covered_days": 1})
    t0 = datetime(2020, 3, 1)
    lines = []
    gap = {t0 + timedelta(days=1, hours=12, minutes=m) for m in (30, 45, 60)}
    moment = t0
    while moment < t0 + timedelta(days=3):
        if moment not in gap:
            lines.append(row("39003", 5, moment))
        moment += timedelta(minutes=15)
    first = t0 + timedelta(hours=10)
    for minutes, out in ((0, 50), (15, 200), (30, 1000), (45, 600), (60, 90)):
        lines.append(row("39001", out, first + timedelta(minutes=minutes)))
    lines.append(row("39001", 1000, first + timedelta(minutes=30)))
    lines.append(row("39001", 650, first + timedelta(minutes=45)))
    second = t0 + timedelta(days=1, hours=12)
    lines += [row("39001", 500, second), row("39001", 450, second + timedelta(minutes=15))]
    third = t0 + timedelta(days=2, hours=6)
    for minutes, out in ((0, 2000), (15, 1500), (30, 150), (45, 120), (60, 110)):
        lines.append(row("39001", out, third + timedelta(minutes=minutes)))
    last = t0 + timedelta(days=2, hours=23, minutes=30)
    lines += [row("39001", 300, last), row("39001", 300, last + timedelta(minutes=15))]
    lines.append(row("9001", 100, t0 + timedelta(hours=12), b"Fairfield"))
    lines.append(row("09110", 200, t0 + timedelta(hours=12), b"Capitol"))
    lines.append(row("39999", 7, t0 + timedelta(hours=12), b"Espa\xf1ola"))
    lines.append(row("02020", 9, t0 + timedelta(hours=12), b"Anchorage"))
    lines.append(row("39001", 45, "2020-03-01 05:00:00+00:00"))
    lines.append(row("39001", 5, "not-a-date"))
    lines.append(row("39001", "abc", t0 + timedelta(hours=3)))
    lines.append(row("39001", 40, "03/01/20 04:00"))
    write_csv(cfg, lines)
    con = build_db(cfg)
    stats = outages.silver_year(cfg, con, 2020)
    outages.create_work_tables(con)
    outages.gold_year(cfg, con, 2020)
    outages.finish_gold(cfg, con)
    yield cfg, con, stats
    con.close()


def test_header_aliases():
    assert outages.ALIASES["sum"] == "out"
    assert outages.ALIASES["customers_out"] == "out"


def test_header_without_value_column_fails(tmp_path):
    path = tmp_path / "eaglei_outages_2020.csv"
    path.write_bytes(b"\xef\xbb\xbfFIPS_CODE,county,state,run_start_time\n")
    with pytest.raises(ValueError, match="one out column"):
        outages.eaglei_columns(path)
    path.write_bytes(b"\xef\xbb\xbfFIPS_CODE,county,state,Customers_Out,Run_Start_Time\n")
    assert list(outages.eaglei_columns(path)) == ["fips", "county", "state", "out", "ts"]


def test_silver_statistics(scenario):
    cfg, con, stats = scenario
    assert stats["bad_timestamps"] == 1
    assert stats["bad_values"] == 1
    assert stats["duplicate_rows"] == 2
    assert stats["conflicting_duplicates"] == 1
    assert stats["unmatched_rows"] == 1
    assert stats["unmatched_sample"] == "39999"
    assert stats["outside_rows"] == 1
    silver = con.execute(
        f"SELECT * FROM read_parquet('{cfg.paths.silver}/outages/2020.parquet')"
    ).df()
    ct = silver[silver["unit_id"] == "09000"]
    assert ct["out"].tolist() == [300]
    parsed = silver[
        (silver["unit_id"] == "39001") & (silver["ts"] == pd.Timestamp("2020-03-01 04:00"))
    ]
    assert parsed["out"].tolist() == [40]
    suffixed = silver[
        (silver["unit_id"] == "39001") & (silver["ts"] == pd.Timestamp("2020-03-01 05:00"))
    ]
    assert suffixed["out"].tolist() == [45]
    conflict = silver[
        (silver["unit_id"] == "39001") & (silver["ts"] == pd.Timestamp("2020-03-01 10:45"))
    ]
    assert conflict["out"].tolist() == [650]


def test_events_restoration_and_censoring(scenario):
    _, con, _ = scenario
    events = con.execute("""
        SELECT start_ts, peak_ts, peak_out, end_ts, duration_h, censored
        FROM events WHERE unit_id = '39001' ORDER BY start_ts
    """).df()
    assert len(events) == 4
    first, second, third, fourth = events.to_dict(orient="records")
    assert first["start_ts"] == pd.Timestamp("2020-03-01 10:15")
    assert first["peak_out"] == 1000
    assert first["end_ts"] == pd.Timestamp("2020-03-01 11:00")
    assert first["duration_h"] == pytest.approx(0.5)
    assert not first["censored"]
    assert second["censored"]
    assert second["duration_h"] == pytest.approx(0.5)
    assert third["end_ts"] == pd.Timestamp("2020-03-03 06:30")
    assert third["duration_h"] == pytest.approx(0.5)
    assert not third["censored"]
    assert fourth["censored"]


def test_state_load_uses_state_total(scenario):
    _, con, _ = scenario
    load = con.execute("""
        SELECT state_load FROM events WHERE unit_id = '39001' AND peak_out = 2000
    """).fetchone()[0]
    assert load == pytest.approx((2000 + 5) / 15000)


def test_county_day_and_gaps(scenario):
    _, con, _ = scenario
    gaps = con.execute("SELECT count(*) FROM snapshot_flags WHERE gap").fetchone()[0]
    assert gaps == 3
    day = con.execute("""
        SELECT peak_out, cust_minutes, gap_minutes, complete, unit_covered
        FROM unit_day WHERE unit_id = '39001' AND day = DATE '2020-03-02'
    """).fetchone()
    assert day == (500, (500 + 450) * 15, 45, True, True)


def test_rows_far_above_customers_are_dropped(tmp_path):
    cfg = make_cfg(tmp_path, outages={"min_covered_days": 1})
    t0 = datetime(2020, 5, 1)
    lines = [row("39003", 5, t0 + timedelta(minutes=15 * i)) for i in range(400)]
    lines.append(row("39003", 99999, t0 + timedelta(hours=34)))
    write_csv(cfg, lines)
    con = build_db(cfg)
    outages.silver_year(cfg, con, 2020)
    outages.create_work_tables(con)
    outages.gold_year(cfg, con, 2020)
    outages.finish_gold(cfg, con)
    peak, over = con.execute(
        "SELECT max(peak_out), sum(over_rows) FROM unit_day WHERE unit_id = '39003'"
    ).fetchone()
    assert peak == 5 and over == 1
    assert con.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    con.close()


def test_coverage_year_formats(tmp_path):
    assert [coverage_year(v) for v in ("1/1/18", "2018-11-01", "2018", "12/31/2019")] == [
        2018,
        2018,
        2018,
        2019,
    ]
    with pytest.raises(ValueError):
        coverage_year("last winter")
    path = tmp_path / "coverage_history.csv"
    path.write_text(
        "year,state,total_customers,min_covered,max_covered,min_pct_covered,max_pct_covered\n"
        "1/1/18,AK,100,60,67,0.6,0.67\n"
        "1/1/19,Alabama,100,80,90,0.8,0.9\n",
        encoding="utf-8",
    )
    frame = read_coverage(path).sort_values("year").reset_index(drop=True)
    assert list(frame["year"]) == [2018, 2019]
    assert list(frame["abbr"]) == ["AK", "AL"]
    assert list(frame["max_pct"]) == [0.67, 0.9]
    percent = tmp_path / "percent.csv"
    percent.write_text(
        "Year,State,Total Customers,Min_covered,Max_covered,Min_pct_covered,Max_pct_covered\n"
        "2018-11-01,Ohio,100,60,67,60.0,67.0\n",
        encoding="utf-8",
    )
    assert read_coverage(percent)["max_pct"].iloc[0] == pytest.approx(0.67)


def test_bad_value_share_uses_the_config_limit(tmp_path):
    cfg = make_cfg(tmp_path, outages={"max_bad_share": 0.2})
    con = connect(cfg)
    stats = [
        {
            "rows": 1000,
            "bad_timestamps": 0,
            "bad_values": 100,
            "bad_fips": 0,
            "valid_rows": 900,
            "duplicate_rows": 0,
            "conflicting_duplicates": 0,
            "unmatched_rows": 0,
            "unmatched_sample": "",
            "outside_rows": 0,
        }
    ]
    outages.record_silver_checks(cfg, con, stats)
    status, value = con.execute(
        "SELECT status, value FROM quality WHERE check_name = 'bad_value_share'"
    ).fetchone()
    assert status == "warn"
    assert value == pytest.approx(0.1)
    outages.record_silver_checks(make_cfg(tmp_path, outages={"max_bad_share": 0.05}), con, stats)
    assert (
        con.execute("SELECT status FROM quality WHERE check_name = 'bad_value_share'").fetchone()[0]
        == "fail"
    )
    stats[0]["bad_values"] = 0
    outages.record_silver_checks(cfg, con, stats)
    assert (
        con.execute("SELECT status FROM quality WHERE check_name = 'bad_value_share'").fetchone()[0]
        == "pass"
    )
    con.close()
