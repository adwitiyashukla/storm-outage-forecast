from __future__ import annotations

import contextlib
from pathlib import Path

import duckdb

from .config import Config
from .db import record_check, sql_path
from .refdata import states

ALIASES = {
    "fips_code": "fips",
    "fips": "fips",
    "county_fips": "fips",
    "geoid": "fips",
    "customers_out": "out",
    "sum": "out",
    "customers": "out",
    "customer_out": "out",
    "run_start_time": "ts",
    "timestamp": "ts",
    "county": "county",
    "state": "state",
}


def eaglei_columns(path: Path) -> dict[str, str]:
    with open(path, "rb") as handle:
        line = handle.readline().removeprefix(b"\xef\xbb\xbf").decode("latin-1").strip()
    names = [
        ALIASES.get(n.strip().strip('"').lower(), n.strip().strip('"').lower())
        for n in line.split(",")
    ]
    for needed in ("fips", "out", "ts"):
        if names.count(needed) != 1:
            raise ValueError(f"{path.name}: expected one {needed} column in header {line!r}")
    columns = {}
    for i, name in enumerate(names):
        columns[name if name not in columns else f"extra_{i}"] = "VARCHAR"
    return columns


def stage_year(cfg: Config, con: duckdb.DuckDBPyConnection, year: int) -> Path:
    src = cfg.paths.raw / "eaglei" / f"eaglei_outages_{year}.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} is missing, run the download step first")
    stage = cfg.paths.tmp / f"eaglei_stage_{year}.parquet"
    stage.parent.mkdir(parents=True, exist_ok=True)
    columns = eaglei_columns(src)
    spec = "{" + ", ".join(f"'{k}': '{v}'" for k, v in columns.items()) + "}"
    con.execute(f"""
        COPY (
            WITH parsed AS (
                SELECT
                    lpad(split_part(trim(fips), '.', 1), 5, '0') AS fips,
                    CASE WHEN substr(trim(ts), 5, 1) = '-'
                         THEN TRY_CAST(regexp_replace(trim(ts), '(Z|[+]00:?00)$', '') AS TIMESTAMP)
                         ELSE coalesce(try_strptime(trim(ts), '%m/%d/%y %H:%M'),
                                       try_strptime(trim(ts), '%m/%d/%Y %H:%M'),
                                       try_strptime(trim(ts), '%m/%d/%y %H:%M:%S'))
                    END AS ts,
                    TRY_CAST(TRY_CAST(trim("out") AS DOUBLE) AS BIGINT) AS out
                FROM read_csv('{sql_path(src)}', header = true, columns = {spec},
                              encoding = 'latin-1', auto_detect = false, quote = '"', escape = '"')
            )
            SELECT fips,
                   CASE WHEN ts >= TIMESTAMP '{year}-01-01' - INTERVAL 1 DAY
                         AND ts < TIMESTAMP '{year + 1}-01-01' + INTERVAL 1 DAY THEN ts END AS ts,
                   out
            FROM parsed
        ) TO '{sql_path(stage)}' (FORMAT parquet, COMPRESSION zstd)
    """)
    return stage


def remove(path: Path) -> None:
    with contextlib.suppress(PermissionError):
        path.unlink(missing_ok=True)


def silver_year(cfg: Config, con: duckdb.DuckDBPyConnection, year: int) -> dict:
    stage = stage_year(cfg, con, year)
    dest = cfg.paths.silver / "outages" / f"{year}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    s = sql_path(stage)
    conus = ", ".join(f"'{code}'" for code in states()["state_fips"])
    stats = con.execute(f"""
        WITH raw AS (SELECT * FROM read_parquet('{s}')),
        valid AS (
            SELECT fips, time_bucket(INTERVAL 15 MINUTE, ts) AS ts, out FROM raw
            WHERE ts IS NOT NULL AND out IS NOT NULL AND out > 0
              AND regexp_full_match(fips, '[0-9]{{5}}')
        ),
        grouped AS (SELECT fips, ts, count(*) AS n, max(out) <> min(out) AS conflict
                    FROM valid GROUP BY fips, ts)
        SELECT
            (SELECT count(*) FROM raw),
            (SELECT count(*) FROM raw WHERE ts IS NULL),
            (SELECT count(*) FROM raw WHERE out IS NULL OR out < 0),
            (SELECT count(*) FROM raw WHERE NOT regexp_full_match(fips, '[0-9]{{5}}')),
            (SELECT count(*) FROM valid),
            (SELECT sum(n - 1) FROM grouped),
            (SELECT count(*) FROM grouped WHERE conflict),
            (SELECT count(*) FROM valid v ANTI JOIN fips_unit f USING (fips)
             WHERE substr(v.fips, 1, 2) IN ({conus})),
            (SELECT string_agg(DISTINCT fips, ' ') FROM
                (SELECT v.fips FROM valid v ANTI JOIN fips_unit f USING (fips)
                 WHERE substr(v.fips, 1, 2) IN ({conus}) LIMIT 200)),
            (SELECT count(*) FROM valid WHERE substr(fips, 1, 2) NOT IN ({conus}))
    """).fetchone()
    con.execute(f"""
        COPY (
            SELECT f.unit_id, d.ts, CAST(sum(d.out) AS INTEGER) AS out
            FROM (
                SELECT fips, time_bucket(INTERVAL 15 MINUTE, ts) AS ts, max(out) AS out
                FROM read_parquet('{s}')
                WHERE ts IS NOT NULL AND out IS NOT NULL AND out > 0
                GROUP BY ALL
            ) d JOIN fips_unit f USING (fips)
            GROUP BY ALL
            ORDER BY f.unit_id, d.ts
        ) TO '{sql_path(dest)}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
    """)
    remove(stage)
    keys = [
        "rows",
        "bad_timestamps",
        "bad_values",
        "bad_fips",
        "valid_rows",
        "duplicate_rows",
        "conflicting_duplicates",
        "unmatched_rows",
        "unmatched_sample",
        "outside_rows",
    ]
    result = dict(zip(keys, stats, strict=True))
    result["year"] = year
    return result


def record_silver_checks(cfg: Config, con: duckdb.DuckDBPyConnection, stats: list[dict]) -> None:
    total = sum(s["rows"] for s in stats) or 1
    valid = sum(s["valid_rows"] for s in stats) or 1
    bad_ts = sum(s["bad_timestamps"] for s in stats)
    bad_val = sum(s["bad_values"] for s in stats)
    dup = sum(s["duplicate_rows"] or 0 for s in stats)
    conflict = sum(s["conflicting_duplicates"] or 0 for s in stats)
    unmatched = sum(s["unmatched_rows"] for s in stats)
    outside = sum(s["outside_rows"] for s in stats)
    record_check(
        con,
        "outages",
        "outside_conus_share",
        "pass",
        outside / valid,
        "Alaska, Hawaii and territories, left out by design",
    )
    sample = " ".join(
        sorted({f for s in stats for f in (s["unmatched_sample"] or "").split()})[:30]
    )
    record_check(con, "outages", "raw_rows", "pass", total)
    limit = cfg.outages.max_bad_share
    record_check(
        con,
        "outages",
        "bad_timestamp_share",
        "fail" if bad_ts / total > limit else ("warn" if bad_ts else "pass"),
        bad_ts / total,
        "rows whose run_start_time could not be read",
    )
    record_check(
        con,
        "outages",
        "bad_value_share",
        "fail" if bad_val / total > limit else ("warn" if bad_val else "pass"),
        bad_val / total,
        "rows with a blank or negative customer count, dropped",
    )
    record_check(
        con, "outages", "duplicate_share", "warn" if dup / valid > 0.01 else "pass", dup / valid
    )
    record_check(con, "outages", "conflicting_duplicates", "warn" if conflict else "pass", conflict)
    share = unmatched / valid
    status = "fail" if share > 0.05 else ("warn" if share > 0.002 else "pass")
    record_check(con, "outages", "unmatched_fips_share", status, share, sample)


def gold_year(cfg: Config, con: duckdb.DuckDBPyConnection, year: int) -> None:
    o = cfg.outages
    src = sql_path(cfg.paths.silver / "outages" / f"{year}.parquet")
    prev = sql_path(cfg.paths.silver / "outages" / f"{year - 1}.parquet")
    nxt = sql_path(cfg.paths.silver / "outages" / f"{year + 1}.parquet")
    ok = "t.out <= 1.5 * u.customers"
    con.execute(f"""
        INSERT INTO unit_day_parts
        SELECT t.unit_id, CAST(t.ts + to_hours(u.offset_hours) AS DATE) AS day,
               coalesce(max(t.out) FILTER (WHERE {ok}), 0) AS peak_out,
               coalesce(sum(t.out) FILTER (WHERE {ok}), 0) * 15 AS cust_minutes,
               count(*) AS rows,
               max(t.out) FILTER (WHERE {ok} AND hour(t.ts + to_hours(u.offset_hours)) = 0
                                  AND minute(t.ts) = 0) AS start_out,
               count(*) FILTER (WHERE NOT ({ok})) AS over_rows
        FROM read_parquet('{src}') t JOIN units u USING (unit_id)
        GROUP BY ALL
    """)
    con.execute(f"""
        INSERT INTO snapshots
        SELECT ts, count(*) FROM read_parquet('{src}') GROUP BY ts
    """)
    con.execute(f"""
        INSERT INTO state_ts
        SELECT u.abbr, t.ts, sum(t.out) FROM read_parquet('{src}') t JOIN units u USING (unit_id)
        WHERE {ok} GROUP BY ALL
    """)
    files = [f"'{src}'"]
    if Path(prev).exists():
        files.append(f"'{prev}'")
    if Path(nxt).exists():
        files.append(f"'{nxt}'")
    level = f"greatest({o.event_floor}, {o.event_share} * u.customers)"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE active AS
        SELECT t.unit_id, t.ts, t.out
        FROM read_parquet([{", ".join(files)}]) t JOIN units u USING (unit_id)
        WHERE t.ts >= TIMESTAMP '{year}-01-01' - INTERVAL 3 DAY
          AND t.ts < TIMESTAMP '{year + 1}-01-01' + INTERVAL 15 DAY
          AND t.out >= {level} AND {ok}
    """)
    window_end = f"TIMESTAMP '{year + 1}-01-01' + INTERVAL 15 DAY"
    gap = int(o.merge_gap_hours * 60)
    con.execute(f"""
        INSERT INTO events_raw
        WITH marked AS (
            SELECT unit_id, ts, out,
                   CASE WHEN lag(ts) OVER w IS NULL
                          OR ts - lag(ts) OVER w > INTERVAL '{gap} minutes' THEN 1 ELSE 0 END AS brk
            FROM active WINDOW w AS (PARTITION BY unit_id ORDER BY ts)
        ),
        seqd AS (
            SELECT *,
                   sum(brk) OVER (PARTITION BY unit_id ORDER BY ts ROWS UNBOUNDED PRECEDING) AS seq
            FROM marked
        ),
        agg AS (
            SELECT unit_id, seq, min(ts) AS start_ts, max(ts) AS last_ts, max(out) AS peak_out,
                   sum(out) * 15 AS cust_minutes, count(*) AS n_rows
            FROM seqd GROUP BY unit_id, seq
        ),
        peaks AS (
            SELECT s.unit_id, s.seq, min(s.ts) AS peak_ts
            FROM seqd s JOIN agg a USING (unit_id, seq)
            WHERE s.out = a.peak_out GROUP BY ALL
        ),
        restored AS (
            SELECT s.unit_id, s.seq, min(s.ts) AS restore_ts
            FROM seqd s
            JOIN agg a USING (unit_id, seq)
            JOIN peaks p USING (unit_id, seq)
            JOIN units u USING (unit_id)
            WHERE s.ts > p.peak_ts AND s.out <= greatest({level}, {o.restore_share} * a.peak_out)
            GROUP BY ALL
        )
        SELECT a.unit_id, a.start_ts, p.peak_ts, a.last_ts, r.restore_ts, a.peak_out,
               a.cust_minutes, a.n_rows, a.last_ts + INTERVAL '{gap} minutes' >= {window_end}
        FROM agg a JOIN peaks p USING (unit_id, seq) LEFT JOIN restored r USING (unit_id, seq)
        WHERE a.start_ts >= TIMESTAMP '{year}-01-01' AND a.start_ts < TIMESTAMP '{year + 1}-01-01'
    """)
    con.execute("DROP TABLE active")


def create_work_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE unit_day_parts (
            unit_id VARCHAR, day DATE, peak_out BIGINT, cust_minutes BIGINT, rows BIGINT,
            start_out BIGINT, over_rows BIGINT)
    """)
    con.execute("CREATE OR REPLACE TABLE snapshots (ts TIMESTAMP, rows BIGINT)")
    con.execute("CREATE OR REPLACE TABLE state_ts (abbr VARCHAR, ts TIMESTAMP, out BIGINT)")
    con.execute("""
        CREATE OR REPLACE TABLE events_raw (
            unit_id VARCHAR, start_ts TIMESTAMP, peak_ts TIMESTAMP, last_ts TIMESTAMP,
            restore_ts TIMESTAMP, peak_out BIGINT, cust_minutes BIGINT, n_rows BIGINT,
            window_cut BOOLEAN)
    """)


def finish_gold(cfg: Config, con: duckdb.DuckDBPyConnection) -> None:
    o = cfg.outages
    y = cfg.years
    gap = int(o.merge_gap_hours * 60)
    con.execute("""
        CREATE OR REPLACE TABLE unit_day_obs AS
        SELECT unit_id, day, max(peak_out) AS peak_out, sum(cust_minutes) AS cust_minutes,
               sum(rows) AS rows, max(start_out) AS start_out, sum(over_rows) AS over_rows
        FROM unit_day_parts GROUP BY ALL
    """)
    con.execute("""
        CREATE OR REPLACE TABLE snapshot_flags AS
        WITH snaps AS (SELECT ts, sum(rows) AS rows FROM snapshots GROUP BY ts),
        bounds AS (SELECT min(ts) AS lo, max(ts) AS hi FROM snaps),
        grid AS (SELECT unnest(generate_series(lo, hi, INTERVAL 15 MINUTE)) AS ts FROM bounds),
        joined AS (
            SELECT g.ts, coalesce(s.rows, 0) AS rows FROM grid g LEFT JOIN snaps s USING (ts)
        ),
        rolled AS (
            SELECT ts, rows,
                   median(rows) OVER (ORDER BY ts ROWS BETWEEN 96 PRECEDING AND 96 FOLLOWING) AS med
            FROM joined
        )
        SELECT ts, rows, med, rows = 0 OR (med >= 20 AND rows < 0.2 * med) AS gap FROM rolled
    """)
    con.execute("""
        CREATE OR REPLACE TABLE gap_day AS
        SELECT o.offset_hours, CAST(f.ts + to_hours(o.offset_hours) AS DATE) AS day,
               count(*) * 15 AS gap_minutes
        FROM snapshot_flags f CROSS JOIN (SELECT DISTINCT offset_hours FROM units) o
        WHERE f.gap GROUP BY ALL
    """)
    con.execute("""
        CREATE OR REPLACE TABLE unit_year AS
        SELECT unit_id, year(day) AS year, count(*) FILTER (WHERE rows > 0) AS days_with_rows
        FROM unit_day_obs GROUP BY ALL
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE unit_day AS
        WITH days AS (
            SELECT CAST(unnest(generate_series(DATE '{y.first}-01-01', DATE '{y.last}-12-31',
                                               INTERVAL 1 DAY)) AS DATE) AS day
        ),
        bounds AS (
            SELECT CAST(min(ts) AS DATE) + 1 AS lo, CAST(max(ts) AS DATE) - 1 AS hi
            FROM snapshot_flags
        )
        SELECT u.unit_id, d.day, year(d.day) AS year,
               coalesce(o.peak_out, 0) AS peak_out,
               coalesce(o.cust_minutes, 0) AS cust_minutes,
               coalesce(o.rows, 0) AS rows,
               coalesce(o.start_out, 0) AS start_out,
               coalesce(o.over_rows, 0) AS over_rows,
               coalesce(g.gap_minutes, 0) AS gap_minutes,
               coalesce(yr.days_with_rows, 0) >= {o.min_covered_days} AS unit_covered,
               coalesce(c.max_pct, 1.0) >= {o.min_state_coverage} AS state_covered,
               coalesce(g.gap_minutes, 0) <= {o.max_gap_hours * 60} AS complete
        FROM units u
        CROSS JOIN days d
        CROSS JOIN bounds b
        LEFT JOIN unit_day_obs o ON o.unit_id = u.unit_id AND o.day = d.day
        LEFT JOIN gap_day g ON g.offset_hours = u.offset_hours AND g.day = d.day
        LEFT JOIN unit_year yr ON yr.unit_id = u.unit_id AND yr.year = year(d.day)
        LEFT JOIN state_coverage c ON c.abbr = u.abbr AND c.year = year(d.day)
        WHERE d.day BETWEEN b.lo AND b.hi
        ORDER BY u.unit_id, d.day
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE events AS
        WITH ends AS (SELECT max(ts) AS data_end FROM snapshot_flags),
        state_customers AS (SELECT abbr, sum(customers) AS customers FROM units GROUP BY abbr),
        flagged AS (
            SELECT e.*,
                   EXISTS (
                       SELECT 1 FROM snapshot_flags f
                       WHERE f.gap AND f.ts > e.last_ts
                         AND f.ts <= e.last_ts + INTERVAL '{gap} minutes'
                   ) AS gap_after
            FROM events_raw e
        )
        SELECT row_number() OVER (ORDER BY e.start_ts, e.unit_id) AS event_id,
               e.unit_id, u.abbr, e.start_ts, e.peak_ts, e.last_ts, e.peak_out, e.cust_minutes,
               e.n_rows,
               coalesce(e.restore_ts, e.last_ts + INTERVAL 15 MINUTE) AS end_ts,
               e.restore_ts IS NULL AND (e.gap_after OR e.window_cut
                   OR e.last_ts + INTERVAL '{gap} minutes' >= x.data_end) AS censored,
               e.peak_out / u.customers AS peak_share,
               coalesce(s.out, e.peak_out) / sc.customers AS state_load,
               u.customers, u.density, u.lat, u.lon, 'none' AS cause
        FROM flagged e
        JOIN units u USING (unit_id)
        JOIN state_customers sc ON sc.abbr = u.abbr
        CROSS JOIN ends x
        LEFT JOIN state_ts s ON s.abbr = u.abbr AND s.ts = e.peak_ts
    """)
    con.execute("""
        ALTER TABLE events ADD COLUMN duration_h DOUBLE;
        UPDATE events SET duration_h = CASE WHEN censored
            THEN epoch(last_ts - peak_ts) / 3600.0 + 0.25
            ELSE epoch(end_ts - peak_ts) / 3600.0 END;
    """)
    con.execute("DROP TABLE unit_day_parts")
    con.execute("DROP TABLE events_raw")


def record_gold_checks(cfg: Config, con: duckdb.DuckDBPyConnection) -> None:
    gaps = con.execute(
        "SELECT count(*) FILTER (WHERE gap), count(*) FROM snapshot_flags"
    ).fetchone()
    share = gaps[0] / max(gaps[1], 1)
    record_check(
        con,
        "outages",
        "gap_snapshot_share",
        "warn" if share > 0.02 else "pass",
        share,
        f"{gaps[0]} of {gaps[1]} 15 minute snapshots missing or thin",
    )
    over = con.execute("SELECT sum(over_rows), sum(rows) FROM unit_day").fetchone()
    share = (over[0] or 0) / max(over[1] or 0, 1)
    record_check(
        con,
        "outages",
        "over_customer_share",
        "warn" if share > 0.001 else "pass",
        share,
        "rows with more customers out than 1.5 times the modeled customer count",
    )
    covered = con.execute("""
        SELECT year, avg(CASE WHEN unit_covered AND state_covered THEN 1 ELSE 0 END)
        FROM unit_day GROUP BY year ORDER BY year
    """).fetchall()
    detail = ", ".join(f"{y}: {v:.2f}" for y, v in covered)
    low = min(v for _, v in covered) if covered else 0
    record_check(
        con, "outages", "covered_unit_share_min", "warn" if low < 0.5 else "pass", low, detail
    )
    ev = con.execute("SELECT count(*), avg(censored::INT) FROM events").fetchone()
    record_check(
        con,
        "outages",
        "events",
        "pass" if ev[0] else "fail",
        ev[0],
        f"censored share {ev[1] or 0:.3f}",
    )


def build(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> None:
    stats = []
    for year in cfg.years.all:
        log(f"outages {year}: parsing raw csv")
        stats.append(silver_year(cfg, con, year))
    record_silver_checks(cfg, con, stats)
    create_work_tables(con)
    for year in cfg.years.all:
        log(f"outages {year}: county-day table and events")
        gold_year(cfg, con, year)
    finish_gold(cfg, con)
    record_gold_checks(cfg, con)
