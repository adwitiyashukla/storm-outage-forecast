from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .config import Config
from .db import record_check
from .storms import PRIORITY

FEATURES = [
    "gust",
    "gust2",
    "wind",
    "gust_max",
    "gust_rel",
    "n_gust",
    "prcp",
    "prcp_3d",
    "prcp_7d",
    "snow",
    "snwd",
    "snwd_change",
    "tmax",
    "tmin",
    "tmax_anom",
    "cold_prcp",
    "ice",
    "thunder",
    "log_customers",
    "log_density",
    "lat",
    "lon",
    "state_code",
    "month",
    "doy_sin",
    "doy_cos",
    "start_share",
    "prev_share",
    "prev_sig",
    "max_share_7d",
]
CATEGORICAL = ["state_code"]
KEYS = ["unit_id", "abbr", "cause"]


def split_expr(cfg: Config) -> str:
    y = cfg.years
    return (
        f"CASE WHEN year <= {y.train_end} THEN 'train' WHEN year = {y.valid} THEN 'valid' "
        f"ELSE 'test' END"
    )


def build(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> None:
    o = cfg.outages
    w = cfg.weather
    y = cfg.years
    log("features: building the county-day model table")
    con.execute(f"""
        CREATE OR REPLACE TABLE climate AS
        SELECT d.unit_id,
               quantile_cont(wd.gust, {w.gust_quantile}) AS gust_clim
        FROM unit_day d JOIN weather_day wd USING (unit_id, day)
        WHERE d.year <= {y.train_end} AND wd.gust IS NOT NULL
        GROUP BY d.unit_id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE climate_month AS
        SELECT d.unit_id, month(d.day) AS month, avg(wd.tmax) AS tmax_clim
        FROM unit_day d JOIN weather_day wd USING (unit_id, day)
        WHERE d.year <= {y.train_end} AND wd.tmax IS NOT NULL
        GROUP BY ALL
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE model_rows AS
        WITH states AS (
            SELECT abbr, row_number() OVER (ORDER BY abbr) - 1 AS state_code
            FROM (SELECT DISTINCT abbr FROM units)
        ),
        base AS (
            SELECT d.unit_id, d.day, d.year, d.peak_out, d.start_out, d.cust_minutes,
                   d.unit_covered AND d.state_covered AND d.complete AS usable,
                   u.customers, u.density, u.lat, u.lon, u.abbr, s.state_code,
                   d.peak_out
                       >= greatest({o.significant_floor}, {o.significant_share} * u.customers)
                       AS sig,
                   wd.gust, wd.gust2, wd.wind, wd.gust_max, wd.n_gust, wd.prcp, wd.snow, wd.snwd,
                   wd.tmax, wd.tmin, wd.ice, wd.thunder,
                   c.gust_clim, cm.tmax_clim
            FROM unit_day d
            JOIN units u USING (unit_id)
            JOIN states s USING (abbr)
            LEFT JOIN weather_day wd USING (unit_id, day)
            LEFT JOIN climate c USING (unit_id)
            LEFT JOIN climate_month cm ON cm.unit_id = d.unit_id AND cm.month = month(d.day)
        )
        SELECT unit_id, day, year, {split_expr(cfg)} AS split, usable, sig, peak_out,
               least(peak_out / customers, 1.0) AS peak_share, customers, abbr,
               coalesce(cause.cause, 'none') AS cause,
               gust, gust2, wind, gust_max,
               gust / nullif(gust_clim, 0) AS gust_rel,
               n_gust, prcp,
               sum(prcp) OVER (PARTITION BY unit_id ORDER BY day
                               ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS prcp_3d,
               sum(prcp) OVER (PARTITION BY unit_id ORDER BY day
                               ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS prcp_7d,
               snow, snwd,
               snwd - lag(snwd) OVER (PARTITION BY unit_id ORDER BY day) AS snwd_change,
               tmax, tmin,
               tmax - tmax_clim AS tmax_anom,
               CASE WHEN tmax <= 1.0 THEN prcp ELSE 0 END AS cold_prcp,
               ice, thunder,
               ln(customers) AS log_customers,
               ln(greatest(density, 0.01)) AS log_density,
               lat, lon, state_code,
               month(day) AS month,
               sin(2 * pi() * dayofyear(day) / 365.25) AS doy_sin,
               cos(2 * pi() * dayofyear(day) / 365.25) AS doy_cos,
               least(start_out / customers, 1.0) AS start_share,
               least(lag(peak_out) OVER (PARTITION BY unit_id ORDER BY day) / customers, 1.0)
                   AS prev_share,
               CAST(lag(sig) OVER (PARTITION BY unit_id ORDER BY day) AS INTEGER) AS prev_sig,
               least(max(peak_out) OVER (PARTITION BY unit_id ORDER BY day
                                         ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) / customers, 1.0)
                   AS max_share_7d
        FROM base
        LEFT JOIN unit_cause_day cause USING (unit_id, day)
        ORDER BY unit_id, day
    """)
    stats = con.execute("""
        SELECT split, count(*), sum(usable::INT), avg(sig::INT) FILTER (WHERE usable)
        FROM model_rows GROUP BY split ORDER BY split
    """).fetchall()
    for split, rows, usable, rate in stats:
        record_check(
            con,
            "features",
            f"{split}_positive_rate",
            "pass" if rate else "fail",
            rate or 0.0,
            f"{usable} usable of {rows} county-days",
        )
    dup = con.execute("""
        SELECT count(*) - count(DISTINCT (unit_id, day)) FROM model_rows
    """).fetchone()[0]
    record_check(con, "features", "duplicate_keys", "fail" if dup else "pass", dup)


def load(
    con: duckdb.DuckDBPyConnection, split: str, extra: str = "", climate: bool = False
) -> pd.DataFrame:
    casts = ", ".join(f"CAST(m.{c} AS FLOAT) AS {c}" for c in FEATURES)
    joins = ""
    clim = ""
    if climate:
        clim = ", c.p_clim, s.mu, s.sd, s.base"
        joins = (
            "LEFT JOIN clim_rate c ON c.unit_id = m.unit_id AND c.month = m.month "
            "LEFT JOIN clim_size s ON s.unit_id = m.unit_id"
        )
    query = f"""
        SELECT m.unit_id, m.day, m.year, m.sig, m.peak_out, m.customers, m.abbr, m.cause,
               {casts} {clim}
        FROM model_rows m {joins}
        WHERE m.usable AND m.split = '{split}' {extra}
        ORDER BY m.unit_id, m.day
    """
    frame = con.execute(query).df()
    units = con.execute("SELECT unit_id FROM units ORDER BY unit_id").df()["unit_id"].tolist()
    abbrs = sorted(set(con.execute("SELECT abbr FROM units").df()["abbr"]))
    levels = {"unit_id": units, "abbr": abbrs, "cause": sorted([*PRIORITY, "none"])}
    for column in KEYS:
        frame[column] = pd.Categorical(frame[column], categories=levels[column])
    frame["sig"] = frame["sig"].astype(bool)
    return frame


def load_sampled(
    con: duckdb.DuckDBPyConnection, split: str, rate: float, seed: int
) -> pd.DataFrame:
    keep = int(round(rate * 100000))
    extra = (
        f"AND (m.sig OR hash(m.unit_id || CAST(m.day AS VARCHAR) || '{seed}') % 100000 < {keep})"
    )
    frame = load(con, split, extra)
    frame["weight"] = np.where(frame["sig"], 1.0, 1.0 / rate).astype(np.float32)
    return frame
