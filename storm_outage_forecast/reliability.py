from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import Config
from .db import record_check
from .refdata import states


def norm(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower() if value is not None else ""


def workbook_bytes(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        names = [
            n
            for n in archive.namelist()
            if Path(n).name.lower().startswith("reliability") and n.lower().endswith(".xlsx")
        ]
        if not names:
            raise ValueError(f"{path.name} has no Reliability xlsx workbook")
        return archive.read(sorted(names)[0])


def header_row(frame: pd.DataFrame) -> int:
    for i in range(min(len(frame), 12)):
        if "utility number" in [norm(v) for v in frame.iloc[i].tolist()]:
            return i
    raise ValueError("no header row with 'Utility Number' in the first 12 rows")


def parse_reliability(path: Path, year: int) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(workbook_bytes(path)), sheet_name=0, header=None, dtype=object)
    h = header_row(frame)
    names = [norm(v) for v in frame.iloc[h].tolist()]
    body = frame.iloc[h + 1 :].reset_index(drop=True)
    saidi = [i for i, n in enumerate(names) if n.startswith("saidi")]
    if not saidi:
        raise ValueError(f"{path.name}: no SAIDI columns")
    first = saidi[0]
    customers = next((i for i in range(first, len(names)) if "customers" in names[i]), None)
    if customers is None:
        raise ValueError(f"{path.name}: no customer count column after SAIDI")

    def column(position):
        if position not in body.columns:
            return pd.Series(np.nan, index=body.index)
        return pd.to_numeric(body[position], errors="coerce")

    out = pd.DataFrame(
        {
            "year": year,
            "utility": pd.to_numeric(body[names.index("utility number")], errors="coerce"),
            "state": body[names.index("state")].astype(str).str.strip().str.upper(),
            "saidi_med": column(first),
            "saifi_med": column(first + 1),
            "saidi_no_med": column(first + 3),
            "customers": column(customers),
        }
    )
    return out[out["utility"].notna()].reset_index(drop=True)


def state_year(utilities: pd.DataFrame) -> pd.DataFrame:
    conus = set(states()["abbr"])
    ok = utilities[
        utilities["state"].isin(conus)
        & (utilities["customers"] > 0)
        & utilities["saidi_med"].between(0, 100000)
    ]
    grouped = ok.assign(weighted=ok["saidi_med"] * ok["customers"]).groupby(["year", "state"])
    frame = grouped.agg(
        weighted=("weighted", "sum"),
        customers_eia=("customers", "sum"),
        utilities=("utility", "nunique"),
    ).reset_index()
    frame["saidi_eia"] = frame["weighted"] / frame["customers_eia"]
    return frame.drop(columns=["weighted"]).rename(columns={"state": "abbr"})


def build(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> dict:
    d = cfg.download
    years = [y for y in range(d.eia861_first, d.eia861_last + 1) if y in cfg.years.all]
    frames = []
    for year in years:
        path = cfg.paths.raw / "eia861" / f"f861{year}.zip"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing, run the download step first")
        log(f"reliability {year}: reading EIA-861")
        frames.append(parse_reliability(path, year))
    utilities = pd.concat(frames, ignore_index=True)
    table = state_year(utilities)
    con.register("eia_df", table)
    con.execute("CREATE OR REPLACE TABLE eia_state_year AS SELECT * FROM eia_df")
    con.unregister("eia_df")
    con.execute("""
        CREATE OR REPLACE TABLE saidi_check AS
        WITH uy AS (
            SELECT unit_id, year, sum(cust_minutes) AS minutes, bool_and(unit_covered) AS covered
            FROM unit_day GROUP BY ALL
        ),
        eaglei AS (
            SELECT u.abbr, uy.year, sum(uy.minutes) / sum(u.customers) AS saidi_eaglei,
                   count(*) AS units, sum(u.customers) AS customers_covered
            FROM uy JOIN units u USING (unit_id)
            WHERE uy.covered GROUP BY ALL
        )
        SELECT e.abbr, e.year, e.saidi_eaglei, s.saidi_eia, e.units, e.customers_covered,
               s.customers_eia, s.utilities
        FROM eaglei e JOIN eia_state_year s USING (abbr, year)
        ORDER BY e.year, e.abbr
    """)
    check = con.execute("SELECT * FROM saidi_check").df()
    result = {"state_years": int(len(check))}
    if len(check) >= 5:
        rho = spearmanr(check["saidi_eaglei"], check["saidi_eia"]).statistic
        ratio = (check["saidi_eaglei"] / check["saidi_eia"]).replace([np.inf, -np.inf], np.nan)
        result.update(
            {
                "spearman": float(rho),
                "median_ratio": float(ratio.median()),
                "log_corr": float(
                    np.corrcoef(np.log1p(check["saidi_eaglei"]), np.log1p(check["saidi_eia"]))[0, 1]
                ),
            }
        )
        status = "pass" if rho >= 0.3 else "warn"
        record_check(
            con,
            "reliability",
            "saidi_spearman",
            status,
            rho,
            f"{len(check)} state-years, median ratio {result['median_ratio']:.2f}",
        )
    else:
        record_check(con, "reliability", "saidi_spearman", "warn", 0.0, "fewer than 5 state-years")
    per_year = table.groupby("year")["abbr"].nunique()
    record_check(
        con,
        "reliability",
        "states_per_year_min",
        "pass",
        float(per_year.min()),
        ", ".join(f"{y}: {n}" for y, n in per_year.items()),
    )
    return result
