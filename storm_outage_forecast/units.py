from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from .config import Config
from .db import record_check
from .download import nws_date
from .geo import county_frame, read_geojson
from .refdata import TZ_OFFSETS, state_lookup, states, unit_remap


def latest_zone_file(folder: Path) -> Path:
    files = sorted(folder.glob("bp*.dbx"))
    if not files:
        raise FileNotFoundError(f"no zone-county file in {folder}")
    dated = [(nws_date(f.name), f) for f in files if nws_date(f.name)]
    return max(dated)[1] if dated else files[-1]


def read_zone_county(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="latin-1") as handle:
        for line in handle:
            parts = [p.strip() for p in line.rstrip("\r\n").split("|")]
            if len(parts) < 11 or parts[0] == "STATE":
                continue
            fips = parts[6]
            if not re.fullmatch(r"\d{4,5}", fips):
                continue
            rows.append(
                {
                    "state_zone": parts[4].upper(),
                    "fips": fips.zfill(5),
                    "tz_code": parts[7][:1].upper() if parts[7] else "",
                }
            )
    if not rows:
        raise ValueError(f"{path} has no usable zone-county rows")
    return pd.DataFrame(rows).drop_duplicates()


def read_mcc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    frame.columns = [c.strip().lower() for c in frame.columns]
    fips_col = next(c for c in frame.columns if "fips" in c)
    cust_col = next(c for c in frame.columns if "customer" in c)
    out = pd.DataFrame(
        {
            "fips": frame[fips_col].str.strip().str.split(".").str[0].str.zfill(5),
            "customers": pd.to_numeric(frame[cust_col], errors="coerce"),
        }
    )
    return out.dropna().groupby("fips", as_index=False)["customers"].sum()


def coverage_year(value: str) -> int:
    text = str(value).strip()
    match = re.search(r"\d{4}", text)
    if match:
        return int(match.group())
    tail = re.split(r"[/.-]", text)[-1]
    if not tail.isdigit():
        raise ValueError(f"cannot read a year from {value!r}")
    number = int(tail)
    return number + 2000 if number < 100 else number


def read_coverage(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    frame.columns = [c.strip().lower() for c in frame.columns]
    names = {name.upper(): abbr for name, abbr in state_lookup("name", "abbr").items()}
    state = frame["state"].str.strip().str.upper()
    out = pd.DataFrame(
        {
            "year": frame["year"].map(coverage_year),
            "abbr": state.map(names).fillna(state),
            "max_pct": pd.to_numeric(frame["max_pct_covered"], errors="coerce"),
            "min_pct": pd.to_numeric(frame["min_pct_covered"], errors="coerce"),
        }
    )
    if out["max_pct"].max() > 1.5:
        out["max_pct"] /= 100
        out["min_pct"] /= 100
    return out.groupby(["year", "abbr"], as_index=False).max()


def build_units(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> pd.DataFrame:
    raw = cfg.paths.raw
    counties = county_frame(read_geojson(raw / "geo" / "geojson-counties-fips.json"))
    remap = unit_remap()
    to_unit = dict(zip(remap["fips"], remap["unit_id"], strict=True))
    counties["unit_id"] = counties["geo_id"].map(to_unit).fillna(counties["geo_id"])
    zones = read_zone_county(latest_zone_file(raw / "nws"))
    zones["unit_id"] = zones["fips"].map(to_unit).fillna(zones["fips"])
    info = states()
    tz_state = dict(zip(info["state_fips"], info["tz"], strict=True))
    tz_unit = (
        zones[zones["tz_code"].isin(TZ_OFFSETS)]
        .groupby("unit_id")["tz_code"]
        .agg(lambda s: s.value_counts().index[0])
    )
    mcc = read_mcc(raw / "eaglei" / "MCC.csv")
    mcc["unit_id"] = mcc["fips"].map(to_unit).fillna(mcc["fips"])
    customers = mcc.groupby("unit_id")["customers"].sum()
    counties["w"] = counties["land_sqmi"].clip(lower=0.01)
    grouped = counties.groupby("unit_id")
    units = pd.DataFrame(
        {
            "lat": grouped.apply(
                lambda g: (g["lat"] * g["w"]).sum() / g["w"].sum(), include_groups=False
            ),
            "lon": grouped.apply(
                lambda g: (g["lon"] * g["w"]).sum() / g["w"].sum(), include_groups=False
            ),
            "land_sqmi": grouped["land_sqmi"].sum(),
            "name": grouped["name"].first(),
            "state_fips": grouped["state_fips"].first(),
        }
    ).reset_index()
    units.loc[units["unit_id"] == "09000", "name"] = "Connecticut"
    units["tz"] = units["unit_id"].map(tz_unit).fillna(units["state_fips"].map(tz_state))
    units["offset_hours"] = units["tz"].map(TZ_OFFSETS).astype(int)
    units["customers"] = units["unit_id"].map(customers)
    units = units.merge(info[["state_fips", "abbr"]], on="state_fips", how="left")
    missing = int(units["customers"].isna().sum())
    record_check(
        con,
        "units",
        "units_without_customers",
        "warn" if missing else "pass",
        missing,
        ", ".join(units.loc[units["customers"].isna(), "unit_id"].head(10)),
    )
    units = units.dropna(subset=["customers"]).copy()
    units["customers"] = units["customers"].round().astype(int)
    units["density"] = units["customers"] / units["land_sqmi"].clip(lower=1.0)
    known = set(units["unit_id"])
    codes = pd.concat(
        [
            pd.DataFrame({"fips": counties["geo_id"], "unit_id": counties["unit_id"]}),
            remap[["fips", "unit_id"]],
            pd.DataFrame({"fips": mcc["fips"], "unit_id": mcc["unit_id"]}),
            pd.DataFrame({"fips": zones["fips"], "unit_id": zones["unit_id"]}),
        ]
    ).drop_duplicates("fips")
    codes = codes[codes["unit_id"].isin(known)]
    con.register("units_df", units)
    con.execute("CREATE OR REPLACE TABLE units AS SELECT * FROM units_df ORDER BY unit_id")
    con.register("codes_df", codes)
    con.execute("CREATE OR REPLACE TABLE fips_unit AS SELECT * FROM codes_df")
    geo = counties[counties["unit_id"].isin(known)][["geo_id", "unit_id"]]
    con.register("geo_df", geo)
    con.execute("CREATE OR REPLACE TABLE unit_geo AS SELECT * FROM geo_df")
    zones = zones[zones["unit_id"].isin(known)][["state_zone", "unit_id"]].drop_duplicates()
    con.register("zones_df", zones)
    con.execute("CREATE OR REPLACE TABLE zone_unit AS SELECT * FROM zones_df")
    coverage = read_coverage(raw / "eaglei" / "coverage_history.csv")
    con.register("coverage_df", coverage)
    con.execute("CREATE OR REPLACE TABLE state_coverage AS SELECT * FROM coverage_df")
    for name in ("units_df", "codes_df", "geo_df", "zones_df", "coverage_df"):
        con.unregister(name)
    record_check(con, "units", "unit_count", "pass", len(units))
    log(f"units: {len(units)} counties, {len(codes)} fips codes")
    return units
