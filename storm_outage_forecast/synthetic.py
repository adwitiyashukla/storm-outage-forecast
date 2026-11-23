from __future__ import annotations

import calendar
import csv
import gzip
import io
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from openpyxl import Workbook

from .config import Config
from .geo import county_frame, haversine_km, read_geojson
from .refdata import TZ_OFFSETS, states

PER_HOUR = 4
PER_DAY = 96
TAIL = 14 * PER_DAY
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
TZ_NAMES = {"E": "EST", "C": "CST", "M": "MST", "P": "PST"}
CT_REGIONS = ["09110", "09120", "09130", "09140", "09150", "09160", "09170", "09180"]
LEGACY = {"46113": "46102"}

PROFILES = {
    "tiny": {
        "storms": {"convective": 12, "wind": 6, "winter": 5, "ice": 2, "tropical": 3},
        "background_events": 400,
        "other_us_stations": 40,
        "foreign_stations": 30,
        "incident_rate": 3.0,
        "big_share": 0.03,
        "global_gaps": 3,
        "partial_gaps": 2,
        "percent_coverage": False,
    },
    "full": {
        "storms": {"convective": 32, "wind": 16, "winter": 11, "ice": 4, "tropical": 7},
        "background_events": 45000,
        "other_us_stations": 15000,
        "foreign_stations": 20000,
        "incident_rate": 2.5,
        "big_share": 0.03,
        "global_gaps": 6,
        "partial_gaps": 15,
        "percent_coverage": True,
    },
}

KINDS = {
    "convective": {
        "months": (4, 5, 6, 7, 8),
        "hours": (6, 15),
        "radius": (60, 150),
        "speed": (40, 70),
        "gust": (24, 40),
        "rain": (4, 12),
        "snow": (0, 0),
        "ice": (0, 0),
        "k": 1.1,
        "scale": 8.0,
        "heading": (60, 100),
        "bias": 0.0,
    },
    "wind": {
        "months": (10, 11, 12, 1, 2, 3, 4),
        "hours": (12, 31),
        "radius": (250, 500),
        "speed": (30, 60),
        "gust": (20, 32),
        "rain": (0.5, 2),
        "snow": (0, 0),
        "ice": (0, 0),
        "k": 1.0,
        "scale": 10.0,
        "heading": (45, 110),
        "bias": 0.0,
    },
    "winter": {
        "months": (12, 1, 2, 3),
        "hours": (24, 61),
        "radius": (250, 450),
        "speed": (30, 50),
        "gust": (15, 24),
        "rain": (0.5, 1.5),
        "snow": (0.8, 2.5),
        "ice": (0, 0.2),
        "k": 0.9,
        "scale": 14.0,
        "heading": (50, 90),
        "bias": -0.2,
    },
    "ice": {
        "months": (12, 1, 2),
        "hours": (18, 41),
        "radius": (150, 300),
        "speed": (25, 45),
        "gust": (10, 18),
        "rain": (0.5, 1.5),
        "snow": (0, 0.3),
        "ice": (0.6, 1.4),
        "k": 1.2,
        "scale": 30.0,
        "heading": (45, 80),
        "bias": 0.0,
    },
    "tropical": {
        "months": (8, 9, 10),
        "hours": (36, 61),
        "radius": (150, 350),
        "speed": (15, 30),
        "gust": (30, 65),
        "rain": (3, 8),
        "snow": (0, 0),
        "ice": (0, 0),
        "k": 1.3,
        "scale": 40.0,
        "heading": (-20, 40),
        "bias": 0.3,
    },
}

GHCN_ELEMENTS = ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND", "WDF2", "WDF5", "WSF2", "WSF5"]


@dataclass
class Storm:
    kind: str
    episode: int
    start: datetime
    hours: int
    lat0: float
    lon0: float
    heading: float
    speed: float
    radius: float
    gust: float
    rain: float
    snow: float
    ice: float


@dataclass
class World:
    geo: dict
    counties: pd.DataFrame
    stations: pd.DataFrame
    zones: pd.DataFrame
    zone_cache: tuple | None = None


def tiny_geojson() -> dict:
    specs = []
    codes = {"39": 1, "21": 1, "18": 1}
    order = ["39", "21", "18"]
    for row in range(6):
        for col in range(6):
            st = order[col // 2]
            specs.append(
                (f"{st}{codes[st]:03d}", f"Grid {row}{col}", 37.6 + row * 0.6, -86.8 + col * 0.9)
            )
            codes[st] += 2
    specs += [
        ("09001", "Fairfield", 41.2, -85.0),
        ("09003", "Hartford", 41.2, -84.1),
        ("46113", "Shannon", 41.2, -83.2),
    ]
    features = []
    for code, name, lat, lon in specs:
        h, w = 0.3, 0.45
        ring = [
            [lon - w, lat - h],
            [lon + w, lat - h],
            [lon + w, lat + h],
            [lon - w, lat + h],
            [lon - w, lat - h],
        ]
        area = (2 * w * 69.17 * math.cos(math.radians(lat))) * (2 * h * 69.17)
        features.append(
            {
                "type": "Feature",
                "id": code,
                "properties": {
                    "GEO_ID": f"0500000US{code}",
                    "STATE": code[:2],
                    "COUNTY": code[2:],
                    "NAME": name,
                    "LSAD": "County",
                    "CENSUSAREA": round(area, 3),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_counties(geo: dict, rng: np.random.Generator, first_year: int) -> pd.DataFrame:
    frame = county_frame(geo)
    frame = frame[frame["geo_id"] != "51515"].reset_index(drop=True)
    info = states().set_index("state_fips")
    frame["abbr"] = frame["state_fips"].map(info["abbr"])
    frame["state_name"] = frame["state_fips"].map(info["name"])
    frame["tz"] = frame["state_fips"].map(info["tz"])
    rules = [
        ("TX", frame["lon"] < -104.9, "M"),
        ("FL", frame["lon"] < -85.0, "C"),
        ("KY", frame["lon"] < -86.0, "C"),
        ("TN", frame["lon"] > -85.2, "E"),
        ("ND", frame["lon"] < -101.0, "M"),
        ("SD", frame["lon"] < -100.5, "M"),
        ("NE", frame["lon"] < -101.2, "M"),
        ("KS", frame["lon"] < -101.5, "M"),
        ("ID", frame["lat"] > 45.6, "P"),
        ("OR", frame["lon"] > -117.9, "M"),
    ]
    for abbr, mask, tz in rules:
        frame.loc[(frame["abbr"] == abbr) & mask, "tz"] = tz
    frame["offset"] = frame["tz"].map(TZ_OFFSETS).astype(int)
    n = len(frame)
    frame["customers"] = np.clip(np.exp(rng.normal(9.3, 1.25, n)), 400, 3_000_000).round()
    frame["customers"] = frame["customers"].astype(np.int64)
    east = (frame["lon"] > -100).to_numpy()
    frame["vuln"] = np.exp(rng.normal(0, 0.35, n)) * np.where(east, 1.2, 0.8)
    plains = np.exp(-0.5 * ((frame["lon"].to_numpy() + 100) / 6) ** 2)
    frame["clim"] = 16 + 4 * plains + rng.normal(0, 1, n)
    density = frame["customers"] / frame["land_sqmi"].clip(lower=1)
    frame["dens_factor"] = np.clip(1.25 - 0.1 * np.log10(density.clip(lower=1)), 0.8, 1.3)
    draw = rng.random(n)
    frame["onboard_year"] = np.where(
        draw < 0.72, first_year - 5, first_year + ((draw - 0.72) / 0.03).astype(int)
    )
    frame["onboard_day"] = rng.integers(0, 365, n)
    frame["eaglei_fips"] = frame["geo_id"].map(LEGACY).fillna(frame["geo_id"])
    frame["mcc"] = np.round(frame["customers"] * np.exp(rng.normal(0, 0.05, n))).astype(np.int64)
    return frame


def build_stations(counties: pd.DataFrame, rng: np.random.Generator, profile: dict) -> pd.DataFrame:
    n = len(counties)
    n_usw = min(n, max(8, int(round(n * 0.29))))
    weights = np.sqrt(counties["customers"].to_numpy(float))
    pick = rng.choice(n, n_usw, replace=False, p=weights / weights.sum())
    usw = pd.DataFrame(
        {
            "id": [f"USW000{10000 + i:05d}" for i in range(n_usw)],
            "lat": counties["lat"].to_numpy()[pick] + rng.normal(0, 0.12, n_usw),
            "lon": counties["lon"].to_numpy()[pick] + rng.normal(0, 0.12, n_usw),
            "elev": rng.uniform(5, 1500, n_usw).round(1),
            "state": counties["abbr"].to_numpy()[pick],
            "name": [f"AIRPORT {i}" for i in range(n_usw)],
            "offset": counties["offset"].to_numpy()[pick],
            "clim": counties["clim"].to_numpy()[pick],
            "wind": rng.random(n_usw) > 0.1,
            "snowobs": rng.random(n_usw) > 0.05,
            "kind": "usw",
        }
    )
    k = profile["other_us_stations"]
    pick = rng.integers(0, n, k)
    abbrs = counties["abbr"].to_numpy()[pick]
    ids = [
        f"USC00{i:06d}" if i % 3 == 0 else f"US1{a}XX{i % 10000:04d}" for i, a in enumerate(abbrs)
    ]
    other = pd.DataFrame(
        {
            "id": ids,
            "lat": counties["lat"].to_numpy()[pick] + rng.normal(0, 0.2, k),
            "lon": counties["lon"].to_numpy()[pick] + rng.normal(0, 0.2, k),
            "elev": rng.uniform(5, 2000, k).round(1),
            "state": abbrs,
            "name": [f"SITE {i}" for i in range(k)],
            "offset": 0,
            "clim": 0.0,
            "wind": False,
            "snowobs": True,
            "kind": "other",
        }
    )
    m = profile["foreign_stations"]
    prefixes = np.array(["CA", "MX", "AS", "GM", "SW"])
    foreign = pd.DataFrame(
        {
            "id": [f"{prefixes[i % 5]}{i:09d}" for i in range(m)],
            "lat": rng.uniform(50, 70, m).round(4),
            "lon": rng.uniform(-140, 30, m).round(4),
            "elev": rng.uniform(0, 3000, m).round(1),
            "state": "",
            "name": [f"FOREIGN {i}" for i in range(m)],
            "offset": 0,
            "clim": 0.0,
            "wind": False,
            "snowobs": False,
            "kind": "foreign",
        }
    )
    return pd.concat([usw, other, foreign], ignore_index=True)


def build_zones(counties: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for abbr, group in counties.groupby("abbr", sort=True):
        zone = 0
        for i, rec in enumerate(group.sort_values("geo_id").itertuples(index=False)):
            zone = zone if (i % 7 == 6 and zone > 0) else zone + 1
            zones = [zone]
            if i % 11 == 10:
                zone += 1
                zones.append(zone)
            for z in zones:
                rows.append(
                    {
                        "STATE": abbr,
                        "ZONE": f"{z:03d}",
                        "CWA": f"{abbr}X",
                        "NAME": f"{rec.name} Zone {z}",
                        "STATE_ZONE": f"{abbr}{z:03d}",
                        "COUNTY": rec.name,
                        "FIPS": rec.eaglei_fips,
                        "TIME_ZONE": rec.tz,
                        "FE_AREA": "nc",
                        "LAT": round(rec.lat, 5),
                        "LON": round(rec.lon, 5),
                    }
                )
    return pd.DataFrame(rows)


def make_world(geo: dict, seed: int, first_year: int, profile: dict) -> World:
    rng = np.random.default_rng(seed)
    counties = build_counties(geo, rng, first_year)
    stations = build_stations(counties, rng, profile)
    return World(geo, counties, stations, build_zones(counties))


def make_storms(
    year: int, counties: pd.DataFrame, rng: np.random.Generator, profile: dict, episode: int
) -> list[Storm]:
    lat = counties["lat"].to_numpy()
    lon = counties["lon"].to_numpy()
    qlat = np.quantile(lat, [0.25, 0.5])
    qlon = np.quantile(lon, [0.1, 0.25, 0.6])
    pools = {
        "convective": np.flatnonzero(lon > qlon[0]),
        "wind": np.arange(len(lat)),
        "winter": np.flatnonzero((lat > qlat[0]) & (lon < qlon[2])),
        "ice": np.flatnonzero(lat < qlat[1]),
        "tropical": np.flatnonzero((lat < qlat[0]) & (lon > qlon[1])),
    }
    storms = []
    for kind, count in profile["storms"].items():
        spec = KINDS[kind]
        pool = pools[kind] if len(pools[kind]) else pools["wind"]
        for _ in range(count):
            month = int(rng.choice(spec["months"]))
            day = int(rng.integers(1, calendar.monthrange(year, month)[1] + 1))
            start = datetime(year, month, day, int(rng.integers(0, 24)))
            idx = int(rng.choice(pool))
            storms.append(
                Storm(
                    kind=kind,
                    episode=episode,
                    start=start,
                    hours=int(rng.integers(*spec["hours"])),
                    lat0=float(lat[idx] + rng.normal(0, 0.2)),
                    lon0=float(lon[idx] + rng.normal(0, 0.2)),
                    heading=float(rng.uniform(*spec["heading"])),
                    speed=float(rng.uniform(*spec["speed"])),
                    radius=float(rng.uniform(*spec["radius"])),
                    gust=float(rng.uniform(*spec["gust"])),
                    rain=float(rng.uniform(*spec["rain"])),
                    snow=float(rng.uniform(*spec["snow"])),
                    ice=float(rng.uniform(*spec["ice"])),
                )
            )
            episode += 1
    storms.sort(key=lambda s: s.start)
    return storms


def storm_shape(storm: Storm, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    t = np.arange(storm.hours) + 0.5
    travel = storm.speed * t
    heading = math.radians(storm.heading)
    clat = storm.lat0 + travel * math.cos(heading) / 111.0
    clon = storm.lon0 + travel * math.sin(heading) / (111.0 * np.cos(np.radians(clat)))
    dist = haversine_km(lat[None, :], lon[None, :], clat[:, None], clon[:, None])
    frac = t / storm.hours
    decay = np.exp(-2.2 * frac)
    envelope = decay if storm.kind == "tropical" else np.sin(np.pi * frac) ** 0.6
    return np.exp(-0.5 * (dist / storm.radius) ** 2) * envelope[:, None]


def year_steps(year: int) -> int:
    return (366 if calendar.isleap(year) else 365) * PER_DAY


def blue_sky(
    counties: pd.DataFrame, year: int, rng: np.random.Generator, profile: dict, steps: int
) -> np.ndarray:
    n = len(counties)
    customers = counties["customers"].to_numpy(float)
    lam = np.minimum(profile["incident_rate"] * np.sqrt(customers / 10000), 12)
    counts = rng.poisson(lam * steps / PER_DAY)
    cidx = np.repeat(np.arange(n), counts)
    total = len(cidx)
    start = rng.integers(0, steps, total)
    season = 1 + 0.3 * np.sin(2 * np.pi * (start / steps - 0.3))
    keep = rng.random(total) < season / 1.3
    cidx, start = cidx[keep], start[keep]
    total = len(cidx)
    big = rng.random(total) < profile["big_share"]
    dur = np.where(
        big, rng.lognormal(math.log(12), 0.6, total), rng.lognormal(math.log(7), 0.7, total)
    )
    dur = np.maximum(1, np.ceil(dur)).astype(np.int64)
    size = np.where(
        big, rng.lognormal(math.log(400), 0.9, total), rng.lognormal(math.log(12), 1.4, total)
    )
    size = np.maximum(1, np.minimum(size, 0.2 * customers[cidx])).astype(np.int32)
    end = np.minimum(start + dur, steps + TAIL)
    arr = np.zeros((steps + TAIL + 1, n), dtype=np.int32)
    np.add.at(arr, (start, cidx), size)
    np.add.at(arr, (end, cidx), -size)
    np.cumsum(arr, axis=0, out=arr)
    return arr[:-1]


def storm_outages(
    storm: Storm,
    shape: np.ndarray,
    counties: pd.DataFrame,
    arr: np.ndarray,
    year_start: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    spec = KINDS[storm.kind]
    n = len(counties)
    customers = counties["customers"].to_numpy(float)
    gmax = storm.gust * shape.max(axis=0)
    tstar = shape.argmax(axis=0)
    total = shape.sum(axis=0)
    ice = storm.ice * total
    rain = storm.rain * total
    snow = storm.snow * total
    x = gmax / counties["clim"].to_numpy()
    logit = (
        -11
        + 4.2 * x
        + 0.35 * np.minimum(ice, 20)
        + 0.5 * np.log(counties["vuln"].to_numpy())
        + 0.01 * np.minimum(rain, 150)
        + 0.02 * np.minimum(snow, 60)
        + spec["bias"]
        + rng.normal(0, 0.6, n)
    )
    frac = 1 / (1 + np.exp(-logit))
    out = np.minimum(np.round(customers * frac * np.exp(rng.normal(0, 0.2, n))), 0.98 * customers)
    state_idx = pd.factorize(counties["abbr"])[0]
    load = np.bincount(state_idx, out) / np.bincount(state_idx, customers)
    hit = np.flatnonzero(out >= 20)
    offset_steps = int((storm.start - year_start).total_seconds() // 900)
    limit = arr.shape[0]
    records = []
    for c in hit:
        k = spec["k"] * math.exp(rng.normal(0, 0.1))
        scale = (
            spec["scale"]
            * (1 + 2.5 * load[state_idx[c]])
            * counties["dens_factor"].iat[c]
            * math.exp(rng.normal(0, 0.3))
        )
        onset = offset_steps + int((tstar[c] + rng.uniform(-0.5, 1.0)) * PER_HOUR)
        ramp = int(rng.integers(2, 9))
        length = int(math.ceil(scale * (-math.log(0.003)) ** (1 / k) * PER_HOUR)) + ramp
        length = min(length, TAIL, limit - max(onset, 0))
        if length <= 0:
            continue
        steps = np.arange(length) + max(0, -onset)
        hours = np.maximum(steps - ramp, 0) / PER_HOUR
        curve = out[c] * np.minimum(1, (steps + 1) / ramp) * np.exp(-((hours / scale) ** k))
        lo = max(onset, 0)
        arr[lo : lo + length, c] += curve.astype(np.int32)
        records.append((c, onset, out[c], scale, k))
    return pd.DataFrame(
        {
            "county": [r[0] for r in records],
            "onset": [r[1] for r in records],
            "peak": [r[2] for r in records],
            "scale": [r[3] for r in records],
            "shape_k": [r[4] for r in records],
        }
    ).assign(
        gmax=gmax[[r[0] for r in records]] if records else [],
        kind=storm.kind,
        episode=storm.episode,
    )


def local_stamp(moment: datetime) -> str:
    return f"{moment.day:02d}-{MONTHS[moment.month - 1]}-{moment.year % 100:02d} {moment:%H:%M}:00"


def event_row(
    event_id: int,
    episode: int,
    kind_label: str,
    cz_type: str,
    cz_fips: int,
    cz_name: str,
    state_name: str,
    state_fips: str,
    tz_label: str,
    begin: datetime,
    end: datetime,
    magnitude,
    magnitude_type: str,
    narrative: str,
    rng: np.random.Generator,
) -> dict:
    return {
        "BEGIN_YEARMONTH": begin.year * 100 + begin.month,
        "BEGIN_DAY": begin.day,
        "BEGIN_TIME": begin.hour * 100 + begin.minute,
        "END_YEARMONTH": end.year * 100 + end.month,
        "END_DAY": end.day,
        "END_TIME": end.hour * 100 + end.minute,
        "EPISODE_ID": episode,
        "EVENT_ID": event_id,
        "STATE": state_name.upper(),
        "STATE_FIPS": int(state_fips),
        "YEAR": begin.year,
        "MONTH_NAME": begin.strftime("%B"),
        "EVENT_TYPE": kind_label,
        "CZ_TYPE": cz_type,
        "CZ_FIPS": cz_fips,
        "CZ_NAME": cz_name.upper(),
        "WFO": "XXX",
        "BEGIN_DATE_TIME": local_stamp(begin),
        "CZ_TIMEZONE": tz_label,
        "END_DATE_TIME": local_stamp(end),
        "INJURIES_DIRECT": 0,
        "INJURIES_INDIRECT": 0,
        "DEATHS_DIRECT": 0,
        "DEATHS_INDIRECT": 0,
        "DAMAGE_PROPERTY": f"{rng.uniform(0, 50):.2f}K",
        "DAMAGE_CROPS": "0.00K",
        "SOURCE": "Trained Spotter",
        "MAGNITUDE": magnitude,
        "MAGNITUDE_TYPE": magnitude_type,
        "FLOOD_CAUSE": None,
        "CATEGORY": None,
        "TOR_F_SCALE": None,
        "TOR_LENGTH": None,
        "TOR_WIDTH": None,
        "TOR_OTHER_WFO": None,
        "TOR_OTHER_CZ_STATE": None,
        "TOR_OTHER_CZ_FIPS": None,
        "TOR_OTHER_CZ_NAME": None,
        "BEGIN_RANGE": None,
        "BEGIN_AZIMUTH": None,
        "BEGIN_LOCATION": None,
        "END_RANGE": None,
        "END_AZIMUTH": None,
        "END_LOCATION": None,
        "BEGIN_LAT": None,
        "BEGIN_LON": None,
        "END_LAT": None,
        "END_LON": None,
        "EPISODE_NARRATIVE": "Synthetic episode.",
        "EVENT_NARRATIVE": narrative,
        "DATA_SOURCE": "CSV",
    }


def tz_label(tz: str, offset: int, rng: np.random.Generator) -> str:
    name = TZ_NAMES.get(tz, "EST")
    draw = rng.random()
    if draw < 0.03:
        return name
    if draw < 0.04:
        return f"{name[0]}SC{offset}"
    return f"{name}{offset}"


def station_weather(
    storm: Storm, shape: np.ndarray, offsets: np.ndarray, wx: dict, year_start: datetime, days: int
) -> None:
    base = (storm.start - year_start).total_seconds() / 3600
    hours = base + np.arange(storm.hours)[:, None] + offsets[None, :]
    day = np.floor(hours / 24).astype(int)
    valid = (day >= 0) & (day < days) & (shape > 1e-4)
    cols = np.broadcast_to(np.arange(shape.shape[1]), shape.shape)
    d, c, s = day[valid], cols[valid], shape[valid]
    np.maximum.at(wx["gust"], (d, c), storm.gust * s)
    np.add.at(wx["rain"], (d, c), storm.rain * s)
    np.add.at(wx["snow"], (d, c), storm.snow * s)
    np.add.at(wx["ice"], (d, c), storm.ice * s)
    if storm.kind in ("convective", "tropical"):
        hot = storm.gust * s > 12
        wx["thunder"][d[hot], c[hot]] = True


class Ids:
    def __init__(self, start: int):
        self.value = start

    def __call__(self) -> int:
        self.value += 1
        return self.value


def zone_index(world: World) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if world.zone_cache is None:
        index = {f: i for i, f in enumerate(world.counties["eaglei_fips"])}
        flat, starts, firsts = [], [], []
        for _, group in world.zones.groupby("STATE_ZONE", sort=False):
            members = [index[f] for f in group["FIPS"] if f in index]
            if not members:
                continue
            starts.append(len(flat))
            flat.extend(members)
            firsts.append(group.iloc[0].to_dict())
        world.zone_cache = (np.array(flat), np.array(starts), firsts)
    return world.zone_cache


def storm_reports(
    storm: Storm,
    shape: np.ndarray,
    rec: pd.DataFrame,
    world: World,
    rng: np.random.Generator,
    ids: Ids,
) -> list[dict]:
    counties = world.counties
    gmax = storm.gust * shape.max(axis=0)
    tstar = shape.argmax(axis=0)
    total = shape.sum(axis=0)
    ice, rain, snow = storm.ice * total, storm.rain * total, storm.snow * total
    customers = counties["customers"].to_numpy()
    rows = []

    def county_row(c: int, label: str, magnitude, mtype: str, narrative: str, hours: float) -> None:
        r = counties.iloc[c]
        begin = storm.start + timedelta(hours=float(tstar[c]) + int(r["offset"]))
        end = begin + timedelta(hours=hours)
        rows.append(
            event_row(
                ids(),
                storm.episode,
                label,
                "C",
                int(r["eaglei_fips"][2:]),
                r["name"],
                r["state_name"],
                r["state_fips"],
                tz_label(r["tz"], int(r["offset"]), rng),
                begin,
                end,
                magnitude,
                mtype,
                narrative,
                rng,
            )
        )

    if storm.kind == "convective":
        for c in np.flatnonzero(gmax >= 25.7):
            county_row(
                c,
                "Thunderstorm Wind",
                int(round(gmax[c] * 1.94384)),
                "MG",
                "Wind gust measured by ASOS.",
                0.5,
            )
            if gmax[c] >= 32 and rng.random() < 0.05:
                county_row(c, "Tornado", None, None, "A brief tornado touched down.", 0.2)
        for c, peak in zip(rec["county"], rec["peak"], strict=True):
            if peak >= 0.01 * customers[c] and gmax[c] < 25.7:
                county_row(
                    c,
                    "Thunderstorm Wind",
                    50,
                    "EG",
                    "Several trees and power lines were blown down.",
                    0.5,
                )
        for c in np.flatnonzero((gmax >= 18) & (rng.random(len(gmax)) < 0.2)):
            county_row(c, "Hail", 1.0, None, "Quarter size hail.", 0.2)
    if storm.kind == "tropical":
        for c in np.flatnonzero(rain >= 100):
            county_row(c, "Flash Flood", None, None, "Roads flooded.", 6.0)
    if storm.kind == "convective":
        return rows
    info = states().set_index("abbr")
    flat, starts, firsts = zone_index(world)
    zone_g = np.maximum.reduceat(gmax[flat], starts)
    zone_i = np.maximum.reduceat(ice[flat], starts)
    zone_r = np.maximum.reduceat(rain[flat], starts)
    zone_s = np.maximum.reduceat(snow[flat], starts)
    zone_t = np.minimum.reduceat(tstar[flat], starts)
    for z in range(len(starts)):
        g, i_, r_, s_ = zone_g[z], zone_i[z], zone_r[z], zone_s[z]
        labels = []
        if storm.kind == "wind":
            labels = ["High Wind"] if g >= 25.7 else (["Strong Wind"] if g >= 18 else [])
        elif storm.kind == "winter":
            labels = ["Winter Storm"] if s_ >= 15 else (["Winter Weather"] if s_ >= 5 else [])
            if s_ >= 30:
                labels.append("Heavy Snow")
        elif storm.kind == "ice":
            labels = ["Ice Storm"] if i_ >= 6 else (["Winter Weather"] if i_ >= 1 else [])
        elif storm.kind == "tropical":
            labels = ["Hurricane"] if g >= 33 else (["Tropical Storm"] if g >= 17 else [])
            if r_ >= 150 and not labels:
                labels = ["Heavy Rain"]
        if not labels:
            continue
        first = firsts[z]
        tz = first["TIME_ZONE"]
        offset = TZ_OFFSETS[tz]
        start_hour = max(0.0, float(zone_t[z]) - 3)
        state = info.loc[first["STATE"]]
        for label in labels:
            begin = storm.start + timedelta(hours=start_hour + offset)
            end = begin + timedelta(hours=max(6.0, storm.hours / 2))
            magnitude = int(round(g * 1.94384)) if label in ("High Wind", "Strong Wind") else None
            rows.append(
                event_row(
                    ids(),
                    storm.episode,
                    label,
                    "Z",
                    int(first["ZONE"]),
                    first["NAME"],
                    state["name"],
                    state["state_fips"],
                    tz_label(tz, offset, rng),
                    begin,
                    end,
                    magnitude,
                    "MS" if magnitude else None,
                    f"{label} affected the zone.",
                    rng,
                )
            )
    return rows


BACKGROUND = [
    ("Hail", "C", 0.2),
    ("Flash Flood", "C", 0.12),
    ("Heat", "Z", 0.1),
    ("Drought", "Z", 0.1),
    ("Dense Fog", "Z", 0.07),
    ("Frost/Freeze", "Z", 0.07),
    ("Flood", "C", 0.08),
    ("Excessive Heat", "Z", 0.05),
    ("Cold/Wind Chill", "Z", 0.06),
    ("Heavy Rain", "C", 0.05),
    ("Lightning", "C", 0.03),
    ("Marine Thunderstorm Wind", "M", 0.07),
]


def background_reports(
    year: int, world: World, rng: np.random.Generator, count: int, ids: Ids
) -> list[dict]:
    counties = world.counties
    zones = world.zones.drop_duplicates("STATE_ZONE").reset_index(drop=True)
    info = states().set_index("abbr")
    probs = np.array([b[2] for b in BACKGROUND])
    picks = rng.choice(len(BACKGROUND), count, p=probs / probs.sum())
    days = 366 if calendar.isleap(year) else 365
    rows = []
    for i, pick in enumerate(picks):
        label, cz_type, _ = BACKGROUND[pick]
        begin = datetime(year, 1, 1) + timedelta(minutes=int(rng.integers(0, days * 1440)))
        end = begin + timedelta(hours=float(rng.uniform(0.2, 48)))
        episode = year * 1000 + 500 + i // 10
        if cz_type == "C" or cz_type == "M":
            r = counties.iloc[int(rng.integers(0, len(counties)))]
            fips = int(r["eaglei_fips"][2:]) if cz_type == "C" else 630 + int(rng.integers(0, 50))
            name = r["name"] if cz_type == "C" else "Lake Waters"
            rows.append(
                event_row(
                    ids(),
                    episode,
                    label,
                    cz_type,
                    fips,
                    name,
                    r["state_name"],
                    r["state_fips"],
                    tz_label(r["tz"], int(r["offset"]), rng),
                    begin,
                    end,
                    None,
                    None,
                    "Background event.",
                    rng,
                )
            )
        else:
            z = zones.iloc[int(rng.integers(0, len(zones)))]
            state = info.loc[z["STATE"]]
            tz = z["TIME_ZONE"]
            rows.append(
                event_row(
                    ids(),
                    episode,
                    label,
                    "Z",
                    int(z["ZONE"]),
                    z["NAME"],
                    state["name"],
                    state["state_fips"],
                    tz_label(tz, TZ_OFFSETS[tz], rng),
                    begin,
                    end,
                    None,
                    None,
                    "Background event.",
                    rng,
                )
            )
    return rows


def eaglei_codes(counties: pd.DataFrame, year: int, padded: bool) -> list[str]:
    codes = counties["eaglei_fips"].tolist()
    if year >= 2023:
        ct = sorted(c for c in codes if c.startswith("09"))
        remap = {c: CT_REGIONS[i % len(CT_REGIONS)] for i, c in enumerate(ct)}
        codes = [remap.get(c, c) for c in codes]
    return codes if padded else [str(int(c)) for c in codes]


def write_eaglei_year(
    path: Path,
    arr: np.ndarray,
    steps: int,
    counties: pd.DataFrame,
    year: int,
    style: dict,
    rng: np.random.Generator,
) -> int:
    fips = pa.array(eaglei_codes(counties, year, style["padded"]))
    names = pa.array(counties["name"].tolist())
    state_names = pa.array(counties["state_name"].tolist())
    base = int(calendar.timegm(datetime(year, 1, 1).timetuple()))
    options = pacsv.WriteOptions(include_header=False, quoting_style="none")
    rows = 0
    with open(path, "wb") as handle:
        handle.write(f"fips_code,county,state,{style['value']},run_start_time\n".encode())
        for month in range(1, 13):
            t0 = (datetime(year, month, 1) - datetime(year, 1, 1)).days * PER_DAY
            t1 = t0 + calendar.monthrange(year, month)[1] * PER_DAY
            block = arr[t0:t1]
            ti, ci = np.nonzero(block)
            vals = block[ti, ci].astype(np.int64)
            dup = np.flatnonzero(rng.random(len(ti)) < 0.0005)
            ti = np.concatenate([ti, ti[dup]])
            ci = np.concatenate([ci, ci[dup]])
            vals = np.concatenate([vals, vals[dup]])
            spikes = rng.random(len(vals)) < 2e-6
            vals[spikes] = vals[spikes] * 50 + 100000
            stamps = pa.array(base + (t0 + ti.astype(np.int64)) * 900, type=pa.timestamp("s"))
            table = pa.table(
                {
                    "fips_code": pc.take(fips, pa.array(ci)),
                    "county": pc.take(names, pa.array(ci)),
                    "state": pc.take(state_names, pa.array(ci)),
                    "value": pa.array(vals),
                    "run_start_time": pc.strftime(stamps, format=style["fmt"]),
                }
            )
            pacsv.write_csv(table, handle, options)
            rows += len(vals)
    return rows


def apply_gaps(
    arr: np.ndarray, steps: int, counties: pd.DataFrame, rng: np.random.Generator, profile: dict
) -> list[tuple[int, int]]:
    gaps = []
    for _ in range(profile["global_gaps"]):
        length = int(rng.choice([1, 2, 4, 8, 16, 48, 96]))
        start = int(rng.integers(0, steps - length))
        arr[start : start + length] = 0
        gaps.append((start, length))
    if rng.random() < 0.3:
        start = int(rng.integers(0, steps - 300))
        arr[start : start + 288] = 0
        gaps.append((start, 288))
    abbrs = counties["abbr"].to_numpy()
    for _ in range(profile["partial_gaps"]):
        state = rng.choice(np.unique(abbrs))
        members = np.flatnonzero((abbrs == state) & (rng.random(len(abbrs)) < 0.6))
        length = int(rng.integers(8, 193))
        start = int(rng.integers(0, steps - length))
        arr[start : start + length, members] = 0
    return gaps


def apply_coverage(arr: np.ndarray, counties: pd.DataFrame, year: int, steps: int) -> None:
    onboard = counties["onboard_year"].to_numpy()
    days = counties["onboard_day"].to_numpy()
    arr[:steps, onboard > year] = 0
    for c in np.flatnonzero(onboard == year):
        arr[: days[c] * PER_DAY, c] = 0


def ghcn_chunk(
    ids: np.ndarray,
    di: np.ndarray,
    si: np.ndarray,
    element: str,
    values: np.ndarray,
    dates: pa.Array,
    rng: np.random.Generator,
) -> pa.Table:
    n = len(di)
    qflag = np.where(rng.random(n) < 0.001, "I", None)
    return pa.table(
        {
            "id": pc.take(pa.array(ids), pa.array(si)),
            "date": pc.take(dates, pa.array(di)),
            "element": pa.array(np.full(n, element)),
            "value": pa.array(values.astype(np.int64)),
            "mflag": pa.nulls(n, pa.string()),
            "qflag": pa.array(qflag, pa.string()),
            "sflag": pa.array(np.full(n, "W")),
            "obstime": pa.nulls(n, pa.string()),
        }
    )


def write_ghcn_year(path: Path, world: World, year: int, wx: dict, rng: np.random.Generator) -> int:
    stations = world.stations
    usw = stations[stations["kind"] == "usw"].reset_index(drop=True)
    days = wx["gust"].shape[0]
    dates = pa.array(
        [(datetime(year, 1, 1) + timedelta(days=d)).strftime("%Y%m%d") for d in range(days)]
    )
    doy = np.arange(days)[:, None]
    lat = usw["lat"].to_numpy()[None, :]
    lon = usw["lon"].to_numpy()[None, :]
    shape = (days, len(usw))
    noise = rng.normal(0, 2.5, shape)
    noise = (noise + np.roll(noise, 1, axis=0) + np.roll(noise, 2, axis=0)) / 1.7
    tmax = 16 + 13 * np.sin(2 * np.pi * (doy - 105) / 365.25) - 0.7 * (lat - 37) + noise
    tmin = tmax - (9 + rng.normal(0, 2, shape))
    plains = np.exp(-0.5 * ((lon + 100) / 6) ** 2)
    awnd = rng.gamma(4, 0.9, shape) * (1 + 0.3 * plains)
    wsf2 = awnd * 1.9 + rng.gamma(2, 0.8, shape)
    wsf5 = wsf2 * 1.28 + rng.gamma(2, 0.6, shape)
    gust = wx["gust"]
    wsf5 = np.where(gust > 0.5, np.maximum(wsf5, gust + rng.normal(0, 1.5, shape)), wsf5)
    wsf2 = np.maximum(wsf2, wsf5 / 1.3)
    awnd = np.maximum(awnd, 0.4 * gust)
    wet = rng.random(shape) < np.where(lon > -104, 0.3, 0.15)
    rain = np.where(wet, rng.gamma(0.7, 8, shape), 0) + wx["rain"]
    iceday = wx["ice"] > 0.5
    snowday = wx["snow"] > 0.5
    tmax = np.where(iceday, np.minimum(tmax, rng.normal(-1, 1, shape)), tmax)
    tmax = np.where(snowday, np.minimum(tmax, 1.0), tmax)
    tmin = np.minimum(tmin, tmax - 2)
    snow = np.where(tmax < 1, rain * 10 * rng.uniform(0.6, 1.0, shape), 0) + wx["snow"] * 10
    prcp = rain + wx["snow"] + wx["ice"]
    snwd = np.zeros(shape)
    for d in range(days):
        prev = snwd[d - 1] if d else 0
        melt = np.maximum(tmax[d], 0) * 8
        snwd[d] = np.maximum(0, prev * 0.95 + snow[d] - melt)
    summer = ((doy > 120) & (doy < 260)) & (rng.random(shape) < 0.06)
    thunder = wx["thunder"] | summer
    wind = usw["wind"].to_numpy()[None, :]
    snowobs = usw["snowobs"].to_numpy()[None, :]
    present = rng.random((len(GHCN_ELEMENTS) + 3, *shape)) > 0.02
    elements = {
        "PRCP": (np.round(prcp * 10), present[0]),
        "SNOW": (np.round(snow), present[1] & snowobs),
        "SNWD": (np.round(snwd), present[2] & snowobs),
        "TMAX": (np.round(tmax * 10), present[3]),
        "TMIN": (np.round(tmin * 10), present[4]),
        "AWND": (np.round(awnd * 10), present[5] & wind),
        "WDF2": (rng.integers(1, 37, shape) * 10, present[6] & wind),
        "WDF5": (rng.integers(1, 37, shape) * 10, present[7] & wind),
        "WSF2": (np.round(wsf2 * 10), present[8] & wind),
        "WSF5": (np.round(wsf5 * 10), present[9] & wind),
        "WT03": (np.ones(shape), present[10] & thunder),
        "WT06": (np.ones(shape), present[11] & iceday & (rng.random(shape) < 0.8)),
        "WT17": (np.ones(shape), present[12] & iceday & (rng.random(shape) < 0.6)),
    }
    options = pacsv.WriteOptions(include_header=False, quoting_style="none")
    ids = usw["id"].to_numpy()
    rows = 0
    with gzip.open(path, "wb", compresslevel=4) as handle:
        for element, (values, mask) in elements.items():
            di, si = np.nonzero(mask)
            table = ghcn_chunk(ids, di, si, element, values[di, si], dates, rng)
            pacsv.write_csv(table, handle, options)
            rows += table.num_rows
        others = stations[stations["kind"] != "usw"].reset_index(drop=True)
        for lo in range(0, len(others), 2000):
            part = others.iloc[lo : lo + 2000]
            part_ids = part["id"].to_numpy()
            cols = len(part)
            coop = np.array([i.startswith("USC") for i in part_ids])
            foreign = (part["kind"] == "foreign").to_numpy()
            for element, share, who in (
                ("PRCP", 0.8, None),
                ("TMAX", 0.8, "temp"),
                ("TMIN", 0.8, "temp"),
                ("SNOW", 0.5, "us"),
            ):
                allowed = np.ones(cols, bool)
                if who == "temp":
                    allowed = coop | foreign
                elif who == "us":
                    allowed = ~foreign
                mask = (rng.random((days, cols)) < share) & allowed[None, :]
                di, si = np.nonzero(mask)
                values = (
                    rng.integers(-150, 350, len(di))
                    if element.startswith("T")
                    else rng.integers(0, 300, len(di))
                )
                table = ghcn_chunk(part_ids, di, si, element, values, dates, rng)
                pacsv.write_csv(table, handle, options)
                rows += table.num_rows
    return rows


def write_storm_events(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC, compression="gzip")


IEEE_OLD = [
    "SAIDI With MED",
    "SAIFI With MED",
    "CAIDI With MED",
    "SAIDI Without MED",
    "SAIFI Without MED",
    "CAIDI Without MED",
    "SAIDI With MED Minus Loss of Supply",
    "SAIFI With MED Minus Loss of Supply",
    "CAIDI With MED Minus Loss of Supply",
    "Number of Customers",
    "Highest Distribution Voltage",
    "Outages Recorded Automatically",
]
OTHER_OLD = [
    "SAIDI With MED",
    "SAIFI With MED",
    "CAIDI With MED",
    "SAIDI Without MED",
    "SAIFI Without MED",
    "CAIDI Without MED",
    "Number of Customers",
    "Inactive Accounts Included",
    "Momentary Interruption Definition",
    "Highest Distribution Voltage",
    "Outages Recorded Automatically",
]
IEEE_NEW = [
    "SAIDI (minutes per year)",
    "SAIFI (times per year)",
    "CAIDI (minutes per interruption)",
] * 3
IEEE_NEW += ["Number of Customers", "Highest Dist. Voltage", "Outages Recorded Automatically"]
OTHER_NEW = [
    "SAIDI (minutes per year)",
    "SAIFI (times per year)",
    "CAIDI With MED",
    "SAIDI (minutes per year)",
    "SAIFI (times per year)",
    "CAIDI Without MED",
    "Number of Customers",
    "Inactive Accounts Included",
    "Momentary Interruptions",
    "Highest Dist. Voltage",
    "Outages Recorded Automatically",
]


def reliability_rows(
    world: World, year: int, minutes: np.ndarray, rng: np.random.Generator
) -> list[list]:
    counties = world.counties
    state_min = pd.Series(minutes, dtype=float).groupby(counties["abbr"].to_numpy()).sum()
    state_cust = counties.groupby("abbr")["customers"].sum()
    rows = []
    for s_index, abbr in enumerate(sorted(state_min.index)):
        truth = state_min[abbr] / state_cust[abbr]
        count = int(rng.integers(2, 9))
        shares = rng.dirichlet(np.ones(count))
        for k in range(count):
            customers = max(100, int(round(state_cust[abbr] * shares[k])))
            saidi = truth * math.exp(rng.normal(-0.05, 0.3))
            caidi = rng.uniform(90, 200)
            saifi = saidi / caidi
            wo = saidi * rng.uniform(0.35, 0.8)
            wo_saifi = saifi * rng.uniform(0.5, 0.9)
            row = [
                year,
                10000 + s_index * 20 + k,
                f"{abbr} Power Company {k + 1}",
                abbr,
                str(rng.choice(["Investor Owned", "Cooperative", "Municipal"])),
            ]
            if year == 2019:
                row.append("N")
            ieee = rng.random() > 0.15
            if ieee:
                block = [
                    saidi,
                    saifi,
                    caidi,
                    wo,
                    wo_saifi,
                    wo / wo_saifi,
                    saidi * 0.97,
                    saifi * 0.97,
                    caidi,
                    customers,
                    12.5,
                    "Y",
                ]
                if rng.random() < 0.03:
                    block[0] = "."
            else:
                block = [None] * 12
            if not ieee or rng.random() < 0.3:
                other = [
                    saidi * 1.1,
                    saifi,
                    caidi,
                    wo * 1.1,
                    wo_saifi,
                    wo / wo_saifi,
                    customers,
                    "N",
                    "5 minutes",
                    12.5,
                    "Y",
                ]
            else:
                other = [None] * 11
            rows.append(row + block + other)
    for abbr, number in (("AK", 90001), ("HI", 90002)):
        row = [year, number, f"{abbr} Island Power", abbr, "Investor Owned"]
        if year == 2019:
            row.append("N")
        rows.append(
            row
            + [150.0, 1.2, 125.0, 90.0, 0.9, 100.0, 140.0, 1.1, 127.0, 50000, 12.5, "Y"]
            + [None] * 11
        )
    return rows


def write_eia861(path: Path, rows: list[list], year: int) -> None:
    lead = ["Data Year", "Utility Number", "Utility Name", "State", "Ownership"]
    if year == 2019:
        lead.append("Short Form")
    names = lead + (IEEE_NEW + OTHER_NEW if year >= 2024 else IEEE_OLD + OTHER_OLD)
    group = [""] * len(lead) + ["IEEE Standard"] + [""] * 11 + ["Other Standard"] + [""] * 10
    book = Workbook()
    sheet = book.active
    sheet.title = "States"
    if year >= 2021:
        sheet.append([f"Reliability - Data Year {year}"])
    if year >= 2025:
        sheet.append(["Early Release data, subject to revision"])
    sheet.append(group)
    sheet.append(names)
    for row in rows:
        sheet.append(row)
    sheet.append(["Note: Data are final."])
    territories = book.create_sheet("Territories")
    territories.append(names)
    territories.append(
        [year, 95000, "PR Power Authority", "PR", "Public"] + [None] * (len(names) - 5)
    )
    buffer = io.BytesIO()
    book.save(buffer)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"Reliability_{year}.xlsx", buffer.getvalue())
        archive.writestr(f"Sales_Ult_Cust_{year}.xlsx", b"placeholder")


def write_static(raw: Path, world: World, years: list[int], profile: dict) -> None:
    stations = world.stations
    ghcn = raw / "ghcn"
    with open(ghcn / "ghcnd-stations.txt", "w", encoding="ascii") as handle:
        for r in stations.itertuples(index=False):
            handle.write(
                f"{r.id:<11} {r.lat:>8.4f} {r.lon:>9.4f} {r.elev:>6.1f} {r.state:<2} "
                f"{r.name:<30} {'':<3} {'':<3} {'':<5}\n"
            )
    with open(ghcn / "ghcnd-inventory.txt", "w", encoding="ascii") as handle:
        for r in stations.itertuples(index=False):
            elements = ["PRCP"]
            if r.kind == "usw":
                elements = (
                    ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]
                    if r.snowobs
                    else ["PRCP", "TMAX", "TMIN"]
                )
                if r.wind:
                    elements += ["AWND", "WDF2", "WDF5", "WSF2", "WSF5"]
            for element in elements:
                handle.write(
                    f"{r.id:<11} {r.lat:>8.4f} {r.lon:>9.4f} {element:<4} 1948 {years[-1] + 1}\n"
                )
    zones = world.zones
    with open(raw / "nws" / "bp16ap26.dbx", "w", encoding="utf-8") as handle:
        for r in zones.itertuples(index=False):
            handle.write("|".join(str(v) for v in r) + "\n")
    counties = world.counties
    mcc = pd.DataFrame(
        {"County_FIPS": counties["eaglei_fips"].astype(int), "Customers": counties["mcc"]}
    )
    mcc.to_csv(raw / "eaglei" / "MCC.csv", index=False)
    rows = []
    reported = years[1:-1] if len(years) > 2 else years
    for year in reported:
        for abbr, group in counties.groupby("abbr"):
            total = group["mcc"].sum()
            lo = group.loc[group["onboard_year"] < year, "mcc"].sum()
            hi = group.loc[group["onboard_year"] <= year, "mcc"].sum()
            scale = 100.0 if profile["percent_coverage"] else 1.0
            rows.append(
                {
                    "Year": f"1/1/{year % 100:02d}",
                    "State": abbr,
                    "Total Customers": total,
                    "Min_covered": lo,
                    "Max_covered": hi,
                    "Min_pct_covered": round(scale * lo / total, 4),
                    "Max_pct_covered": round(scale * hi / total, 4),
                }
            )
    pd.DataFrame(rows).to_csv(raw / "eaglei" / "coverage_history.csv", index=False)


def eaglei_style(profile_name: str, index: int, year: int) -> dict:
    if profile_name == "tiny":
        if index == 0:
            return {"value": "sum", "padded": False, "fmt": "%Y-%m-%d %H:%M:%S"}
        if index == 1:
            return {"value": "customers_out", "padded": True, "fmt": "%m/%d/%y %H:%M"}
        return {"value": "customers_out", "padded": True, "fmt": "%Y-%m-%d %H:%M:%S"}
    if year < 2019:
        return {"value": "sum", "padded": False, "fmt": "%Y-%m-%d %H:%M:%S"}
    return {"value": "customers_out", "padded": True, "fmt": "%Y-%m-%d %H:%M:%S"}


def generate(cfg: Config, profile_name: str, log=print) -> dict:
    profile = PROFILES[profile_name]
    raw = cfg.paths.raw
    for sub in ("eaglei", "ghcn", "storm_events", "nws", "eia861", "geo", "truth"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    geo_path = raw / "geo" / "geojson-counties-fips.json"
    if profile_name == "tiny":
        geo_path.write_text(json.dumps(tiny_geojson()), encoding="utf-8")
    elif not geo_path.exists():
        raise FileNotFoundError(f"{geo_path} is missing, run the download step for geo first")
    world = make_world(read_geojson(geo_path), cfg.runtime.seed, cfg.years.first, profile)
    write_static(raw, world, cfg.years.all, profile)
    offsets = world.stations.loc[world.stations["kind"] == "usw", "offset"].to_numpy()
    usw = world.stations[world.stations["kind"] == "usw"]
    lat_all = np.concatenate([world.counties["lat"].to_numpy(), usw["lat"].to_numpy()])
    lon_all = np.concatenate([world.counties["lon"].to_numpy(), usw["lon"].to_numpy()])
    n = len(world.counties)
    carry = None
    summary = {"profile": profile_name, "counties": n, "years": {}}
    truth = []
    for index, year in enumerate(cfg.years.all):
        rng = np.random.default_rng([cfg.runtime.seed, year])
        steps = year_steps(year)
        year_start = datetime(year, 1, 1)
        arr = blue_sky(world.counties, year, rng, profile, steps)
        if carry is not None:
            arr[:TAIL] += carry
        storms = make_storms(year, world.counties, rng, profile, year * 1000)
        days = steps // PER_DAY
        wx = {key: np.zeros((days, len(usw))) for key in ("gust", "rain", "snow", "ice")}
        wx["thunder"] = np.zeros((days, len(usw)), dtype=bool)
        ids = Ids(year * 1_000_000)
        reports = []
        for storm in storms:
            shape = storm_shape(storm, lat_all, lon_all)
            rec = storm_outages(storm, shape[:, :n], world.counties, arr, year_start, rng)
            truth.append(rec.assign(year=year))
            station_weather(storm, shape[:, n:], offsets, wx, year_start, days)
            reports.extend(storm_reports(storm, shape[:, :n], rec, world, rng, ids))
        reports.extend(background_reports(year, world, rng, profile["background_events"], ids))
        minutes = arr[:steps].sum(axis=0, dtype=np.int64) * 15
        carry = arr[steps : steps + TAIL].copy()
        apply_coverage(arr, world.counties, year, steps)
        gaps = apply_gaps(arr, steps, world.counties, rng, profile)
        style = eaglei_style(profile_name, index, year)
        rows = write_eaglei_year(
            raw / "eaglei" / f"eaglei_outages_{year}.csv",
            arr,
            steps,
            world.counties,
            year,
            style,
            rng,
        )
        del arr
        ghcn_rows = write_ghcn_year(raw / "ghcn" / f"{year}.csv.gz", world, year, wx, rng)
        stamp = "c20260323"
        write_storm_events(
            raw / "storm_events" / f"StormEvents_details-ftp_v1.0_d{year}_{stamp}.csv.gz", reports
        )
        if profile_name == "tiny" and index == 0:
            write_storm_events(
                raw / "storm_events" / f"StormEvents_details-ftp_v1.0_d{year}_c20240101.csv.gz",
                reports[:5],
            )
        if cfg.download.eia861_first <= year <= cfg.download.eia861_last:
            write_eia861(
                raw / "eia861" / f"f861{year}.zip",
                reliability_rows(world, year, minutes, rng),
                year,
            )
        summary["years"][year] = {
            "eaglei_rows": rows,
            "ghcn_rows": ghcn_rows,
            "storms": len(storms),
            "reports": len(reports),
            "global_gaps": len(gaps),
        }
        log(
            f"synthetic {year}: {rows:,} outage rows, {ghcn_rows:,} weather rows, "
            f"{len(reports):,} storm reports"
        )
    pd.concat(truth, ignore_index=True).assign(
        geo_id=lambda f: world.counties["geo_id"].to_numpy()[f["county"].to_numpy(int)]
    ).to_parquet(raw / "truth" / "storm_outages.parquet", index=False)
    world.counties.drop(columns=[]).to_parquet(raw / "truth" / "counties.parquet", index=False)
    manifest = {
        "kind": "synthetic",
        "profile": profile_name,
        "seed": cfg.runtime.seed,
        "created": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
    }
    (raw / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return summary
