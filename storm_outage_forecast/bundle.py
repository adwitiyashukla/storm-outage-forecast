from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from .config import ROOT, Config
from .geo import read_geojson

SPACE_README = """---
title: Storm Outage Forecast
emoji: {emoji}
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Dashboard for the storm-outage-forecast project: county-level power outage risk from weather,
restoration times and crew staging, tested on held-out years.
"""

SPACE_REQUIREMENTS = [
    "duckdb==1.5.5",
    "numpy==2.4.6",
    "pandas==2.3.3",
    "plotly==7.0.0",
    "streamlit==1.63.0",
]

DOCKERFILE = """FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH BUNDLE_DIR=/home/user/app/bundle
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
"""


def round_coords(value, digits: int = 3):
    if isinstance(value, list):
        if value and isinstance(value[0], (int, float)):
            return [round(float(v), digits) for v in value]
        return [round_coords(v, digits) for v in value]
    return value


def write_geojson(cfg: Config, con: duckdb.DuckDBPyConnection, folder: Path) -> None:
    geo = read_geojson(cfg.paths.raw / "geo" / "geojson-counties-fips.json")
    keep = set(con.execute("SELECT geo_id FROM unit_geo").df()["geo_id"])
    features = []
    for feature in geo["features"]:
        geo_id = str(feature["id"]).zfill(5)
        if geo_id in keep:
            geometry = dict(feature["geometry"])
            geometry["coordinates"] = round_coords(geometry["coordinates"])
            features.append(
                {
                    "type": "Feature",
                    "id": geo_id,
                    "properties": {"NAME": feature.get("properties", {}).get("NAME", "")},
                    "geometry": geometry,
                }
            )
    text = json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":"))
    (folder / "counties.json").write_text(text, encoding="utf-8")


def frames(cfg: Config, con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    top = cfg.app.top_days
    q = con.execute
    out = {
        "units": q("SELECT unit_id, name, abbr, customers, lat, lon FROM units").df(),
        "unit_geo": q("SELECT geo_id, unit_id FROM unit_geo").df(),
        "days": q("SELECT * FROM test_days ORDER BY day").df(),
        "map_days": q(f"""
            SELECT p.unit_id, p.day, p.p, p.p_clim, p.q10, p.q50, p.q90, p.expected, p.peak_out,
                   p.sig, p.cause, p.customers
            FROM predictions p
            WHERE p.day IN (
                SELECT day FROM (SELECT day FROM test_days ORDER BY actual DESC LIMIT {top})
                UNION
                SELECT day FROM (
                    SELECT day FROM test_days ORDER BY expected DESC LIMIT {max(1, top // 4)}
                )
            )
        """).df(),
        "shap": q("SELECT * FROM risk_shap").df(),
        "by_cause": q("SELECT * FROM risk_by_cause").df(),
        "survival_curves": q("SELECT * FROM survival_curves").df(),
        "survival_coefficients": q("SELECT * FROM survival_coefficients").df(),
        "survival_calibration": q("SELECT * FROM survival_calibration").df(),
        "event_summary": q(f"""
            SELECT cause, count(*) AS events,
                   median(duration_h) FILTER (WHERE NOT censored) AS median_hours,
                   median(peak_out) AS median_peak, max(peak_out) AS max_peak
            FROM events WHERE peak_out >= {cfg.outages.event_min_peak}
            GROUP BY cause ORDER BY events DESC
        """).df(),
        "staging_results": q("SELECT * FROM staging_results").df(),
        "staging_allocations": q("SELECT * FROM staging_allocations").df(),
        "saidi_check": q("SELECT * FROM saidi_check").df(),
        "quality": q("SELECT * FROM quality ORDER BY stage, check_name").df(),
        "coverage": q("""
            SELECT year,
                   avg(CASE WHEN unit_covered AND state_covered THEN 1.0 ELSE 0.0 END)
                       AS covered_share,
                   avg(CASE WHEN complete THEN 1.0 ELSE 0.0 END) AS complete_share,
                   sum(rows) AS outage_rows
            FROM unit_day GROUP BY year ORDER BY year
        """).df(),
    }
    reliability, curves = [], []
    for model in ("lightgbm", "logistic", "climatology"):
        reliability.append(q(f"SELECT * FROM risk_reliability_{model}").df().assign(model=model))
        curves.append(q(f"SELECT * FROM risk_pr_{model}").df().assign(model=model))
    out["reliability"] = pd.concat(reliability, ignore_index=True)
    out["pr_curve"] = pd.concat(curves, ignore_index=True)
    return out


def export(cfg: Config, con: duckdb.DuckDBPyConnection, metrics: dict, log=print) -> dict:
    folder = cfg.paths.bundle
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "app.duckdb"
    for path in (target, target.with_name(target.name + ".wal")):
        try:
            path.unlink(missing_ok=True)
        except PermissionError as err:
            raise RuntimeError(f"{path} is in use, stop the dashboard before rebuilding") from err
    tables = frames(cfg, con)
    out = duckdb.connect(str(target))
    try:
        for name, frame in tables.items():
            out.register("t_df", frame)
            out.execute(f"CREATE TABLE {name} AS SELECT * FROM t_df")
            out.unregister("t_df")
        out.execute("CHECKPOINT")
    finally:
        out.close()
    write_geojson(cfg, con, folder)
    meta = {
        "data_kind": metrics.get("data_kind"),
        "years": [cfg.years.first, cfg.years.last],
        "train_years": [cfg.years.first, cfg.years.train_end],
        "valid_year": cfg.years.valid,
        "test_years": [cfg.years.test_start, cfg.years.last],
        "significant": {
            "share": cfg.outages.significant_share,
            "floor": cfg.outages.significant_floor,
        },
        "staging": {
            "crews": cfg.staging.mutual_aid_crews,
            "restore_rate": cfg.staging.restore_rate,
            "customers_per_base_crew": cfg.staging.customers_per_base_crew,
        },
        "created": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            key: metrics.get(key) for key in ("data", "reliability", "restoration", "staging")
        },
        "risk": {
            key: metrics.get("risk", {}).get(key) for key in ("test", "slices", "size", "days")
        },
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    size = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    log(f"bundle: {len(tables)} tables, {size / 2**20:.1f} MB in {folder}")
    return {"tables": len(tables), "bytes": size}


def build_space(cfg: Config, dest: Path, log=print) -> Path:
    bundle = cfg.paths.bundle
    for name in ("app.duckdb", "counties.json", "meta.json"):
        if not (bundle / name).exists():
            raise FileNotFoundError(f"{bundle / name} is missing, run the pipeline first")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "bundle").mkdir(exist_ok=True)
    for name in ("app.duckdb", "counties.json", "meta.json"):
        shutil.copy2(bundle / name, dest / "bundle" / name)
    shutil.copy2(ROOT / "app" / "app.py", dest / "app.py")
    (dest / ".streamlit").mkdir(exist_ok=True)
    shutil.copy2(ROOT / ".streamlit" / "config.toml", dest / ".streamlit" / "config.toml")
    (dest / "requirements.txt").write_text("\n".join(SPACE_REQUIREMENTS) + "\n", encoding="utf-8")
    (dest / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    readme = SPACE_README.format(emoji=chr(0x26A1))
    (dest / "README.md").write_text(readme, encoding="utf-8")
    (dest / ".gitattributes").write_text(
        "*.duckdb filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )
    log(f"space files written to {dest}")
    return dest
