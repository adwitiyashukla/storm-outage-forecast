from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime
from importlib import metadata

from . import (
    bundle,
    features,
    outages,
    reliability,
    risk,
    staging,
    storms,
    survival,
    units,
    weather,
)
from .config import Config
from .db import connect

PACKAGES = ["duckdb", "lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]
STAGES = ("units", "weather", "outages", "storms", "reliability", "features")


class QualityError(RuntimeError):
    pass


def peak_memory_mb() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
        read.restype = wintypes.BOOL
        if not read(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return 0.0
        return counters.PeakWorkingSetSize / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "packages": versions,
    }


def data_kind(cfg: Config) -> str:
    path = cfg.paths.raw / "manifest.json"
    if not path.exists():
        return "unknown"
    return json.loads(path.read_text(encoding="utf-8")).get("kind", "unknown")


def read_metrics(cfg: Config) -> dict:
    path = cfg.paths.reports / "metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_metrics(cfg: Config, metrics: dict) -> None:
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    metrics["updated"] = datetime.now().isoformat(timespec="seconds")
    text = json.dumps(metrics, indent=2, default=str)
    (cfg.paths.reports / "metrics.json").write_text(text, encoding="utf-8")


def check_quality(con, stages: tuple[str, ...]) -> list[dict]:
    rows = con.execute("SELECT * FROM quality ORDER BY stage, check_name").df()
    failed = rows[(rows["status"] == "fail") & rows["stage"].isin(stages)]
    if len(failed):
        lines = [f"{r.stage}.{r.check_name} = {r.value} {r.detail}" for r in failed.itertuples()]
        raise QualityError("quality checks failed:\n" + "\n".join(lines))
    return rows.to_dict(orient="records")


def timed(timing: dict, name: str, func, *args, log=print):
    started = time.perf_counter()
    result = func(*args, log=log)
    timing[name] = round(time.perf_counter() - started, 1)
    log(f"{name} finished in {timing[name]:.1f} s")
    return result


def build(cfg: Config, log=print, start: str | None = None) -> dict:
    if start is not None and start not in STAGES:
        raise ValueError(f"unknown stage {start}, pick from {', '.join(STAGES)}")
    stages = STAGES[STAGES.index(start) :] if start else STAGES
    metrics = read_metrics(cfg) if start else {}
    metrics.setdefault("timing_s", {})
    metrics.setdefault("peak_memory_mb", {})
    timing = metrics["timing_s"]
    if start is None:
        for path in (cfg.paths.gold, cfg.paths.gold.with_name(cfg.paths.gold.name + ".wal")):
            if path.exists():
                path.unlink()
    runners = {
        "units": units.build_units,
        "weather": weather.build,
        "outages": outages.build,
        "storms": storms.build,
        "reliability": reliability.build,
        "features": features.build,
    }
    con = connect(cfg)
    try:
        for name in stages:
            result = timed(timing, name, runners[name], cfg, con, log=log)
            if name == "reliability":
                metrics["reliability"] = result
        metrics["quality"] = check_quality(
            con, ("units", "outages", "weather", "storms", "features")
        )
        metrics["data"] = summary(cfg, con)
    finally:
        con.close()
    metrics["data_kind"] = data_kind(cfg)
    metrics["environment"] = environment()
    metrics["peak_memory_mb"]["build"] = round(peak_memory_mb(), 1)
    write_metrics(cfg, metrics)
    return metrics


def summary(cfg: Config, con) -> dict:
    row = con.execute("""
        SELECT (SELECT count(*) FROM units), (SELECT sum(customers) FROM units),
               (SELECT sum(rows) FROM snapshots), (SELECT count(*) FROM unit_day),
               (SELECT count(*) FROM events), (SELECT count(*) FROM storm_reports),
               (SELECT count(DISTINCT station) FROM station_map WHERE grp = 'wind')
    """).fetchone()
    splits = con.execute("""
        SELECT split, count(*) AS rows, avg(sig::INT) AS positive_rate
        FROM model_rows WHERE usable GROUP BY split ORDER BY split
    """).df()
    return {
        "years": [cfg.years.first, cfg.years.last],
        "units": row[0],
        "customers": row[1],
        "outage_rows": row[2],
        "county_days": row[3],
        "events": row[4],
        "storm_reports_mapped": row[5],
        "wind_stations": row[6],
        "splits": splits.to_dict(orient="records"),
    }


def train(cfg: Config, log=print) -> dict:
    metrics = read_metrics(cfg)
    timing = metrics.setdefault("timing_s", {})
    con = connect(cfg, memory_gb=cfg.runtime.train_memory_gb)
    try:
        metrics["risk"] = timed(timing, "risk", risk.run, cfg, con, log=log)
        metrics["restoration"] = timed(timing, "restoration", survival.run, cfg, con, log=log)
        metrics["staging"] = timed(timing, "staging", staging.run, cfg, con, log=log)
        metrics["quality"] = check_quality(con, ("risk", "restoration", "staging"))
    finally:
        con.close()
    memory = metrics.get("peak_memory_mb")
    metrics["peak_memory_mb"] = {
        **(memory if isinstance(memory, dict) else {}),
        "train": round(peak_memory_mb(), 1),
    }
    write_metrics(cfg, metrics)
    return metrics


def make_bundle(cfg: Config, log=print) -> dict:
    metrics = read_metrics(cfg)
    con = connect(cfg, read_only=True)
    try:
        info = bundle.export(cfg, con, metrics, log=log)
    finally:
        con.close()
    return info


def run(cfg: Config, log=print, start: str | None = None) -> dict:
    started = time.perf_counter()
    build(cfg, log, start=start)
    metrics = train(cfg, log)
    make_bundle(cfg, log)
    metrics = read_metrics(cfg)
    metrics["timing_s"]["total"] = round(time.perf_counter() - started, 1)
    write_metrics(cfg, metrics)
    return metrics
