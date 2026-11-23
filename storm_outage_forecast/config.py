from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REFDATA = ROOT / "refdata"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Paths:
    raw: Path
    silver: Path
    gold: Path
    models: Path
    reports: Path
    bundle: Path
    tmp: Path


@dataclass(frozen=True)
class Years:
    first: int
    last: int
    train_end: int
    valid: int
    test_start: int

    @property
    def all(self) -> list[int]:
        return list(range(self.first, self.last + 1))

    @property
    def train(self) -> list[int]:
        return list(range(self.first, self.train_end + 1))

    @property
    def test(self) -> list[int]:
        return list(range(self.test_start, self.last + 1))


@dataclass(frozen=True)
class Runtime:
    duckdb_memory_gb: float
    train_memory_gb: float
    threads: int
    seed: int


@dataclass(frozen=True)
class Download:
    user_agent: str
    timeout_s: int
    retries: int
    eaglei_article: str
    storm_events_index: str
    ghcn_by_year: str
    ghcn_stations: str
    ghcn_inventory: str
    zone_county: str
    zone_county_index: str
    eia861: tuple[str, ...]
    eia861_first: int
    eia861_last: int
    counties_geojson: str


@dataclass(frozen=True)
class Outages:
    significant_share: float
    significant_floor: int
    event_share: float
    event_floor: int
    event_min_peak: int
    merge_gap_hours: float
    restore_share: float
    max_gap_hours: float
    min_covered_days: int
    min_state_coverage: float
    max_bad_share: float


@dataclass(frozen=True)
class Weather:
    neighbors: int
    radius_km: float
    idw_power: float
    gust_quantile: float


@dataclass(frozen=True)
class Storms:
    lookback_hours: float
    lookahead_hours: float


@dataclass(frozen=True)
class Risk:
    negative_rate: float
    logistic_rows: int
    num_leaves: int
    learning_rate: float
    max_rounds: int
    early_stopping: int
    min_child_samples: int
    feature_fraction: float
    bagging_fraction: float
    lambda_l2: float
    quantiles: tuple[float, ...]
    size_learning_rate: float
    size_num_leaves: int
    size_min_child_samples: int
    size_max_rounds: int
    climatology_prior: float
    calibration_bins: int
    shap_rows: int
    top_days: int


@dataclass(frozen=True)
class Survival:
    min_events: int
    max_hours: float


@dataclass(frozen=True)
class Staging:
    mutual_aid_crews: int
    customers_per_base_crew: int
    restore_rate: float
    samples: int
    top_days: int


@dataclass(frozen=True)
class App:
    top_days: int


@dataclass(frozen=True)
class Config:
    source: Path
    paths: Paths
    years: Years
    runtime: Runtime
    download: Download
    outages: Outages
    weather: Weather
    storms: Storms
    risk: Risk
    survival: Survival
    staging: Staging
    app: App


SECTIONS = {
    "paths": Paths,
    "years": Years,
    "runtime": Runtime,
    "download": Download,
    "outages": Outages,
    "weather": Weather,
    "storms": Storms,
    "risk": Risk,
    "survival": Survival,
    "staging": Staging,
    "app": App,
}


def merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def check_type(section: str, key: str, value: Any, kind: type) -> Any:
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise ConfigError(f"[{section}] {key} must be an integer, got {value!r}")
    if kind is float and not isinstance(value, float):
        raise ConfigError(f"[{section}] {key} must be a number, got {value!r}")
    if kind is str and (not isinstance(value, str) or not value.strip()):
        raise ConfigError(f"[{section}] {key} must be a non-empty string, got {value!r}")
    return value


def build_section(name: str, raw: dict[str, Any], base_dir: Path) -> Any:
    cls = SECTIONS[name]
    fields = cls.__dataclass_fields__
    unknown = sorted(set(raw) - set(fields))
    missing = sorted(set(fields) - set(raw))
    if unknown:
        raise ConfigError(f"[{name}] unknown keys: {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"[{name}] missing keys: {', '.join(missing)}")
    values = {}
    for key, field in fields.items():
        value = raw[key]
        kind = field.type
        if kind == "Path":
            value = check_type(name, key, value, str)
            path = Path(value)
            values[key] = path if path.is_absolute() else base_dir / path
        elif kind.startswith("tuple"):
            if not isinstance(value, list) or not value:
                raise ConfigError(f"[{name}] {key} must be a non-empty list")
            item = float if "float" in kind else str
            values[key] = tuple(check_type(name, key, v, item) for v in value)
        else:
            values[key] = check_type(
                name, key, value, {"int": int, "float": float, "str": str}[kind]
            )
    return cls(**values)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate(cfg: Config) -> None:
    y = cfg.years
    require(2014 <= y.first <= y.last <= 2030, "[years] need 2014 <= first <= last <= 2030")
    require(
        y.first <= y.train_end < y.valid < y.test_start <= y.last,
        "[years] need first <= train_end < valid < test_start <= last",
    )
    require(y.valid == y.train_end + 1, "[years] valid must be the year right after train_end")
    require(y.test_start == y.valid + 1, "[years] test_start must be the year right after valid")
    r = cfg.runtime
    require(0.5 <= r.duckdb_memory_gb <= 64, "[runtime] duckdb_memory_gb must be in [0.5, 64]")
    require(0.25 <= r.train_memory_gb <= 64, "[runtime] train_memory_gb must be in [0.25, 64]")
    require(1 <= r.threads <= 64, "[runtime] threads must be in [1, 64]")
    require(r.seed >= 0, "[runtime] seed must be >= 0")
    d = cfg.download
    require(5 <= d.timeout_s <= 600, "[download] timeout_s must be in [5, 600]")
    require(0 <= d.retries <= 20, "[download] retries must be in [0, 20]")
    for key in (
        "eaglei_article",
        "storm_events_index",
        "ghcn_by_year",
        "ghcn_stations",
        "ghcn_inventory",
        "zone_county",
        "zone_county_index",
        "counties_geojson",
    ):
        require(getattr(d, key).startswith("https://"), f"[download] {key} must be an https url")
    require("{year}" in d.ghcn_by_year, "[download] ghcn_by_year needs a {year} placeholder")
    for url in d.eia861:
        require(
            url.startswith("https://") and "{year}" in url,
            "[download] eia861 urls must be https and contain {year}",
        )
    require(
        2015 <= d.eia861_first <= d.eia861_last <= 2030,
        "[download] need 2015 <= eia861_first <= eia861_last <= 2030",
    )
    o = cfg.outages
    for key in ("significant_share", "event_share", "restore_share", "min_state_coverage"):
        require(0 < getattr(o, key) < 1, f"[outages] {key} must be in (0, 1)")
    require(0.001 < o.max_bad_share <= 1, "[outages] max_bad_share must be in (0.001, 1]")
    require(o.significant_floor >= 1 and o.event_floor >= 1, "[outages] floors must be >= 1")
    require(o.event_min_peak >= o.event_floor, "[outages] event_min_peak must be >= event_floor")
    require(0.25 <= o.merge_gap_hours <= 48, "[outages] merge_gap_hours must be in [0.25, 48]")
    require(0 <= o.max_gap_hours <= 24, "[outages] max_gap_hours must be in [0, 24]")
    require(1 <= o.min_covered_days <= 366, "[outages] min_covered_days must be in [1, 366]")
    w = cfg.weather
    require(1 <= w.neighbors <= 10, "[weather] neighbors must be in [1, 10]")
    require(10 <= w.radius_km <= 1000, "[weather] radius_km must be in [10, 1000]")
    require(0 <= w.idw_power <= 4, "[weather] idw_power must be in [0, 4]")
    require(0.5 <= w.gust_quantile < 1, "[weather] gust_quantile must be in [0.5, 1)")
    s = cfg.storms
    require(
        0 <= s.lookback_hours <= 72 and 0 <= s.lookahead_hours <= 72,
        "[storms] lookback_hours and lookahead_hours must be in [0, 72]",
    )
    k = cfg.risk
    require(0 < k.negative_rate <= 1, "[risk] negative_rate must be in (0, 1]")
    require(k.logistic_rows >= 1000, "[risk] logistic_rows must be >= 1000")
    require(2 <= k.num_leaves <= 1024, "[risk] num_leaves must be in [2, 1024]")
    require(0 < k.learning_rate <= 1, "[risk] learning_rate must be in (0, 1]")
    require(1 <= k.early_stopping <= k.max_rounds, "[risk] need 1 <= early_stopping <= max_rounds")
    require(k.min_child_samples >= 1, "[risk] min_child_samples must be >= 1")
    require(
        0 < k.feature_fraction <= 1 and 0 < k.bagging_fraction <= 1,
        "[risk] feature_fraction and bagging_fraction must be in (0, 1]",
    )
    require(k.lambda_l2 >= 0, "[risk] lambda_l2 must be >= 0")
    require(
        list(k.quantiles) == sorted(set(k.quantiles))
        and len(k.quantiles) == 3
        and all(0 < q < 1 for q in k.quantiles)
        and abs(k.quantiles[1] - 0.5) < 1e-9,
        "[risk] quantiles must be three increasing values in (0, 1) with 0.5 in the middle",
    )
    require(0 < k.size_learning_rate <= 1, "[risk] size_learning_rate must be in (0, 1]")
    require(2 <= k.size_num_leaves <= 1024, "[risk] size_num_leaves must be in [2, 1024]")
    require(k.size_min_child_samples >= 1, "[risk] size_min_child_samples must be >= 1")
    require(
        k.early_stopping <= k.size_max_rounds,
        "[risk] size_max_rounds must be >= early_stopping",
    )
    require(k.climatology_prior > 0, "[risk] climatology_prior must be > 0")
    require(2 <= k.calibration_bins <= 50, "[risk] calibration_bins must be in [2, 50]")
    require(k.shap_rows >= 100 and k.top_days >= 1, "[risk] shap_rows >= 100 and top_days >= 1")
    v = cfg.survival
    require(v.min_events >= 5, "[survival] min_events must be >= 5")
    require(24 <= v.max_hours <= 2000, "[survival] max_hours must be in [24, 2000]")
    g = cfg.staging
    require(
        g.mutual_aid_crews >= 1 and g.customers_per_base_crew >= 1,
        "[staging] mutual_aid_crews and customers_per_base_crew must be >= 1",
    )
    require(g.restore_rate > 0, "[staging] restore_rate must be > 0")
    require(g.samples >= 10 and g.top_days >= 1, "[staging] samples >= 10 and top_days >= 1")
    require(cfg.app.top_days >= 1, "[app] top_days must be >= 1")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    source = Path(path) if path else ROOT / "config.toml"
    if not source.exists():
        raise ConfigError(f"config file not found: {source}")
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    if overrides:
        raw = merge(raw, overrides)
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ConfigError(f"unknown sections: {', '.join(unknown)}")
    base_dir = source.resolve().parent
    parts = {}
    for name in SECTIONS:
        if name not in raw or not isinstance(raw[name], dict):
            raise ConfigError(f"missing section [{name}]")
        parts[name] = build_section(name, raw[name], base_dir)
    cfg = Config(source=source.resolve(), **parts)
    validate(cfg)
    return cfg


def with_paths(cfg: Config, root: Path) -> Config:
    fields = Paths.__dataclass_fields__
    paths = {key: root / getattr(cfg.paths, key).relative_to(cfg.source.parent) for key in fields}
    return Config(**{**cfg.__dict__, "paths": Paths(**paths)})
