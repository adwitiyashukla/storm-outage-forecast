from __future__ import annotations

import gzip
import hashlib
import json

from storm_outage_forecast import reliability, storms, synthetic, weather
from storm_outage_forecast.config import load_config

from .conftest import tiny_overrides


def first_lines(path, count=3):
    with open(path, encoding="utf-8") as handle:
        return [next(handle).rstrip("\n") for _ in range(count)]


def test_manifest_marks_synthetic(replica):
    manifest = json.loads((replica.paths.raw / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "synthetic"
    assert manifest["profile"] == "tiny"


def test_eaglei_layout_variants(replica):
    folder = replica.paths.raw / "eaglei"
    assert first_lines(folder / "eaglei_outages_2019.csv", 1) == [
        "fips_code,county,state,sum,run_start_time"
    ]
    text_2019 = (folder / "eaglei_outages_2019.csv").read_text(encoding="utf-8")
    assert "\n9001," in text_2019 or "\n9003," in text_2019
    line_2020 = first_lines(folder / "eaglei_outages_2020.csv", 2)[1]
    assert line_2020.split(",")[-1][2] == "/"
    text_2023 = (folder / "eaglei_outages_2023.csv").read_text(encoding="utf-8")
    assert "\n09110," in text_2023 or "\n09120," in text_2023
    assert "\n09001," not in text_2023


def test_ghcn_rows_and_station_file(replica):
    folder = replica.paths.raw / "ghcn"
    with gzip.open(folder / "2021.csv.gz", "rt") as handle:
        rows = [next(handle).rstrip("\n").split(",") for _ in range(2000)]
    assert all(len(r) == 8 and len(r[0]) == 11 for r in rows)
    stations = weather.read_stations(folder / "ghcnd-stations.txt")
    assert len(stations) > 0
    assert stations["station"].str.startswith("USW").all()


def test_storm_events_and_eia_files(replica):
    files = storms.latest_files(replica.paths.raw / "storm_events", replica.years.all)
    assert files[2019].name.endswith("_c20260323.csv.gz")
    with gzip.open(files[2019], "rt") as handle:
        header = next(handle).strip().split(",")
    assert len(header) == 51
    for year in replica.years.all:
        parsed = reliability.parse_reliability(
            replica.paths.raw / "eia861" / f"f861{year}.zip", year
        )
        assert parsed["saidi_med"].notna().sum() > 0


def test_generation_is_deterministic(replica, tmp_path):
    cfg = load_config(overrides=tiny_overrides(tmp_path))
    synthetic.generate(cfg, "tiny", log=lambda message: None)
    for name in ("eaglei/eaglei_outages_2021.csv", "ghcn/2022.csv.gz", "nws/bp16ap26.dbx"):
        ours = hashlib.md5((cfg.paths.raw / name).read_bytes()).hexdigest()
        theirs = hashlib.md5((replica.paths.raw / name).read_bytes()).hexdigest()
        if name.endswith(".gz"):
            ours = hashlib.md5(gzip.decompress((cfg.paths.raw / name).read_bytes())).hexdigest()
            theirs = hashlib.md5(
                gzip.decompress((replica.paths.raw / name).read_bytes())
            ).hexdigest()
        assert ours == theirs, name
