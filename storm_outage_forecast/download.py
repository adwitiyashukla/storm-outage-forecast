from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .config import Config

CHUNK = 1 << 20
SOURCES = ("geo", "nws", "eia861", "storm_events", "ghcn", "eaglei")
NWS_MONTHS = {
    "ja": 1,
    "fe": 2,
    "mr": 3,
    "ap": 4,
    "my": 5,
    "jn": 6,
    "jl": 7,
    "au": 8,
    "se": 9,
    "oc": 10,
    "no": 11,
    "de": 12,
}


class DownloadError(RuntimeError):
    pass


@dataclass
class Item:
    source: str
    url: str
    dest: Path
    size: int | None = None
    md5: str | None = None
    alternates: tuple[str, ...] = ()


def open_url(url: str, cfg: Config, start: int = 0):
    headers = {"User-Agent": cfg.download.user_agent, "Accept-Encoding": "identity"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=cfg.download.timeout_s)


def get_bytes(url: str, cfg: Config) -> bytes:
    last = None
    for attempt in range(cfg.download.retries + 1):
        try:
            with open_url(url, cfg) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            if err.code in (403, 404, 410):
                raise DownloadError(f"{url} answered HTTP {err.code}") from err
            last = err
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as err:
            last = err
        time.sleep(min(30, 2**attempt))
    raise DownloadError(f"could not read {url}: {last}")


def looks_like_html(head: bytes) -> bool:
    text = head[:1024].lstrip().lower()
    return text.startswith((b"<!doctype html", b"<html")) or b"<html" in text[:300]


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK * 8):
            digest.update(block)
    return digest.hexdigest()


def check_packed(path: Path, name: str) -> None:
    try:
        if name.endswith(".gz"):
            with gzip.open(path, "rb") as handle:
                while handle.read(CHUNK * 16):
                    pass
        elif name.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
            if bad:
                raise DownloadError(f"{name} has a corrupt member {bad}")
        else:
            with open(path, encoding="utf-8") as handle:
                json.load(handle)
    except (OSError, EOFError, zlib.error, zipfile.BadZipFile, ValueError) as err:
        raise DownloadError(f"{name} is damaged: {err}") from err


def check_text(path: Path, name: str) -> None:
    with open(path, "rb") as handle:
        head = handle.read(4096)
    if not head.strip():
        raise DownloadError(f"{name} is empty")
    if looks_like_html(head):
        raise DownloadError(f"{name} is an html page, not data")
    if name.startswith("eaglei_outages") and b"run_start_time" not in head.split(b"\n", 1)[0]:
        raise DownloadError(f"{name} does not start with the expected EAGLE-I header")
    if name.endswith(".dbx") and b"|" not in head:
        raise DownloadError(f"{name} is not a pipe delimited zone-county file")


def check_content(path: Path, dest: Path) -> None:
    if dest.name.endswith((".gz", ".zip", ".json")):
        check_packed(path, dest.name)
    else:
        check_text(path, dest.name)


def complete(dest: Path, item: Item) -> bool:
    if not dest.exists():
        return False
    if item.size is not None and dest.stat().st_size != item.size:
        return False
    return not (item.md5 and md5_of(dest) != item.md5)


def fetch(item: Item, cfg: Config, log=print) -> dict:
    dest = item.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if complete(dest, item):
        log(f"skip {dest.name} (already complete)")
        return {"url": item.url, "size": dest.stat().st_size, "md5": md5_of(dest), "status": "kept"}
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for url in (item.url, *item.alternates):
        for attempt in range(cfg.download.retries + 1):
            try:
                have = part.stat().st_size if part.exists() else 0
                with open_url(url, cfg, have) as response:
                    if have and response.status != 206:
                        have = 0
                    length = response.headers.get("Content-Length")
                    expected = have + int(length) if length is not None else None
                    content_type = response.headers.get("Content-Type", "")
                    started = time.time()
                    done = have
                    mark = have
                    with open(part, "ab" if have else "wb") as out:
                        first = True
                        while block := response.read(CHUNK):
                            if (
                                first
                                and not have
                                and ("text/html" in content_type.lower() or looks_like_html(block))
                            ):
                                raise DownloadError(f"{url} returned an html page instead of data")
                            first = False
                            out.write(block)
                            done += len(block)
                            if done - mark >= 200 * CHUNK:
                                rate = (done - have) / max(time.time() - started, 1e-6) / CHUNK
                                log(f"  {dest.name}: {done / CHUNK:,.0f} MB at {rate:,.1f} MB/s")
                                mark = done
                size = part.stat().st_size
                if expected is not None and size != expected:
                    raise DownloadError(
                        f"{dest.name}: transfer stopped at {size} of {expected} bytes"
                    )
                if item.size is not None and size != item.size:
                    part.unlink()
                    raise DownloadError(f"{dest.name}: got {size} bytes, expected {item.size}")
                digest = md5_of(part)
                if item.md5 and digest != item.md5:
                    part.unlink()
                    raise DownloadError(f"{dest.name}: md5 {digest} does not match {item.md5}")
                try:
                    check_content(part, dest)
                except DownloadError:
                    part.unlink()
                    raise
                part.replace(dest)
                log(f"done {dest.name} ({size / CHUNK:,.1f} MB)")
                return {"url": url, "size": size, "md5": digest, "status": "downloaded"}
            except urllib.error.HTTPError as err:
                last_error = err
                if err.code == 416 and part.exists():
                    part.unlink()
                if err.code in (403, 404, 410):
                    break
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                ConnectionError,
                OSError,
                DownloadError,
            ) as err:
                last_error = err
                if isinstance(err, DownloadError) and "html page" in str(err) and part.exists():
                    part.unlink()
                    break
            log(f"  retry {dest.name} after error: {last_error}")
            time.sleep(min(60, 2**attempt))
    raise DownloadError(f"could not download {dest.name}: {last_error}")


def plan_geo(cfg: Config) -> list[Item]:
    url = cfg.download.counties_geojson
    return [Item("geo", url, cfg.paths.raw / "geo" / "geojson-counties-fips.json")]


def nws_date(name: str) -> date | None:
    match = re.fullmatch(r"bp(\d{2})([a-z]{2})(\d{2})\.dbx", name)
    if not match or match.group(2) not in NWS_MONTHS:
        return None
    day, month, year = int(match.group(1)), NWS_MONTHS[match.group(2)], 2000 + int(match.group(3))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def latest_nws(html: str) -> str | None:
    names = set(re.findall(r"(bp\d{2}[a-z]{2}\d{2}\.dbx)", html))
    dated = [(nws_date(n), n) for n in names if nws_date(n)]
    return max(dated)[1] if dated else None


def plan_nws(cfg: Config) -> list[Item]:
    url = cfg.download.zone_county
    try:
        with open_url(url, cfg) as response:
            response.read(64)
        name = url.rsplit("/", 1)[-1]
    except (urllib.error.URLError, TimeoutError, OSError):
        html = get_bytes(cfg.download.zone_county_index, cfg).decode("utf-8", "replace")
        name = latest_nws(html)
        if not name:
            raise DownloadError("no zone-county file found on the NWS page") from None
        url = url.rsplit("/", 1)[0] + "/" + name
    return [Item("nws", url, cfg.paths.raw / "nws" / name)]


def plan_eia861(cfg: Config) -> list[Item]:
    d = cfg.download
    items = []
    for year in range(d.eia861_first, d.eia861_last + 1):
        urls = [template.format(year=year) for template in d.eia861]
        items.append(
            Item(
                "eia861",
                urls[0],
                cfg.paths.raw / "eia861" / f"f861{year}.zip",
                alternates=tuple(urls[1:]),
            )
        )
    return items


def storm_event_files(html: str, years: list[int]) -> dict[int, str]:
    found = re.findall(r"(StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz)", html)
    latest: dict[int, tuple[str, str]] = {}
    for name, year, stamp in found:
        y = int(year)
        if y in years and (y not in latest or stamp > latest[y][1]):
            latest[y] = (name, stamp)
    return {y: latest[y][0] for y in sorted(latest)}


def plan_storm_events(cfg: Config) -> list[Item]:
    base = cfg.download.storm_events_index
    html = get_bytes(base, cfg).decode("utf-8", "replace")
    files = storm_event_files(html, cfg.years.all)
    missing = sorted(set(cfg.years.all) - set(files))
    if missing:
        raise DownloadError(f"Storm Events index has no details file for {missing}")
    folder = cfg.paths.raw / "storm_events"
    return [
        Item("storm_events", base.rstrip("/") + "/" + name, folder / name)
        for name in files.values()
    ]


def plan_ghcn(cfg: Config) -> list[Item]:
    d = cfg.download
    folder = cfg.paths.raw / "ghcn"
    items = [
        Item("ghcn", d.ghcn_stations, folder / "ghcnd-stations.txt"),
        Item("ghcn", d.ghcn_inventory, folder / "ghcnd-inventory.txt"),
    ]
    for year in cfg.years.all:
        items.append(Item("ghcn", d.ghcn_by_year.format(year=year), folder / f"{year}.csv.gz"))
    return items


def plan_eaglei(cfg: Config) -> list[Item]:
    try:
        article = json.loads(get_bytes(cfg.download.eaglei_article, cfg))
    except ValueError as err:
        raise DownloadError(f"figshare did not answer with json: {err}") from err
    files = {f["name"]: f for f in article.get("files", [])}
    wanted = [f"eaglei_outages_{year}.csv" for year in cfg.years.all]
    wanted += ["MCC.csv", "coverage_history.csv"]
    missing = [name for name in wanted if name not in files]
    if missing:
        raise DownloadError(f"figshare article is missing {missing}; it lists {sorted(files)}")
    folder = cfg.paths.raw / "eaglei"
    items = []
    for name in wanted:
        meta = files[name]
        md5 = meta.get("computed_md5") or meta.get("supplied_md5") or None
        items.append(
            Item("eaglei", meta["download_url"], folder / name, size=int(meta["size"]), md5=md5)
        )
    return items


PLANNERS = {
    "geo": plan_geo,
    "nws": plan_nws,
    "eia861": plan_eia861,
    "storm_events": plan_storm_events,
    "ghcn": plan_ghcn,
    "eaglei": plan_eaglei,
}


def read_manifest(raw: Path) -> dict:
    path = raw / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"kind": "real", "files": {}}


def write_manifest(raw: Path, manifest: dict) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    manifest["updated"] = datetime.now().isoformat(timespec="seconds")
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(cfg: Config, only: list[str] | None = None, log=print) -> dict:
    raw = cfg.paths.raw
    manifest = read_manifest(raw)
    if manifest.get("kind") == "synthetic":
        raise DownloadError(f"{raw} holds a synthetic replica, point paths.raw somewhere else")
    sources = list(only or SOURCES)
    unknown = [source for source in sources if source not in PLANNERS]
    if unknown:
        raise DownloadError(f"unknown source {', '.join(unknown)}, pick from {', '.join(SOURCES)}")
    manifest["kind"] = "real"
    manifest.setdefault("files", {})
    failures = []
    for source in sources:
        log(f"planning {source}")
        try:
            items = PLANNERS[source](cfg)
        except DownloadError as err:
            failures.append(f"{source}: {err}")
            log(f"failed {source}: {err}")
            continue
        for item in items:
            try:
                record = fetch(item, cfg, log)
            except DownloadError as err:
                failures.append(f"{source}: {err}")
                log(f"failed {item.dest.name}: {err}")
                continue
            key = item.dest.relative_to(raw).as_posix()
            manifest["files"][key] = {**record, "source": source}
            write_manifest(raw, manifest)
    manifest["failures"] = failures
    write_manifest(raw, manifest)
    if failures:
        raise DownloadError(f"{len(failures)} download problem(s):\n" + "\n".join(failures))
    total = sum(record["size"] for record in manifest["files"].values())
    log(f"download finished, {len(manifest['files'])} files, {total / 2**30:.1f} GB in {raw}")
    return manifest


def probe(cfg: Config, log=print) -> list[dict]:
    rows = []
    for source in SOURCES:
        try:
            items = PLANNERS[source](cfg)
        except DownloadError as err:
            rows.append({"source": source, "file": "-", "status": f"planning failed: {err}"})
            log(f"{source:13s} {'-':45s} planning failed: {err}")
            continue
        for item in items[:3] if source in ("ghcn", "eaglei", "eia861") else items:
            status = "ok"
            for url in (item.url, *item.alternates):
                try:
                    with open_url(url, cfg, 0) as response:
                        head = response.read(2048)
                    status = "html page" if looks_like_html(head) else "ok"
                except urllib.error.HTTPError as err:
                    status = f"HTTP {err.code}"
                except (urllib.error.URLError, TimeoutError, OSError) as err:
                    status = f"error {err}"
                if status == "ok":
                    break
            rows.append({"source": source, "file": item.dest.name, "status": status})
            log(f"{source:13s} {item.dest.name:45s} {status}")
    return rows
