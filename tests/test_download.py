from __future__ import annotations

import gzip
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from storm_outage_forecast import download
from storm_outage_forecast.download import DownloadError, Item, fetch

from .conftest import make_cfg

PAYLOAD = bytes(range(256)) * 4000


class Handler(BaseHTTPRequestHandler):
    files: dict = {}
    calls: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.calls.append((self.path, self.headers.get("Range")))
        spec = self.files.get(self.path)
        if spec is None:
            self.send_response(404)
            self.end_headers()
            return
        body = spec["body"]
        mode = spec.get("mode", "ok")
        start = 0
        rng = self.headers.get("Range")
        if rng and mode != "no-range":
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Type", spec.get("type", "application/octet-stream"))
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        if mode == "truncate-once" and not spec.get("done"):
            spec["done"] = True
            self.wfile.write(body[start : start + len(body) // 3])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body[start:])


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    Handler.files = {}
    Handler.calls = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path, download={"retries": 3})


def test_complete_download_with_md5(server, cfg):
    Handler.files["/a.bin"] = {"body": PAYLOAD}
    item = Item(
        "x",
        f"{server}/a.bin",
        cfg.paths.raw / "a.bin",
        size=len(PAYLOAD),
        md5=hashlib.md5(PAYLOAD).hexdigest(),
    )
    record = fetch(item, cfg, log=lambda m: None)
    assert (cfg.paths.raw / "a.bin").read_bytes() == PAYLOAD
    assert record["status"] == "downloaded"
    again = fetch(item, cfg, log=lambda m: None)
    assert again["status"] == "kept"


def test_truncated_transfer_is_resumed(server, cfg):
    Handler.files["/b.bin"] = {"body": PAYLOAD, "mode": "truncate-once"}
    item = Item("x", f"{server}/b.bin", cfg.paths.raw / "b.bin", size=len(PAYLOAD))
    fetch(item, cfg, log=lambda m: None)
    assert (cfg.paths.raw / "b.bin").read_bytes() == PAYLOAD
    ranges = [r for p, r in Handler.calls if p == "/b.bin"]
    assert ranges[0] is None and ranges[1].startswith("bytes=")


def test_server_ignoring_range_restarts_cleanly(server, cfg):
    Handler.files["/c.bin"] = {"body": PAYLOAD, "mode": "no-range"}
    dest = cfg.paths.raw / "c.bin"
    dest.parent.mkdir(parents=True)
    dest.with_name("c.bin.part").write_bytes(b"stale-bytes")
    fetch(Item("x", f"{server}/c.bin", dest, size=len(PAYLOAD)), cfg, log=lambda m: None)
    assert dest.read_bytes() == PAYLOAD


def test_html_error_page_is_rejected(server, cfg):
    Handler.files["/d.csv"] = {
        "body": b"<!DOCTYPE html><html><body>Maintenance</body></html>",
        "type": "text/html",
    }
    with pytest.raises(DownloadError, match="html page"):
        fetch(Item("x", f"{server}/d.csv", cfg.paths.raw / "d.csv"), cfg, log=lambda m: None)
    assert not (cfg.paths.raw / "d.csv").exists()


def test_md5_mismatch_is_an_error(server, cfg):
    Handler.files["/e.bin"] = {"body": PAYLOAD}
    item = Item("x", f"{server}/e.bin", cfg.paths.raw / "e.bin", md5="0" * 32)
    with pytest.raises(DownloadError, match="md5"):
        fetch(item, cfg, log=lambda m: None)
    assert not (cfg.paths.raw / "e.bin").exists()


def test_alternate_url_after_404(server, cfg):
    Handler.files["/archive/f.zip.bin"] = {"body": PAYLOAD}
    item = Item(
        "x",
        f"{server}/f.zip.bin",
        cfg.paths.raw / "f.bin",
        alternates=(f"{server}/archive/f.zip.bin",),
    )
    record = fetch(item, cfg, log=lambda m: None)
    assert record["url"].endswith("/archive/f.zip.bin")


def test_html_primary_falls_back_to_the_alternate(server, cfg):
    Handler.files["/zip/h.zip"] = {"body": b"<html>not here</html>", "type": "text/html"}
    Handler.files["/archive/zip/h.zip"] = {"body": PAYLOAD}
    item = Item(
        "eia861",
        f"{server}/zip/h.zip",
        cfg.paths.raw / "h.bin",
        alternates=(f"{server}/archive/zip/h.zip",),
    )
    record = fetch(item, cfg, log=lambda m: None)
    assert record["url"].endswith("/archive/zip/h.zip")
    assert (cfg.paths.raw / "h.bin").read_bytes() == PAYLOAD
    assert not (cfg.paths.raw / "h.bin.part").exists()


def test_probe_tries_the_alternate_when_the_first_url_is_html(server, monkeypatch, tmp_path):
    Handler.files["/zip/p.zip"] = {"body": b"<html>not here</html>", "type": "text/html"}
    Handler.files["/archive/zip/p.zip"] = {"body": PAYLOAD}
    cfg = make_cfg(tmp_path)

    def plan(c):
        return [
            Item(
                "eia861",
                f"{server}/zip/p.zip",
                c.paths.raw / "p.zip",
                alternates=(f"{server}/archive/zip/p.zip",),
            )
        ]

    for source in download.SOURCES:
        monkeypatch.setitem(download.PLANNERS, source, plan)
    rows = download.probe(cfg, log=lambda m: None)
    assert {r["status"] for r in rows} == {"ok"}


def test_corrupt_gzip_is_rejected(server, cfg):
    good = gzip.compress(b"USW00010000,20200101,PRCP,5,,,W,\n" * 100)
    Handler.files["/g.csv.gz"] = {"body": good[:-12] + b"\x00" * 12}
    with pytest.raises(DownloadError):
        fetch(Item("x", f"{server}/g.csv.gz", cfg.paths.raw / "g.csv.gz"), cfg, log=lambda m: None)


def test_eaglei_header_is_checked(server, cfg):
    Handler.files["/eaglei_outages_2020.csv"] = {"body": b"a,b,c\n1,2,3\n"}
    item = Item("x", f"{server}/eaglei_outages_2020.csv", cfg.paths.raw / "eaglei_outages_2020.csv")
    with pytest.raises(DownloadError, match="EAGLE-I header"):
        fetch(item, cfg, log=lambda m: None)


def test_storm_event_listing_keeps_latest_stamp():
    html = """
    <a href="StormEvents_details-ftp_v1.0_d2020_c20240101.csv.gz">x</a>
    <a href="StormEvents_details-ftp_v1.0_d2020_c20260323.csv.gz">x</a>
    <a href="StormEvents_details-ftp_v1.0_d2021_c20250101.csv.gz">x</a>
    <a href="StormEvents_fatalities-ftp_v1.0_d2021_c20260323.csv.gz">x</a>
    """
    files = download.storm_event_files(html, [2020, 2021, 2022])
    assert files == {
        2020: "StormEvents_details-ftp_v1.0_d2020_c20260323.csv.gz",
        2021: "StormEvents_details-ftp_v1.0_d2021_c20250101.csv.gz",
    }


def test_nws_file_names():
    assert str(download.nws_date("bp16ap26.dbx")) == "2026-04-16"
    assert download.nws_date("bp31fe26.dbx") is None
    assert download.nws_date("readme.txt") is None
    html = 'href="bp05mr24.dbx" href="bp16ap26.dbx" href="bp18mr25.dbx"'
    assert download.latest_nws(html) == "bp16ap26.dbx"


def test_eaglei_plan_uses_figshare_sizes_and_md5(server, monkeypatch, tmp_path):
    files = [
        {"name": n, "size": 10, "download_url": f"{server}/files/{i}", "computed_md5": "ab" * 16}
        for i, n in enumerate(
            [
                "eaglei_outages_2019.csv",
                "eaglei_outages_2020.csv",
                "eaglei_outages_2021.csv",
                "eaglei_outages_2022.csv",
                "eaglei_outages_2023.csv",
                "MCC.csv",
                "coverage_history.csv",
                "DQI.csv",
                "Uri_Map.R",
            ]
        )
    ]
    Handler.files["/article"] = {
        "body": json.dumps({"files": files}).encode(),
        "type": "application/json",
    }
    cfg = make_cfg(tmp_path, download={"eaglei_article": "https://placeholder.test/article"})
    monkeypatch.setattr(download, "get_bytes", lambda url, c: json.dumps({"files": files}).encode())
    items = download.plan_eaglei(cfg)
    assert [i.dest.name for i in items][:2] == [
        "eaglei_outages_2019.csv",
        "eaglei_outages_2020.csv",
    ]
    assert all(i.size == 10 and i.md5 == "ab" * 16 for i in items)
    assert len(items) == 7
    monkeypatch.setattr(
        download, "get_bytes", lambda url, c: json.dumps({"files": files[:2]}).encode()
    )
    with pytest.raises(DownloadError, match="missing"):
        download.plan_eaglei(cfg)


def test_run_keeps_going_after_a_failed_source(server, cfg, monkeypatch):
    Handler.files["/good.bin"] = {"body": PAYLOAD}

    def broken(c):
        raise DownloadError("index page is down")

    def good(c):
        return [Item("geo", f"{server}/good.bin", c.paths.raw / "geo" / "good.bin")]

    def missing(c):
        return [Item("nws", f"{server}/nope.dbx", c.paths.raw / "nws" / "nope.dbx")]

    monkeypatch.setitem(download.PLANNERS, "eia861", broken)
    monkeypatch.setitem(download.PLANNERS, "nws", missing)
    monkeypatch.setitem(download.PLANNERS, "geo", good)
    with pytest.raises(DownloadError) as info:
        download.run(cfg, ["eia861", "nws", "geo"], log=lambda m: None)
    assert "index page is down" in str(info.value)
    assert "nope.dbx" in str(info.value)
    assert (cfg.paths.raw / "geo" / "good.bin").read_bytes() == PAYLOAD
    manifest = json.loads((cfg.paths.raw / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest["files"]) == ["geo/good.bin"]
    assert len(manifest["failures"]) == 2
    monkeypatch.setitem(download.PLANNERS, "nws", good)
    monkeypatch.setitem(download.PLANNERS, "eia861", good)
    lines = []
    download.run(cfg, ["nws", "eia861"], log=lines.append)
    assert lines[-1].startswith("download finished, 1 files")


def test_unknown_source_is_rejected_before_any_download(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(DownloadError, match="unknown source weather"):
        download.run(cfg, ["geo", "weather"], log=lambda m: None)
    assert not cfg.paths.raw.exists()


def test_probe_reports_planning_failures(monkeypatch, tmp_path):
    cfg = make_cfg(tmp_path)

    def broken(c):
        raise DownloadError("index page is down")

    for source in download.SOURCES:
        monkeypatch.setitem(download.PLANNERS, source, broken)
    lines = []
    rows = download.probe(cfg, log=lines.append)
    assert len(rows) == len(lines) == len(download.SOURCES)
    assert all("planning failed: index page is down" in line for line in lines)


def test_figshare_html_answer_is_a_download_error(monkeypatch, tmp_path):
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(download, "get_bytes", lambda url, c: b"<html>busy</html>")
    with pytest.raises(DownloadError, match="json"):
        download.plan_eaglei(cfg)


def test_download_refuses_a_synthetic_folder(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.paths.raw.mkdir(parents=True)
    (cfg.paths.raw / "manifest.json").write_text('{"kind": "synthetic"}', encoding="utf-8")
    with pytest.raises(DownloadError, match="synthetic"):
        download.run(cfg, ["geo"], log=lambda m: None)
