from __future__ import annotations

from pathlib import Path

import duckdb

from .config import Config


def connect(
    cfg: Config, read_only: bool = False, memory_gb: float | None = None
) -> duckdb.DuckDBPyConnection:
    cfg.paths.gold.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cfg.paths.gold), read_only=read_only)
    con.execute(f"SET memory_limit = '{memory_gb or cfg.runtime.duckdb_memory_gb}GB'")
    con.execute(f"SET threads = {cfg.runtime.threads}")
    con.execute(f"SET temp_directory = '{sql_path(cfg.paths.tmp)}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET TimeZone = 'UTC'")
    return con


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    query = "SELECT count(*) FROM information_schema.tables WHERE table_name = ?"
    return con.execute(query, [name]).fetchone()[0] > 0


def record_check(
    con: duckdb.DuckDBPyConnection,
    stage: str,
    check: str,
    status: str,
    value: float,
    detail: str = "",
) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS quality (
            stage VARCHAR, check_name VARCHAR, status VARCHAR, value DOUBLE, detail VARCHAR)
    """)
    con.execute("DELETE FROM quality WHERE stage = ? AND check_name = ?", [stage, check])
    con.execute(
        "INSERT INTO quality VALUES (?, ?, ?, ?, ?)", [stage, check, status, float(value), detail]
    )
