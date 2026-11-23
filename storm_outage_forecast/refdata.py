from __future__ import annotations

from functools import cache

import pandas as pd

from .config import REFDATA

TZ_OFFSETS = {"V": -4, "E": -5, "C": -6, "M": -7, "P": -8, "A": -9, "H": -10, "S": -11, "G": 10}


@cache
def states() -> pd.DataFrame:
    frame = pd.read_csv(REFDATA / "states.csv", dtype={"state_fips": str, "tz": str})
    frame["offset"] = frame["tz"].map(TZ_OFFSETS).astype(int)
    return frame


@cache
def unit_remap() -> pd.DataFrame:
    return pd.read_csv(REFDATA / "unit_remap.csv", dtype={"fips": str, "unit_id": str})


def state_lookup(key: str, value: str) -> dict[str, str]:
    frame = states()
    return dict(zip(frame[key], frame[value], strict=True))
