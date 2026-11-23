from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from storm_outage_forecast import reliability, synthetic


def utility_rows(year: int) -> list[list]:
    rows = []
    for number, state, saidi, customers in (
        (1, "OH", 100.0, 1000),
        (2, "OH", 200.0, 3000),
        (3, "AK", 500.0, 900),
        (4, "TX", ".", 5000),
        (5, "TX", 80.0, 0),
    ):
        lead = [year, number, f"Utility {number}", state, "Cooperative"]
        if year == 2019:
            lead.append("N")
        ieee = [saidi, 1.0, 100.0, 50.0, 0.8, 62.5, 90.0, 0.9, 100.0, customers, 12.5, "Y"]
        rows.append(lead + ieee + [None] * 11)
    return rows


@pytest.mark.parametrize("year", [2019, 2021, 2024, 2025])
def test_parse_handles_each_layout(tmp_path, year):
    path = tmp_path / f"f861{year}.zip"
    synthetic.write_eia861(path, utility_rows(year), year)
    parsed = reliability.parse_reliability(path, year)
    assert parsed["utility"].tolist() == [1, 2, 3, 4, 5]
    assert parsed["saidi_med"].tolist()[:3] == [100.0, 200.0, 500.0]
    assert pd.isna(parsed["saidi_med"].iloc[3])
    assert parsed["saidi_no_med"].iloc[0] == 50.0
    assert parsed["customers"].tolist()[:2] == [1000, 3000]


def test_state_year_is_customer_weighted(tmp_path):
    path = tmp_path / "f8612021.zip"
    synthetic.write_eia861(path, utility_rows(2021), 2021)
    table = reliability.state_year(reliability.parse_reliability(path, 2021))
    assert table["abbr"].tolist() == ["OH"]
    row = table.iloc[0]
    assert row["saidi_eia"] == pytest.approx((100 * 1000 + 200 * 3000) / 4000)
    assert row["customers_eia"] == 4000
    assert row["utilities"] == 2


def test_missing_workbook_is_an_error(tmp_path):
    path = tmp_path / "f8612021.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Sales_Ult_Cust_2021.xlsx", b"x")
    with pytest.raises(ValueError, match="no Reliability"):
        reliability.parse_reliability(path, 2021)
