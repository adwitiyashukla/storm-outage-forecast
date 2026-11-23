from __future__ import annotations

from streamlit.testing.v1 import AppTest

from storm_outage_forecast.config import ROOT


def test_dashboard_renders_and_reacts(built, monkeypatch):
    cfg, _ = built
    monkeypatch.setenv("BUNDLE_DIR", str(cfg.paths.bundle))
    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert len(app.tabs) == 5
    assert app.warning[0].value.startswith("These numbers come from the synthetic")
    app.radio[0].set_value("Actual share of customers out").run()
    app.selectbox[0].select_index(1).run()
    app.selectbox[1].select_index(0).run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception


def test_dashboard_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("BUNDLE_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=60).run()
    assert app.error and "No dashboard data" in app.error[0].value
