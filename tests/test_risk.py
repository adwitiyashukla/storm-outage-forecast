from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from storm_outage_forecast import risk
from storm_outage_forecast.db import connect
from storm_outage_forecast.features import FEATURES


def test_ece_and_reliability():
    p = np.array([0.1] * 10 + [0.9] * 10)
    y = np.array([1] + [0] * 9 + [1] * 9 + [0])
    assert risk.ece(y, p, 2) == pytest.approx(0.0)
    table = risk.reliability(y, p, 2)
    assert table["n"].sum() == 20
    assert risk.ece(np.zeros(20), p, 2) == pytest.approx(0.5)


def test_pinball_loss():
    y = np.array([1.0, 3.0])
    q = np.array([2.0, 2.0])
    assert risk.pinball(y, q, 0.9) == pytest.approx((0.1 * 1 + 0.9 * 1) / 2)


def test_classifier_metrics_reference():
    y = np.array([0, 0, 1, 1])
    out = risk.classifier_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 2, np.full(4, 0.5))
    assert out["roc_auc"] == 1.0
    assert out["brier_skill"] == pytest.approx(1 - 0.025 / 0.25)


class Stub:
    best_iteration = 0

    def __init__(self, value):
        self.value = value

    def predict(self, x, num_iteration=None):
        return np.full(len(x), self.value)


def test_conformal_adjustment_is_exact():
    size = {0.1: Stub(2.0), 0.5: Stub(3.0), 0.9: Stub(4.0)}
    model = risk.RiskModel(None, None, size, None, (0.1, 0.5, 0.9))
    y = np.array([1.0, 2.5, 3.0, 3.5, 4.5, 6.0, 3.1, 3.2, 3.3, 3.4])
    frame = pd.DataFrame({f: np.zeros(len(y)) for f in FEATURES}).assign(peak_out=np.expm1(y))
    adjust = model.conformalize(frame)
    scores = np.sort(np.maximum(2.0 - y, y - 4.0))
    assert adjust == pytest.approx(scores[int(np.ceil(11 * 0.8)) - 1])


def test_trained_model_outputs(built):
    cfg, metrics = built
    test = metrics["risk"]["test"]
    assert test["lightgbm"]["roc_auc"] > test["climatology"]["roc_auc"]
    assert test["lightgbm"]["brier_skill"] > 0
    con = connect(cfg, read_only=True)
    pred = con.execute("SELECT * FROM predictions").df()
    months = con.execute("SELECT count(*), min(p_clim), max(p_clim) FROM clim_rate").fetchone()
    units = con.execute("SELECT count(*) FROM units").fetchone()[0]
    con.close()
    assert set(pred["year"]) == set(cfg.years.test)
    assert pred["p"].between(0, 1).all()
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()
    assert months[0] == 12 * units and 0 < months[1] <= months[2] < 1
    saved = json.loads((cfg.paths.models / "calibration.json").read_text(encoding="utf-8"))
    assert "interval_adjust" in saved
    for name in ("classifier.txt", "size_q10.txt", "size_q50.txt", "size_q90.txt"):
        assert (cfg.paths.models / name).stat().st_size > 0
