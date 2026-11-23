from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

from storm_outage_forecast import survival


def simulate(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 2, n).astype(float)
    t = np.exp(2.0 + 0.5 * x1 - 0.3 * x2 + 0.7 * np.log(rng.exponential(size=n)))
    c = rng.exponential(scale=15, size=n)
    observed = t <= c
    durations = np.round(np.minimum(t, c), 1) + 0.1
    return pd.DataFrame({"x1": x1, "x2": x2}), durations, observed


def test_kaplan_meier_hand_example():
    table = survival.kaplan_meier(np.array([1, 2, 2, 3, 4]), np.array([1, 1, 0, 1, 0], bool))
    assert table["time"].tolist() == [1, 2, 3]
    assert table["at_risk"].tolist() == [5, 4, 2]
    assert table["survival"].tolist() == pytest.approx([0.8, 0.6, 0.3])
    assert survival.km_median(table) == 3
    assert survival.km_at(table, np.array([0.5, 2.5, 10])).tolist() == pytest.approx(
        [1.0, 0.6, 0.3]
    )


def test_kaplan_meier_matches_lifelines():
    _, durations, observed = simulate(2000)
    table = survival.kaplan_meier(durations, observed)
    reference = KaplanMeierFitter().fit(durations, observed)
    expected = reference.survival_function_at_times(table["time"]).to_numpy()
    assert table["survival"].to_numpy() == pytest.approx(expected, abs=1e-12)
    assert survival.km_median(table) == reference.median_survival_time_


def test_weibull_matches_lifelines():
    x, durations, observed = simulate(3000)
    model = survival.WeibullAFT(["x1", "x2"]).fit(x, durations, observed)
    scaled = (x - x.mean()) / x.std(ddof=0)
    reference = WeibullAFTFitter().fit(scaled.assign(T=durations, E=observed), "T", "E")
    lam = reference.params_["lambda_"]
    assert model.params[1] == pytest.approx(lam["x1"], abs=1e-4)
    assert model.params[2] == pytest.approx(lam["x2"], abs=1e-4)
    assert model.params[0] == pytest.approx(lam["Intercept"], abs=1e-4)
    assert -model.params[-1] == pytest.approx(reference.params_["rho_"]["Intercept"], abs=1e-4)
    assert model.log_likelihood == pytest.approx(reference.log_likelihood_, abs=1e-4)
    se = reference.standard_errors_["lambda_"]
    assert model.se[1] == pytest.approx(se["x1"], rel=1e-3)


def test_weibull_recovers_true_parameters():
    x, durations, observed = simulate(40000, seed=3)
    model = survival.WeibullAFT(["x1", "x2"]).fit(x, durations, observed)
    raw = model.params[1:3] / model.scale
    assert raw == pytest.approx([0.5, -0.3], abs=0.03)
    assert model.sigma == pytest.approx(0.7, abs=0.03)


def test_hessian_matches_finite_differences():
    x, durations, observed = simulate(300, seed=5)
    model = survival.WeibullAFT(["x1", "x2"]).fit(x, durations, observed)
    design = model.design(x)
    log_t = np.log(durations)
    obs = observed.astype(float)
    exact = survival.weibull_hessian(model.params, design, log_t, obs)
    step = 1e-5
    numeric = np.zeros_like(exact)
    for i in range(len(model.params)):
        up = model.params.copy()
        down = model.params.copy()
        up[i] += step
        down[i] -= step
        g_up = -survival.weibull_nll(up, design, log_t, obs)[1]
        g_down = -survival.weibull_nll(down, design, log_t, obs)[1]
        numeric[:, i] = (g_up - g_down) / (2 * step)
    assert exact == pytest.approx(numeric, rel=1e-4, abs=1e-4)


def test_predicted_median_has_half_survival():
    x, durations, observed = simulate(500)
    model = survival.WeibullAFT(["x1", "x2"]).fit(x, durations, observed)
    medians = model.predict_median(x.head(5))
    values = [model.survival(x.iloc[[i]], np.array([m]))[0, 0] for i, m in enumerate(medians)]
    assert values == pytest.approx([0.5] * 5)


@pytest.mark.parametrize("rounding", [None, 0])
def test_concordance_matches_lifelines(rounding):
    x, durations, observed = simulate(2500, seed=9)
    predicted = survival.WeibullAFT(["x1", "x2"]).fit(x, durations, observed).predict_median(x)
    if rounding is not None:
        predicted = np.round(predicted, rounding)
    ours = survival.concordance(durations, predicted, observed)
    assert ours == pytest.approx(concordance_index(durations, predicted, observed), abs=1e-12)


def test_concordance_small_cases():
    assert survival.concordance(np.array([1, 2, 3]), np.array([1, 2, 3]), np.ones(3, bool)) == 1.0
    assert survival.concordance(np.array([1, 2, 3]), np.array([3, 2, 1]), np.ones(3, bool)) == 0.0
    assert survival.concordance(np.array([1, 2]), np.array([5, 5]), np.array([True, True])) == 0.5
    assert np.isnan(survival.concordance(np.array([1, 2]), np.array([1, 2]), np.zeros(2, bool)))
