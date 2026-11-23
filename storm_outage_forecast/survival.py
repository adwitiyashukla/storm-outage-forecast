from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import Config
from .db import record_check

CAUSES = ["tropical", "ice", "winter", "convective", "wind", "flood"]
NUMERIC = ["log_peak_out", "log_peak_share", "log_density", "state_load", "summer", "night"]
LOG_LOG_2 = math.log(math.log(2))


def kaplan_meier(durations: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
    durations = np.asarray(durations, float)
    observed = np.asarray(observed, bool)
    times = np.unique(durations[observed])
    ordered = np.sort(durations)
    at_risk = len(durations) - np.searchsorted(ordered, times, side="left")
    event_times = np.sort(durations[observed])
    deaths = np.searchsorted(event_times, times, side="right") - np.searchsorted(
        event_times, times, side="left"
    )
    survival = np.cumprod(1 - deaths / at_risk)
    return pd.DataFrame({"time": times, "at_risk": at_risk, "events": deaths, "survival": survival})


def km_median(table: pd.DataFrame) -> float:
    below = table.loc[table["survival"] <= 0.5, "time"]
    return float(below.iloc[0]) if len(below) else float("nan")


def km_at(table: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(table["time"].to_numpy(), grid, side="right") - 1
    values = np.ones(len(grid))
    ok = idx >= 0
    values[ok] = table["survival"].to_numpy()[idx[ok]]
    return values


def weibull_terms(params: np.ndarray, x: np.ndarray, log_t: np.ndarray):
    beta, log_sigma = params[:-1], params[-1]
    sigma = math.exp(log_sigma)
    z = (log_t - x @ beta) / sigma
    return beta, log_sigma, sigma, z, np.exp(np.clip(z, -700, 700))


def weibull_nll(params: np.ndarray, x: np.ndarray, log_t: np.ndarray, observed: np.ndarray):
    beta, log_sigma, sigma, z, a = weibull_terms(params, x, log_t)
    ll = observed * (z - log_sigma - log_t) - a
    grad_beta = x.T @ ((a - observed) / sigma)
    grad_sigma = np.sum(z * (a - observed) - observed)
    return -float(ll.sum()), -np.append(grad_beta, grad_sigma)


def weibull_hessian(
    params: np.ndarray, x: np.ndarray, log_t: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    beta, log_sigma, sigma, z, a = weibull_terms(params, x, log_t)
    k = x.shape[1]
    hess = np.zeros((k + 1, k + 1))
    hess[:k, :k] = -(x * (a / sigma**2)[:, None]).T @ x
    cross = -(x.T @ ((a * z + a - observed) / sigma))
    hess[:k, k] = cross
    hess[k, :k] = cross
    hess[k, k] = np.sum(-z * (a - observed) - z**2 * a)
    return hess


class WeibullAFT:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.mean = None
        self.scale = None
        self.params = None
        self.se = None
        self.log_likelihood = None

    def design(self, frame: pd.DataFrame) -> np.ndarray:
        values = (frame[self.columns].to_numpy(float) - self.mean) / self.scale
        return np.column_stack([np.ones(len(frame)), values])

    def fit(self, frame: pd.DataFrame, durations: np.ndarray, observed: np.ndarray) -> WeibullAFT:
        raw = frame[self.columns].to_numpy(float)
        self.mean = raw.mean(axis=0)
        self.scale = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)
        x = self.design(frame)
        log_t = np.log(durations)
        obs = observed.astype(float)
        start = np.zeros(x.shape[1] + 1)
        start[0] = log_t.mean()
        start[-1] = math.log(max(log_t.std(), 0.1))
        result = minimize(
            weibull_nll,
            start,
            args=(x, log_t, obs),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 2000, "gtol": 1e-8},
        )
        if not result.success and result.nit < 5:
            raise RuntimeError(f"Weibull fit failed: {result.message}")
        self.params = result.x
        self.log_likelihood = -float(result.fun)
        info = -weibull_hessian(self.params, x, log_t, obs)
        try:
            self.se = np.sqrt(np.clip(np.diag(np.linalg.inv(info)), 0, None))
        except np.linalg.LinAlgError:
            self.se = np.full(len(self.params), np.nan)
        return self

    @property
    def sigma(self) -> float:
        return math.exp(self.params[-1])

    def predict_median(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.design(frame)
        return np.exp(x @ self.params[:-1] + self.sigma * LOG_LOG_2)

    def survival(self, frame: pd.DataFrame, hours: np.ndarray) -> np.ndarray:
        x = self.design(frame)
        loc = x @ self.params[:-1]
        z = (np.log(np.maximum(hours, 1e-9))[None, :] - loc[:, None]) / self.sigma
        return np.exp(-np.exp(z))

    def coefficients(self) -> pd.DataFrame:
        names = ["intercept"] + self.columns + ["log_sigma"]
        return pd.DataFrame(
            {
                "term": names,
                "coef": self.params,
                "se": self.se,
                "time_ratio_per_sd": np.exp(self.params),
            }
        )


def concordance(durations: np.ndarray, predicted: np.ndarray, observed: np.ndarray) -> float:
    durations = np.asarray(durations, float)
    observed = np.asarray(observed, bool)
    ranks = np.unique(predicted, return_inverse=True)[1] + 1
    size = int(ranks.max()) + 1
    tree = np.zeros(size + 1, dtype=np.int64)

    def add(i: int) -> None:
        while i <= size:
            tree[i] += 1
            i += i & (-i)

    def total(i: int) -> int:
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return int(s)

    order = np.argsort(-durations, kind="stable")
    inserted = 0
    concordant = discordant = tied = 0.0
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j < n and durations[order[j]] == durations[order[i]]:
            j += 1
        group = order[i:j]
        for k in group[~observed[group]]:
            add(int(ranks[k]))
            inserted += 1
        for k in group[observed[group]]:
            r = int(ranks[k])
            below = total(r - 1)
            same = total(r) - below
            concordant += inserted - below - same
            discordant += below
            tied += same
        for k in group[observed[group]]:
            add(int(ranks[k]))
            inserted += 1
        i = j
    pairs = concordant + discordant + tied
    return float((concordant + 0.5 * tied) / pairs) if pairs else float("nan")


def event_frame(cfg: Config, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frame = con.execute(f"""
        SELECT e.event_id, e.unit_id, e.abbr, e.start_ts, e.peak_ts, e.peak_out, e.peak_share,
               e.state_load, e.density, e.cause, e.duration_h, e.censored,
               year(e.start_ts) AS year,
               hour(e.peak_ts + to_hours(u.offset_hours)) AS local_hour,
               month(e.peak_ts + to_hours(u.offset_hours)) AS month
        FROM events e JOIN units u USING (unit_id)
        WHERE e.peak_out >= {cfg.outages.event_min_peak} AND e.duration_h > 0
    """).df()
    cap = cfg.survival.max_hours
    frame["observed"] = ~frame["censored"].astype(bool) & (frame["duration_h"] <= cap)
    frame["duration_h"] = frame["duration_h"].clip(upper=cap)
    frame["log_peak_out"] = np.log(frame["peak_out"])
    frame["log_peak_share"] = np.log(frame["peak_share"].clip(lower=1e-6))
    frame["log_density"] = np.log(frame["density"].clip(lower=0.01))
    frame["state_load"] = frame["state_load"].clip(0, 1)
    frame["summer"] = frame["month"].between(6, 9).astype(float)
    frame["night"] = ((frame["local_hour"] >= 20) | (frame["local_hour"] < 6)).astype(float)
    for cause in CAUSES:
        frame[f"cause_{cause}"] = (frame["cause"] == cause).astype(float)
    return frame


def run(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> dict:
    events = event_frame(cfg, con)
    fit_part = events[events["year"] <= cfg.years.valid]
    test = events[events["year"] >= cfg.years.test_start].copy()
    if len(fit_part) < cfg.survival.min_events or len(test) < cfg.survival.min_events:
        raise RuntimeError(f"not enough outage events: {len(fit_part)} fit, {len(test)} test")
    log(f"restoration: fitting on {len(fit_part):,} events, testing on {len(test):,}")
    columns = NUMERIC + [f"cause_{c}" for c in CAUSES if fit_part[f"cause_{c}"].sum() >= 5]
    model = WeibullAFT(columns).fit(
        fit_part, fit_part["duration_h"].to_numpy(), fit_part["observed"].to_numpy()
    )
    test["pred_median"] = model.predict_median(test)
    base = {}
    overall = km_median(kaplan_meier(fit_part["duration_h"], fit_part["observed"]))
    for cause, part in fit_part.groupby("cause"):
        value = km_median(kaplan_meier(part["duration_h"], part["observed"]))
        base[cause] = value if np.isfinite(value) else overall
    test["base_median"] = test["cause"].map(base).fillna(overall)
    durations = test["duration_h"].to_numpy()
    observed = test["observed"].to_numpy()
    obs = test[test["observed"]]
    log_err = np.abs(np.log(obs["pred_median"]) - np.log(obs["duration_h"]))
    base_err = np.abs(np.log(obs["base_median"]) - np.log(obs["duration_h"]))
    test["decile"] = pd.qcut(test["pred_median"].rank(method="first"), 10, labels=False)
    calibration = []
    for decile, part in test.groupby("decile"):
        km = kaplan_meier(part["duration_h"], part["observed"])
        calibration.append(
            {
                "decile": int(decile),
                "events": len(part),
                "predicted_median": float(part["pred_median"].median()),
                "observed_median": km_median(km),
            }
        )
    grid = np.concatenate([np.arange(0, 48, 1.0), np.arange(48, 169, 6.0)])
    curves = []
    for label, part in (("fit", fit_part), ("test", test)):
        for cause, group in part.groupby("cause"):
            if len(group) < 10:
                continue
            km = kaplan_meier(group["duration_h"], group["observed"])
            for hour, value in zip(grid, km_at(km, grid), strict=True):
                curves.append(
                    {
                        "sample": label,
                        "cause": cause,
                        "hours": hour,
                        "survival": value,
                        "events": len(group),
                    }
                )
    metrics = {
        "fit_events": int(len(fit_part)),
        "test_events": int(len(test)),
        "test_censored_share": float(1 - observed.mean()),
        "c_index_model": concordance(durations, test["pred_median"].to_numpy(), observed),
        "c_index_cause_median": concordance(durations, test["base_median"].to_numpy(), observed),
        "median_abs_log_error_model": float(np.median(log_err)),
        "median_abs_log_error_cause_median": float(np.median(base_err)),
        "within_2x_model": float(np.mean(log_err <= math.log(2))),
        "within_2x_cause_median": float(np.mean(base_err <= math.log(2))),
        "sigma": model.sigma,
        "log_likelihood": model.log_likelihood,
        "cause_medians_fit": {k: float(v) for k, v in base.items()},
    }
    tables = {
        "survival_coefficients": model.coefficients(),
        "survival_calibration": pd.DataFrame(calibration),
        "survival_curves": pd.DataFrame(curves),
        "event_predictions": test[
            [
                "event_id",
                "unit_id",
                "abbr",
                "start_ts",
                "peak_ts",
                "peak_out",
                "cause",
                "duration_h",
                "observed",
                "pred_median",
                "base_median",
            ]
        ],
    }
    for name, table in tables.items():
        con.register("t_df", table)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM t_df")
        con.unregister("t_df")
    gain = metrics["c_index_model"] - metrics["c_index_cause_median"]
    record_check(
        con, "restoration", "c_index_gain_over_cause_median", "pass" if gain > 0 else "warn", gain
    )
    return metrics
