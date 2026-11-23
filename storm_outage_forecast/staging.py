from __future__ import annotations

import heapq

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import Config
from .db import record_check

POLICIES = ["none", "proportional", "climatology", "forecast", "perfect"]


def allocate(weights: np.ndarray, base: np.ndarray, crews: int) -> np.ndarray:
    weights = np.asarray(weights, float)
    base = np.asarray(base, float)
    extra = np.zeros(len(weights), dtype=int)
    heap = [(-(w / b - w / (b + 1)), s) for s, (w, b) in enumerate(zip(weights, base, strict=True))]
    heapq.heapify(heap)
    for _ in range(crews):
        _, s = heapq.heappop(heap)
        extra[s] += 1
        b = base[s] + extra[s]
        heapq.heappush(heap, (-(weights[s] / b - weights[s] / (b + 1)), s))
    return extra


def proportional(customers: np.ndarray, crews: int) -> np.ndarray:
    share = crews * np.asarray(customers, float) / np.sum(customers)
    extra = np.floor(share).astype(int)
    left = crews - extra.sum()
    order = np.argsort(-(share - extra), kind="stable")
    extra[order[:left]] += 1
    return extra


def customer_hours(
    n_squared: np.ndarray, base: np.ndarray, extra: np.ndarray, rate: float
) -> float:
    return float(np.sum(np.asarray(n_squared, float) / (2 * rate * (base + extra))))


def quantile_draw(u: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> np.ndarray:
    lo = (q50 - q10) / 0.4
    hi = (q90 - q50) / 0.4
    below = q50 - lo * (0.5 - u)
    above = q50 + hi * (u - 0.5)
    return np.where(u < 0.5, below, above)


def state_samples(
    day: pd.DataFrame, states: list[str], samples: int, rng: np.random.Generator, mode: str
) -> np.ndarray:
    n = len(day)
    index = pd.Categorical(day["abbr"], categories=states).codes
    onehot = np.zeros((len(states), n))
    onehot[index, np.arange(n)] = 1.0
    u_sig = rng.random((samples, n))
    u_size = rng.random((samples, n))
    if mode == "forecast":
        sig = u_sig < day["p"].to_numpy()
        log_size = quantile_draw(
            u_size,
            np.log1p(day["q10"].to_numpy()),
            np.log1p(day["q50"].to_numpy()),
            np.log1p(day["q90"].to_numpy()),
        )
    else:
        sig = u_sig < day["p_clim"].to_numpy()
        log_size = day["mu"].to_numpy() + day["sd"].to_numpy() * norm.ppf(
            np.clip(u_size, 1e-6, 1 - 1e-6)
        )
    size = np.expm1(np.clip(log_size, 0, 16))
    size = np.minimum(size, day["customers"].to_numpy())
    values = np.where(sig, size, day["base"].to_numpy())
    return values @ onehot.T


def run(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> dict:
    g = cfg.staging
    rng = np.random.default_rng(cfg.runtime.seed)
    states = con.execute("""
        SELECT abbr, sum(customers) AS customers FROM units GROUP BY abbr ORDER BY abbr
    """).df()
    names = states["abbr"].tolist()
    customers = states["customers"].to_numpy(float)
    base = np.maximum(1, np.round(customers / g.customers_per_base_crew))
    days = (
        con.execute(f"""
        SELECT day FROM test_days ORDER BY actual DESC LIMIT {g.top_days}
    """)
        .df()["day"]
        .tolist()
    )
    rows = []
    allocations = []
    for day in days:
        part = con.execute(
            """
            SELECT unit_id, abbr, customers, peak_out, p, q10, q50, q90, p_clim, mu, sd, base
            FROM predictions WHERE day = ?
        """,
            [day],
        ).df()
        actual = part.groupby("abbr")["peak_out"].sum().reindex(names, fill_value=0).to_numpy(float)
        model = state_samples(part, names, g.samples, rng, "forecast")
        clim = state_samples(part, names, g.samples, rng, "climatology")
        weights = {
            "climatology": np.mean(clim**2, axis=0),
            "forecast": np.mean(model**2, axis=0),
            "perfect": actual**2,
        }
        plans = {
            "none": np.zeros(len(names), dtype=int),
            "proportional": proportional(customers, g.mutual_aid_crews),
        }
        for policy, w in weights.items():
            plans[policy] = allocate(w, base, g.mutual_aid_crews)
        for policy in POLICIES:
            cost = customer_hours(actual**2, base, plans[policy], g.restore_rate)
            rows.append(
                {
                    "day": day,
                    "policy": policy,
                    "customer_hours": cost,
                    "actual_out": float(actual.sum()),
                }
            )
            for s, name in enumerate(names):
                if plans[policy][s] or actual[s]:
                    allocations.append(
                        {
                            "day": day,
                            "policy": policy,
                            "abbr": name,
                            "crews": int(plans[policy][s]),
                            "base_crews": int(base[s]),
                            "actual_out": float(actual[s]),
                            "expected_out": float(np.mean(model[:, s])),
                        }
                    )
    results = pd.DataFrame(rows)
    totals = results.groupby("policy")["customer_hours"].sum()
    gap = totals["proportional"] - totals["perfect"]
    captured = {
        p: float((totals["proportional"] - totals[p]) / gap) if gap > 0 else float("nan")
        for p in ("climatology", "forecast")
    }
    for name, table in (
        ("staging_results", results),
        ("staging_allocations", pd.DataFrame(allocations)),
    ):
        con.register("t_df", table)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM t_df")
        con.unregister("t_df")
    record_check(
        con,
        "staging",
        "forecast_beats_climatology",
        "pass" if totals["forecast"] <= totals["climatology"] else "warn",
        float(totals["climatology"] - totals["forecast"]),
    )
    log(
        f"staging: {len(days)} storm days, forecast captures {captured['forecast']:.2f} of the "
        f"perfect-foresight savings"
    )
    return {
        "days": len(days),
        "crews": g.mutual_aid_crews,
        "customer_hours": {p: float(totals[p]) for p in POLICIES},
        "share_of_perfect_savings": captured,
    }
