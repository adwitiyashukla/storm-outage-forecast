from __future__ import annotations

import itertools

import numpy as np
import pytest

from storm_outage_forecast import staging


def brute_force(weights, base, crews):
    best = None
    for plan in itertools.product(range(crews + 1), repeat=len(weights)):
        if sum(plan) == crews:
            cost = staging.customer_hours(weights, base, np.array(plan), 1.0)
            best = cost if best is None else min(best, cost)
    return best


def test_greedy_is_optimal_on_random_cases():
    rng = np.random.default_rng(11)
    for _ in range(150):
        k = int(rng.integers(2, 5))
        crews = int(rng.integers(0, 8))
        weights = rng.uniform(0, 1000, k) ** 2
        base = rng.integers(1, 6, k).astype(float)
        plan = staging.allocate(weights, base, crews)
        assert plan.sum() == crews
        assert staging.customer_hours(weights, base, plan, 1.0) == pytest.approx(
            brute_force(weights, base, crews)
        )


def test_customer_hours_formula():
    assert staging.customer_hours(
        np.array([400.0**2]), np.array([2.0]), np.array([2]), 50.0
    ) == pytest.approx(400.0**2 / (2 * 50.0 * 4))


def test_proportional_split():
    plan = staging.proportional(np.array([10, 20, 30]), 7)
    assert plan.sum() == 7
    assert plan.tolist() == [1, 2, 4]


def test_quantile_draw_passes_through_the_quantiles():
    q10, q50, q90 = np.array([1.0]), np.array([2.0]), np.array([4.0])
    got = [staging.quantile_draw(np.array([u]), q10, q50, q90)[0] for u in (0.1, 0.5, 0.9)]
    assert got == pytest.approx([1.0, 2.0, 4.0])
    grid = staging.quantile_draw(np.linspace(0.01, 0.99, 50), q10, q50, q90)
    assert np.all(np.diff(grid) > 0)
