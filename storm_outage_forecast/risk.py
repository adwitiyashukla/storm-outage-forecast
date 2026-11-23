from __future__ import annotations

import gc
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .db import record_check
from .features import CATEGORICAL, FEATURES, load, load_sampled

LOGIT_FEATURES = [
    "gust",
    "gust_rel",
    "gust2",
    "wind",
    "prcp",
    "prcp_3d",
    "snow",
    "ice",
    "thunder",
    "cold_prcp",
    "log_customers",
    "log_density",
    "start_share",
    "prev_share",
    "doy_sin",
    "doy_cos",
]
Z90 = 1.2815515655446004
EPS = 1e-6


def lgb_params(cfg: Config, **extra) -> dict:
    r = cfg.risk
    params = {
        "learning_rate": r.learning_rate,
        "num_leaves": r.num_leaves,
        "min_data_in_leaf": r.min_child_samples,
        "feature_fraction": r.feature_fraction,
        "bagging_fraction": r.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l2": r.lambda_l2,
        "seed": cfg.runtime.seed,
        "num_threads": cfg.runtime.threads,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
    }
    params.update(extra)
    return params


def build_climatology(cfg: Config, con: duckdb.DuckDBPyConnection) -> None:
    prior = cfg.risk.climatology_prior
    con.execute(f"""
        CREATE OR REPLACE TABLE clim_rate AS
        WITH tr AS (
            SELECT unit_id, abbr, month, CAST(sig AS INTEGER) AS y
            FROM model_rows WHERE usable AND split = 'train'
        ),
        g AS (SELECT avg(y) AS p FROM tr),
        sm AS (SELECT abbr, month, sum(y) AS k, count(*) AS n FROM tr GROUP BY ALL),
        um AS (SELECT unit_id, month, sum(y) AS k, count(*) AS n FROM tr GROUP BY ALL),
        grid AS (
            SELECT u.unit_id, u.abbr, CAST(m.range AS INTEGER) AS month
            FROM units u CROSS JOIN range(1, 13) m
        )
        SELECT grid.unit_id, grid.month,
               (coalesce(um.k, 0) + {prior} * ((coalesce(sm.k, 0) + {prior} * g.p)
                                               / (coalesce(sm.n, 0) + {prior})))
               / (coalesce(um.n, 0) + {prior}) AS p_clim
        FROM grid CROSS JOIN g
        LEFT JOIN sm ON sm.abbr = grid.abbr AND sm.month = grid.month
        LEFT JOIN um ON um.unit_id = grid.unit_id AND um.month = grid.month
    """)
    con.execute("""
        CREATE OR REPLACE TABLE clim_size AS
        WITH s AS (
            SELECT unit_id, ln(1 + peak_out) AS z FROM model_rows
            WHERE usable AND split = 'train' AND sig
        ),
        g AS (SELECT avg(z) AS mu, stddev_samp(z) AS sd FROM s),
        us AS (SELECT unit_id, avg(z) AS mu, count(*) AS n FROM s GROUP BY unit_id),
        b AS (
            SELECT unit_id, avg(peak_out) AS base FROM model_rows
            WHERE usable AND split = 'train' AND NOT sig GROUP BY unit_id
        )
        SELECT u.unit_id,
               (coalesce(us.n, 0) * coalesce(us.mu, 0) + 10 * g.mu)
                   / (coalesce(us.n, 0) + 10) AS mu,
               coalesce(g.sd, 1.0) AS sd,
               coalesce(b.base, 0) AS base
        FROM units u CROSS JOIN g
        LEFT JOIN us USING (unit_id)
        LEFT JOIN b USING (unit_id)
    """)


def ece(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def reliability(y: np.ndarray, p: np.ndarray, bins: int) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    frame = pd.DataFrame({"bin": idx, "p": p, "y": y})
    return (
        frame.groupby("bin")
        .agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size"))
        .reset_index()
    )


def classifier_metrics(
    y: np.ndarray, p: np.ndarray, bins: int, reference: np.ndarray | None
) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    out = {"rows": int(len(y)), "base_rate": float(y.mean())}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    out["ece"] = ece(y, p, bins)
    if reference is not None:
        ref = float(brier_score_loss(y, np.clip(reference, EPS, 1 - EPS)))
        out["brier_skill"] = float(1 - out["brier"] / ref) if ref > 0 else 0.0
    return out


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    diff = y - q
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def pr_points(y: np.ndarray, p: np.ndarray, limit: int = 200) -> pd.DataFrame:
    precision, recall, _ = precision_recall_curve(y, p)
    keep = np.unique(np.linspace(0, len(precision) - 1, min(limit, len(precision))).astype(int))
    return pd.DataFrame({"recall": recall[keep], "precision": precision[keep]})


class RiskModel:
    def __init__(
        self,
        classifier: lgb.Booster,
        iso: IsotonicRegression,
        size: dict[float, lgb.Booster],
        logit,
        quantiles: tuple[float, ...],
    ):
        self.classifier = classifier
        self.iso = iso
        self.size = size
        self.logit = logit
        self.quantiles = quantiles
        self.interval_adjust = 0.0

    def size_quantiles(self, x) -> np.ndarray:
        qs = np.column_stack(
            [
                self.size[q].predict(x, num_iteration=self.size[q].best_iteration)
                for q in self.quantiles
            ]
        )
        return np.sort(qs, axis=1)

    def conformalize(self, frame: pd.DataFrame) -> float:
        qs = self.size_quantiles(frame[FEATURES].to_numpy(dtype=np.float32))
        y = np.log1p(frame["peak_out"].to_numpy())
        scores = np.maximum(qs[:, 0] - y, y - qs[:, 2])
        level = self.quantiles[2] - self.quantiles[0]
        rank = min(len(scores), int(np.ceil((len(scores) + 1) * level)))
        self.interval_adjust = float(np.sort(scores)[rank - 1])
        return self.interval_adjust

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[FEATURES].to_numpy(dtype=np.float32)
        keep = [
            "unit_id",
            "day",
            "year",
            "abbr",
            "cause",
            "customers",
            "sig",
            "peak_out",
            "prev_sig",
            "p_clim",
            "mu",
            "sd",
            "base",
        ]
        out = frame[keep].copy()
        out["p_raw"] = self.classifier.predict(x, num_iteration=self.classifier.best_iteration)
        out["p"] = self.iso.predict(out["p_raw"].to_numpy())
        if self.logit is not None:
            out["p_logit"] = self.logit.predict_proba(frame[LOGIT_FEATURES])[:, 1]
        qs = self.size_quantiles(x)
        lo = np.minimum(qs[:, 0] - self.interval_adjust, qs[:, 1])
        hi = np.maximum(qs[:, 2] + self.interval_adjust, qs[:, 1])
        out["q10"] = np.expm1(np.clip(lo, 0, 20))
        out["q50"] = np.expm1(np.clip(qs[:, 1], 0, 20))
        out["q90"] = np.expm1(np.clip(hi, 0, 20))
        out["q10_raw"] = np.expm1(np.clip(qs[:, 0], 0, 20))
        out["q90_raw"] = np.expm1(np.clip(qs[:, 2], 0, 20))
        sigma = np.maximum((hi - lo) / (2 * Z90), 0.05)
        mean_size = np.expm1(np.clip(qs[:, 1] + sigma**2 / 2, 0, 20))
        out["mean_size"] = mean_size
        out["expected"] = out["p"] * mean_size + (1 - out["p"]) * out["base"]
        clim_mean = np.expm1(np.clip(out["mu"] + out["sd"] ** 2 / 2, 0, 20))
        out["expected_clim"] = out["p_clim"] * clim_mean + (1 - out["p_clim"]) * out["base"]
        floats = out.select_dtypes(include="float64").columns
        out[floats] = out[floats].astype(np.float32)
        return out

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(
            str(folder / "classifier.txt"), num_iteration=self.classifier.best_iteration
        )
        for q, model in self.size.items():
            model.save_model(
                str(folder / f"size_q{int(round(q * 100)):02d}.txt"),
                num_iteration=model.best_iteration,
            )
        calibration = {
            "x": self.iso.X_thresholds_.tolist(),
            "y": self.iso.y_thresholds_.tolist(),
            "interval_adjust": self.interval_adjust,
        }
        (folder / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")


def fit(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> RiskModel:
    r = cfg.risk
    train = load_sampled(con, "train", r.negative_rate, cfg.runtime.seed)
    valid = load_sampled(con, "valid", r.negative_rate, cfg.runtime.seed)
    log(
        f"risk: {len(train):,} train rows ({train['sig'].mean():.3f} positive), "
        f"{len(valid):,} valid rows"
    )
    sample = train.sample(n=min(len(train), r.logistic_rows), random_state=cfg.runtime.seed)
    logit = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)
    )
    logit.fit(
        sample[LOGIT_FEATURES],
        sample["sig"].astype(int),
        logisticregression__sample_weight=sample["weight"],
    )
    del sample
    stop = [lgb.early_stopping(r.early_stopping, verbose=False)]
    dtrain = lgb.Dataset(
        train[FEATURES],
        label=train["sig"].astype(int),
        weight=train["weight"],
        categorical_feature=CATEGORICAL,
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        valid[FEATURES],
        label=valid["sig"].astype(int),
        weight=valid["weight"],
        reference=dtrain,
        categorical_feature=CATEGORICAL,
        free_raw_data=True,
    )
    classifier = lgb.train(
        lgb_params(cfg, objective="binary", metric="binary_logloss"),
        dtrain,
        num_boost_round=r.max_rounds,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=stop,
    )
    del dtrain, dvalid
    gc.collect()
    log(f"risk: classifier stopped at {classifier.best_iteration} trees")
    raw = classifier.predict(valid[FEATURES], num_iteration=classifier.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    iso.fit(raw, valid["sig"].astype(int), sample_weight=valid["weight"])
    pos_train = train[train["sig"]]
    pos_valid = valid[valid["sig"]]
    size = {}
    for q in r.quantiles:
        d1 = lgb.Dataset(
            pos_train[FEATURES],
            label=np.log1p(pos_train["peak_out"]),
            categorical_feature=CATEGORICAL,
            free_raw_data=True,
        )
        d2 = lgb.Dataset(
            pos_valid[FEATURES],
            label=np.log1p(pos_valid["peak_out"]),
            reference=d1,
            categorical_feature=CATEGORICAL,
            free_raw_data=True,
        )
        params = lgb_params(
            cfg,
            objective="quantile",
            alpha=q,
            metric="quantile",
            learning_rate=r.size_learning_rate,
            num_leaves=r.size_num_leaves,
            min_data_in_leaf=r.size_min_child_samples,
        )
        size[q] = lgb.train(
            params,
            d1,
            num_boost_round=r.size_max_rounds,
            valid_sets=[d2],
            valid_names=["valid"],
            callbacks=stop,
        )
        log(f"risk: size q{q:.2f} stopped at {size[q].best_iteration} trees")
    model = RiskModel(classifier, iso, size, logit, r.quantiles)
    adjust = model.conformalize(pos_valid)
    log(f"risk: conformal interval adjustment {adjust:+.3f} on the log scale")
    return model


def shap_importance(model: RiskModel, frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    pos = frame[frame["sig"]]
    neg = frame[~frame["sig"]]
    half = rows // 2
    sample = pd.concat(
        [
            pos.sample(n=min(len(pos), half), random_state=seed),
            neg.sample(n=min(len(neg), rows - min(len(pos), half)), random_state=seed),
        ]
    )
    contrib = model.classifier.predict(
        sample[FEATURES], num_iteration=model.classifier.best_iteration, pred_contrib=True
    )
    values = np.abs(contrib[:, :-1]).mean(axis=0)
    return (
        pd.DataFrame({"feature": FEATURES, "mean_abs_shap": values})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


PRED_COLUMNS = [
    "unit_id",
    "day",
    "year",
    "abbr",
    "cause",
    "customers",
    "sig",
    "peak_out",
    "prev_sig",
    "p",
    "p_raw",
    "p_logit",
    "p_clim",
    "q10",
    "q50",
    "q90",
    "q10_raw",
    "q90_raw",
    "expected",
    "expected_clim",
    "mean_size",
    "mu",
    "sd",
    "base",
]


def write_predictions(con: duckdb.DuckDBPyConnection, scored: pd.DataFrame, first: bool) -> None:
    con.register("pred_df", scored[PRED_COLUMNS])
    select = """
        SELECT * REPLACE (CAST(unit_id AS VARCHAR) AS unit_id, CAST(abbr AS VARCHAR) AS abbr,
                          CAST(cause AS VARCHAR) AS cause)
        FROM pred_df
    """
    if first:
        con.execute(f"CREATE OR REPLACE TABLE predictions AS {select}")
    else:
        con.execute(f"INSERT INTO predictions {select}")
    con.unregister("pred_df")


def arrays(con: duckdb.DuckDBPyConnection, columns: list[str], where: str = "") -> pd.DataFrame:
    body = ", ".join(f"CAST({c} AS FLOAT) AS {c}" for c in columns)
    return con.execute(f"SELECT CAST(sig AS TINYINT) AS y, {body} FROM predictions {where}").df()


def cause_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    for (cause,) in con.execute("SELECT DISTINCT cause FROM predictions ORDER BY cause").fetchall():
        part = arrays(con, ["p", "p_clim"], f"WHERE cause = '{cause}'")
        y = part["y"].to_numpy()
        row = {
            "cause": cause,
            "rows": len(part),
            "positives": int(y.sum()),
            "base_rate": float(y.mean()),
        }
        if 0 < y.sum() < len(y):
            row["ap_lightgbm"] = float(average_precision_score(y, part["p"]))
            row["ap_climatology"] = float(average_precision_score(y, part["p_clim"]))
            row["auc_lightgbm"] = float(roc_auc_score(y, part["p"]))
        rows.append(row)
    return pd.DataFrame(rows)


def size_metrics(con: duckdb.DuckDBPyConnection, model: RiskModel, quantiles) -> dict:
    pos = arrays(
        con, ["peak_out", "q10", "q50", "q90", "q10_raw", "q90_raw", "mu", "sd"], "WHERE sig"
    )
    out = pos["peak_out"].to_numpy(float)
    ly = np.log1p(out)
    lq = {
        q: np.log1p(pos[c].to_numpy(float))
        for q, c in zip(quantiles, ["q10", "q50", "q90"], strict=True)
    }
    clim_q = {
        q: pos["mu"].to_numpy(float) + pos["sd"].to_numpy(float) * z
        for q, z in zip(quantiles, [-Z90, 0.0, Z90], strict=True)
    }
    inside = (out >= pos["q10"]) & (out <= pos["q90"])
    inside_raw = (out >= pos["q10_raw"]) & (out <= pos["q90_raw"])
    miss = np.abs(np.log10(1 + out) - np.log10(1 + pos["q50"].to_numpy(float)))
    return {
        "rows": int(len(pos)),
        "interval_coverage": float(np.mean(inside)),
        "interval_coverage_raw": float(np.mean(inside_raw)),
        "interval_adjust": model.interval_adjust,
        "nominal_coverage": float(quantiles[2] - quantiles[0]),
        "median_abs_log10_error": float(np.median(miss)),
        "pinball_model": {str(q): pinball(ly, lq[q], q) for q in quantiles},
        "pinball_climatology": {str(q): pinball(ly, clim_q[q], q) for q in quantiles},
    }


def day_table(con: duckdb.DuckDBPyConnection, top_days: int) -> dict:
    con.execute("""
        CREATE OR REPLACE TABLE test_days AS
        SELECT day, sum(peak_out) AS actual, sum(expected) AS expected,
               sum(expected_clim) AS expected_clim, sum(CAST(sig AS INTEGER)) AS sig_actual,
               sum(p) AS sig_expected, sum(p_clim) AS sig_clim, count(*) AS units
        FROM predictions GROUP BY day ORDER BY day
    """)
    days = con.execute("SELECT * FROM test_days").df()
    out = {
        "days": int(len(days)),
        "spearman_model": float(spearmanr(days["actual"], days["expected"]).statistic),
        "spearman_climatology": float(spearmanr(days["actual"], days["expected_clim"]).statistic),
        "spearman_sig_model": float(spearmanr(days["sig_actual"], days["sig_expected"]).statistic),
        "spearman_sig_climatology": float(
            spearmanr(days["sig_actual"], days["sig_clim"]).statistic
        ),
    }
    top = days.sort_values("actual", ascending=False).head(top_days)
    out["top_days"] = [
        {
            "day": str(pd.Timestamp(row.day).date()),
            "actual": int(row.actual),
            "expected": float(row.expected),
            "expected_clim": float(row.expected_clim),
            "sig_actual": int(row.sig_actual),
            "sig_expected": float(row.sig_expected),
        }
        for row in top.itertuples()
    ]
    total = max(float(top["actual"].sum()), 1.0)
    out["top_days_ratio_model"] = float(top["expected"].sum() / total)
    out["top_days_ratio_climatology"] = float(top["expected_clim"].sum() / total)
    return out


def evaluate(cfg: Config, con: duckdb.DuckDBPyConnection, model: RiskModel, log=print) -> dict:
    r = cfg.risk
    importance = None
    first = True
    for year in cfg.years.test:
        frame = load(con, "test", f"AND m.year = {year}", climate=True)
        if frame.empty:
            continue
        if importance is None:
            importance = shap_importance(model, frame, r.shap_rows, cfg.runtime.seed)
        scored = model.score(frame)
        rows = len(frame)
        del frame
        write_predictions(con, scored, first)
        first = False
        del scored
        gc.collect()
        log(f"risk: scored {rows:,} test county-days for {year}")
    if first:
        raise RuntimeError("no usable county-days in the test years")
    columns = {
        "climatology": "p_clim",
        "logistic": "p_logit",
        "lightgbm_raw": "p_raw",
        "lightgbm": "p",
    }
    test = arrays(con, [*columns.values(), "prev_sig"])
    y = test["y"].to_numpy()
    clim = test["p_clim"].to_numpy(float)
    probs = {name: test[col].to_numpy(float) for name, col in columns.items()}
    onset = (test["prev_sig"] == 0).to_numpy()
    del test
    metrics = {
        name: classifier_metrics(y, p, r.calibration_bins, clim) for name, p in probs.items()
    }
    slices = {
        "onset_days": {
            name: classifier_metrics(y[onset], p[onset], r.calibration_bins, clim[onset])
            for name, p in probs.items()
        }
    }
    tables = {
        "reliability_lightgbm": reliability(y, probs["lightgbm"], r.calibration_bins),
        "reliability_logistic": reliability(y, probs["logistic"], r.calibration_bins),
        "reliability_climatology": reliability(y, clim, r.calibration_bins),
        "pr_lightgbm": pr_points(y, probs["lightgbm"]),
        "pr_logistic": pr_points(y, probs["logistic"]),
        "pr_climatology": pr_points(y, clim),
        "shap": importance,
        "by_cause": cause_table(con),
    }
    del probs
    for name, table in tables.items():
        con.register("t_df", table)
        con.execute(f"CREATE OR REPLACE TABLE risk_{name} AS SELECT * FROM t_df")
        con.unregister("t_df")
    auc_gain = metrics["lightgbm"].get("roc_auc", 0) - metrics["climatology"].get("roc_auc", 0)
    status = "pass" if auc_gain > 0 else "warn"
    record_check(con, "risk", "test_auc_gain_over_climatology", status, auc_gain)
    return {
        "classifier_trees": int(model.classifier.best_iteration),
        "size_trees": {str(q): int(m.best_iteration) for q, m in model.size.items()},
        "test": metrics,
        "slices": slices,
        "size": size_metrics(con, model, r.quantiles),
        "days": day_table(con, r.top_days),
        "shap_top": importance.head(10).to_dict(orient="records"),
    }


def run(cfg: Config, con: duckdb.DuckDBPyConnection, log=print) -> dict:
    build_climatology(cfg, con)
    model = fit(cfg, con, log)
    model.save(cfg.paths.models)
    return evaluate(cfg, con, model, log)
