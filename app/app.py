from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "bundle"
DEFAULT = LOCAL if LOCAL.exists() else HERE.parent / "data" / "bundle"
BUNDLE = Path(os.environ.get("BUNDLE_DIR") or DEFAULT)

SURFACE = "#1a1a19"
RAISED = "#262624"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
ACTUAL = "#c3c2b7"
MODELS = {
    "lightgbm": ("LightGBM", BLUE),
    "logistic": ("Logistic regression", ORANGE),
    "climatology": ("Climatology", AQUA),
}
CAUSES = {
    "convective": "#3987e5",
    "tropical": "#d95926",
    "winter": "#199e70",
    "wind": "#c98500",
    "ice": "#d55181",
    "flood": "#008300",
    "none": "#9085e9",
    "other": "#e66767",
}
SEQUENTIAL = [
    [0.0, "#0d366b"],
    [0.2, "#184f95"],
    [0.4, "#256abf"],
    [0.6, "#3987e5"],
    [0.8, "#86b6ef"],
    [1.0, "#cde2fb"],
]
POLICIES = {
    "none": "No mutual aid",
    "proportional": "Split by customers",
    "climatology": "Climatology plan",
    "forecast": "Forecast plan",
    "perfect": "Perfect foresight",
}
MAP_STYLE = {
    "version": 8,
    "sources": {},
    "layers": [{"id": "background", "type": "background", "paint": {"background-color": SURFACE}}],
}

st.set_page_config(page_title="Storm Outage Forecast", layout="wide")


@st.cache_resource
def connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(BUNDLE / "app.duckdb"), read_only=True)


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    return connection().execute(f"SELECT * FROM {name}").df()


def table(name: str) -> pd.DataFrame:
    return load_table(name).copy()


@st.cache_data
def meta() -> dict:
    return json.loads((BUNDLE / "meta.json").read_text(encoding="utf-8"))


@st.cache_data
def counties() -> dict:
    return json.loads((BUNDLE / "counties.json").read_text(encoding="utf-8"))


def compact(value: float) -> str:
    value = float(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"{value / limit:.1f}{suffix}"
    return f"{value:,.0f}"


def style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        font={"family": FONT, "color": INK_2, "size": 13},
        margin={"l": 12, "r": 16, "t": 40, "b": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "title": None,
            "font": {"color": INK_2},
        },
        hoverlabel={"bgcolor": RAISED, "font": {"color": INK, "family": FONT}},
        barcornerradius=4,
    )
    axis = {
        "gridcolor": GRID,
        "linecolor": AXIS,
        "zerolinecolor": AXIS,
        "tickfont": {"color": MUTED},
        "title_font": {"color": INK_2},
        "automargin": True,
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def chart(fig: go.Figure, title: str, height: int = 380) -> None:
    st.markdown(f"##### {title}")
    st.plotly_chart(style(fig, height), theme=None, config={"displaylogo": False})


def log_axis(fig: go.Figure, axis: str, fmt: str) -> None:
    values = np.concatenate([np.asarray(getattr(t, axis), dtype=float) for t in fig.data])
    values = values[np.isfinite(values) & (values > 0)]
    ratio = values.max() / values.min() if len(values) else 10.0
    dtick = "D1" if ratio < 30 else ("D2" if ratio < 3000 else 1)
    update = fig.update_xaxes if axis == "x" else fig.update_yaxes
    update(type="log", dtick=dtick, tickformat=fmt, minor={"showgrid": False})


def county_map(shapes: pd.DataFrame, column: str, top: float, fmt: str) -> go.Figure:
    custom = np.column_stack(
        [
            shapes["name"],
            shapes["abbr"],
            shapes["p"],
            shapes["share"],
            shapes["peak_out"],
            shapes["expected"],
            shapes["cause"],
        ]
    )
    trace = go.Choroplethmap(
        geojson=counties(),
        locations=shapes["geo_id"],
        z=shapes[column],
        featureidkey="id",
        colorscale=SEQUENTIAL,
        zmin=0,
        zmax=top,
        marker_line_width=0,
        customdata=custom,
        colorbar={
            "tickformat": fmt,
            "tickfont": {"color": MUTED},
            "outlinewidth": 0,
            "thickness": 14,
        },
        hovertemplate=(
            "<b>%{customdata[0]}, %{customdata[1]}</b><br>"
            "Forecast probability %{customdata[2]:.2f}<br>"
            "Share out %{customdata[3]:.1%}<br>"
            "Peak customers out %{customdata[4]:,}<br>"
            "Forecast customers out %{customdata[5]:,.0f}<br>"
            "Storm report %{customdata[6]}<extra></extra>"
        ),
    )
    fig = go.Figure(trace)
    fig.update_layout(map={"style": MAP_STYLE, "center": {"lat": 38.5, "lon": -96.5}, "zoom": 3.1})
    return fig


def replay_tab() -> None:
    days = table("days")
    map_days = table("map_days")
    units = table("units")
    geo = table("unit_geo")
    days["key"] = pd.to_datetime(days["day"]).dt.strftime("%Y-%m-%d")
    map_days["key"] = pd.to_datetime(map_days["day"]).dt.strftime("%Y-%m-%d")
    options = days[days["key"].isin(map_days["key"].unique())].sort_values(
        "actual", ascending=False
    )
    labels = {
        k: f"{k}   {compact(a)} customers out"
        for k, a in zip(options["key"], options["actual"], strict=True)
    }
    day = st.selectbox(
        "Storm day in the test years",
        options["key"].tolist(),
        format_func=lambda k: labels.get(k, k),
    )
    row = days[days["key"] == day].iloc[0]
    cols = st.columns(4)
    cols[0].metric(
        "Customers out, actual",
        compact(row["actual"]),
        help="Sum over counties of the day's peak customers without power",
    )
    cols[1].metric("Forecast, expected", compact(row["expected"]))
    cols[2].metric("Climatology, expected", compact(row["expected_clim"]))
    cols[3].metric(
        "Significant counties, actual / forecast",
        f"{row['sig_actual']:.0f} / {row['sig_expected']:.0f}",
    )
    view = st.radio(
        "Map layer", ["Forecast probability", "Actual share of customers out"], horizontal=True
    )
    part = map_days[map_days["key"] == day].merge(units[["unit_id", "name", "abbr"]], on="unit_id")
    part["share"] = (part["peak_out"] / part["customers"]).clip(0, 1)
    shapes = geo.merge(part, on="unit_id")
    if view == "Forecast probability":
        fig = county_map(shapes, "p", 1.0, ".1f")
    else:
        top = max(0.02, float(shapes["share"].quantile(0.995)))
        fig = county_map(shapes, "share", top, ".0%")
    chart(fig, f"{view} by county, {day}", 540)
    ranked = part.sort_values("p", ascending=False).head(15)
    st.dataframe(
        ranked[["name", "abbr", "p", "q10", "q50", "q90", "peak_out", "cause"]],
        hide_index=True,
        column_config={
            "name": "County",
            "abbr": "State",
            "p": st.column_config.ProgressColumn(
                "Forecast probability", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "q10": st.column_config.NumberColumn("Size P10", format="%d"),
            "q50": st.column_config.NumberColumn("Size P50", format="%d"),
            "q90": st.column_config.NumberColumn("Size P90", format="%d"),
            "peak_out": st.column_config.NumberColumn("Actual peak out", format="%d"),
            "cause": "Storm report",
        },
    )
    full = days[days["units"] >= 0.9 * days["units"].median()]
    line = go.Figure()
    line.add_scatter(
        x=full["day"], y=full["actual"], name="Actual", line={"color": ACTUAL, "width": 2}
    )
    line.add_scatter(
        x=full["day"],
        y=full["expected"],
        name="LightGBM forecast",
        line={"color": BLUE, "width": 2},
    )
    line.add_scatter(
        x=full["day"], y=full["expected_clim"], name="Climatology", line={"color": AQUA, "width": 2}
    )
    line.add_shape(
        type="line", x0=day, x1=day, y0=0, y1=1, yref="paper", line={"color": MUTED, "width": 1}
    )
    line.update_layout(hovermode="x unified", yaxis_title="Customers out")
    log_axis(line, "y", "~s")
    chart(line, "Every test day, sum of county peaks", 380)
    st.caption(
        "Days with reporting gaps in more than 10 percent of counties are left out of this chart."
    )


def metric_rows(block: dict) -> pd.DataFrame:
    names = {
        "lightgbm": "LightGBM",
        "lightgbm_raw": "LightGBM before calibration",
        "logistic": "Logistic regression",
        "climatology": "County-month climatology",
    }
    rows = []
    for key, label in names.items():
        m = block.get(key, {})
        rows.append(
            {
                "Model": label,
                "ROC AUC": m.get("roc_auc"),
                "Average precision": m.get("average_precision"),
                "Brier skill": m.get("brier_skill"),
                "Calibration error": m.get("ece"),
                "Log loss": m.get("log_loss"),
            }
        )
    return pd.DataFrame(rows)


def model_tab(info: dict) -> None:
    risk = info["risk"]
    test = risk["test"]
    st.markdown(
        f"Positive rate in the test years: {test['lightgbm']['base_rate']:.2%} of "
        f"{test['lightgbm']['rows']:,} county-days."
    )
    fmt = {
        "ROC AUC": "%.3f",
        "Average precision": "%.3f",
        "Brier skill": "%.3f",
        "Calibration error": "%.4f",
        "Log loss": "%.4f",
    }
    config = {k: st.column_config.NumberColumn(k, format=v) for k, v in fmt.items()}
    st.markdown("#### All county-days")
    st.dataframe(metric_rows(test), hide_index=True, column_config=config)
    st.markdown("#### Onset days only (no significant outage the day before)")
    st.dataframe(metric_rows(risk["slices"]["onset_days"]), hide_index=True, column_config=config)
    left, right = st.columns(2)
    curves = table("pr_curve")
    pr = go.Figure()
    for key, (label, color) in MODELS.items():
        part = curves[curves["model"] == key]
        pr.add_scatter(
            x=part["recall"], y=part["precision"], name=label, line={"color": color, "width": 2}
        )
    pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", hovermode="closest")
    pr.update_xaxes(range=[0, 1])
    pr.update_yaxes(range=[0, 1.02])
    with left:
        chart(pr, "Precision and recall")
    rel = table("reliability")
    low = max(1e-4, float(rel["p_mean"].min()) * 0.8)
    cal = go.Figure()
    cal.add_scatter(
        x=[low, 1],
        y=[low, 1],
        name="Perfect calibration",
        mode="lines",
        line={"color": AXIS, "width": 1},
        hoverinfo="skip",
    )
    for key, (label, color) in MODELS.items():
        part = rel[rel["model"] == key]
        cal.add_scatter(
            x=part["p_mean"],
            y=part["y_rate"].clip(lower=low),
            name=label,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
        )
    cal.update_layout(xaxis_title="Predicted probability (bin mean)", yaxis_title="Observed rate")
    log_axis(cal, "x", ".1~%")
    log_axis(cal, "y", ".1~%")
    with right:
        chart(cal, "Calibration by probability decile")
    shap = table("shap").head(15).iloc[::-1]
    bars = go.Figure(
        go.Bar(
            x=shap["mean_abs_shap"],
            y=shap["feature"],
            orientation="h",
            marker={"color": BLUE},
            width=0.6,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    bars.update_layout(xaxis_title="Mean |SHAP| on the log-odds scale")
    chart(bars, "What drives the LightGBM probability", 480)
    st.markdown("#### By storm report on the day")
    cause = table("by_cause").sort_values("positives", ascending=False)
    st.dataframe(
        cause,
        hide_index=True,
        column_config={
            "cause": "Storm report",
            "rows": "County-days",
            "positives": "Significant",
            "base_rate": st.column_config.NumberColumn("Rate", format="%.3f"),
            "ap_lightgbm": st.column_config.NumberColumn("AP LightGBM", format="%.3f"),
            "ap_climatology": st.column_config.NumberColumn("AP climatology", format="%.3f"),
            "auc_lightgbm": st.column_config.NumberColumn("AUC LightGBM", format="%.3f"),
        },
    )
    size = risk["size"]
    st.markdown("#### How many customers, given a significant outage")
    cols = st.columns(3)
    cols[0].metric(
        f"P10 to P90 coverage (target {size['nominal_coverage']:.0%})",
        f"{size['interval_coverage']:.1%}",
    )
    cols[1].metric("Coverage before the conformal step", f"{size['interval_coverage_raw']:.1%}")
    cols[2].metric(
        "Typical P50 miss",
        f"x{10 ** size['median_abs_log10_error']:.2f}",
        help="Median of the ratio between actual and predicted peak, in either direction",
    )
    pin = pd.DataFrame(
        {
            "Quantile": list(size["pinball_model"]),
            "LightGBM": list(size["pinball_model"].values()),
            "Climatology": list(size["pinball_climatology"].values()),
        }
    )
    st.dataframe(
        pin,
        hide_index=True,
        column_config={
            "LightGBM": st.column_config.NumberColumn("Pinball loss, LightGBM", format="%.4f"),
            "Climatology": st.column_config.NumberColumn(
                "Pinball loss, climatology", format="%.4f"
            ),
        },
    )


def restoration_tab(info: dict) -> None:
    rest = info["metrics"]["restoration"]
    cols = st.columns(4)
    cols[0].metric("C-index, Weibull model", f"{rest['c_index_model']:.3f}")
    cols[1].metric("C-index, cause medians", f"{rest['c_index_cause_median']:.3f}")
    cols[2].metric("Within 2x, Weibull model", f"{rest['within_2x_model']:.1%}")
    cols[3].metric(
        f"Test events ({rest['test_censored_share']:.1%} censored)", f"{rest['test_events']:,}"
    )
    curves = table("survival_curves")
    test = curves[curves["sample"] == "test"]
    ranked = test.groupby("cause")["events"].max().sort_values(ascending=False).index.tolist()
    chosen = st.multiselect("Storm report", ranked, default=ranked[:5])
    km = go.Figure()
    for cause in chosen:
        part = test[test["cause"] == cause]
        km.add_scatter(
            x=part["hours"],
            y=part["survival"],
            name=cause,
            mode="lines",
            line={"color": CAUSES.get(cause, CAUSES["other"]), "width": 2, "shape": "hv"},
        )
    km.update_layout(
        xaxis_title="Hours after the peak",
        yaxis_title="Share of events not yet restored",
        hovermode="x unified",
    )
    km.update_yaxes(tickformat=".0%", range=[0, 1.02])
    chart(km, "Kaplan-Meier restoration curves, test years", 420)
    left, right = st.columns(2)
    calib = table("survival_calibration").dropna()
    with left:
        fig = go.Figure()
        if len(calib):
            low = float(min(calib["predicted_median"].min(), calib["observed_median"].min()))
            high = float(max(calib["predicted_median"].max(), calib["observed_median"].max()))
            fig.add_scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                name="Perfect",
                line={"color": AXIS, "width": 1},
                hoverinfo="skip",
            )
            fig.add_scatter(
                x=calib["predicted_median"],
                y=calib["observed_median"],
                mode="lines+markers",
                name="Weibull model",
                line={"color": BLUE, "width": 2},
                marker={"size": 9, "line": {"color": SURFACE, "width": 2}},
            )
        fig.update_layout(xaxis_title="Predicted median hours", yaxis_title="Observed median hours")
        log_axis(fig, "x", "~g")
        log_axis(fig, "y", "~g")
        chart(fig, "Calibration by predicted decile")
    with right:
        st.markdown("##### Weibull AFT terms")
        st.dataframe(
            table("survival_coefficients"),
            hide_index=True,
            column_config={
                "term": "Term",
                "coef": st.column_config.NumberColumn("Coefficient", format="%.3f"),
                "se": st.column_config.NumberColumn("Std. error", format="%.3f"),
                "time_ratio_per_sd": st.column_config.NumberColumn(
                    "Time ratio per sd", format="%.2f"
                ),
            },
        )
    st.markdown("#### Events by storm report")
    st.dataframe(
        table("event_summary"),
        hide_index=True,
        column_config={
            "cause": "Storm report",
            "events": "Events",
            "median_hours": st.column_config.NumberColumn("Median hours", format="%.1f"),
            "median_peak": st.column_config.NumberColumn("Median peak", format="%d"),
            "max_peak": st.column_config.NumberColumn("Largest peak", format="%d"),
        },
    )


def staging_tab(info: dict) -> None:
    stage = info["metrics"]["staging"]
    setup = info["staging"]
    st.markdown(
        f"{setup['crews']} mutual-aid crews are placed across states the night before each "
        f"of the {stage['days']} biggest test days. Each state also has one local crew per "
        f"{setup['customers_per_base_crew']:,} customers, and one crew restores "
        f"{setup['restore_rate']:.0f} customers an hour."
    )
    share = stage["share_of_perfect_savings"]
    cols = st.columns(2)
    cols[0].metric("Forecast plan, share of perfect-foresight savings", f"{share['forecast']:.0%}")
    cols[1].metric(
        "Climatology plan, share of perfect-foresight savings", f"{share['climatology']:.0%}"
    )
    order = list(POLICIES)
    peak = max(stage["customer_hours"].values())
    scale, unit = (1e9, "Billion") if peak >= 1e9 else (1e6, "Million")
    hours = [stage["customer_hours"][p] / scale for p in order]
    colors = [BLUE if p == "forecast" else AXIS for p in order]
    fig = go.Figure(
        go.Bar(
            x=hours,
            y=[POLICIES[p] for p in order],
            orientation="h",
            marker={"color": colors},
            width=0.6,
            hovertemplate="%{y}: %{x:,.2f} " + unit.lower() + " customer-hours<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title=f"{unit} customer-hours without power, all storm days")
    fig.update_yaxes(autorange="reversed")
    chart(fig, "Outage duration under each plan", 320)
    alloc = table("staging_allocations")
    alloc["key"] = pd.to_datetime(alloc["day"]).dt.strftime("%Y-%m-%d")
    day = st.selectbox("Storm day", sorted(alloc["key"].unique()))
    part = alloc[alloc["key"] == day]
    pivot = part.pivot_table(index="abbr", columns="policy", values="crews", fill_value=0)
    pivot = pivot.reindex(columns=order, fill_value=0)
    pivot = pivot.join(part.groupby("abbr")["actual_out"].max())
    pivot = pivot.sort_values("actual_out", ascending=False).head(12)
    fig = go.Figure()
    fig.add_bar(x=pivot.index, y=pivot["forecast"], name="Forecast plan", marker={"color": BLUE})
    fig.add_bar(
        x=pivot.index, y=pivot["perfect"], name="Perfect foresight", marker={"color": ACTUAL}
    )
    fig.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.12,
        yaxis_title="Extra crews",
        hovermode="x unified",
    )
    chart(fig, "Extra crews for the 12 hardest-hit states", 360)
    view = pivot.reset_index().rename(
        columns={**POLICIES, "abbr": "State", "actual_out": "Customers out"}
    )
    st.dataframe(view, hide_index=True)


def checks_tab(info: dict) -> None:
    data = info["metrics"].get("data") or {}
    cols = st.columns(5)
    cols[0].metric("Counties", f"{data.get('units', 0):,}")
    cols[1].metric("Outage rows", compact(data.get("outage_rows", 0)))
    cols[2].metric("County-days", compact(data.get("county_days", 0)))
    cols[3].metric("Outage events", compact(data.get("events", 0)))
    cols[4].metric("Wind stations", f"{data.get('wind_stations', 0):,}")
    rel = info["metrics"].get("reliability") or {}
    saidi = table("saidi_check")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        if len(saidi):
            low = float(min(saidi["saidi_eaglei"].min(), saidi["saidi_eia"].min()))
            high = float(max(saidi["saidi_eaglei"].max(), saidi["saidi_eia"].max()))
            fig.add_scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                name="Equal",
                line={"color": AXIS, "width": 1},
                hoverinfo="skip",
            )
            fig.add_scatter(
                x=saidi["saidi_eia"],
                y=saidi["saidi_eaglei"],
                mode="markers",
                name="State-year",
                marker={"size": 8, "color": BLUE, "line": {"color": SURFACE, "width": 2}},
                text=saidi["abbr"] + " " + saidi["year"].astype(str),
                hovertemplate="%{text}<br>EIA-861 %{x:,.0f} min<br>"
                "EAGLE-I %{y:,.0f} min<extra></extra>",
            )
        fig.update_layout(
            xaxis_title="EIA-861 SAIDI with major events (minutes)",
            yaxis_title="EAGLE-I outage minutes per customer",
        )
        log_axis(fig, "x", "~s")
        log_axis(fig, "y", "~s")
        spearman = rel.get("spearman")
        title = "Outage minutes vs EIA-861"
        if spearman is not None:
            title += f", Spearman {spearman:.2f}"
        chart(fig, title, 400)
    with right:
        coverage = table("coverage")
        fig = go.Figure(
            go.Bar(
                x=coverage["year"],
                y=coverage["covered_share"],
                marker={"color": BLUE},
                width=0.6,
                hovertemplate="%{x}: %{y:.0%}<extra></extra>",
            )
        )
        fig.update_layout(yaxis_title="County-days used for modeling")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_xaxes(dtick=1)
        chart(fig, "Reporting coverage by year", 400)
    st.markdown("#### Data quality checks")
    st.dataframe(
        table("quality"),
        hide_index=True,
        column_config={
            "stage": "Stage",
            "check_name": "Check",
            "status": "Status",
            "value": st.column_config.NumberColumn("Value", format="%.4g"),
            "detail": "Detail",
        },
    )


def main() -> None:
    if not (BUNDLE / "app.duckdb").exists():
        st.error(f"No dashboard data found in {BUNDLE}. Run the pipeline first.")
        st.stop()
    info = meta()
    st.title("Storm Outage Forecast")
    test = info["test_years"]
    st.caption(
        f"County-level power outage risk from weather for the contiguous US. Trained on "
        f"{info['train_years'][0]}-{info['train_years'][1]}, tuned on {info['valid_year']}, "
        f"scored on {test[0]}-{test[1]}. A significant outage is a day when at least "
        f"{info['significant']['share']:.0%} of a county's customers (and at least "
        f"{info['significant']['floor']}) lose power."
    )
    if info.get("data_kind") == "synthetic":
        st.warning("These numbers come from the synthetic test replica, not the public data.")
    tabs = st.tabs(["Storm replay", "Model report", "Restoration", "Crew staging", "Data checks"])
    with tabs[0]:
        replay_tab()
    with tabs[1]:
        model_tab(info)
    with tabs[2]:
        restoration_tab(info)
    with tabs[3]:
        staging_tab(info)
    with tabs[4]:
        checks_tab(info)


main()
