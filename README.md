# storm-outage-forecast

County level power outage risk for the contiguous US, plus how long the lights stay off and where
to send extra crews. Everything is built from public data: EAGLE-I outage snapshots, NOAA weather,
NOAA Storm Events and EIA-861 filings, covering 2015 to 2025. Models train on 2015 to 2022, tune on
2023 and are scored on 2024 and 2025, which they never see during training.

Live demo: https://huggingface.co/spaces/adwitiyashukla/storm-outage-forecast

## What it answers

A storm is coming tonight, so three questions follow:

1. Which counties lose power tomorrow, and how many customers in each.
2. Once a county goes dark, how long until it comes back.
3. Where should the 400 mutual aid crews go before it hits.

A county-day counts as a significant outage when the peak is at least 1% of the county's customers
and at least 250 customers. In the two test years that happens on 6.79% of 2,089,900 county-days.

## Data

| Source | What it gives | Scale |
| --- | --- | --- |
| EAGLE-I (Oak Ridge) | customers out per county every 15 minutes | 247.6M rows, 12.8 GB |
| NOAA GHCN-Daily | gust, rain, snow, temperature per station | 1,025 first order stations |
| NOAA Storm Events | storm type and time, used only to label causes | 710,359 reports mapped |
| EIA-861 | utility SAIDI and customer counts by state | 536 state-years |
| NWS zone-county file | maps forecast zones to counties | one file |

Storm Events never becomes a model input. Those reports are written after the storm, so a model
that read them would be seeing the future. They only say what caused an outage after the fact.

## Pipeline

`build` turns the raw files into a DuckDB warehouse: 3,091 counties, 152.0M customers, 12.4M
county-days and 1,067,687 outage events. Weather comes from the 3 nearest first order stations
within 200 km, inverse distance weighted.

The part that decides whether any of this means anything is coverage. EAGLE-I only sees the
utilities that report, and that changed a lot over the years, so a day a county was not reporting
is dropped instead of being counted as zero outages. Covered county-days go from 73% in 2015 to 93%
in 2025.

`train` fits the three models and writes the metrics. `bundle` exports the data the dashboard
reads.

## Risk model

Features are that day's weather and what was already out at the start of the day, plus county size,
density, location, season and the last week of outage history. The model predicts whether the day
turns into a significant outage, and three quantile models predict how large it gets.

| Model | AUC | Average precision | Brier skill | Calibration error |
| --- | --- | --- | --- | --- |
| County-month climatology | 0.695 | 0.147 | 0.000 | 0.0125 |
| Logistic regression | 0.770 | 0.345 | 0.126 | 0.0051 |
| LightGBM | 0.825 | 0.445 | 0.213 | 0.0038 |

Because a county that is already out tends to stay out, I also score only the days where the county
was clean the day before. That slice is 1,948,440 county-days at a 5.4% base rate:

| Model | AUC | Average precision | Brier skill |
| --- | --- | --- | --- |
| County-month climatology | 0.690 | 0.113 | 0.000 |
| Logistic regression | 0.731 | 0.185 | 0.037 |
| LightGBM | 0.794 | 0.275 | 0.100 |

Size is three LightGBM quantile models on log customers out, with a split conformal adjustment fit
on the tuning year. The 80% interval covers 80.3% of test county-days, against 78.8% before the
adjustment. The median prediction is off by a factor of 1.47 at the median.

Summed over counties, the daily national total tracks the real one with a Spearman correlation of
0.910, against 0.432 for climatology. The biggest test day was 2024-09-27, when Helene came ashore:
6.43M customers actually out, 4.73M expected by the model, 0.52M expected by climatology.

By cause, average precision on test county-days:

| Cause | County-days | Base rate | LightGBM | Climatology |
| --- | --- | --- | --- | --- |
| Tropical | 721 | 0.720 | 0.974 | 0.723 |
| Ice | 394 | 0.464 | 0.838 | 0.507 |
| Wind | 8,987 | 0.282 | 0.777 | 0.504 |
| Convective | 25,185 | 0.431 | 0.738 | 0.604 |
| Winter | 29,315 | 0.122 | 0.589 | 0.215 |
| No storm report | 1,961,181 | 0.060 | 0.397 | 0.134 |

The features that move the prediction most are the share already out at the start of the day, how
many customers the county has, the gust relative to that county's own windy days, and rainfall.

## Restoration

Outage spells become survival data, censored when a county stops reporting before power returns.
A Weibull accelerated failure time model, written from scratch and checked against lifelines,
scores a c-index of 0.660 on 119,619 test events, against 0.569 for per cause medians. It ranks
better than the medians but its point predictions are not sharper, so the dashboard shows both.

Median restoration runs 11.25 hours for tropical storms, 10.75 for ice, 3.5 for convective and wind,
and about an hour for days with no storm report. A one standard deviation rise in the share of
customers out multiplies the expected time by 1.45.

## Crew staging

Crew hours lost grow with the square of unmet demand, so the greedy allocation over the forecast
distribution is the optimal one. On the 20 worst test days, staging 400 crews on the forecast
captures 94% of what perfect foresight would save against splitting them by customer count.
Climatology staging lands at -10%, meaning it is worse than not trying.

## Data checks

Every stage writes its checks into a quality table that the dashboard shows. The one that matters
most: state SAIDI computed from EAGLE-I correlates 0.882 with what utilities filed with EIA over
536 state-years, and the median ratio between the two is 1.01.

The rest are there to catch bad input instead of letting it turn into a quietly wrong number.
3.5% of raw rows carry a blank customer count and get dropped, 0.6% of 15 minute snapshots are
missing, and 0.3% of county codes match no county, all reported rather than hidden.

## Running it

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m storm_outage_forecast download
.venv\Scripts\python -m storm_outage_forecast run
.venv\Scripts\python -m streamlit run app\app.py
```

The download is 12.8 GB and needs about 20 GB of free disk once the warehouse is built. On my
laptop, an 8 GB HP OmniBook, parsing the outage files takes 34 minutes and training takes 29, with
DuckDB held to 1 GB and 3 threads so it fits. If a stage fails you can pick up where it stopped
with `run --from weather`.

Every setting lives in config.toml and is checked when it loads. `synth` writes a synthetic copy of
all four data sets in the real file formats, which is what the tests run on, so nothing here needs
the 12.8 GB download to be tested.

## Tests

101 tests over the parsers, the coverage rules, the conformal interval, the survival math against
lifelines, the staging optimum against brute force, and the dashboard itself.
