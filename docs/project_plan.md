# Project Plan

## Goal

Build a reproducible NYC yellow-taxi demand forecasting project that
combines real public data with clean software-engineering practices
(packaging, tests, docs, CI).

## Forecasting question

For each NYC borough, can we predict hourly yellow-taxi pickup demand
for the upcoming week using the previous month(s) of trip records?

## Pipeline

1. Download two months of NYC TLC yellow-taxi parquet files plus the
   zone lookup CSV (`scripts/download_data.py`).
2. Load and concatenate the parquet files, keeping only the columns we
   need (`load_yellow_taxi_data`).
3. Map pickup `LocationID` to borough names via the zone lookup
   (`attach_pickup_borough`); restrict to Manhattan, Queens, Brooklyn,
   Bronx.
4. Aggregate raw trips into hourly pickup counts per borough
   (`aggregate_hourly_pickups`) and fill any gap hours with zero
   (`fill_missing_hours`).
5. Hold out the last 168 hours of each borough's series for testing
   (`train_test_split_by_time`).
6. Fit and score two baselines (`hourly_profile_baseline`,
   `seasonal_naive_baseline`); compute MAE and RMSE.
7. Save results into `data/processed/` and figures into
   `docs/figures/`.

## Models implemented

- **Hourly profile baseline**: predict the historical mean for each
  `(borough, hour-of-day)` pair, with fallbacks to the borough mean
  and global train mean.
- **Seasonal naive (24h)**: for each test timestamp, walk back through
  multiples of 24h until a *training* timestamp is found. Train-only
  by construction - no test-set leakage even when the test horizon
  exceeds the seasonal period.

## Engineering choices

- `src/`-layout Python package, installable via `pip install -e .[dev]`.
- Unit tests cover both reusable utilities (`dataset`, `metrics`) and
  taxi-specific code paths, including the parquet/CSV loaders.
- Coverage gate set to 80% in `pyproject.toml` and enforced by CI.
- GitHub Actions workflow (`.github/workflows/tests.yml`) runs the
  test suite against Python 3.10, 3.11 and 3.12 on every push and PR.
- Data files are not committed; the download script reproduces them.

## Stretch ideas

- Add a richer model (e.g. SARIMAX, gradient-boosted regression on
  calendar/weather features) and compare against the baselines.
- Break results down by hour-of-day and day-of-week to show *where*
  each model struggles.
- Add a Sphinx HTML documentation build (matches the syllabus's
  "HTML doc & visualization" bullet).
