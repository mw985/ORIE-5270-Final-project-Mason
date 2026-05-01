# ORIE 5270 Final Project: NYC Taxi Demand Forecasting

A reproducible computational project that turns raw New York City TLC
yellow-taxi trip records into an hourly borough-level demand time series
and benchmarks two simple forecasting baselines against the held-out last
week of data.

## 1. Purpose of the project

The project asks a concrete forecasting question:

> *Given the last several months of NYC yellow-taxi pickups, how well can
> we predict next week's hourly pickup demand for each borough?*

Rather than predicting individual trips, we forecast aggregate hourly
demand because that is the quantity an operator (TLC, Uber, Lyft, the
city) actually needs in order to do dispatching, pricing or planning.
The repository emphasises reproducibility and code quality over modelling
complexity: the pipeline runs end to end from a single command and every
reusable piece of logic is unit tested.

## 2. Dataset

We use the public **NYC Taxi & Limousine Commission Yellow Taxi Trip
Records** for January and February 2025, plus the official taxi-zone
lookup table.

- Source: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
- Files used:
  - `yellow_tripdata_2025-01.parquet`
  - `yellow_tripdata_2025-02.parquet`
  - `taxi_zone_lookup.csv`

The two parquet files contain one row per trip; we only read
`tpep_pickup_datetime` and `PULocationID` to keep memory usage low. The
lookup CSV maps numeric zone IDs to borough and zone names.

The data files are **not committed to git** (they're large and
re-downloadable). Use the bundled download script:

```bash
python scripts/download_data.py
```

This populates `data/` with the three files listed above.

## 3. Installation

The project is organised as a standard Python package and installs with
`pip`. Cross-platform; tested on Linux, macOS and Windows.

```bash
# (recommended) create and activate a virtual environment
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

# install the package and its dev dependencies
python -m pip install -e .[dev]
```

After installation, `orie5270_project` is importable from anywhere and
the `pytest` test suite runs without any `PYTHONPATH` tweaks.

The dependency list pins NumPy to `<2` for compatibility with older
Anaconda/Jupyter environments whose compiled optional packages may not
yet support NumPy 2.x. This does not change the analysis; it makes local
reproduction more reliable.

## 4. How to run

### Run the unit tests with coverage

```bash
pytest
```

Coverage is enforced at **80%** through `pyproject.toml`; the suite will
fail if coverage drops below that threshold.

### Run the full taxi analysis

First, make sure the data files are present (see Section 2):

```bash
python scripts/download_data.py
```

Then run the analysis:

```bash
python scripts/run_taxi_analysis.py
```

This will:

1. read the two yellow-taxi parquet files,
2. join them against the zone lookup to attach borough names,
3. restrict the frame to Manhattan, Queens, Brooklyn and Bronx,
4. aggregate trips into hourly pickup counts and fill any gap hours with zero,
5. hold out the last 168 hours (7 days) of each borough's series for testing,
6. fit and score two baseline forecasters (`hourly_profile` and
   `seasonal_naive_24h`),
7. write metrics, predictions and figures into `data/processed/` and
   `docs/figures/`.

### Run the small synthetic demo

```bash
python scripts/run_demo.py
```

Useful for confirming the package imports cleanly without needing the
real data files.

## 5. How to import and use the package

```python
from orie5270_project.taxi import (
    load_yellow_taxi_data, load_zone_lookup,
    attach_pickup_borough, aggregate_hourly_pickups,
    fill_missing_hours, train_test_split_by_time,
    hourly_profile_baseline, seasonal_naive_baseline,
)
from orie5270_project.metrics import mae, rmse

trips = load_yellow_taxi_data(["data/yellow_tripdata_2025-01.parquet"])
zones = load_zone_lookup("data/taxi_zone_lookup.csv")

trips_with_borough = attach_pickup_borough(trips, zones)
hourly = fill_missing_hours(aggregate_hourly_pickups(trips_with_borough))
manhattan = hourly[hourly["Borough"] == "Manhattan"]
train, test = train_test_split_by_time(manhattan, test_hours=24 * 7)
forecast = seasonal_naive_baseline(train, test)

print("MAE:", mae(forecast["pickup_count"], forecast["prediction"]))
```

## 6. Project layout

```
data/                      # raw inputs (gitignored, fetched by download script)
docs/
  figures/                 # generated PNG visualizations
  project_plan.md
scripts/
  download_data.py         # fetches the TLC parquet + zone CSV
  run_demo.py              # synthetic demo of the utility functions
  run_taxi_analysis.py     # full pipeline producing CSVs + figures
src/orie5270_project/
  __init__.py              # re-exports the public API
  dataset.py               # generic dataframe utilities
  metrics.py               # MAE / RMSE / MAPE
  taxi.py                  # taxi-specific pipeline + baselines
tests/                     # pytest suite (>80% coverage enforced)
pyproject.toml
README.md
```

## 7. Models

### Hourly profile baseline

For each `(borough, hour-of-day)` pair, predict the historical training
mean. Falls back first to the borough mean and then to the global train
mean for unseen combinations.

### Seasonal naive (24h) baseline

For each test timestamp `t`, predict the most recent training-set value
at the same hour of day, walking back through multiples of the
seasonal period until a training timestamp is found. The lookup is
**train-only** by construction so longer test horizons cannot leak
information from later test points back into earlier predictions.

Both methods are scored with MAE and RMSE on the last 168 hours of each
borough series.

## 8. Outputs produced by `run_taxi_analysis.py`

- `data/processed/borough_model_metrics.csv` - one row per `(borough, model)`
- `data/processed/borough_model_predictions.csv` - full prediction frame
- `docs/figures/borough_model_comparison.png` - bar chart of MAE/RMSE
- `docs/figures/manhattan_forecast_last_72_hours.png` - actual vs. predicted

## 9. License

MIT. See `LICENSE`.
