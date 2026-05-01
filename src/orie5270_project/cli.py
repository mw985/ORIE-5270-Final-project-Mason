"""Command-line entrypoints for the orie5270_project package.

Exposed via ``[project.scripts]`` in ``pyproject.toml`` so users can run::

    orie5270-run-taxi-analysis

after ``pip install -e .``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .metrics import mae, rmse
from .taxi import (
    aggregate_hourly_pickups,
    attach_pickup_borough,
    fill_missing_hours,
    hourly_profile_baseline,
    load_yellow_taxi_data,
    load_zone_lookup,
    seasonal_naive_baseline,
    train_test_split_by_time,
)


_TARGET_BOROUGHS = ("Manhattan", "Queens", "Brooklyn", "Bronx")
_TEST_HOURS = 24 * 7  # last week held out for evaluation


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _save_metric_plot(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    melted = metrics_frame.melt(
        id_vars=["borough", "model"],
        value_vars=["mae", "rmse"],
        var_name="metric",
        value_name="value",
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="borough", y="value", hue="model", ax=axis)
    axis.set_title("Forecast Error by Borough and Model")
    axis.set_xlabel("Borough")
    axis.set_ylabel("Error")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_manhattan_forecast_plot(
    predictions_frame: pd.DataFrame, output_path: Path
) -> None:
    subset = predictions_frame[predictions_frame["Borough"] == "Manhattan"].copy()
    subset = subset.sort_values("pickup_hour").tail(72)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(subset["pickup_hour"], subset["pickup_count"], label="Actual", linewidth=2)
    axis.plot(subset["pickup_hour"], subset["prediction"], label="Predicted", linewidth=2)
    axis.set_title("Manhattan Hourly Pickup Demand: Last 72 Test Hours")
    axis.set_xlabel("Pickup Hour")
    axis.set_ylabel("Pickup Count")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _score(
    forecast: pd.DataFrame,
    borough: str,
    model_name: str,
    train_rows: int,
) -> dict:
    return {
        "borough": borough,
        "model": model_name,
        "train_rows": train_rows,
        "test_rows": len(forecast),
        "mae": round(mae(forecast["pickup_count"], forecast["prediction"]), 3),
        "rmse": round(rmse(forecast["pickup_count"], forecast["prediction"]), 3),
    }


def run_taxi_analysis() -> None:
    """End-to-end pipeline: read parquet -> hourly demand -> baselines -> outputs."""
    sns.set_theme(style="whitegrid")

    root = _project_root()
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    figures_dir = root / "docs" / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = [
        data_dir / "yellow_tripdata_2025-01.parquet",
        data_dir / "yellow_tripdata_2025-02.parquet",
    ]
    zones_path = data_dir / "taxi_zone_lookup.csv"

    missing = [str(p) for p in parquet_paths + [zones_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data files; run `python scripts/download_data.py` first.\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    trips = load_yellow_taxi_data(parquet_paths)
    zones = load_zone_lookup(zones_path)
    borough_trips = attach_pickup_borough(trips, zones)
    borough_trips = borough_trips[
        borough_trips["Borough"].isin(_TARGET_BOROUGHS)
    ]

    hourly = aggregate_hourly_pickups(borough_trips)
    hourly = fill_missing_hours(hourly)

    metrics = []
    all_predictions = []
    for borough, subset in hourly.groupby("Borough", observed=True):
        train, test = train_test_split_by_time(subset, test_hours=_TEST_HOURS)

        hourly_forecast = hourly_profile_baseline(train, test)
        seasonal_forecast = seasonal_naive_baseline(train, test)

        metrics.append(_score(hourly_forecast, borough, "hourly_profile", len(train)))
        metrics.append(
            _score(seasonal_forecast, borough, "seasonal_naive_24h", len(train))
        )

        all_predictions.append(
            hourly_forecast.assign(
                model="hourly_profile",
                Borough=borough,
                prediction=lambda df: df["prediction"].round(3),
            )
        )
        all_predictions.append(
            seasonal_forecast.assign(
                model="seasonal_naive_24h",
                Borough=borough,
                prediction=lambda df: df["prediction"].round(3),
            )
        )

    metrics_frame = (
        pd.DataFrame(metrics)
        .sort_values(["borough", "mae"])
        .reset_index(drop=True)
    )
    predictions_frame = pd.concat(all_predictions, ignore_index=True)

    metrics_path = processed_dir / "borough_model_metrics.csv"
    predictions_path = processed_dir / "borough_model_predictions.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    predictions_frame.to_csv(predictions_path, index=False)

    _save_metric_plot(metrics_frame, figures_dir / "borough_model_comparison.png")
    _save_manhattan_forecast_plot(
        predictions_frame[predictions_frame["model"] == "seasonal_naive_24h"],
        figures_dir / "manhattan_forecast_last_72_hours.png",
    )

    print("Model comparison metrics by borough:")
    print(metrics_frame.to_string(index=False))
    print(f"\nSaved metrics to:     {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved figures to:     {figures_dir}")


if __name__ == "__main__":
    run_taxi_analysis()
