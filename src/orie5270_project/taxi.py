"""Taxi-demand utilities for the ORIE 5270 final project.

This module turns raw NYC TLC yellow-taxi trip records into an hourly
borough-level demand time series and provides simple, leakage-free
forecasting baselines that can be unit tested in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .dataset import validate_required_columns

DEFAULT_TAXI_COLUMNS = ["tpep_pickup_datetime", "PULocationID"]


def load_zone_lookup(path: str | Path) -> pd.DataFrame:
    """Load the TLC taxi zone lookup table.

    Parameters
    ----------
    path:
        Filesystem path to ``taxi_zone_lookup.csv``.

    Returns
    -------
    pandas.DataFrame
        Frame with at least ``LocationID``, ``Borough`` and ``Zone`` columns.
    """
    zones = pd.read_csv(path)
    validate_required_columns(zones, ["LocationID", "Borough", "Zone"])
    return zones


def load_yellow_taxi_data(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate yellow-taxi parquet files.

    Only the columns required by downstream aggregation are read in order
    to keep memory usage low.

    Parameters
    ----------
    paths:
        Iterable of parquet file paths.

    Returns
    -------
    pandas.DataFrame
        Concatenated trip records.

    Raises
    ------
    ValueError
        If ``paths`` is empty.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("At least one parquet file is required.")
    frames = [pd.read_parquet(path, columns=DEFAULT_TAXI_COLUMNS) for path in paths]
    return pd.concat(frames, ignore_index=True)


def attach_pickup_borough(trips: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """Map pickup ``LocationID`` values to borough names.

    Trips with location IDs missing from ``zones`` are tagged ``"Unknown"``.
    """
    validate_required_columns(trips, DEFAULT_TAXI_COLUMNS)
    validate_required_columns(zones, ["LocationID", "Borough"])

    merged = trips.merge(
        zones[["LocationID", "Borough"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    )
    merged["Borough"] = merged["Borough"].fillna("Unknown")
    return merged.drop(columns=["LocationID"])


def aggregate_hourly_pickups(
    trips: pd.DataFrame,
    group_column: str = "Borough",
) -> pd.DataFrame:
    """Aggregate raw trip rows into hourly pickup counts per group."""
    validate_required_columns(trips, ["tpep_pickup_datetime", group_column])

    frame = trips.copy()
    frame["pickup_hour"] = pd.to_datetime(frame["tpep_pickup_datetime"]).dt.floor("h")
    hourly = (
        frame.groupby([group_column, "pickup_hour"], observed=True)
        .size()
        .reset_index(name="pickup_count")
        .sort_values([group_column, "pickup_hour"])
        .reset_index(drop=True)
    )
    return hourly


def fill_missing_hours(hourly: pd.DataFrame, group_column: str = "Borough") -> pd.DataFrame:
    """Insert zero-count rows for hours with no observed pickups in each group."""
    validate_required_columns(hourly, [group_column, "pickup_hour", "pickup_count"])

    groups = []
    for group_value, subset in hourly.groupby(group_column, observed=True):
        full_hours = pd.date_range(
            subset["pickup_hour"].min(),
            subset["pickup_hour"].max(),
            freq="h",
        )
        expanded = (
            subset.set_index("pickup_hour")
            .reindex(full_hours)
            .rename_axis("pickup_hour")
            .reset_index()
        )
        expanded[group_column] = group_value
        expanded["pickup_count"] = expanded["pickup_count"].fillna(0).astype(int)
        groups.append(expanded[[group_column, "pickup_hour", "pickup_count"]])

    return (
        pd.concat(groups, ignore_index=True)
        .sort_values([group_column, "pickup_hour"])
        .reset_index(drop=True)
    )


def train_test_split_by_time(
    hourly: pd.DataFrame,
    time_column: str = "pickup_hour",
    test_hours: int = 24 * 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent ``test_hours`` rows for testing.

    Parameters
    ----------
    hourly:
        Time-indexed dataframe (one row per hour and group).
    time_column:
        Column to sort by before splitting.
    test_hours:
        Number of trailing rows to assign to the test set.

    Raises
    ------
    ValueError
        If ``test_hours`` is not strictly between 0 and ``len(hourly)``.
    """
    validate_required_columns(hourly, [time_column])
    if test_hours <= 0 or test_hours >= len(hourly):
        raise ValueError("test_hours must be positive and smaller than the dataset size.")

    ordered = hourly.sort_values(time_column).reset_index(drop=True)
    split_index = len(ordered) - test_hours
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def hourly_profile_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_column: str = "Borough",
    time_column: str = "pickup_hour",
    target_column: str = "pickup_count",
) -> pd.DataFrame:
    """Predict each test hour as the historical mean for its (group, hour-of-day).

    Falls back first to the group-level mean and finally to the global train
    mean when a (group, hour) combination is absent from training.
    """
    validate_required_columns(train, [group_column, time_column, target_column])
    validate_required_columns(test, [group_column, time_column, target_column])

    train = train.copy()
    test = test.copy()
    train["hour_of_day"] = pd.to_datetime(train[time_column]).dt.hour
    test["hour_of_day"] = pd.to_datetime(test[time_column]).dt.hour

    hourly_means = (
        train.groupby([group_column, "hour_of_day"], observed=True)[target_column]
        .mean()
        .rename("prediction")
        .reset_index()
    )
    group_means = (
        train.groupby(group_column, observed=True)[target_column]
        .mean()
        .rename("group_mean")
        .reset_index()
    )
    global_mean = float(train[target_column].mean())

    predicted = test.merge(hourly_means, on=[group_column, "hour_of_day"], how="left")
    predicted = predicted.merge(group_means, on=group_column, how="left")
    predicted["prediction"] = (
        predicted["prediction"].fillna(predicted["group_mean"]).fillna(global_mean)
    )
    return predicted.drop(columns=["group_mean"])


def seasonal_naive_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seasonal_lag_hours: int = 24,
    time_column: str = "pickup_hour",
    target_column: str = "pickup_count",
) -> pd.DataFrame:
    """Strict seasonal-naive forecast that uses ONLY training data.

    For each test timestamp ``t`` we look up the most recent training
    timestamp ``t - k * seasonal_lag_hours`` (with the smallest ``k >= 1``)
    that exists in the training set, and use its value as the prediction.
    This avoids leaking test labels into later test predictions, which is
    important when the test horizon is longer than the seasonal period.

    Parameters
    ----------
    train, test:
        Frames containing ``time_column`` and ``target_column``.
    seasonal_lag_hours:
        Length of the seasonal cycle in hours (default ``24``).
    time_column, target_column:
        Names of the timestamp and target columns.

    Raises
    ------
    ValueError
        If ``seasonal_lag_hours`` is not positive.
    """
    validate_required_columns(train, [time_column, target_column])
    validate_required_columns(test, [time_column, target_column])
    if seasonal_lag_hours <= 0:
        raise ValueError("seasonal_lag_hours must be positive.")

    # Use a plain dict keyed by Timestamp for O(1) hash lookups; this keeps
    # the loop below cheap even for long test horizons.
    train_times = pd.to_datetime(train[time_column])
    train_lookup: dict = dict(zip(train_times, train[target_column]))
    train_min_time = train_times.min()

    fallback = float(train[target_column].mean())
    lag = pd.Timedelta(hours=seasonal_lag_hours)

    predictions = []
    for ts in pd.to_datetime(test[time_column]).tolist():
        # Walk back in multiples of the seasonal period until we land in train.
        candidate = ts - lag
        value: float | None = None
        # Bound the loop so we never spin forever; one full year of hourly
        # lookups is far more than enough for any realistic test horizon.
        for _ in range(8760):
            if candidate in train_lookup:
                value = float(train_lookup[candidate])
                break
            if candidate < train_min_time:
                break
            candidate = candidate - lag
        predictions.append(fallback if value is None else value)

    result = test.copy()
    result["prediction"] = predictions
    return result
