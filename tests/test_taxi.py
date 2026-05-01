"""Unit tests for the taxi module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from orie5270_project.taxi import (
    aggregate_hourly_pickups,
    attach_pickup_borough,
    fill_missing_hours,
    hourly_profile_baseline,
    load_yellow_taxi_data,
    load_zone_lookup,
    seasonal_naive_baseline,
    train_test_split_by_time,
)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_load_zone_lookup_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "zones.csv"
    pd.DataFrame(
        {
            "LocationID": [1, 2],
            "Borough": ["Manhattan", "Queens"],
            "Zone": ["Midtown", "Astoria"],
            "service_zone": ["Yellow Zone", "Boro Zone"],
        }
    ).to_csv(csv_path, index=False)

    zones = load_zone_lookup(csv_path)

    assert list(zones["Borough"]) == ["Manhattan", "Queens"]
    assert "LocationID" in zones.columns


def test_load_zone_lookup_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad_zones.csv"
    pd.DataFrame({"LocationID": [1], "Zone": ["Midtown"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_zone_lookup(csv_path)


def test_load_yellow_taxi_data_concatenates_parquet(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-01 00:00:00"]),
            "PULocationID": [1],
            "extra": ["ignored"],
        }
    ).to_parquet(path_a, index=False)
    pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-02 00:00:00"]),
            "PULocationID": [2],
            "extra": ["ignored"],
        }
    ).to_parquet(path_b, index=False)

    trips = load_yellow_taxi_data([path_a, path_b])

    assert len(trips) == 2
    assert set(trips.columns) == {"tpep_pickup_datetime", "PULocationID"}


def test_load_yellow_taxi_data_rejects_empty_paths() -> None:
    with pytest.raises(ValueError, match="At least one parquet file"):
        load_yellow_taxi_data([])


# ---------------------------------------------------------------------------
# Borough mapping
# ---------------------------------------------------------------------------


def test_attach_pickup_borough_maps_unknown_values() -> None:
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 01:00:00"]
            ),
            "PULocationID": [1, 999],
        }
    )
    zones = pd.DataFrame(
        {"LocationID": [1], "Borough": ["EWR"], "Zone": ["Newark Airport"]}
    )

    merged = attach_pickup_borough(trips, zones)

    assert list(merged["Borough"]) == ["EWR", "Unknown"]
    assert "LocationID" not in merged.columns


def test_attach_pickup_borough_validates_inputs() -> None:
    bad_trips = pd.DataFrame({"PULocationID": [1]})  # no datetime column
    zones = pd.DataFrame(
        {"LocationID": [1], "Borough": ["Manhattan"], "Zone": ["Midtown"]}
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        attach_pickup_borough(bad_trips, zones)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_hourly_pickups_counts_rows_per_hour_and_group() -> None:
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2025-01-01 00:05:00",
                    "2025-01-01 00:45:00",
                    "2025-01-01 01:05:00",
                ]
            ),
            "Borough": ["Manhattan", "Manhattan", "Queens"],
        }
    )

    hourly = aggregate_hourly_pickups(trips)

    assert hourly.to_dict("records") == [
        {
            "Borough": "Manhattan",
            "pickup_hour": pd.Timestamp("2025-01-01 00:00:00"),
            "pickup_count": 2,
        },
        {
            "Borough": "Queens",
            "pickup_hour": pd.Timestamp("2025-01-01 01:00:00"),
            "pickup_count": 1,
        },
    ]


def test_fill_missing_hours_inserts_zero_rows() -> None:
    hourly = pd.DataFrame(
        {
            "Borough": ["Manhattan", "Manhattan"],
            "pickup_hour": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 02:00:00"]
            ),
            "pickup_count": [5, 7],
        }
    )

    filled = fill_missing_hours(hourly)

    assert list(filled["pickup_count"]) == [5, 0, 7]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_train_test_split_by_time_holds_out_last_hours() -> None:
    hourly = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01", periods=10, freq="h"),
            "pickup_count": range(10),
        }
    )

    train, test = train_test_split_by_time(hourly, test_hours=3)

    assert len(train) == 7
    assert len(test) == 3
    assert test["pickup_hour"].iloc[0] == pd.Timestamp("2025-01-01 07:00:00")


def test_train_test_split_by_time_rejects_bad_test_hours() -> None:
    hourly = pd.DataFrame(
        {"pickup_hour": pd.date_range("2025-01-01", periods=5, freq="h")}
    )

    with pytest.raises(ValueError, match="positive and smaller"):
        train_test_split_by_time(hourly, test_hours=0)
    with pytest.raises(ValueError, match="positive and smaller"):
        train_test_split_by_time(hourly, test_hours=10)


# ---------------------------------------------------------------------------
# Forecasting baselines
# ---------------------------------------------------------------------------


def test_hourly_profile_baseline_uses_group_hour_average() -> None:
    train = pd.DataFrame(
        {
            "Borough": ["Manhattan", "Manhattan", "Manhattan", "Queens"],
            "pickup_hour": pd.to_datetime(
                [
                    "2025-01-01 08:00:00",
                    "2025-01-02 08:00:00",
                    "2025-01-01 09:00:00",
                    "2025-01-01 08:00:00",
                ]
            ),
            "pickup_count": [10, 14, 20, 8],
        }
    )
    test = pd.DataFrame(
        {
            "Borough": ["Manhattan", "Queens"],
            "pickup_hour": pd.to_datetime(
                ["2025-01-03 08:00:00", "2025-01-03 09:00:00"]
            ),
            "pickup_count": [12, 9],
        }
    )

    predicted = hourly_profile_baseline(train, test)

    assert predicted.loc[0, "prediction"] == 12  # mean of Manhattan 8 AM (10, 14)
    # Queens has no 9 AM in train, falls back to Queens group mean (8)
    assert predicted.loc[1, "prediction"] == 8


def test_seasonal_naive_baseline_uses_previous_day_same_hour() -> None:
    train = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01 00:00:00", periods=48, freq="h"),
            "pickup_count": list(range(48)),
        }
    )
    test = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-03 00:00:00", periods=3, freq="h"),
            "pickup_count": [100, 101, 102],
        }
    )

    predicted = seasonal_naive_baseline(train, test)

    assert list(predicted["prediction"]) == [24, 25, 26]


def test_seasonal_naive_baseline_walks_back_to_train_only() -> None:
    """When the test horizon exceeds the seasonal period, we must keep
    walking back through multiples of the lag rather than reading from the
    test set itself.
    """
    train = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01 00:00:00", periods=48, freq="h"),
            "pickup_count": list(range(48)),
        }
    )
    # Test starts on Jan 4 (two days after train ends), three hours.
    test = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-04 00:00:00", periods=3, freq="h"),
            "pickup_count": [999, 999, 999],
        }
    )

    predicted = seasonal_naive_baseline(train, test)

    # Jan 4 00:00 - 24h = Jan 3 00:00 (not in train) - 24h = Jan 2 00:00 (in
    # train, value = 24). Same idea for the other two hours.
    assert list(predicted["prediction"]) == [24, 25, 26]


def test_seasonal_naive_baseline_rejects_non_positive_lag() -> None:
    train = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01", periods=3, freq="h"),
            "pickup_count": [1, 2, 3],
        }
    )
    test = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01 03:00:00", periods=1, freq="h"),
            "pickup_count": [4],
        }
    )

    with pytest.raises(ValueError, match="positive"):
        seasonal_naive_baseline(train, test, seasonal_lag_hours=0)


def test_seasonal_naive_baseline_falls_back_to_train_mean() -> None:
    """If no historical match exists at any seasonal lag, use the train mean."""
    train = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-10 00:00:00", periods=24, freq="h"),
            "pickup_count": [10] * 24,
        }
    )
    # Test is BEFORE train - no possible lag lookup hits train.
    test = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01 00:00:00", periods=2, freq="h"),
            "pickup_count": [0, 0],
        }
    )

    predicted = seasonal_naive_baseline(train, test)

    assert list(predicted["prediction"]) == [10.0, 10.0]
