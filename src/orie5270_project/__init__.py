"""Utilities for the ORIE 5270 final project.

Public API is re-exported from this package for convenience.
"""

from .dataset import (
    summarize_missing_values,
    time_based_train_test_split,
    validate_required_columns,
)
from .metrics import mae, mape, rmse
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "aggregate_hourly_pickups",
    "attach_pickup_borough",
    "fill_missing_hours",
    "hourly_profile_baseline",
    "load_yellow_taxi_data",
    "load_zone_lookup",
    "mae",
    "mape",
    "rmse",
    "seasonal_naive_baseline",
    "summarize_missing_values",
    "time_based_train_test_split",
    "train_test_split_by_time",
    "validate_required_columns",
]
