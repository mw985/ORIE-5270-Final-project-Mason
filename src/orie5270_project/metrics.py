"""Common regression metrics for forecasting experiments."""

from __future__ import annotations

import numpy as np


def _to_numpy(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    return array


def mae(actual, predicted) -> float:
    actual_array = _to_numpy(actual)
    predicted_array = _to_numpy(predicted)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def rmse(actual, predicted) -> float:
    actual_array = _to_numpy(actual)
    predicted_array = _to_numpy(predicted)
    return float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))


def mape(actual, predicted) -> float:
    actual_array = _to_numpy(actual)
    predicted_array = _to_numpy(predicted)
    if np.any(actual_array == 0):
        raise ValueError("MAPE is undefined when actual values contain zero.")
    return float(np.mean(np.abs((actual_array - predicted_array) / actual_array)))