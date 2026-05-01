"""Unit tests for the metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from orie5270_project.metrics import mae, mape, rmse


def test_mae_returns_expected_value() -> None:
    assert mae([1, 2, 3], [1, 4, 5]) == pytest.approx(4 / 3)


def test_mae_handles_perfect_predictions() -> None:
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_returns_expected_value() -> None:
    assert rmse([1, 2], [1, 4]) == pytest.approx(2**0.5)


def test_rmse_penalizes_outliers_more_than_mae() -> None:
    actual = [0, 0, 0, 0]
    predicted = [0, 0, 0, 10]

    assert rmse(actual, predicted) > mae(actual, predicted)


def test_mape_returns_expected_value() -> None:
    assert mape([2, 4], [1, 5]) == pytest.approx(0.375)


def test_mape_rejects_zero_actual_values() -> None:
    with pytest.raises(ValueError, match="contain zero"):
        mape([0, 1], [0, 1])


def test_metrics_reject_multidimensional_inputs() -> None:
    actual = np.array([[1, 2], [3, 4]])
    predicted = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="one-dimensional"):
        mae(actual, predicted)
    with pytest.raises(ValueError, match="one-dimensional"):
        rmse(actual, predicted)
    with pytest.raises(ValueError, match="one-dimensional"):
        mape(actual, predicted)
