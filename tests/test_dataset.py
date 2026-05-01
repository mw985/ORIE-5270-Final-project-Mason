"""Unit tests for the dataset module."""

from __future__ import annotations

import pandas as pd
import pytest

from orie5270_project.dataset import (
    summarize_missing_values,
    time_based_train_test_split,
    validate_required_columns,
)


def test_validate_required_columns_passes_when_all_present() -> None:
    frame = pd.DataFrame({"a": [1], "b": [2]})

    # Should not raise.
    validate_required_columns(frame, ["a", "b"])


def test_validate_required_columns_raises_for_missing_columns() -> None:
    frame = pd.DataFrame({"a": [1], "b": [2]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(frame, ["a", "c"])


def test_summarize_missing_values_returns_counts_and_rates() -> None:
    frame = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 2]})

    summary = summarize_missing_values(frame)

    assert summary.loc["a", "missing_count"] == 1
    assert summary.loc["a", "missing_rate"] == pytest.approx(1 / 3)
    assert summary.loc["b", "missing_count"] == 2
    assert summary.loc["b", "missing_rate"] == pytest.approx(2 / 3)


def test_summarize_missing_values_handles_empty_frame() -> None:
    frame = pd.DataFrame({"a": [], "b": []})

    summary = summarize_missing_values(frame)

    assert summary.loc["a", "missing_count"] == 0
    assert summary.loc["b", "missing_count"] == 0


def test_time_based_train_test_split_sorts_before_splitting() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"]
            ),
            "value": [3, 1, 2],
        }
    )

    train, test = time_based_train_test_split(frame, "timestamp", train_fraction=2 / 3)

    assert list(train["value"]) == [1, 2]
    assert list(test["value"]) == [3]


def test_time_based_train_test_split_rejects_bad_fraction() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"])}
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        time_based_train_test_split(frame, "timestamp", train_fraction=1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        time_based_train_test_split(frame, "timestamp", train_fraction=0.0)


def test_time_based_train_test_split_rejects_tiny_frames() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"])})

    with pytest.raises(ValueError, match="at least two rows"):
        time_based_train_test_split(frame, "timestamp", train_fraction=0.5)


def test_time_based_train_test_split_guarantees_nonempty_split() -> None:
    """Even with a very lopsided fraction, both partitions must have rows."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "value": [1, 2, 3, 4],
        }
    )

    train, test = time_based_train_test_split(
        frame, "timestamp", train_fraction=0.99
    )

    assert len(train) >= 1
    assert len(test) >= 1


def test_time_based_train_test_split_requires_time_column() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing required columns"):
        time_based_train_test_split(frame, "timestamp", train_fraction=0.5)
