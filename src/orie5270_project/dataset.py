"""Dataset utilities designed to be easy to test and reuse."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_required_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise ``ValueError`` when any of ``required_columns`` is missing.

    Parameters
    ----------
    frame:
        Dataframe to inspect.
    required_columns:
        Iterable of column names that must be present.
    """
    required = list(required_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def summarize_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing-value counts and rates."""
    total_rows = len(frame)
    counts = frame.isna().sum()
    rates = counts / total_rows if total_rows else counts.astype(float)
    return pd.DataFrame(
        {
            "missing_count": counts,
            "missing_rate": rates,
        }
    )


def time_based_train_test_split(
    frame: pd.DataFrame,
    time_column: str,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe chronologically to avoid leakage.

    Uses ``round`` on the split index rather than ``int`` truncation so the
    function is robust to floating-point representation issues (e.g.
    ``2/3 * 3`` evaluating to a value just below 2.0).

    Parameters
    ----------
    frame:
        Dataframe to split.
    time_column:
        Column to sort by before splitting.
    train_fraction:
        Fraction of rows to assign to the train set; must be strictly
        between 0 and 1.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(train, test)``, both copies, both at least one row.

    Raises
    ------
    ValueError
        If ``train_fraction`` is outside ``(0, 1)`` or the input is empty.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if len(frame) < 2:
        raise ValueError("Need at least two rows to perform a train/test split.")

    validate_required_columns(frame, [time_column])

    ordered = frame.sort_values(time_column).reset_index(drop=True)
    raw_split = round(len(ordered) * train_fraction)
    split_index = max(1, min(len(ordered) - 1, raw_split))
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()
