"""Small demo script showing how package utilities can be used."""

from __future__ import annotations

import pandas as pd

from orie5270_project.dataset import summarize_missing_values, time_based_train_test_split
from orie5270_project.metrics import mae, rmse


def main() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
            "demand": [10, 12, 13, 12, 15, 16],
            "weather_score": [1.0, 0.9, None, 0.8, 0.7, 0.6],
        }
    )

    missing_summary = summarize_missing_values(frame)
    train, test = time_based_train_test_split(frame, "timestamp", train_fraction=0.67)

    baseline_prediction = [train["demand"].mean()] * len(test)
    print("Missing-value summary:")
    print(missing_summary)
    print("\nTrain rows:", len(train))
    print("Test rows:", len(test))
    print("Baseline MAE:", round(mae(test["demand"], baseline_prediction), 3))
    print("Baseline RMSE:", round(rmse(test["demand"], baseline_prediction), 3))


if __name__ == "__main__":
    main()