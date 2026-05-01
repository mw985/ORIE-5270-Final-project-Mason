"""Tests for the command-line taxi analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from orie5270_project import cli


def _write_minimal_taxi_inputs(root: Path, periods: int = 220) -> None:
    data_dir = root / "data"
    data_dir.mkdir()

    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.date_range(
                "2025-01-01 00:00:00", periods=periods, freq="h"
            ),
            "PULocationID": [1] * periods,
        }
    )
    midpoint = periods // 2
    trips.iloc[:midpoint].to_parquet(
        data_dir / "yellow_tripdata_2025-01.parquet", index=False
    )
    trips.iloc[midpoint:].to_parquet(
        data_dir / "yellow_tripdata_2025-02.parquet", index=False
    )

    pd.DataFrame(
        {
            "LocationID": [1],
            "Borough": ["Manhattan"],
            "Zone": ["Test Zone"],
            "service_zone": ["Yellow Zone"],
        }
    ).to_csv(data_dir / "taxi_zone_lookup.csv", index=False)


def test_score_reports_actual_train_and_test_rows() -> None:
    forecast = pd.DataFrame(
        {
            "pickup_count": [10, 12, 14],
            "prediction": [9, 12, 16],
        }
    )

    result = cli._score(forecast, "Manhattan", "demo", train_rows=52)

    assert result["train_rows"] == 52
    assert result["test_rows"] == 3
    assert result["mae"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(1.291, abs=0.001)


def test_run_taxi_analysis_writes_outputs(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_minimal_taxi_inputs(tmp_path)
    monkeypatch.setattr(cli, "_project_root", lambda: tmp_path)

    cli.run_taxi_analysis()

    metrics_path = tmp_path / "data" / "processed" / "borough_model_metrics.csv"
    predictions_path = tmp_path / "data" / "processed" / "borough_model_predictions.csv"
    comparison_plot = tmp_path / "docs" / "figures" / "borough_model_comparison.png"
    forecast_plot = tmp_path / "docs" / "figures" / "manhattan_forecast_last_72_hours.png"

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["model"]) == {"hourly_profile", "seasonal_naive_24h"}
    assert set(metrics["train_rows"]) == {52}
    assert set(metrics["test_rows"]) == {168}
    assert predictions_path.exists()
    assert comparison_plot.exists()
    assert forecast_plot.exists()
    assert "Model comparison metrics by borough" in capsys.readouterr().out


def test_run_taxi_analysis_explains_missing_data(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_project_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="scripts/download_data.py"):
        cli.run_taxi_analysis()
