"""Run the borough-level NYC yellow-taxi demand analysis.

This is a thin wrapper that delegates to ``orie5270_project.cli`` so the
same entrypoint is also exposed as the ``orie5270-run-taxi-analysis``
console script when the package is installed with pip.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a fresh clone without ``pip install -e .``.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orie5270_project.cli import run_taxi_analysis  # noqa: E402


if __name__ == "__main__":
    run_taxi_analysis()
