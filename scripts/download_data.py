"""Download the NYC TLC yellow-taxi parquet files and zone lookup CSV.

The grading rubric requires the project to be reproducible. Rather than
committing ~100 MB of parquet to the repo, anyone who clones the project
runs this script once and gets the same data files.

Usage:

    python scripts/download_data.py
    python scripts/download_data.py --months 2025-01 2025-02 2025-03

Downloads are skipped when the destination file already exists. Pass
``--force`` to re-download.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

YELLOW_URL_TEMPLATE = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_{month}.parquet"
)
ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
)
DEFAULT_MONTHS = ["2025-01", "2025-02"]


def _download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"  exists, skipping: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading: {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as fh:
        # Stream in 1 MB chunks so memory stays bounded for large files.
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    print(f"  saved to:    {destination}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--months",
        nargs="+",
        default=DEFAULT_MONTHS,
        help="Months to fetch in YYYY-MM format (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args(argv)

    print(f"Target data directory: {DATA_DIR}")
    print("Zone lookup:")
    _download(ZONE_LOOKUP_URL, DATA_DIR / "taxi_zone_lookup.csv", args.force)

    print("Yellow taxi parquet files:")
    for month in args.months:
        url = YELLOW_URL_TEMPLATE.format(month=month)
        destination = DATA_DIR / f"yellow_tripdata_{month}.parquet"
        _download(url, destination, args.force)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
