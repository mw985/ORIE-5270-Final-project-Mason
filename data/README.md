# Data Folder

This folder is populated at install time by running:

```bash
python scripts/download_data.py
```

The script downloads:

- `yellow_tripdata_2025-01.parquet` (NYC TLC yellow taxi trip records)
- `yellow_tripdata_2025-02.parquet`
- `taxi_zone_lookup.csv` (TLC zone -> borough mapping)

Files in this folder are **gitignored** because they are large and easily
re-downloadable. Generated tabular outputs from `run_taxi_analysis.py` land in
`data/processed/` and are also gitignored. The small PNG figures are saved in
`docs/figures/` so they can be included in the GitHub submission.
