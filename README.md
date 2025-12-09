# Restaurant Success – Cloud Run app

Lean copy of the Streamlit app plus supporting `src/` utilities. Data artifacts are kept in GCS to stay under GitHub storage limits.

## Quick start (local)
```bash
cd restaurant-success-app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# pull data (needs gcloud + access to the project)
GCS_DATA_BUCKET=restaurant-success-data-gen-lang-client-0301771744 \
  bash scripts/download_data.sh

streamlit run streamlit_app.py
```

## What’s included
- `streamlit_app.py` – production app (identical to the final submission).
- `src/` – app utilities (data loading, geo helpers, scoring).
- `scripts/download_data.sh` – syncs `data/processed` from GCS into the repo before running or building.
- `Dockerfile` – runs the app on port 8080 (Cloud Run ready).
- `data/processed/.gitkeep` – placeholder; actual parquet files are in GCS.

## Required data (in GCS)
Bucket: `gs://restaurant-success-data-gen-lang-client-0301771744`
Path: `data/processed` containing `grid_concepts_*`, `grid_best_*`, `area_perf.parquet`, review CSVs, and the survival dataset. All Swiss cities in the app come from these files.

## Deploy (overview)
1. Sync data from GCS before building the image (GitHub Action handles this in CI).
2. Build and push the image to Artifact Registry.
3. Deploy to Cloud Run with port `8080` and `--allow-unauthenticated`.

See `.github/workflows/deploy.yml` for the exact CI pipeline.***
