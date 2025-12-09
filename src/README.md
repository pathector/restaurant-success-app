# Source Code

Python modules that build features, train models, and support the Streamlit app.

- `data/merge_sources.py` – documented merger for Google + OSM supply (standardize, dedupe, spatial/name link). Final merged CSVs are already in `data/final/`; reuse or extend this stub if you need to re-run matching.
- `features/build_features.py` – main feature builder (LV95 projection, microlocation buffers, nearest distances, competition radius + KNN blocks, price/cuisine/establishment flags, city dummies, log transforms). Defaults read `data/final/` merged tables and write `data/final/features_location_competition.parquet`.
- `models/train_models.py` – orchestrates Model 1/2 training (random search + GroupKFold CV). `train_random_forest.py` provides the simpler baseline invoked by `run_all.sh`.
- `models/train_survival_rsf.py` – trains the RandomSurvivalForest on `data/final/closure_survival_dataset.parquet`.
- `models/train_review_dynamics.py` – city-specific recent-performance regressors/classifiers using pre-period signals.
- `models/train_monthly_seasonal.py` – LightGBM regressors for next-month counts with lags + seasonal terms (Interlaken/Sion).
- `app_utils.py` – utilities for the Streamlit app (grid loaders, LV95<->WGS transforms, KDTree nearest-grid lookup, geocoding, geojson conversion).
- `utils/` – placeholder for shared helpers (none required for the current release).

Scripts under `scripts/` call these modules via `python -m ...` or direct imports; consult `scripts/README.md` for CLI parameters.***
