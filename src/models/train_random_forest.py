"""Train the all-cities success model (Model 1) using RandomForestRegressor.

This script expects the feature table produced by ``src.features.build_features``.
It filters to restaurants with Google review counts, builds the feature matrix
based on naming patterns (location, competition, restaurant attributes), trains
the RandomForestRegressor, evaluates it, and saves the model plus diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

DEFAULT_FEATURES_PATH = Path("data/final/features_location_competition.parquet")
DEFAULT_MODEL_PATH = Path("models/model1_random_forest.pkl")
DEFAULT_METRICS_PATH = Path("artifacts/model1_metrics.json")
DEFAULT_IMPORTANCE_PATH = Path("artifacts/model1_feature_importances.csv")

NON_FEATURE_COLUMNS = {
    "id",
    "display_name",
    "formatted_address",
    "latitude",
    "longitude",
    "google_maps_uri",
    "business_status",
    "price_level",
    "types",
    "locality",
    "admin_area_level_1",
    "postal_code",
    "country",
    "country_code",
    "postal_address_lines",
    "weekday_descriptions",
    "accessibility_options",
    "__SelectedCity__",
    "osm_id",
    "primary_amenity",
    "name_osm",
    "name_google",
    "closest_city",
    "inside_any_city_bounds",
    "y_success",
    "target_log_reviews",
}

PREFIX_FEATURES: Sequence[str] = (
    "log_",
    "dist_",
    "nn_",
    "comp_",
    "comp200",
    "cuisine_",
    "is_",
    "city_",
    "parking_",
    "busstop_",
    "stations_",
    "offices_",
    "hotels_",
    "schools_",
    "attractions_",
    "supermarkets_",
    "gyms_",
)
EXACT_FEATURES: Sequence[str] = ("price_level_num",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model 1 RandomForestRegressor for restaurant success.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to the feature table (parquet or csv).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to store the fitted RandomForestRegressor.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to store evaluation metrics (JSON).",
    )
    parser.add_argument(
        "--importance-path",
        type=Path,
        default=DEFAULT_IMPORTANCE_PATH,
        help="Where to store feature importances (CSV).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for evaluation.",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=3,
        help="Minimum number of Google reviews required for inclusion.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees in the RandomForest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max depth for RandomForestRegressor (None = unrestricted).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=5,
        help="Minimum samples per leaf.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallelism for RandomForestRegressor.",
    )
    parser.add_argument(
        "--extra-features",
        nargs="*",
        default=(),
        help="Optional explicit feature column names to include.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="__SelectedCity__",
        help="Column used for grouped cross-validation (default: __SelectedCity__).",
    )
    parser.add_argument(
        "--group-cv-folds",
        type=int,
        default=0,
        help="Number of GroupKFold splits for evaluation (0 disables grouped CV).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logger verbosity.",
    )
    return parser.parse_args()


def load_features_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".tsv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature format for {path}")


def select_feature_columns(df: pd.DataFrame, extra: Iterable[str]) -> List[str]:
    columns: List[str] = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col in EXACT_FEATURES:
            columns.append(col)
            continue
        if any(col.startswith(prefix) for prefix in PREFIX_FEATURES):
            columns.append(col)
    for col in extra:
        if col in df.columns and col not in columns:
            columns.append(col)
    if not columns:
        raise ValueError("No feature columns were selected – check feature prefixes or inputs.")
    return columns


def ensure_output_dirs(paths: Sequence[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Loading feature table from %s", args.features_path)
    df = load_features_table(args.features_path)

    labelled = df[df["user_rating_count"].notna()].copy()
    if args.min_reviews > 0:
        labelled = labelled[labelled["user_rating_count"] >= args.min_reviews]
    if labelled.empty:
        raise ValueError("No labelled rows remain after filtering – reduce min-reviews or inspect data.")

    labelled["target_log_reviews"] = np.log1p(labelled["user_rating_count"])

    feature_cols = select_feature_columns(labelled, args.extra_features)
    logging.info("Using %d feature columns.", len(feature_cols))

    X = labelled[feature_cols].fillna(0)
    y = labelled["target_log_reviews"].to_numpy()
    if args.group_column in labelled.columns:
        groups = labelled[args.group_column].fillna("Unknown").astype(str).reset_index(drop=True)
    elif "__SelectedCity__" in labelled.columns:
        groups = labelled["__SelectedCity__"].fillna("Unknown").astype(str).reset_index(drop=True)
    elif "locality" in labelled.columns:
        groups = labelled["locality"].fillna("Unknown").astype(str).reset_index(drop=True)
    else:
        groups = pd.Series(["Unknown"] * len(labelled))

    group_cv_results = []
    if args.group_cv_folds and args.group_cv_folds >= 2:
        logging.info("Running grouped %d-fold CV using column '%s'.", args.group_cv_folds, args.group_column)
        gkf = GroupKFold(n_splits=args.group_cv_folds)
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            model_cv = RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                n_jobs=args.n_jobs,
                random_state=args.random_state + fold_idx,
                oob_score=False,
            )
            model_cv.fit(X.iloc[train_idx], y[train_idx])
            preds = model_cv.predict(X.iloc[test_idx])
            fold_metrics = compute_metrics(y[test_idx], preds)
            fold_metrics["fold"] = fold_idx
            group_cv_results.append(fold_metrics)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        oob_score=False,
    )

    logging.info(
        "Training RandomForestRegressor on %d rows (%d features)…",
        X_train.shape[0],
        X_train.shape[1],
    )
    model.fit(X_train, y_train)

    train_metrics = compute_metrics(y_train, model.predict(X_train))
    test_pred = model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_pred)

    metrics_payload = {
        "train": train_metrics,
        "test": test_metrics,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "min_reviews": args.min_reviews,
    }
    if group_cv_results:
        metrics_payload["group_cv"] = {
            "folds": group_cv_results,
            "mean": {
                "r2": float(np.mean([m["r2"] for m in group_cv_results])),
                "mae": float(np.mean([m["mae"] for m in group_cv_results])),
                "rmse": float(np.mean([m["rmse"] for m in group_cv_results])),
            },
            "n_folds": len(group_cv_results),
            "group_column": args.group_column,
        }

    ensure_output_dirs([args.model_path, args.metrics_path, args.importance_path])
    joblib.dump(model, args.model_path)
    logging.info("Saved model to %s", args.model_path)

    with args.metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)
    logging.info("Saved metrics to %s", args.metrics_path)

    importances = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(args.importance_path, index=False)
    logging.info("Saved feature importances to %s", args.importance_path)

    logging.info(
        "Model 1 complete – test R^2=%.3f, MAE=%.3f, RMSE=%.3f",
        test_metrics["r2"],
        test_metrics["mae"],
        test_metrics["rmse"],
    )


if __name__ == "__main__":
    main()
