"""Train multiple models for existing restaurants and new locations.

Models:
- Model 1 (existing restaurants, regression): RandomForest + LightGBM
- Model 2 (new locations, classification): RandomForest + LightGBM

Each model uses GroupKFold (by city) random search for hyperparameters and
evaluates a random holdout split as a secondary check.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Paths
FEATURE_PATH = Path("data/final/features_location_competition.parquet")
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")
COVID_TS_FILES = [
    Path("data/final/reviews_timeseries_interlaken.csv"),
    Path("data/final/reviews_timeseries_sion.csv"),
]
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-12-31")
COVID_WEIGHT_ALPHA = 0.5

# Feature helpers
NUMERIC_PREFIXES = (
    "log_",
    "dist_",
    "comp",
    "nn_",
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
CATEGORICAL_PREFIXES = ("cuisine_", "is_", "city_")
MODEL2_EXCLUDE_COLS = {
    "rating",
    "rating_sq",
    "rating_centered",
    "user_rating_count",
    "city_review_threshold",
    "target_log_reviews",
    "success_class",
    "y_exist",
}
MODEL2_EXCLUDE_CONTAINS = ("rating_gap",)


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF and LightGBM models for existing and new locations.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=FEATURE_PATH,
        help="Path to the feature table produced by src.features.build_features.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["model1_rf", "model1_lgbm", "model2_rf", "model2_lgbm"],
        choices=["model1_rf", "model1_lgbm", "model2_rf", "model2_lgbm"],
        help="Select which models to train.",
    )
    parser.add_argument(
        "--random-search-iter",
        type=int,
        default=20,
        help="Number of parameter samples for RandomizedSearchCV per model.",
    )
    parser.add_argument(
        "--min-reviews-existing",
        type=int,
        default=3,
        help="Minimum number of reviews required for Model 1 training rows.",
    )
    parser.add_argument(
        "--success-rating-threshold",
        type=float,
        default=4.3,
        help="Rating threshold used in Model 2 success label.",
    )
    parser.add_argument(
        "--success-review-quantile",
        type=float,
        default=0.7,
        help="City quantile for review count used in Model 2 success label.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction for random train/test evaluation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Verbosity for the training script.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_covid_weight_map(alpha: float = COVID_WEIGHT_ALPHA) -> Dict[str, float]:
    weight_map: Dict[str, float] = {}
    for path in COVID_TS_FILES:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, parse_dates=["review_date"])
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to load %s (%s)", path, exc)
            continue
        if "place_id" not in df.columns:
            continue
        df["review_date"] = pd.to_datetime(df["review_date"], utc=True, errors="coerce").dt.tz_localize(None)
        if "review_rating" in df.columns:
            df = df.dropna(subset=["review_rating"])
            df["rating"] = df["review_rating"]
        elif "rating" in df.columns:
            df = df.dropna(subset=["rating"])
        else:
            continue
        total = df.groupby("place_id").size()
        covid = (
            df[(df["review_date"] >= COVID_START) & (df["review_date"] <= COVID_END)]
            .groupby("place_id")
            .size()
        )
        fraction = (covid / total).fillna(0.0)
        weights = (1.0 - alpha * fraction).clip(lower=0.2)
        weight_map.update(weights.to_dict())
    logging.info("Loaded COVID weights for %d place_ids.", len(weight_map))
    return weight_map


def prepare_model1_data(
    df: pd.DataFrame, min_reviews: int
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, List[str], List[str]]:
    subset = df[df["user_rating_count"].notna() & (df["user_rating_count"] >= min_reviews)].copy()
    subset = subset.reset_index(drop=True)
    if subset.empty:
        raise ValueError("No rows available for Model 1 after filtering by user_rating_count.")

    subset["y_exist"] = np.log1p(subset["user_rating_count"].astype(float))
    subset["rating"] = pd.to_numeric(subset["rating"], errors="coerce").fillna(0.0)
    subset["rating_sq"] = subset["rating"] ** 2
    subset["rating_centered"] = subset["rating"] - 4.0

    num_cols, cat_cols = select_feature_columns(subset.columns, include_rating_cols=True)
    feature_cols = num_cols + cat_cols
    X = subset[feature_cols].fillna(0).reset_index(drop=True)
    y = subset["y_exist"].to_numpy()
    groups = subset["__SelectedCity__"].fillna("Unknown").reset_index(drop=True)
    place_ids = subset["place_id"].reset_index(drop=True)
    return X, y, groups, num_cols, cat_cols, place_ids


def prepare_model2_data(
    df: pd.DataFrame, rating_thr: float, quantile: float
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, List[str], List[str]]:
    subset = df[df["user_rating_count"].notna() & df["rating"].notna()].copy()
    subset = subset.reset_index(drop=True)
    if subset.empty:
        raise ValueError("No rows available for Model 2 after filtering for rating/review counts.")

    subset["city_review_threshold"] = (
        subset.groupby("__SelectedCity__")["user_rating_count"].transform(lambda s: s.quantile(quantile))
    )
    subset["success_class"] = (
        (subset["rating"] >= rating_thr) & (subset["user_rating_count"] >= subset["city_review_threshold"])
    ).astype(int)

    num_cols, cat_cols = select_feature_columns(subset.columns, include_rating_cols=False)
    feature_cols = num_cols + cat_cols
    X = subset[feature_cols].fillna(0).reset_index(drop=True)
    y = subset["success_class"].to_numpy()
    groups = subset["__SelectedCity__"].fillna("Unknown").reset_index(drop=True)
    place_ids = subset["place_id"].reset_index(drop=True)
    return X, y, groups, num_cols, cat_cols, place_ids


def select_feature_columns(columns: pd.Index, include_rating_cols: bool) -> Tuple[List[str], List[str]]:
    num_cols = set()
    cat_cols = []
    for col in columns:
        if col in {"id", "__SelectedCity__", "success_class", "city_review_threshold", "y_exist"}:
            continue
        if not include_rating_cols and (col in MODEL2_EXCLUDE_COLS):
            continue
        if not include_rating_cols and any(token in col for token in MODEL2_EXCLUDE_CONTAINS):
            continue
        if include_rating_cols and col in {"rating", "rating_sq", "rating_centered"}:
            num_cols.add(col)
            continue
        if col.startswith(CATEGORICAL_PREFIXES):
            cat_cols.append(col)
            continue
        if col == "price_level_num":
            num_cols.add(col)
            continue
        if col.startswith(NUMERIC_PREFIXES):
            num_cols.add(col)
    return sorted(num_cols), sorted(cat_cols)


def build_fit_params(estimator, sample_weight: np.ndarray | None):
    if sample_weight is None:
        return {}
    if isinstance(estimator, Pipeline):
        return {"model__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


def build_lgbm_pipeline(
    task: str, num_cols: List[str], cat_cols: List[str], random_state: int
) -> Pipeline:
    numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", "passthrough", cat_cols),
        ],
        remainder="drop",
    )
    if task == "regression":
        model = lgb.LGBMRegressor(
            boosting_type="gbdt",
            n_estimators=800,
            objective="regression",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            n_estimators=800,
            objective="binary",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def get_param_distributions(model_key: str) -> Dict[str, List[object]]:
    if model_key == "model1_rf":
        return {
            "n_estimators": [400, 600, 800, 1000],
            "max_depth": [10, 14, 18, None],
            "min_samples_leaf": [2, 5, 10, 20],
            "max_features": [0.3, 0.5, 0.7, 0.9],
        }
    if model_key == "model2_rf":
        return {
            "n_estimators": [400, 600, 800, 1000],
            "max_depth": [10, 14, 18, None],
            "min_samples_leaf": [2, 5, 10, 20],
            "max_features": [0.3, 0.5, 0.7, 0.9],
        }
    if model_key == "model1_lgbm":
        return {
            "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
            "model__num_leaves": [31, 63, 127, 255],
            "model__max_depth": [-1, 10, 14, 18],
            "model__min_child_samples": [20, 40, 60, 100],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__reg_lambda": [0.0, 0.1, 1.0, 5.0],
        }
    if model_key == "model2_lgbm":
        return {
            "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
            "model__num_leaves": [31, 63, 127, 255],
            "model__max_depth": [-1, 10, 14, 18],
            "model__min_child_samples": [20, 40, 60, 100],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__reg_lambda": [0.0, 0.1, 1.0, 5.0],
        }
    raise ValueError(f"Unknown model key for parameter distribution: {model_key}")


def evaluate_holdout(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    task: str,
    test_size: float,
    random_state: int,
    sample_weight: np.ndarray | None,
) -> Dict[str, float]:
    stratify = y if (task == "classification") else None
    if sample_weight is not None:
        X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
            X,
            y,
            sample_weight,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        w_train = None
    est = clone(estimator)
    fit_params = build_fit_params(est, w_train)
    est.fit(X_train, y_train, **fit_params)
    if task == "regression":
        preds = est.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        }
    else:
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X_test)[:, 1]
        else:
            proba = est.decision_function(X_test)
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "average_precision": float(average_precision_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        }
    return metrics


    if task == "regression":
        preds = est.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        }
    else:
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X_test)[:, 1]
        else:
            proba = est.decision_function(X_test)
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "average_precision": float(average_precision_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        }
    return metrics


def run_random_search(
    estimator,
    param_dist: Dict[str, List[object]],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    task: str,
    n_iter: int,
    random_state: int,
    desc: str,
    sample_weight: np.ndarray | None,
) -> Tuple[object, Dict[str, object], float, List[Dict[str, object]]]:
    rng = np.random.RandomState(random_state)
    gkf = GroupKFold(n_splits=5)
    best_score = -np.inf
    best_params: Dict[str, object] | None = None
    history: List[Dict[str, object]] = []

    iterator = tqdm(range(n_iter), desc=desc, unit="iter")
    for _ in iterator:
        params = {key: rng.choice(values) for key, values in param_dist.items()}
        fold_scores: List[float] = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            est = clone(estimator).set_params(**params)
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y[val_idx]
            train_weights = sample_weight[train_idx] if sample_weight is not None else None
            fit_params = build_fit_params(est, train_weights)
            est.fit(X_train, y_train, **fit_params)
            if task == "regression":
                preds = est.predict(X_val)
                score = r2_score(y_val, preds)
            else:
                if hasattr(est, "predict_proba"):
                    proba = est.predict_proba(X_val)[:, 1]
                else:
                    proba = est.decision_function(X_val)
                score = roc_auc_score(y_val, proba)
            fold_scores.append(score)
        mean_score = float(np.mean(fold_scores))
        history.append({"params": params, "score": mean_score})
        iterator.set_postfix({"score": f"{mean_score:.3f}"})
        if mean_score > best_score or best_params is None:
            best_score = mean_score
            best_params = params

    best_estimator = clone(estimator).set_params(**(best_params or {}))
    full_fit_params = build_fit_params(best_estimator, sample_weight)
    best_estimator.fit(X, y, **full_fit_params)
    return best_estimator, best_params or {}, best_score, history


def train_model(
    model_key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    args: argparse.Namespace,
    sample_weight: np.ndarray | None,
) -> Dict[str, object]:
    logging.info("Training %s ...", model_key)
    task = "regression" if "model1" in model_key else "classification"

    if "lgbm" in model_key:
        estimator = build_lgbm_pipeline(task, num_cols, cat_cols, args.random_state)
    elif model_key == "model1_rf":
        estimator = RandomForestRegressor(
            n_estimators=600,
            random_state=args.random_state,
            n_jobs=-1,
        )
    else:  # model2_rf
        estimator = RandomForestClassifier(
            n_estimators=600,
            random_state=args.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

    param_dist = get_param_distributions(model_key)
    best_estimator, best_params, best_score, history = run_random_search(
        estimator,
        param_dist,
        X,
        y,
        groups,
        task,
        n_iter=args.random_search_iter,
        random_state=args.random_state,
        desc=f"{model_key} search",
        sample_weight=sample_weight,
    )
    holdout_metrics = evaluate_holdout(
        best_estimator,
        X,
        y,
        task,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_weight=sample_weight,
    )

    model_name = f"{model_key}_best"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(best_estimator, model_path)

    metrics = {
        "model": model_key,
        "task": task,
        "best_params": best_params,
        "group_cv_score": float(best_score),
        "holdout_metrics": holdout_metrics,
        "search_history": history,
        "model_path": str(model_path),
    }

    metrics_path = ARTIFACTS_DIR / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=json_default)

    logging.info("Finished %s. Metrics saved to %s", model_key, metrics_path)
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    ensure_dirs()

    logging.info("Loading feature table from %s", args.features_path)
    df = pd.read_parquet(args.features_path)
    df = df.copy()
    df["place_id"] = df["id"]

    prepared_data = {}
    all_metrics = []
    covid_weight_map = load_covid_weight_map()

    if any(m.startswith("model1") for m in args.models):
        X1, y1, groups1, num_cols1, cat_cols1, place_ids1 = prepare_model1_data(df, args.min_reviews_existing)
        weights1 = place_ids1.map(covid_weight_map).fillna(1.0).to_numpy()
        prepared_data["model1"] = (X1, y1, groups1, num_cols1, cat_cols1, weights1)

    if any(m.startswith("model2") for m in args.models):
        X2, y2, groups2, num_cols2, cat_cols2, place_ids2 = prepare_model2_data(
            df, args.success_rating_threshold, args.success_review_quantile
        )
        weights2 = place_ids2.map(covid_weight_map).fillna(1.0).to_numpy()
        prepared_data["model2"] = (X2, y2, groups2, num_cols2, cat_cols2, weights2)

    for model_key in args.models:
        dataset_key = "model1" if "model1" in model_key else "model2"
        X, y, groups, num_cols, cat_cols, weights = prepared_data[dataset_key]
        metrics = train_model(model_key, X, y, groups, num_cols, cat_cols, args, weights)
        all_metrics.append(metrics)

    summary_path = ARTIFACTS_DIR / "model_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2, default=json_default)
    logging.info("Training complete. Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
