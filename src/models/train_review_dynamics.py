"""Train time-series enhanced review dynamics models for Interlaken and Sion.

Workflow per city:
1. Load review time series (per-review) and static features (microlocation/competition).
2. Split each restaurant's history into a pre window (>365 days ago) and a recent window (last 365 days).
3. Aggregate pre-window features (volume, average rating, volatility, monthly trend, early/late rating shift).
4. Aggregate recent-window targets (log review volume, avg rating, recent success flag).
5. Merge with static features and train two LightGBM models per city:
   - Regression: predict log reviews in last 12 months.
   - Classification: predict whether the restaurant is a recent "strong success".
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Paths
FEATURE_PATH = Path("data/final/features_location_competition.parquet")
TS_TEMPLATE = Path("data/final/reviews_timeseries_{city}.csv")
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")

# Feature configuration
NUMERIC_STATIC_PREFIXES = (
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
TIME_SERIES_FEATURES = [
    "pre_total_reviews",
    "pre_log_reviews",
    "pre_avg_rating",
    "pre_rating_std",
    "pre_count_trend_slope",
    "pre_rating_early",
    "pre_rating_late",
    "pre_rating_delta",
]
STATIC_EXCLUDE_KEYWORDS = ("rating_gap",)
STATIC_EXCLUDE_COLUMNS = {"nn_avg_rating", "comp200_avg_rating"}
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-12-31")


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train review dynamics models for Interlaken and Sion.")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["interlaken", "sion"],
        help="Cities to process (matching reviews_timeseries_<city>.csv).",
    )
    parser.add_argument(
        "--min-total-reviews",
        type=int,
        default=10,
        help="Minimum number of total reviews per restaurant to be included.",
    )
    parser.add_argument(
        "--min-pre-reviews",
        type=int,
        default=3,
        help="Minimum number of pre-window reviews per restaurant.",
    )
    parser.add_argument(
        "--min-post-reviews",
        type=int,
        default=3,
        help="Minimum number of post-window reviews per restaurant.",
    )
    parser.add_argument(
        "--recent-window-days",
        type=int,
        default=365,
        help="Length of the recent window (days) used for targets.",
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.3,
        help="Rating threshold for recent success classification.",
    )
    parser.add_argument(
        "--review-quantile",
        type=float,
        default=0.7,
        help="Quantile (0-1) for review volume threshold in the recent success class.",
    )
    parser.add_argument(
        "--random-search-iter",
        type=int,
        default=20,
        help="Number of random-search iterations per model.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout size for final evaluation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def slugify_city(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_static_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_PATH)
    df = df.copy()
    df["place_id"] = df["id"]
    return df


def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "review_date" not in df.columns:
        raise ValueError(f"Timeseries file {path} is missing 'review_date'.")
    if "place_id" not in df.columns:
        raise ValueError(f"Timeseries file {path} is missing 'place_id'.")
    df["review_date"] = pd.to_datetime(df["review_date"], utc=True, errors="coerce").dt.tz_localize(None)
    if "review_rating" in df.columns:
        df["rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    elif "rating" not in df.columns:
        raise ValueError(f"Timeseries file {path} is missing rating information.")
    df = df.dropna(subset=["rating"])
    return df


def filter_restaurants(ts: pd.DataFrame, static_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_counts = ts.groupby("place_id").size()
    valid_ids = total_counts[total_counts >= args.min_total_reviews].index
    ts = ts[ts["place_id"].isin(valid_ids)].copy()
    static_df = static_df[static_df["place_id"].isin(valid_ids)].copy()
    return ts, static_df


def remove_covid_period(ts: pd.DataFrame) -> pd.DataFrame:
    mask = (ts["review_date"] < COVID_START) | (ts["review_date"] > COVID_END)
    return ts[mask].copy()


def split_pre_post(ts: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    max_date = ts["review_date"].max()
    cutoff = max_date - pd.Timedelta(days=args.recent_window_days)
    ts_pre = ts[ts["review_date"] < cutoff].copy()
    ts_post = ts[ts["review_date"] >= cutoff].copy()
    return ts_pre, ts_post, cutoff


def enforce_pre_post_minimums(
    ts_pre: pd.DataFrame, ts_post: pd.DataFrame, static_df: pd.DataFrame, args: argparse.Namespace
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre_counts = ts_pre.groupby("place_id").size()
    post_counts = ts_post.groupby("place_id").size()
    valid_ids = pre_counts[pre_counts >= args.min_pre_reviews].index.intersection(
        post_counts[post_counts >= args.min_post_reviews].index
    )
    ts_pre = ts_pre[ts_pre["place_id"].isin(valid_ids)].copy()
    ts_post = ts_post[ts_post["place_id"].isin(valid_ids)].copy()
    static_df = static_df[static_df["place_id"].isin(valid_ids)].copy()
    return ts_pre, ts_post, static_df


def compute_pre_features(ts_pre: pd.DataFrame) -> pd.DataFrame:
    if ts_pre.empty:
        return pd.DataFrame(columns=["place_id"] + TIME_SERIES_FEATURES)

    agg = ts_pre.groupby("place_id").agg(
        pre_total_reviews=("rating", "size"),
        pre_avg_rating=("rating", "mean"),
        pre_rating_std=("rating", "std"),
    )
    agg["pre_log_reviews"] = np.log1p(agg["pre_total_reviews"])

    ts_pre = ts_pre.copy()
    ts_pre["month"] = ts_pre["review_date"].dt.to_period("M").dt.to_timestamp()
    monthly = ts_pre.groupby(["place_id", "month"]).size().reset_index(name="count")

    def compute_trend(group: pd.DataFrame) -> pd.Series:
        counts = group["count"].values
        if len(counts) < 2:
            return pd.Series({"pre_count_trend_slope": 0.0})
        t = np.arange(len(counts))
        A = np.vstack([t, np.ones(len(t))]).T
        slope, _ = np.linalg.lstsq(A, counts, rcond=None)[0]
        return pd.Series({"pre_count_trend_slope": float(slope)})

    trend = monthly.groupby("place_id").apply(compute_trend)

    median_date = ts_pre.groupby("place_id")["review_date"].median()
    ts_pre = ts_pre.merge(median_date.rename("median_split"), on="place_id", how="left")
    early = ts_pre[ts_pre["review_date"] < ts_pre["median_split"]]
    late = ts_pre[ts_pre["review_date"] >= ts_pre["median_split"]]

    early_mean = early.groupby("place_id")["rating"].mean().rename("pre_rating_early")
    late_mean = late.groupby("place_id")["rating"].mean().rename("pre_rating_late")

    rating_shift = pd.concat([early_mean, late_mean], axis=1)
    rating_shift["pre_rating_delta"] = rating_shift["pre_rating_late"] - rating_shift["pre_rating_early"]

    features = agg.join(trend, how="left").join(rating_shift, how="left").reset_index()
    return features


def compute_post_targets(
    ts_post: pd.DataFrame, rating_threshold: float, review_quantile: float
) -> pd.DataFrame:
    if ts_post.empty:
        return pd.DataFrame(columns=["place_id", "post_reviews_1y", "post_log_reviews_1y", "post_avg_rating_1y", "recent_success_class"])

    post = ts_post.groupby("place_id").agg(
        post_reviews_1y=("rating", "size"),
        post_avg_rating_1y=("rating", "mean"),
    )
    post["post_log_reviews_1y"] = np.log1p(post["post_reviews_1y"])
    review_threshold = post["post_reviews_1y"].quantile(review_quantile)
    post["recent_success_class"] = (
        (post["post_avg_rating_1y"] >= rating_threshold) & (post["post_reviews_1y"] >= review_threshold)
    ).astype(int)
    return post.reset_index()


def prepare_city_dataset(
    city: str, static_df: pd.DataFrame, args: argparse.Namespace
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    slug = slugify_city(city)
    city_col = f"city_{slug}"
    if city_col not in static_df.columns:
        raise ValueError(f"Column {city_col} not found in static features.")

    ts_path = TS_TEMPLATE.with_name(TS_TEMPLATE.name.format(city=slug))
    if not ts_path.exists():
        raise FileNotFoundError(f"Timeseries file not found for city '{city}': {ts_path}")

    logging.info("Loading time series for %s from %s", city, ts_path)
    ts = load_timeseries(ts_path)
    static_city = static_df[static_df[city_col] == 1].copy()
    ts = ts[ts["place_id"].isin(static_city["place_id"])]

    ts, static_city = filter_restaurants(ts, static_city, args)
    if ts.empty:
        logging.warning("No restaurants with >= %d reviews for %s.", args.min_total_reviews, city)
        return pd.DataFrame(), [], []

    ts_pre, ts_post, cutoff = split_pre_post(ts, args)
    ts_pre = remove_covid_period(ts_pre)
    ts_pre, ts_post, static_city = enforce_pre_post_minimums(ts_pre, ts_post, static_city, args)

    if ts_pre.empty or ts_post.empty:
        logging.warning("Insufficient pre/post data for %s after filtering.", city)
        return pd.DataFrame(), [], []

    pre_feats = compute_pre_features(ts_pre)
    post_targets = compute_post_targets(ts_post, args.rating_threshold, args.review_quantile)

    dataset = static_city.merge(pre_feats, on="place_id", how="inner").merge(post_targets, on="place_id", how="inner")
    if dataset.empty:
        logging.warning("Merged dataset is empty for %s.", city)
        return pd.DataFrame(), [], []

    numeric_cols = ["price_level_num"] + [c for c in dataset.columns if c.startswith(NUMERIC_STATIC_PREFIXES)]
    filtered_numeric_cols = []
    for col in numeric_cols:
        if col in STATIC_EXCLUDE_COLUMNS:
            continue
        if any(keyword in col for keyword in STATIC_EXCLUDE_KEYWORDS):
            continue
        filtered_numeric_cols.append(col)
    numeric_cols = sorted(set(filtered_numeric_cols + TIME_SERIES_FEATURES))
    categorical_cols = [c for c in dataset.columns if c.startswith(CATEGORICAL_PREFIXES)]
    feature_cols = numeric_cols + categorical_cols
    dataset = dataset[["place_id"] + feature_cols + ["post_log_reviews_1y", "recent_success_class"]]
    dataset = dataset.fillna({col: 0.0 for col in numeric_cols})
    dataset = dataset.fillna({col: 0 for col in categorical_cols})
    return dataset, numeric_cols, categorical_cols


def build_pipeline(task: str, num_cols: List[str], cat_cols: List[str], random_state: int):
    numeric_pipeline = Pipeline(
        steps=[
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
        model = LGBMRegressor(
            boosting_type="gbdt",
            objective="regression",
            n_estimators=600,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            n_estimators=600,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def get_param_grid(task: str) -> Dict[str, List[object]]:
    base_grid = {
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
        "model__num_leaves": [31, 63, 127, 255],
        "model__max_depth": [-1, 10, 14, 18],
        "model__min_child_samples": [5, 10, 20, 40, 60, 100],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__reg_lambda": [0.0, 0.1, 1.0, 5.0],
    }
    return base_grid


def random_search_pipeline(
    pipeline,
    param_grid: Dict[str, List[object]],
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: str,
    cv,
    n_iter: int,
    random_state: int,
    desc: str,
) -> Tuple[Pipeline, Dict[str, object], float, List[Dict[str, object]]]:
    rng = np.random.RandomState(random_state)
    best_score = -np.inf
    best_params = None
    history: List[Dict[str, object]] = []

    for _ in tqdm(range(n_iter), desc=desc, unit="iter"):
        params = {key: rng.choice(values) for key, values in param_grid.items()}
        estimator = clone(pipeline).set_params(**params)
        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        mean_score = float(np.mean(scores))
        history.append({"params": params, "score": mean_score})
        if mean_score > best_score or best_params is None:
            best_score = mean_score
            best_params = params

    best_estimator = clone(pipeline).set_params(**(best_params or {})).fit(X, y)
    return best_estimator, best_params or {}, best_score, history


def evaluate_holdout(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    task: str,
    test_size: float,
    random_state: int,
) -> Dict[str, float]:
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    est = clone(estimator).fit(X_train, y_train)
    if task == "regression":
        preds = est.predict(X_test)
        return {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        }
    else:
        proba = est.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "average_precision": float(average_precision_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float((2 * precision_score(y_test, preds, zero_division=0) * recall_score(y_test, preds, zero_division=0)) / max(
                precision_score(y_test, preds, zero_division=0) + recall_score(y_test, preds, zero_division=0), 1e-9
            )),
            "log_loss": float(log_loss(y_test, proba)),
        }


def save_metrics(metrics: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=json_default)


def train_for_city(
    city: str,
    dataset: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    results = []
    if dataset.empty:
        return results

    X = dataset[numeric_cols + categorical_cols]

    # Regression model
    pipeline_reg = build_pipeline("regression", numeric_cols, categorical_cols, args.random_state)
    param_grid_reg = get_param_grid("regression")
    kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    logging.info("Random search (regression) for %s ...", city)
    reg_estimator, reg_params, reg_cv, reg_history = random_search_pipeline(
        pipeline_reg,
        param_grid_reg,
        X,
        dataset["post_log_reviews_1y"].to_numpy(),
        scoring="r2",
        cv=kf,
        n_iter=args.random_search_iter,
        random_state=args.random_state,
        desc=f"{city}-reg",
    )
    reg_holdout = evaluate_holdout(
        reg_estimator,
        X,
        dataset["post_log_reviews_1y"].to_numpy(),
        task="regression",
        test_size=args.test_size,
        random_state=args.random_state,
    )
    reg_model_path = MODELS_DIR / f"review_dynamics_{city}_reg.pkl"
    joblib.dump(reg_estimator, reg_model_path)
    reg_metrics = {
        "city": city,
        "task": "regression",
        "best_params": reg_params,
        "group_cv_score": reg_cv,
        "holdout_metrics": reg_holdout,
        "search_history": reg_history,
        "model_path": str(reg_model_path),
    }
    reg_metrics_path = ARTIFACTS_DIR / f"review_dynamics_{city}_reg_metrics.json"
    save_metrics(reg_metrics, reg_metrics_path)
    logging.info("Saved regression metrics for %s to %s", city, reg_metrics_path)
    results.append(reg_metrics)

    # Classification model
    pipeline_clf = build_pipeline("classification", numeric_cols, categorical_cols, args.random_state)
    param_grid_clf = get_param_grid("classification")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    logging.info("Random search (classification) for %s ...", city)
    clf_estimator, clf_params, clf_cv, clf_history = random_search_pipeline(
        pipeline_clf,
        param_grid_clf,
        X,
        dataset["recent_success_class"].to_numpy(),
        scoring="roc_auc",
        cv=skf,
        n_iter=args.random_search_iter,
        random_state=args.random_state,
        desc=f"{city}-clf",
    )
    clf_holdout = evaluate_holdout(
        clf_estimator,
        X,
        dataset["recent_success_class"].to_numpy(),
        task="classification",
        test_size=args.test_size,
        random_state=args.random_state,
    )
    clf_model_path = MODELS_DIR / f"review_dynamics_{city}_clf.pkl"
    joblib.dump(clf_estimator, clf_model_path)
    clf_metrics = {
        "city": city,
        "task": "classification",
        "best_params": clf_params,
        "group_cv_score": clf_cv,
        "holdout_metrics": clf_holdout,
        "search_history": clf_history,
        "model_path": str(clf_model_path),
    }
    clf_metrics_path = ARTIFACTS_DIR / f"review_dynamics_{city}_clf_metrics.json"
    save_metrics(clf_metrics, clf_metrics_path)
    logging.info("Saved classification metrics for %s to %s", city, clf_metrics_path)
    results.append(clf_metrics)
    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    ensure_dirs()

    static_features = load_static_features()
    summary: List[Dict[str, object]] = []

    for city in args.cities:
        logging.info("==== Processing city: %s ====", city)
        dataset, num_cols, cat_cols = prepare_city_dataset(city, static_features, args)
        if dataset.empty:
            logging.warning("Skipping %s due to insufficient data.", city)
            continue
        city_results = train_for_city(city, dataset, num_cols, cat_cols, args)
        summary.extend(city_results)

    summary_path = ARTIFACTS_DIR / "review_dynamics_summary.json"
    save_metrics(summary, summary_path)
    logging.info("Review dynamics training complete. Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
