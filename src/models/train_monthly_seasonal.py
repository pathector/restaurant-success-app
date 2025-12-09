"""Train monthly seasonal review models for Interlaken and Sion.

Workflow per city:
1. Load per-review time series and static microlocation features.
2. Aggregate reviews into a restaurant-month panel with zero-filled gaps.
3. Engineer seasonal/lagged features plus COVID flags and next-month targets.
4. Merge static features, run time-aware CV LightGBM search, and save artifacts.

The script mirrors the earlier Interlaken/Sion review dynamics work but shifts
the target to monthly granularity so we can surface explicit seasonality.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
STATIC_EXCLUDE_KEYWORDS = ("rating_gap",)
STATIC_EXCLUDE_COLUMNS = {"nn_avg_rating", "comp200_avg_rating"}
MONTHLY_DYNAMIC_FEATURES = [
    "reviews_m",
    "reviews_m_lag1",
    "reviews_m_trailing3",
    "reviews_m_trailing6",
    "avg_rating_m",
    "avg_rating_lag1",
    "avg_rating_trailing3",
    "rating_std_m",
    "t_since_first",
    "month_idx",
    "month_sin",
    "month_cos",
    "is_covid_month",
]
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-12-31")


@dataclass
class FeatureLists:
    numeric_dynamic: List[str]
    numeric_static: List[str]
    categorical: List[str]

    @property
    def all_numeric(self) -> List[str]:
        return self.numeric_dynamic + self.numeric_static


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train monthly seasonal review models.")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["interlaken", "sion"],
        help="Cities to process (matching reviews_timeseries_{city}.csv).",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=FEATURE_PATH,
        help="Static feature table path.",
    )
    parser.add_argument(
        "--timeseries-template",
        type=str,
        default=str(TS_TEMPLATE),
        help="Template for per-review CSV paths (use {city} placeholder).",
    )
    parser.add_argument(
        "--covid-strategy",
        choices=("exclude", "keep", "downweight"),
        default="exclude",
        help="How to handle COVID-period months.",
    )
    parser.add_argument(
        "--covid-weight",
        type=float,
        default=0.3,
        help="Sample weight to apply when covid-strategy=downweight.",
    )
    parser.add_argument(
        "--min-history-months",
        type=int,
        default=3,
        help="Minimum number of past months required before predicting.",
    )
    parser.add_argument(
        "--holdout-months",
        type=int,
        default=6,
        help="Number of most-recent months reserved for holdout evaluation.",
    )
    parser.add_argument(
        "--tscv-splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds for random search.",
    )
    parser.add_argument(
        "--random-search-iter",
        type=int,
        default=20,
        help="Random search iterations per city.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=600,
        help="Baseline number of LightGBM boosting rounds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--cv-jobs",
        type=int,
        default=1,
        help="Parallel jobs for cross-validation.",
    )
    parser.add_argument(
        "--lgbm-jobs",
        type=int,
        default=-1,
        help="n_jobs parameter passed to LightGBM.",
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


def load_static_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df["place_id"] = df["id"].astype(str)
    return df


def load_timeseries(city: str, template: str) -> pd.DataFrame:
    path = Path(template.format(city=city))
    if not path.exists():
        raise FileNotFoundError(f"Missing time series file for {city}: {path}")
    df = pd.read_csv(path)
    for col in ("place_id", "review_date"):
        if col not in df.columns:
            raise ValueError(f"Timeseries file {path} missing column '{col}'")
    df["place_id"] = df["place_id"].astype(str)
    df["review_date"] = (
        pd.to_datetime(df["review_date"], utc=True, errors="coerce").dt.tz_localize(None)
    )
    rating_col = "rating" if "rating" in df.columns else "review_rating"
    if rating_col not in df.columns:
        raise ValueError(f"Timeseries file {path} missing rating information.")
    df["rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=["review_date", "rating"])
    if df.empty:
        raise ValueError(f"No valid reviews after cleaning for {city}")
    return df[["place_id", "review_date", "rating"]].copy()


def aggregate_monthly(ts: pd.DataFrame, city: str) -> pd.DataFrame:
    df = ts.copy()
    df["month"] = df["review_date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby(["place_id", "month"])
        .agg(
            reviews_m=("rating", "size"),
            avg_rating_m=("rating", "mean"),
            rating_std_m=("rating", "std"),
        )
        .reset_index()
    )
    monthly = fill_month_gaps(monthly)
    monthly = monthly.sort_values(["place_id", "month"]).reset_index(drop=True)
    monthly["city"] = city
    monthly["reviews_m"] = monthly["reviews_m"].fillna(0).astype(float)
    return monthly


def fill_month_gaps(monthly: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for place_id, group in monthly.groupby("place_id"):
        min_month = group["month"].min()
        max_month = group["month"].max()
        all_months = pd.date_range(min_month, max_month, freq="MS")
        expanded = pd.DataFrame({"place_id": place_id, "month": all_months})
        expanded = expanded.merge(group, on=["place_id", "month"], how="left")
        frames.append(expanded)
    return pd.concat(frames, ignore_index=True)


def add_lag_features(monthly: pd.DataFrame) -> pd.DataFrame:
    df = monthly.copy()
    grouped = df.groupby("place_id", group_keys=False)
    df["reviews_m_lag1"] = grouped["reviews_m"].shift(1)
    df["reviews_m_trailing3"] = grouped["reviews_m"].apply(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
    )
    df["reviews_m_trailing6"] = grouped["reviews_m"].apply(
        lambda s: s.shift(1).rolling(window=6, min_periods=1).sum()
    )
    df["avg_rating_lag1"] = grouped["avg_rating_m"].shift(1)
    df["avg_rating_trailing3"] = grouped["avg_rating_m"].apply(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["t_since_first"] = grouped.cumcount()
    df["next_reviews"] = grouped["reviews_m"].shift(-1)
    df["next_log_reviews"] = np.log1p(df["next_reviews"])
    df["month_idx"] = df["month"].dt.year * 12 + df["month"].dt.month
    df["month_of_year"] = df["month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12.0)
    df["is_covid_month"] = ((df["month"] >= COVID_START) & (df["month"] <= COVID_END)).astype(int)
    return df


def apply_history_filter(df: pd.DataFrame, min_history: int) -> pd.DataFrame:
    mask = df["t_since_first"] >= min_history
    subset = df[mask].copy()
    subset = subset.dropna(subset=["next_log_reviews"])
    subset = subset.sort_values(["month", "place_id"]).reset_index(drop=True)
    if subset.empty:
        raise ValueError("No rows remaining after enforcing history filter.")
    return subset


def merge_static_features(df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(static_df, on="place_id", how="left", suffixes=("", "_static"))
    duplicated = merged.columns.duplicated()
    if duplicated.any():
        dup_cols = [col for col, dup in zip(merged.columns, duplicated) if dup]
        logging.warning("Dropping duplicated columns after merge: %s", dup_cols)
        merged = merged.loc[:, ~duplicated]
    return merged


def select_static_numeric_columns(df: pd.DataFrame) -> List[str]:
    columns: List[str] = []
    for col in df.columns:
        if col in STATIC_EXCLUDE_COLUMNS:
            continue
        if any(keyword in col for keyword in STATIC_EXCLUDE_KEYWORDS):
            continue
        if any(col.startswith(prefix) for prefix in NUMERIC_STATIC_PREFIXES):
            columns.append(col)
    if "price_level_num" in df.columns:
        columns.append("price_level_num")
    return sorted(set(columns))


def select_categorical_columns(df: pd.DataFrame) -> List[str]:
    columns: List[str] = []
    for col in df.columns:
        if col in MONTHLY_DYNAMIC_FEATURES:
            continue
        if any(col.startswith(prefix) for prefix in CATEGORICAL_PREFIXES):
            columns.append(col)
    return sorted(set(columns))


def apply_covid_strategy(df: pd.DataFrame, strategy: str, weight: float) -> pd.DataFrame:
    if strategy == "exclude":
        return df[df["is_covid_month"] == 0].copy()
    if strategy == "keep":
        return df.copy()
    if strategy == "downweight":
        df = df.copy()
        df["sample_weight"] = np.where(df["is_covid_month"] == 1, weight, 1.0)
        return df
    raise ValueError(f"Unknown covid strategy: {strategy}")


def build_feature_lists(df: pd.DataFrame) -> FeatureLists:
    dynamic = [col for col in MONTHLY_DYNAMIC_FEATURES if col in df.columns]
    static_numeric = select_static_numeric_columns(df)
    categorical = select_categorical_columns(df)
    return FeatureLists(dynamic, static_numeric, categorical)


def get_preprocessor(feature_lists: FeatureLists) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transformers = [("num", numeric_transformer, feature_lists.all_numeric)]
    if feature_lists.categorical:
        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("cat", cat_transformer, feature_lists.categorical))
    return ColumnTransformer(transformers=transformers)


def sample_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "learning_rate": rng.choice([0.02, 0.03, 0.05]),
        "num_leaves": int(rng.choice([31, 63, 127])),
        "max_depth": int(rng.choice([-1, 10, 14])),
        "min_child_samples": int(rng.choice([10, 20, 40])),
        "subsample": float(rng.choice([0.7, 0.8, 0.9])),
        "colsample_bytree": float(rng.choice([0.7, 0.8, 0.9])),
        "reg_lambda": float(rng.choice([0.0, 0.1, 1.0])),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def evaluate_counts(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    true_counts = np.expm1(y_true_log)
    pred_counts = np.expm1(y_pred_log)
    return {
        "count_rmse": float(np.sqrt(mean_squared_error(true_counts, pred_counts))),
        "count_mae": mean_absolute_error(true_counts, pred_counts),
    }


def prepare_city_panel(
    city: str,
    static_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, FeatureLists]:
    monthly = aggregate_monthly(load_timeseries(city, args.timeseries_template), city)
    monthly = add_lag_features(monthly)
    monthly = apply_history_filter(monthly, args.min_history_months)
    monthly = apply_covid_strategy(monthly, args.covid_strategy, args.covid_weight)
    if monthly.empty:
        raise ValueError(f"No training rows remain for {city} after filtering.")
    merged = merge_static_features(monthly, static_df)
    feature_lists = build_feature_lists(merged)
    return merged, feature_lists


def split_train_holdout(
    df: pd.DataFrame, holdout_months: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_months = np.array(sorted(df["month"].unique()))
    if len(unique_months) <= holdout_months:
        raise ValueError("Not enough months to create a holdout segment.")
    cutoff = unique_months[-holdout_months]
    train_df = df[df["month"] < cutoff].copy()
    holdout_df = df[df["month"] >= cutoff].copy()
    if train_df.empty or holdout_df.empty:
        raise ValueError("Holdout split produced an empty partition.")
    return (
        train_df.sort_values(["month", "place_id"]).reset_index(drop=True),
        holdout_df.sort_values(["month", "place_id"]).reset_index(drop=True),
    )


def train_city_model(
    city: str,
    panel: pd.DataFrame,
    feature_lists: FeatureLists,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, object]:
    train_df, holdout_df = split_train_holdout(panel, args.holdout_months)
    X_train = train_df[feature_lists.all_numeric + feature_lists.categorical]
    y_train = train_df["next_log_reviews"].to_numpy()
    sample_weight = train_df["sample_weight"].to_numpy() if "sample_weight" in train_df.columns else None

    preprocessor = get_preprocessor(feature_lists)
    base_model = LGBMRegressor(
        boosting_type="gbdt",
        objective="regression",
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=args.lgbm_jobs,
    )

    best_score = -np.inf
    best_params: Dict[str, float] = {}
    cv_records: List[Dict[str, float]] = []
    tscv = TimeSeriesSplit(n_splits=args.tscv_splits)
    for iter_idx in range(args.random_search_iter):
        params = sample_params(rng)
        model = LGBMRegressor(**base_model.get_params())
        model.set_params(**params)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        fold_scores: List[float] = []
        for train_idx, val_idx in tscv.split(X_train, y_train):
            estimator = clone(pipeline)
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["model__sample_weight"] = sample_weight[train_idx]
            estimator.fit(X_train.iloc[train_idx], y_train[train_idx], **fit_kwargs)
            preds = estimator.predict(X_train.iloc[val_idx])
            fold_scores.append(r2_score(y_train[val_idx], preds))
        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        cv_records.append(
            {"iteration": iter_idx + 1, "params": params, "mean_r2": mean_score, "std_r2": std_score}
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    best_model = LGBMRegressor(**base_model.get_params())
    best_model.set_params(**best_params)
    best_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", best_model)])
    fit_params_final = {"model__sample_weight": sample_weight} if sample_weight is not None else None
    best_pipeline.fit(X_train, y_train, **(fit_params_final or {}))

    X_holdout = holdout_df[feature_lists.all_numeric + feature_lists.categorical]
    y_holdout = holdout_df["next_log_reviews"].to_numpy()
    holdout_pred = best_pipeline.predict(X_holdout)
    holdout_metrics = regression_metrics(y_holdout, holdout_pred)
    holdout_metrics.update(
        {f"train_{k}": v for k, v in regression_metrics(y_train, best_pipeline.predict(X_train)).items()}
    )
    holdout_metrics.update(evaluate_counts(y_holdout, holdout_pred))

    model_path = MODELS_DIR / f"{slugify_city(city)}_monthly_seasonal.pkl"
    joblib.dump(best_pipeline, model_path)

    artifact = {
        "city": city,
        "model_path": str(model_path),
        "best_params": best_params,
        "cv_records": cv_records,
        "holdout_months": args.holdout_months,
        "holdout_start": holdout_df["month"].min(),
        "holdout_end": holdout_df["month"].max(),
        "feature_lists": {
            "numeric_dynamic": feature_lists.numeric_dynamic,
            "numeric_static": feature_lists.numeric_static,
            "categorical": feature_lists.categorical,
        },
        "metrics": holdout_metrics,
        "rows": {
            "train": len(train_df),
            "holdout": len(holdout_df),
            "total": len(panel),
        },
    }
    artifact_path = ARTIFACTS_DIR / f"{slugify_city(city)}_monthly_seasonal.json"
    with artifact_path.open("w", encoding="utf-8") as fp:
        json.dump(artifact, fp, indent=2, default=json_default)
    logging.info(
        "City %s: saved model to %s (holdout R2=%.3f, RMSE=%.3f)",
        city,
        model_path,
        holdout_metrics["r2"],
        holdout_metrics["rmse"],
    )
    return artifact


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    ensure_dirs()
    static_df = load_static_features(args.features_path)
    results = {}
    for idx, city in enumerate(args.cities):
        logging.info("Processing city %s", city)
        rng = np.random.default_rng(args.random_state + idx)
        try:
            panel, feature_lists = prepare_city_panel(city, static_df, args)
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            logging.exception("Failed to prepare city %s: %s", city, exc)
            continue
        artifact = train_city_model(city, panel, feature_lists, args, rng)
        results[city] = artifact
    summary_path = ARTIFACTS_DIR / "monthly_seasonal_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, default=json_default)
    logging.info("Completed monthly seasonal training for %d cities.", len(results))


if __name__ == "__main__":
    main()
