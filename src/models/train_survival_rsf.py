"""Train a Random Survival Forest to estimate restaurant closure risk.

The script:
1. Matches the closed-venue snapshot from OSM/Google quality control data
   to the curated FINAL_MERGED dataset using the same loose matching rules
   relied upon for dataset assembly.
2. Builds a survival-analysis table with closure events and censored rows,
   computing exposure durations inside the 7-year observation window.
3. Fits a scikit-survival RandomSurvivalForest using the existing
   microlocation + competition features and reports evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from analysis import merge_competition_sources as merger
from src.features import build_features as feat

FEATURES_PATH = Path("data/final/features_location_competition.parquet")
FINAL_CORE_PATH = Path("data/final/merged_core_cities.csv")
CLOSED_VENUES_PATH = Path("data/raw/Closed_OSM_ohsome/closed_food_venues_7y copy.csv")
SURVIVAL_DATASET_PATH = Path("data/final/closure_survival_dataset.parquet")
MATCHED_EXPORT_PATH = Path("data/final/matched_closed_venues.csv")
MODEL_OUTPUT_PATH = Path("models/survival_rsf.joblib")
METRICS_OUTPUT_PATH = Path("artifacts/survival_rsf_metrics.json")
MICRO_POI_PATH = feat.DEFAULT_MICRO_POI_PATH
COMPETITION_PATH = feat.DEFAULT_EXTENDED_PATH

META_COLUMNS = {
    "id",
    "display_name",
    "formatted_address",
    "google_maps_uri",
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
    "business_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build survival dataset and train a Random Survival Forest for closure risk."
    )
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH, help="Feature parquet from build_features.")
    parser.add_argument("--core-path", type=Path, default=FINAL_CORE_PATH, help="FINAL_MERGED CSV for matching.")
    parser.add_argument(
        "--closed-path", type=Path, default=CLOSED_VENUES_PATH, help="CSV with closed venues in the last 7 years."
    )
    parser.add_argument(
        "--survival-output",
        type=Path,
        default=SURVIVAL_DATASET_PATH,
        help="Where to write the labeled survival dataset (parquet or csv).",
    )
    parser.add_argument(
        "--matched-output",
        type=Path,
        default=MATCHED_EXPORT_PATH,
        help="Optional CSV export describing the closed-venue matches.",
    )
    parser.add_argument("--model-output", type=Path, default=MODEL_OUTPUT_PATH, help="Path to persist the fitted model.")
    parser.add_argument("--metrics-output", type=Path, default=METRICS_OUTPUT_PATH, help="JSON with evaluation metrics.")
    parser.add_argument(
        "--snapshot-date",
        type=str,
        default=None,
        help="Override for the censoring snapshot (YYYY-MM-DD). Defaults to max closed_at date.",
    )
    parser.add_argument(
        "--lookback-years",
        type=float,
        default=7.0,
        help="Observation window length used to compute exposure durations.",
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of GroupKFold splits for CV.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for random split evaluation.")
    parser.add_argument("--n-estimators", type=int, default=400, help="Trees in the RandomSurvivalForest.")
    parser.add_argument("--min-samples-split", type=int, default=8, help="Minimum samples required to split an internal node.")
    parser.add_argument("--min-samples-leaf", type=int, default=4, help="Minimum samples required to be at a leaf node.")
    parser.add_argument("--max-depth", type=int, default=None, help="Max depth for the RSF trees.")
    parser.add_argument("--max-features", default="sqrt", help="Features considered when looking for the best split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for the RSF.")
    parser.add_argument(
        "--micropoi-path",
        type=Path,
        default=MICRO_POI_PATH,
        help="Microlocation POI CSV used for feature recomputation of closed venues.",
    )
    parser.add_argument(
        "--competition-path",
        type=Path,
        default=COMPETITION_PATH,
        help="Extended competition CSV used for feature recomputation of closed venues.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Logging level.")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_closed_address(value: object) -> Tuple[str, str, str, str]:
    if not isinstance(value, str):
        return "", "", "", ""
    parts = [part.strip() for part in value.split(",") if part and part.strip()]
    street = parts[0] if parts else ""
    number = ""
    postal = ""
    city = ""
    for token in parts[1:]:
        if not number and any(ch.isdigit() for ch in token) and not token.isalpha():
            number = token
            continue
        if not postal and token.replace(" ", "").isdigit() and len(token.replace(" ", "")) in (3, 4):
            postal = token.replace(" ", "")
            continue
        if not city:
            city = token
    return street, number, postal, city


def prepare_closed_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["lat", "lon", "closed_at"]).copy()
    df["latitude"] = pd.to_numeric(df["lat"], errors="coerce").round(7)
    df["longitude"] = pd.to_numeric(df["lon"], errors="coerce").round(7)
    df = df.dropna(subset=["latitude", "longitude"])
    df["osm_id"] = df["osm_id"].astype(str)
    df["primary_amenity"] = df["amenity"].fillna("").str.lower()
    df = df[df["primary_amenity"].isin(merger.AMENITY_TYPES.keys())].reset_index(drop=True)
    df["name_osm"] = df["name"].fillna("")
    df["display_name"] = df["name_osm"]
    df["closest_city"] = df["zone"].fillna("").str.split("_").str[0].str.replace("-", " ").str.replace("_", " ")
    df["__SelectedCity__"] = df["closest_city"].replace("", pd.NA)
    parsed_addr = df["addr"].apply(parse_closed_address)
    df["addr_street"] = parsed_addr.apply(lambda item: item[0])
    df["addr_number"] = parsed_addr.apply(lambda item: item[1])
    df["addr_postcode"] = parsed_addr.apply(lambda item: item[2])
    df["addr_city"] = parsed_addr.apply(lambda item: item[3])
    norm_results = df["name_osm"].fillna("").apply(merger.linker.normalize_label)
    df["__norm_name"] = norm_results.apply(lambda item: item[0])
    df["__tokens"] = norm_results.apply(lambda item: item[1])
    df["__street_norm"] = df["addr_street"].apply(merger.normalize_simple)
    df["__number_norm"] = df["addr_number"].apply(merger.normalize_simple)
    df["__postal_code_norm"] = df["addr_postcode"].apply(merger.normalize_simple)
    df["__city_norm"] = df["addr_city"].apply(merger.normalize_city)
    return df.reset_index(drop=True)


def prepare_google_df(path: Path) -> pd.DataFrame:
    df = merger.prepare_google_df(path)
    df["id"] = df["id"].fillna("")
    df = df[df["id"].astype(str).str.len() > 0].reset_index(drop=True)
    return df


def match_closed_to_open(
    google_df: pd.DataFrame, closed_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    tree, _ = merger.build_ball_tree(google_df)
    matches, stats = merger.run_loose_matching(google_df, closed_df, tree, {}, set())
    records: List[Dict[str, object]] = []
    for row_idx, info in matches.items():
        closed_row = closed_df.iloc[row_idx]
        records.append(
            {
                "closed_osm_id": closed_row.get("osm_id"),
                "closed_name": closed_row.get("name_osm"),
                "closed_at": closed_row.get("closed_at"),
                "zone": closed_row.get("zone"),
                "amenity": closed_row.get("primary_amenity"),
                "google_id": info.google_id,
                "match_reason": info.reason,
                "distance_m": info.distance_m,
                "jw": info.jw,
                "token_subset": info.token_subset,
                "closed_index": row_idx,
            }
        )
    match_df = pd.DataFrame.from_records(records)
    if match_df.empty:
        return match_df, stats
    match_df = match_df.sort_values(["google_id", "closed_at"]).drop_duplicates("google_id", keep="first")
    return match_df.reset_index(drop=True), stats


def prepare_closed_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["id"] = base["osm_id"].apply(lambda value: f"closed_osm_{value}")
    base["display_name"] = base["name_osm"].replace("", pd.NA)
    base["formatted_address"] = df["addr"]
    base["google_maps_uri"] = pd.NA
    base["business_status"] = "CLOSED_PERMANENTLY"
    base["price_level"] = pd.NA
    base["rating"] = np.nan
    base["user_rating_count"] = np.nan
    base["types"] = df["primary_amenity"].apply(merger.map_amenity_types)
    locality = df["addr_city"].replace("", pd.NA).fillna(df["closest_city"])
    base["locality"] = locality
    base["admin_area_level_1"] = df["closest_city"]
    base["postal_code"] = df["addr_postcode"]
    base["country"] = "Switzerland"
    base["country_code"] = "CH"
    base["postal_address_lines"] = df["addr"]
    base["weekday_descriptions"] = pd.NA
    base["accessibility_options"] = pd.NA
    base["__SelectedCity__"] = df["closest_city"].replace("", pd.NA).fillna(locality)
    base["name_google"] = base["display_name"]
    base["closest_city"] = df["closest_city"]
    base["inside_any_city_bounds"] = False
    base["closed_at"] = df["closed_at"]
    return base


def build_closed_feature_table(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    base = prepare_closed_feature_frame(df)
    transformer = Transformer.from_crs(feat.WGS84_EPSG, feat.LV95_EPSG, always_xy=True)
    core_coords, core_mask = feat.project_points(base[feat.LON_COL], base[feat.LAT_COL], transformer)
    microlocation = feat.load_microlocation_table(args.micropoi_path)
    competition_raw = feat.load_restaurant_table(args.competition_path)
    comp_layer = feat.filter_competition_layer(competition_raw)
    comp_coords, _ = feat.project_points(comp_layer[feat.LON_COL], comp_layer[feat.LAT_COL], transformer)
    comp_valid_mask = feat.coords_valid_mask(comp_coords)
    comp_valid = comp_layer.iloc[np.where(comp_valid_mask)[0]].reset_index(drop=True)
    comp_coords_valid = comp_coords[comp_valid_mask]
    nn_pos_by_id: Dict[object, int] = {}
    if not comp_valid.empty and "id" in comp_valid.columns:
        for pos, value in enumerate(comp_valid["id"]):
            if pd.isna(value):
                continue
            nn_pos_by_id.setdefault(value, pos)
    feat.compute_microlocation_counts(base, core_coords, core_mask, microlocation, transformer)
    feat.compute_nearest_distance_features(base, core_coords, core_mask, microlocation, transformer)
    feat.clip_and_log_distances(base, ("dist_station_m", "dist_hotel_m"))
    feat.compute_competition_radius_counts(base, core_coords, core_mask, comp_valid, comp_coords_valid, nn_pos_by_id)
    if comp_valid.empty:
        logging.warning("Competition layer empty when building closed features; KNN features will be NaN.")
    else:
        feat.compute_knn_competition_features(
            base,
            core_coords,
            core_mask,
            comp_valid,
            comp_coords_valid,
            nn_pos_by_id,
            feat.NN_NEIGHBORS_DEFAULT,
        )
    feat.add_price_level_feature(base)
    feat.add_type_keyword_flags(base)
    feat.add_city_dummies(base)
    count_columns = [cfg[0] for cfg in feat.MICRO_POI_FEATURES] + [cfg[0] for cfg in feat.COMP_RADIUS_FEATURES]
    count_columns.append(feat.COMP_HIGH_RATING_FEATURE[0])
    feat.add_log_transforms(base, count_columns)
    return base


def derive_snapshot_date(match_df: pd.DataFrame, closed_df: pd.DataFrame, override: str | None) -> pd.Timestamp:
    if override:
        return pd.Timestamp(override).tz_localize(None)
    candidates = pd.concat([match_df["closed_at"], closed_df["closed_at"]], axis=0).dropna()
    if candidates.empty:
        return pd.Timestamp.today().normalize()
    return candidates.max().normalize()


def build_survival_dataset(
    open_df: pd.DataFrame,
    closed_df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    lookback_years: float,
) -> pd.DataFrame:
    open_part = open_df.copy()
    open_part["event_closed"] = 0
    open_part["event_date"] = snapshot_date
    closed_part = closed_df.copy()
    if not closed_part.empty:
        closed_part["event_closed"] = 1
        closed_part["event_date"] = pd.to_datetime(closed_part["closed_at"], errors="coerce").dt.tz_localize(
            None
        )
        closed_part = closed_part.dropna(subset=["event_date"])
    combined = pd.concat([open_part, closed_part], ignore_index=True, sort=False)
    observation_start = snapshot_date - pd.DateOffset(years=lookback_years)
    combined["reference_date"] = observation_start
    durations = (combined["event_date"] - observation_start).dt.days.clip(lower=1)
    combined["duration_months"] = (durations / 30.0).astype(float)
    return combined


def align_feature_frames(open_df: pd.DataFrame, closed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if closed_df.empty:
        return open_df, closed_df
    open_aligned = open_df.copy()
    closed_aligned = closed_df.copy()
    open_columns = set(open_aligned.columns)
    extra_cols = [col for col in closed_aligned.columns if col not in open_columns and col != "closed_at"]
    if extra_cols:
        closed_aligned = closed_aligned.drop(columns=extra_cols)
    if "closed_at" not in open_aligned.columns:
        open_aligned["closed_at"] = pd.NaT
    if "closed_at" not in closed_aligned.columns:
        closed_aligned["closed_at"] = pd.NaT
    for column in closed_aligned.columns:
        if column not in open_aligned.columns:
            open_aligned[column] = np.nan
    for column in open_aligned.columns:
        if column not in closed_aligned.columns:
            closed_aligned[column] = np.nan
    closed_aligned = closed_aligned[open_aligned.columns]
    return open_aligned, closed_aligned


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = list(df.select_dtypes(include=[np.number, "bool"]).columns)
    cols = [
        col
        for col in numeric_cols
        if col not in META_COLUMNS and col not in {"event_closed", "duration_months"}
    ]
    return cols


def prepare_model_inputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    feature_cols = select_feature_columns(df)
    df = df.copy()
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    X = df[feature_cols]
    y = Surv.from_arrays(event=df["event_closed"].astype(bool), time=df["duration_months"].astype(float))
    groups = df["__SelectedCity__"].fillna("unknown").to_numpy()
    return X, y, groups, feature_cols


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rsf",
                RandomSurvivalForest(
                    n_estimators=args.n_estimators,
                    min_samples_split=args.min_samples_split,
                    min_samples_leaf=args.min_samples_leaf,
                    max_depth=args.max_depth,
                    max_features=args.max_features,
                    n_jobs=args.n_jobs,
                    random_state=args.random_state,
                ),
            ),
        ]
    )


def compute_cv_scores(
    args: argparse.Namespace,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> List[float]:
    scores: List[float] = []
    splitter = GroupKFold(n_splits=args.cv_splits)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y["event"], groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = build_pipeline(args)
        model.fit(X_train, y_train)
        risk_scores = model.predict(X_test)
        c_index = concordance_index_censored(y_test["event"], y_test["time"], risk_scores)[0]
        scores.append(float(c_index))
        logging.debug("CV fold %d → c-index %.4f", fold_idx + 1, c_index)
    return scores


def evaluate_holdout(
    args: argparse.Namespace,
    X: pd.DataFrame,
    y: np.ndarray,
) -> Dict[str, float]:
    events = y["event"]
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=events,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = build_pipeline(args)
    model.fit(X_train, y_train)
    risk_scores = model.predict(X_test)
    holdout_cindex = concordance_index_censored(y_test["event"], y_test["time"], risk_scores)[0]
    X_test_imputed = model.named_steps["imputer"].transform(X_test)
    surv_funcs = model.named_steps["rsf"].predict_survival_function(X_test_imputed)
    eval_times = np.asarray(model.named_steps["rsf"].unique_times_, dtype=float)
    min_allowed = y_test["time"].min()
    max_allowed = y_test["time"].max()
    if max_allowed <= min_allowed:
        max_allowed = min_allowed + 1e-3
    mask = (eval_times >= min_allowed) & (eval_times < max_allowed)
    if not np.any(mask):
        upper = max_allowed - 1e-6
        if upper <= min_allowed:
            upper = min_allowed + 1e-6
        eval_times = np.linspace(min_allowed, upper, num=50)
    else:
        eval_times = eval_times[mask]
    surv_preds = np.asarray([[fn(t) for t in eval_times] for fn in surv_funcs])
    ibs = integrated_brier_score(y_train, y_test, surv_preds, eval_times)
    return {
        "holdout_c_index": float(holdout_cindex),
        "holdout_ibs": float(ibs),
    }


def fit_final_model(args: argparse.Namespace, X: pd.DataFrame, y: np.ndarray) -> Tuple[Pipeline, float]:
    model = build_pipeline(args)
    model.fit(X, y)
    risk_scores = model.predict(X)
    c_index = concordance_index_censored(y["event"], y["time"], risk_scores)[0]
    return model, float(c_index)


def summarize_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: Sequence[str],
    random_state: int,
    n_repeats: int = 2,
    n_jobs: int = 1,
    max_rows: int = 800,
) -> List[Dict[str, float]]:
    if len(X) > max_rows > 0:
        rng = np.random.default_rng(random_state)
        sample_idx = np.sort(rng.choice(len(X), size=max_rows, replace=False))
        X_eval = X.iloc[sample_idx]
        y_eval = y[sample_idx]
    else:
        X_eval = X
        y_eval = y
    try:
        result = permutation_importance(
            model,
            X_eval,
            y_eval,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    except Exception as exc:
        logging.warning("Permutation importance failed: %s", exc)
        return []
    pairs = [
        {"feature": feature_cols[idx], "importance": float(mean)}
        for idx, mean in enumerate(result.importances_mean)
    ]
    pairs.sort(key=lambda item: item["importance"], reverse=True)
    return pairs


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Loading datasets …")
    features_df = pd.read_parquet(args.features_path)
    google_df = prepare_google_df(args.core_path)
    closed_df = prepare_closed_df(args.closed_path)
    match_df, match_stats = match_closed_to_open(google_df, closed_df)
    matched_indices = set(match_df["closed_index"]) if not match_df.empty else set()
    unmatched_closed = closed_df.loc[~closed_df.index.isin(matched_indices)].reset_index(drop=True)
    closed_features = build_closed_feature_table(unmatched_closed, args)
    open_features = features_df.copy()
    open_features["closed_at"] = pd.NaT
    open_features, closed_features = align_feature_frames(open_features, closed_features)
    snapshot_date = derive_snapshot_date(match_df, closed_df, args.snapshot_date)
    survival_df = build_survival_dataset(open_features, closed_features, snapshot_date, args.lookback_years)
    for text_col in ("postal_code", "osm_id"):
        if text_col in survival_df.columns:
            survival_df[text_col] = survival_df[text_col].astype("string")
    logging.info(
        "Closed venues used as events: %d | Matched/ignored closures: %d",
        len(closed_features),
        len(match_df),
    )

    ensure_parent(args.survival_output)
    out_format = args.survival_output.suffix.lower()
    if out_format.endswith(".csv"):
        survival_df.to_csv(args.survival_output, index=False)
    else:
        survival_df.to_parquet(args.survival_output, index=False)
    logging.info("Wrote survival dataset (%d rows) to %s", len(survival_df), args.survival_output)

    if not match_df.empty:
        ensure_parent(args.matched_output)
        match_df.to_csv(args.matched_output, index=False)
        logging.info("Exported %d closed matches to %s", len(match_df), args.matched_output)

    X, y, groups, feature_cols = prepare_model_inputs(survival_df)
    logging.info("Running %d-fold grouped CV …", args.cv_splits)
    cv_scores = compute_cv_scores(args, X, y, groups)
    logging.info("CV c-index: mean=%.4f std=%.4f", np.mean(cv_scores), np.std(cv_scores))

    logging.info("Evaluating random holdout …")
    holdout_metrics = evaluate_holdout(args, X, y)

    logging.info("Fitting final RSF …")
    final_model, train_cindex = fit_final_model(args, X, y)
    ensure_parent(args.model_output)
    joblib.dump({"pipeline": final_model, "feature_columns": feature_cols}, args.model_output)
    logging.info("Saved RSF model to %s", args.model_output)

    feature_importance = summarize_feature_importance(
        final_model,
        X,
        y,
        feature_cols,
        random_state=args.random_state,
        n_jobs=args.n_jobs if args.n_jobs and args.n_jobs > 0 else 1,
    )
    metrics = {
        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "lookback_years": args.lookback_years,
        "n_rows": len(survival_df),
        "n_events": int(survival_df["event_closed"].sum()),
        "cv_c_index_mean": float(np.mean(cv_scores)),
        "cv_c_index_std": float(np.std(cv_scores)),
        "train_c_index": train_cindex,
        **holdout_metrics,
        "match_stats": match_stats,
        "matched_closed_rows": len(match_df),
        "used_closed_rows": len(closed_features),
        "top_feature_importance": feature_importance[:20],
    }
    ensure_parent(args.metrics_output)
    with args.metrics_output.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logging.info("Metrics written to %s", args.metrics_output)


if __name__ == "__main__":
    main()
