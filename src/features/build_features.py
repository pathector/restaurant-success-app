"""Feature builder for restaurant location + competition signals.

Loads the merged restaurant datasets plus microlocation POIs, projects
everything to Swiss LV95 coordinates, and computes:

* Microlocation counts (parking, transit, hotels, offices, schools, etc.)
* Distance-to-nearest POIs for a couple of Swiss-relevant categories
* Competition density within buffered radii
* Swiss-style 20-nearest-neighbour rating/review gap features

The script only reads the source CSVs and writes a new feature table under
``data/final`` – it never modifies the input databases.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import KDTree, NearestNeighbors

# File defaults
DEFAULT_CORE_PATH = Path("data/final/merged_core_cities.csv")
DEFAULT_EXTENDED_PATH = Path("data/final/merged_competition_extended.csv")
DEFAULT_MICRO_POI_PATH = Path("data/reference/collectors/osm/swiss_microlocation_poi.csv")
DEFAULT_OUTPUT_PATH = Path("data/final/features_location_competition.parquet")

# Column names
LAT_COL = "latitude"
LON_COL = "longitude"
POI_LAT_COL = "centroid_lat"
POI_LON_COL = "centroid_lon"

# Projection setup (WGS84 -> LV95 meters)
WGS84_EPSG = "EPSG:4326"
LV95_EPSG = "EPSG:2056"

# Microlocation POI configuration: (feature_name, target_categories, buffer_meters)
MICRO_POI_FEATURES: Tuple[Tuple[str, Sequence[str], float], ...] = (
    ("parking_300m", ("parking",), 300.0),
    ("busstop_200m", ("bus_stop",), 200.0),
    ("stations_600m", ("train_station",), 600.0),
    ("offices_400m", ("office",), 400.0),
    ("hotels_400m", ("hotel",), 400.0),
    ("schools_400m", ("school_university",), 400.0),
    ("attractions_500m", ("tourist_attraction",), 500.0),
    ("supermarkets_400m", ("supermarket",), 400.0),
    ("gyms_400m", ("gym",), 400.0),
)

# Nearest distance features reuse the same category filtering
NEAREST_POI_FEATURES: Tuple[Tuple[str, Sequence[str]], ...] = (
    ("dist_station_m", ("train_station",)),
    ("dist_hotel_m", ("hotel",)),
)

# Competition configs
COMP_RADIUS_FEATURES: Tuple[Tuple[str, float], ...] = (
    ("comp_count_200m", 200.0),
    ("comp_count_500m", 500.0),
)
COMP_HIGH_RATING_FEATURE = ("comp_highrating_200m", 200.0, 4.5)
NN_NEIGHBORS_DEFAULT = 20

COMP_TYPE_REGEX = (
    r"(?:restaurant|bar|pub|cafe|coffee|food|meal_takeaway|meal_delivery|bakery|"
    r"fast_food|bistro|diner|ice_cream|steakhouse|pizzeria|wine)"
)
COMP_AMENITY_ALLOW_LIST = {
    "restaurant",
    "fast_food",
    "cafe",
    "bar",
    "pub",
    "biergarten",
    "food_court",
    "ice_cream",
    "coffee_shop",
    "bbq",
}

# Restaurant attribute config
PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}
CUISINE_KEYWORDS: Tuple[str, ...] = (
    "italian",
    "pizza",
    "pasta",
    "japanese",
    "sushi",
    "chinese",
    "thai",
    "indian",
    "mexican",
    "burger",
    "kebab",
    "bbq",
    "greek",
    "turkish",
    "lebanese",
    "mediterranean",
    "french",
    "spanish",
    "tapas",
    "seafood",
    "steak",
    "vegan",
    "vegetarian",
    "bakery",
    "coffee",
    "brunch",
)
ESTABLISHMENT_KEYWORDS: Tuple[str, ...] = (
    "restaurant",
    "cafe",
    "bar",
    "fast_food",
    "meal_takeaway",
    "meal_delivery",
    "bistro",
    "pub",
    "wine_bar",
    "coffee_shop",
    "dessert",
    "night_club",
)
CITY_TOP_N = 25
GENERIC_TYPE_TOKENS = {
    "restaurant",
    "food",
    "point",
    "point_of_interest",
    "establishment",
    "store",
    "meal",
    "meal_takeaway",
    "meal_delivery",
    "delivery",
    "takeaway",
    "dine_in",
    "dine-in",
    "lodging",
    "place",
    "service",
}
IDENTIFIER_COLUMNS = ("id", "osm_id", "google_maps_uri")
RATING_SMOOTHING_ALPHA = 4.0
DISTANCE_CLIP_METERS = 5000.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build microlocation and competition features for Swiss restaurants."
    )
    parser.add_argument("--core-path", type=Path, default=DEFAULT_CORE_PATH, help="CSV with restaurants inside cities.")
    parser.add_argument(
        "--extended-path",
        type=Path,
        default=DEFAULT_EXTENDED_PATH,
        help="CSV with restaurants inside + buffer (competition pool).",
    )
    parser.add_argument(
        "--micropoi-path", type=Path, default=DEFAULT_MICRO_POI_PATH, help="OSM microlocation POI CSV."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the enriched feature table (parquet or csv).",
    )
    parser.add_argument(
        "--output-format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Storage format for the feature table.",
    )
    parser.add_argument(
        "--nn-k",
        type=int,
        default=NN_NEIGHBORS_DEFAULT,
        help="Number of nearest competitors used for Swiss-style rating gap features.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Verbosity of the feature builder logs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_restaurant_table(csv_path: Path) -> pd.DataFrame:
    """Load restaurants table and ensure critical columns are numeric."""

    df = pd.read_csv(csv_path)
    for col in (LAT_COL, LON_COL, "rating", "user_rating_count"):
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])
    return df


def load_microlocation_table(csv_path: Path) -> pd.DataFrame:
    """Load microlocation POIs (parking, bus stops, etc.)."""

    df = pd.read_csv(csv_path, low_memory=False)
    for col in (POI_LAT_COL, POI_LON_COL):
        df[col] = _coerce_numeric(df[col])
    df["category_primary"] = df["category_primary"].fillna("").str.lower()
    return df


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------


def project_points(lon: pd.Series, lat: pd.Series, transformer: Transformer) -> Tuple[np.ndarray, np.ndarray]:
    """Project lon/lat arrays to LV95 meters, returning coords + validity mask."""

    lon_series = _coerce_numeric(lon)
    lat_series = _coerce_numeric(lat)
    lon_arr = lon_series.to_numpy()
    lat_arr = lat_series.to_numpy()
    lon_arr = lon_arr.astype(float, copy=False)
    lat_arr = lat_arr.astype(float, copy=False)
    mask = np.isfinite(lon_arr) & np.isfinite(lat_arr)

    coords = np.full((len(lon_arr), 2), np.nan, dtype=float)
    if mask.any():
        x, y = transformer.transform(lon_arr[mask], lat_arr[mask])
        coords[mask, 0] = x
        coords[mask, 1] = y
    return coords, mask


def coords_valid_mask(coords: np.ndarray) -> np.ndarray:
    """Valid coordinates have finite X/Y pairs."""

    return np.isfinite(coords).all(axis=1)


def series_or_default(df: pd.DataFrame, column: str, default_value: object, dtype=None) -> pd.Series:
    if column in df.columns:
        return df[column]
    if len(df) == 0:
        return pd.Series(dtype=dtype)
    return pd.Series([default_value] * len(df), index=df.index, dtype=dtype)


def identifier_array(df: pd.DataFrame) -> np.ndarray:
    for col in IDENTIFIER_COLUMNS:
        if col in df.columns:
            arr = df[col].to_numpy()
            if not np.all(pd.isna(arr)):
                return arr
    return np.arange(len(df))


# ---------------------------------------------------------------------------
# Microlocation features
# ---------------------------------------------------------------------------


def _filter_pois(df: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    cat_set = {c.lower() for c in categories}
    return df[df["category_primary"].isin(cat_set)]


def compute_microlocation_counts(
    core_df: pd.DataFrame,
    core_coords: np.ndarray,
    core_valid_mask: np.ndarray,
    poi_df: pd.DataFrame,
    transformer: Transformer,
) -> None:
    """Attach POI counts within buffers for each restaurant."""

    if not core_valid_mask.any():
        logging.warning("No valid coordinates in core dataset – microlocation counts will be NaN.")
        for feature, _, _ in MICRO_POI_FEATURES:
            core_df[feature] = np.nan
        return

    valid_coords = core_coords[core_valid_mask]

    for feature_name, categories, radius in MICRO_POI_FEATURES:
        subset = _filter_pois(poi_df, categories)
        feature_values = np.full(len(core_df), np.nan, dtype=float)

        if subset.empty:
            logging.info("No POIs found for %s – filling zeros for valid rows.", feature_name)
            feature_values[core_valid_mask] = 0.0
            core_df[feature_name] = feature_values
            continue

        poi_coords, poi_mask = project_points(subset[POI_LON_COL], subset[POI_LAT_COL], transformer)
        poi_coords = poi_coords[poi_mask]

        if poi_coords.size == 0:
            feature_values[core_valid_mask] = 0.0
            core_df[feature_name] = feature_values
            continue

        tree = KDTree(poi_coords, metric="euclidean")
        counts = tree.query_radius(valid_coords, r=radius, count_only=True).astype(float)
        feature_values[core_valid_mask] = counts
        core_df[feature_name] = feature_values


def compute_nearest_distance_features(
    core_df: pd.DataFrame,
    core_coords: np.ndarray,
    core_valid_mask: np.ndarray,
    poi_df: pd.DataFrame,
    transformer: Transformer,
) -> None:
    """Attach distance to nearest POI of selected categories."""

    if not core_valid_mask.any():
        logging.warning("No valid coordinates in core dataset – distance features left as NaN.")
        for feature, _ in NEAREST_POI_FEATURES:
            core_df[feature] = np.nan
        return

    valid_coords = core_coords[core_valid_mask]

    for feature_name, categories in NEAREST_POI_FEATURES:
        subset = _filter_pois(poi_df, categories)
        feature_values = np.full(len(core_df), np.nan, dtype=float)

        if subset.empty:
            core_df[feature_name] = feature_values
            continue

        poi_coords, poi_mask = project_points(subset[POI_LON_COL], subset[POI_LAT_COL], transformer)
        poi_coords = poi_coords[poi_mask]

        if poi_coords.size == 0:
            core_df[feature_name] = feature_values
            continue

        tree = KDTree(poi_coords, metric="euclidean")
        distances, _ = tree.query(valid_coords, k=1)
        feature_values[core_valid_mask] = distances[:, 0]
        core_df[feature_name] = feature_values


def clip_and_log_distances(core_df: pd.DataFrame, columns: Sequence[str], clip_value: float = DISTANCE_CLIP_METERS) -> None:
    for col in columns:
        if col in core_df.columns:
            core_df[col] = core_df[col].clip(upper=clip_value)
            core_df[f"log_{col}"] = np.log1p(core_df[col])


# ---------------------------------------------------------------------------
# Competition helpers
# ---------------------------------------------------------------------------


def filter_competition_layer(ext_df: pd.DataFrame) -> pd.DataFrame:
    """Extract rows that behave like restaurants/bars from the extended dataset."""

    type_series = ext_df.get("types", pd.Series(dtype=str)).fillna("").str.lower()
    amenity_series = ext_df.get("primary_amenity", pd.Series(dtype=str)).fillna("").str.lower()

    type_mask = type_series.str.contains(COMP_TYPE_REGEX, regex=True)
    amenity_mask = amenity_series.isin(COMP_AMENITY_ALLOW_LIST)

    comp_df = ext_df[type_mask | amenity_mask].copy()
    for col in ("rating", "user_rating_count"):
        if col in comp_df.columns:
            comp_df[col] = _coerce_numeric(comp_df[col])
    return comp_df


def tokenize_types(series: pd.Series, drop_generic: bool = True) -> List[set]:
    values = series.fillna("").astype(str)
    tokens: List[set] = []
    for value in values:
        token_set: set = set()
        for raw in value.split("|"):
            token = raw.strip().lower()
            if not token:
                continue
            components = {token}
            cleaned = token.replace("-", "_")
            if "_" in cleaned:
                components.update(part for part in cleaned.split("_") if part)
            for comp in components:
                if drop_generic and comp in GENERIC_TYPE_TOKENS:
                    continue
                token_set.add(comp)
        tokens.append(token_set)
    return tokens


def compute_competition_radius_counts(
    core_df: pd.DataFrame,
    core_coords: np.ndarray,
    core_valid_mask: np.ndarray,
    comp_valid: pd.DataFrame,
    comp_coords_valid: np.ndarray,
    nn_pos_by_id: Dict[object, int],
) -> None:
    """Counts competitors within fixed radii plus high-rating density."""

    if not core_valid_mask.any():
        for feature, _ in COMP_RADIUS_FEATURES:
            core_df[feature] = np.nan
        core_df[COMP_HIGH_RATING_FEATURE[0]] = np.nan
        return

    valid_core_coords = core_coords[core_valid_mask]

    if comp_valid.empty or comp_coords_valid.size == 0:
        logging.warning("No valid coordinates in competition layer – radius counts left as NaN.")
        for feature, _ in COMP_RADIUS_FEATURES:
            core_df[feature] = np.nan
        core_df[COMP_HIGH_RATING_FEATURE[0]] = np.nan
        for col in ("comp200_avg_rating", "comp200_rating_gap", "comp200_avg_reviewcount", "log_comp200_sum_reviewcount"):
            core_df[col] = np.nan
        return

    comp_tree = KDTree(comp_coords_valid, metric="euclidean")
    core_ids = identifier_array(core_df)
    valid_core_idx = np.where(core_valid_mask)[0]
    comp_ratings = series_or_default(comp_valid, "rating", np.nan, dtype=float).to_numpy()
    comp_reviews = series_or_default(comp_valid, "user_rating_count", np.nan, dtype=float).to_numpy()
    own_ratings = series_or_default(core_df, "rating", np.nan, dtype=float).to_numpy()
    global_rating = float(np.nanmean(comp_ratings)) if np.isfinite(comp_ratings).any() else np.nan

    comp200_avg_rating = np.full(len(core_df), np.nan, dtype=float)
    comp200_rating_gap = np.full(len(core_df), np.nan, dtype=float)
    comp200_avg_reviewcount = np.full(len(core_df), np.nan, dtype=float)
    log_comp200_sum_reviewcount = np.full(len(core_df), np.nan, dtype=float)

    for feature_name, radius in COMP_RADIUS_FEATURES:
        vals = np.full(len(core_df), np.nan, dtype=float)
        neighbor_lists = comp_tree.query_radius(valid_core_coords, r=radius, return_distance=False)
        for rel_idx, neighbors in enumerate(neighbor_lists):
            core_idx = valid_core_idx[rel_idx]
            core_id = core_ids[core_idx]
            self_pos = nn_pos_by_id.get(core_id)
            filtered = neighbors if self_pos is None else neighbors[neighbors != self_pos]
            vals[core_idx] = float(len(filtered))

            if feature_name == "comp_count_200m" and filtered.size > 0:
                neigh_ratings = comp_ratings[filtered]
                valid_ratings = neigh_ratings[np.isfinite(neigh_ratings)]
                if valid_ratings.size > 0:
                    local_avg = float(np.nanmean(valid_ratings))
                    if np.isfinite(global_rating):
                        n = valid_ratings.size
                        local_avg = (RATING_SMOOTHING_ALPHA * global_rating + n * local_avg) / (
                            RATING_SMOOTHING_ALPHA + n
                        )
                    comp200_avg_rating[core_idx] = local_avg
                    own_rating = own_ratings[core_idx]
                    if np.isfinite(own_rating):
                        comp200_rating_gap[core_idx] = local_avg - own_rating

                neigh_reviews = comp_reviews[filtered]
                valid_reviews = neigh_reviews[np.isfinite(neigh_reviews)]
                if valid_reviews.size > 0:
                    comp200_avg_reviewcount[core_idx] = float(np.nanmean(valid_reviews))
                    log_comp200_sum_reviewcount[core_idx] = float(np.log1p(np.nansum(valid_reviews)))

        core_df[feature_name] = vals

    core_df["comp200_avg_rating"] = comp200_avg_rating
    core_df["comp200_rating_gap"] = comp200_rating_gap
    core_df["comp200_avg_reviewcount"] = comp200_avg_reviewcount
    core_df["log_comp200_sum_reviewcount"] = log_comp200_sum_reviewcount

    # High-rating competition
    thr_feature, thr_radius, thr_rating = COMP_HIGH_RATING_FEATURE
    high_vals = np.full(len(core_df), np.nan, dtype=float)
    high_mask = comp_ratings >= thr_rating
    high_neighbor_lists = comp_tree.query_radius(valid_core_coords, r=thr_radius, return_distance=False)
    for rel_idx, neighbors in enumerate(high_neighbor_lists):
        core_idx = valid_core_idx[rel_idx]
        core_id = core_ids[core_idx]
        self_pos = nn_pos_by_id.get(core_id)
        filtered = neighbors if self_pos is None else neighbors[neighbors != self_pos]
        high_filtered = filtered[high_mask[filtered]]
        high_vals[core_idx] = float(len(high_filtered))
    core_df[thr_feature] = high_vals


def compute_knn_competition_features(
    core_df: pd.DataFrame,
    core_coords: np.ndarray,
    core_valid_mask: np.ndarray,
    comp_valid: pd.DataFrame,
    comp_coords_valid: np.ndarray,
    nn_pos_by_id: Dict[object, int],
    k_neighbors: int,
) -> None:
    """Swiss-style KNN competition block (distance, rating gap, similarity)."""

    feature_templates = {
        "nn_mean_distance_m": np.full(len(core_df), np.nan, dtype=float),
        "nn_avg_rating": np.full(len(core_df), np.nan, dtype=float),
        "nn_avg_reviewcount": np.full(len(core_df), np.nan, dtype=float),
        "nn_rating_gap": np.full(len(core_df), np.nan, dtype=float),
        "nn_similarity_count": np.full(len(core_df), np.nan, dtype=float),
    }

    for key, arr in feature_templates.items():
        core_df[key] = arr

    if not core_valid_mask.any() or comp_valid.empty:
        logging.warning("Cannot compute KNN features (no valid coords or competition rows).")
        return

    valid_core_coords = core_coords[core_valid_mask]
    n_comp = len(comp_valid)
    neighbors = min(k_neighbors + 1, n_comp)  # +1 to safely drop self

    nn_model = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
    nn_model.fit(comp_coords_valid)
    distances, indices = nn_model.kneighbors(valid_core_coords)

    comp_ratings = series_or_default(comp_valid, "rating", np.nan, dtype=float).to_numpy()
    comp_review_counts = series_or_default(comp_valid, "user_rating_count", np.nan, dtype=float).to_numpy()
    comp_tokens = tokenize_types(series_or_default(comp_valid, "types", "", dtype=object), drop_generic=True)
    core_tokens = tokenize_types(series_or_default(core_df, "types", "", dtype=object), drop_generic=True)
    core_ratings = series_or_default(core_df, "rating", np.nan, dtype=float).to_numpy()

    valid_core_idx = np.where(core_valid_mask)[0]

    for row_pos, core_idx in enumerate(valid_core_idx):
        dists = distances[row_pos]
        neigh_idx = indices[row_pos]

        core_id = core_df["id"].iat[core_idx] if "id" in core_df.columns else core_idx
        self_pos = nn_pos_by_id.get(core_id)

        if self_pos is not None:
            keep_mask = neigh_idx != self_pos
            dists = dists[keep_mask]
            neigh_idx = neigh_idx[keep_mask]

        if len(neigh_idx) == 0:
            continue

        # Limit to K neighbors after removing self
        dists = dists[:k_neighbors]
        neigh_idx = neigh_idx[:k_neighbors]

        if len(neigh_idx) == 0:
            continue

        feature_templates["nn_mean_distance_m"][core_idx] = float(np.mean(dists))

        neigh_ratings = comp_ratings[neigh_idx]
        if np.isfinite(neigh_ratings).any():
            avg_rating = float(np.nanmean(neigh_ratings))
            feature_templates["nn_avg_rating"][core_idx] = avg_rating
            own_rating = core_ratings[core_idx] if core_ratings.size > core_idx else np.nan
            if np.isfinite(own_rating):
                feature_templates["nn_rating_gap"][core_idx] = avg_rating - own_rating

        neigh_reviews = comp_review_counts[neigh_idx]
        if np.isfinite(neigh_reviews).any():
            feature_templates["nn_avg_reviewcount"][core_idx] = float(np.nanmean(neigh_reviews))

        tokens_core = core_tokens[core_idx] if core_idx < len(core_tokens) else set()
        if tokens_core:
            sim_count = sum(1 for token_set in (comp_tokens[i] for i in neigh_idx) if token_set & tokens_core)
        else:
            sim_count = 0
        feature_templates["nn_similarity_count"][core_idx] = float(sim_count)

    for key, arr in feature_templates.items():
        core_df[key] = arr


# ---------------------------------------------------------------------------
# Restaurant attribute features
# ---------------------------------------------------------------------------


def slugify_value(value: object) -> str:
    if pd.isna(value) or value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def add_price_level_feature(df: pd.DataFrame) -> None:
    if "price_level" not in df.columns:
        df["price_level_num"] = 0
        return
    series = df["price_level"].fillna("").astype(str).str.upper()
    df["price_level_num"] = series.map(PRICE_LEVEL_MAP).fillna(0).astype(int)


def add_type_keyword_flags(df: pd.DataFrame) -> None:
    types_series = series_or_default(df, "types", "", dtype=object)
    token_sets_clean = tokenize_types(types_series, drop_generic=True)
    token_sets_full = tokenize_types(types_series, drop_generic=False)

    def build_flag(token_sets: List[set], keyword: str) -> pd.Series:
        keyword_norm = keyword.lower().replace(" ", "_")
        data = [1 if keyword_norm in tokens else 0 for tokens in token_sets]
        return pd.Series(data, index=df.index, dtype=int)

    for keyword in CUISINE_KEYWORDS:
        col = f"cuisine_{keyword.replace(' ', '_')}"
        df[col] = build_flag(token_sets_clean, keyword)

    for keyword in ESTABLISHMENT_KEYWORDS:
        col = f"is_{keyword.replace(' ', '_')}"
        df[col] = build_flag(token_sets_full, keyword)


def add_city_dummies(df: pd.DataFrame, top_n: int = CITY_TOP_N) -> None:
    city_col = "__SelectedCity__" if "__SelectedCity__" in df.columns else "locality"
    if city_col not in df.columns:
        return

    series = df[city_col].fillna("Unknown").astype(str)
    if series.eq("").all():
        return

    top_cities = series.value_counts().head(top_n).index.tolist()
    for city in top_cities:
        slug = slugify_value(city)
        df[f"city_{slug}"] = (series == city).astype(int)

    df["city_other"] = (~series.isin(top_cities)).astype(int)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def add_log_transforms(core_df: pd.DataFrame, columns: Iterable[str]) -> None:
    new_cols = {}
    for col in columns:
        if col in core_df.columns:
            new_cols[f"log_{col}"] = np.log1p(core_df[col])
    if new_cols:
        core_df[list(new_cols.keys())] = pd.DataFrame(new_cols)


def write_output(df: pd.DataFrame, output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    logging.info("Features written to %s (%s rows).", output_path, len(df))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Loading source tables…")
    rest_core = load_restaurant_table(args.core_path)
    rest_extended = load_restaurant_table(args.extended_path)
    microlocation = load_microlocation_table(args.micropoi_path)
    logging.info(
        "Loaded %d core restaurants, %d extended restaurants, %d POIs.",
        len(rest_core),
        len(rest_extended),
        len(microlocation),
    )

    transformer = Transformer.from_crs(WGS84_EPSG, LV95_EPSG, always_xy=True)

    core_coords, core_mask = project_points(rest_core[LON_COL], rest_core[LAT_COL], transformer)
    comp_layer = filter_competition_layer(rest_extended)
    comp_coords, _ = project_points(comp_layer[LON_COL], comp_layer[LAT_COL], transformer)
    comp_valid_mask = coords_valid_mask(comp_coords)
    comp_valid_idx = np.where(comp_valid_mask)[0]
    comp_valid = comp_layer.iloc[comp_valid_idx].reset_index(drop=True)
    comp_coords_valid = comp_coords[comp_valid_mask]

    nn_pos_by_id: Dict[object, int] = {}
    if not comp_valid.empty and "id" in comp_valid.columns:
        for pos, value in enumerate(comp_valid["id"]):
            if pd.isna(value):
                continue
            nn_pos_by_id.setdefault(value, pos)

    logging.info("Computing microlocation features…")
    compute_microlocation_counts(rest_core, core_coords, core_mask, microlocation, transformer)
    compute_nearest_distance_features(rest_core, core_coords, core_mask, microlocation, transformer)
    clip_and_log_distances(rest_core, ("dist_station_m", "dist_hotel_m"))

    logging.info("Computing competition density features…")
    compute_competition_radius_counts(rest_core, core_coords, core_mask, comp_valid, comp_coords_valid, nn_pos_by_id)

    logging.info("Computing %d-NN competition features…", args.nn_k)
    if comp_valid.empty:
        logging.warning("Competition layer is empty after filtering; KNN features will be NaN.")
    else:
        compute_knn_competition_features(
            rest_core,
            core_coords,
            core_mask,
            comp_valid,
            comp_coords_valid,
            nn_pos_by_id,
            args.nn_k,
        )

    logging.info("Adding restaurant attribute features…")
    add_price_level_feature(rest_core)
    add_type_keyword_flags(rest_core)
    add_city_dummies(rest_core)

    rest_core = rest_core.copy()
    count_columns = [cfg[0] for cfg in MICRO_POI_FEATURES] + [cfg[0] for cfg in COMP_RADIUS_FEATURES] + [
        COMP_HIGH_RATING_FEATURE[0]
    ]
    add_log_transforms(rest_core, count_columns)

    logging.info("Writing enriched feature table…")
    write_output(rest_core, args.output_path, args.output_format)


if __name__ == "__main__":
    main()
