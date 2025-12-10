import functools
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Transformer
from sklearn.neighbors import KDTree
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
GRID_CONCEPTS_GLOB = "grid_concepts_*.parquet"
GRID_BEST_GLOB = "grid_best_*.parquet"
AREA_PERF_PATH = DATA_DIR / "area_perf.parquet"

# ---------------------------------------------------------------------------
# Coordinate Transformations
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_transformers() -> Tuple[Transformer, Transformer]:
    """
    Returns (lv95_to_wgs, wgs_to_lv95) transformers.
    LV95 = EPSG:2056 (Swiss coordinate system)
    WGS84 = EPSG:4326 (GPS coordinates)
    """
    lv95_to_wgs = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    wgs_to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    return lv95_to_wgs, wgs_to_lv95

def wgs_to_lv95(lon: float, lat: float) -> Tuple[float, float]:
    _, t = get_transformers()
    return t.transform(lon, lat)

def lv95_to_wgs(x: float, y: float) -> Tuple[float, float]:
    t, _ = get_transformers()
    return t.transform(x, y)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def list_available_cities() -> List[str]:
    """Scans data directory for available cities based on grid_concepts files."""
    files = sorted(DATA_DIR.glob(GRID_CONCEPTS_GLOB))
    cities = []
    for path in files:
        # Expected format: grid_concepts_zurich.parquet or grid_concepts_zurich_part1.parquet
        name = path.stem.replace("grid_concepts_", "")
        name = re.sub(r"_part\\d+$", "", name)
        name = name.replace("_", " ").title()
        cities.append(name)
    # Deduplicate while preserving sorted order
    return sorted(list(dict.fromkeys(cities)))

def city_to_slug(city: str) -> str:
    return city.lower().replace(" ", "_")

@st.cache_data
def load_grid_concepts(city: Optional[str] = None) -> pd.DataFrame:
    """Loads grid concepts data. If city is None, loads ALL cities (use with caution)."""
    if city and city != "All":
        slug = city_to_slug(city)
        # Support split files (e.g. grid_concepts_zurich_part1.parquet)
        pattern = f"grid_concepts_{slug}*.parquet"
        files = sorted(DATA_DIR.glob(pattern))
        
        if not files:
            return pd.DataFrame()
            
        dfs = []
        for path in files:
            dfs.append(pd.read_parquet(path))
            
        if not dfs:
            return pd.DataFrame()
            
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure city column exists
        if "city" not in df.columns:
            df["city"] = city
        return df
    else:
        # Load all
        frames = []
        for path in DATA_DIR.glob(GRID_CONCEPTS_GLOB):
            df = pd.read_parquet(path)
            # Try to infer city from filename if not present
            if "city" not in df.columns:
                c_name = path.stem.replace("grid_concepts_", "").replace("_", " ").title()
                df["city"] = c_name
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

@st.cache_data
def load_grid_best(city: Optional[str] = None) -> pd.DataFrame:
    """Loads grid best data."""
    if city and city != "All":
        slug = city_to_slug(city)
        path = DATA_DIR / f"grid_best_{slug}.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if "city" not in df.columns:
            df["city"] = city
        return df
    else:
        frames = []
        for path in DATA_DIR.glob(GRID_BEST_GLOB):
            df = pd.read_parquet(path)
            if "city" not in df.columns:
                c_name = path.stem.replace("grid_best_", "").replace("_", " ").title()
                df["city"] = c_name
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

@st.cache_data
def load_area_perf() -> pd.DataFrame:
    """Loads area performance (saturation) data."""
    if not AREA_PERF_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(AREA_PERF_PATH)

# ---------------------------------------------------------------------------
# Grid & Spatial Logic
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def get_city_kdtree(city: str) -> Tuple[Optional[KDTree], pd.DataFrame]:
    """Builds a KDTree for a specific city's grid centers."""
    df = load_grid_concepts(city)
    if df.empty:
        return None, pd.DataFrame()
    
    # Unique grid cells
    # Assuming x_lv95, y_lv95 are the centers
    coords = df[["grid_id", "x_lv95", "y_lv95"]].drop_duplicates().reset_index(drop=True)
    if coords.empty:
        return None, pd.DataFrame()
        
    tree = KDTree(coords[["x_lv95", "y_lv95"]].to_numpy())
    return tree, coords

def find_nearest_grid(city: str, lon: float, lat: float) -> Optional[str]:
    """Finds the nearest grid_id for a given WGS84 point in a specific city."""
    tree, coords = get_city_kdtree(city)
    if tree is None:
        return None
    
    x, y = wgs_to_lv95(lon, lat)
    dist, idx = tree.query([[x, y]], k=1)
    
    # Optional: Cutoff distance? For now, just take nearest.
    # If dist is too large (e.g. > 200m), maybe return None?
    # Let's stick to nearest for now as requested.
    
    return coords.iloc[idx[0][0]]["grid_id"]

def infer_step_size(df: pd.DataFrame) -> float:
    """Infers grid step size from coordinate differences."""
    if df.empty or len(df) < 2:
        return 100.0 # Default
    
    # Sample a subset for speed
    sample = df.head(1000)
    xs = np.sort(sample["x_lv95"].unique())
    ys = np.sort(sample["y_lv95"].unique())
    
    dx = np.diff(xs)
    dy = np.diff(ys)
    
    # Filter small diffs (floating point noise) and large jumps
    dx = dx[dx > 10] 
    dy = dy[dy > 10]
    
    if len(dx) == 0 and len(dy) == 0:
        return 100.0
        
    # Take median of valid diffs
    candidates = np.concatenate([dx, dy])
    if len(candidates) == 0:
        return 100.0
        
    return float(np.median(candidates))

def df_to_geojson(df: pd.DataFrame, value_col: str, tooltip_cols: List[str], limit: int = 2000) -> Dict:
    """
    Converts a dataframe with x_lv95, y_lv95 to a GeoJSON FeatureCollection of polygons.
    """
    if df.empty:
        return {"type": "FeatureCollection", "features": []}
    
    # Limit rows for performance
    df_subset = df.head(limit).copy()
    
    step = infer_step_size(df_subset)
    half = step / 2.0
    
    features = []
    
    # Batch transform for speed
    centers_x = df_subset["x_lv95"].to_numpy()
    centers_y = df_subset["y_lv95"].to_numpy()
    
    # Create corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left, Top-Left (closed loop)
    # But wait, we need to transform each corner to WGS84.
    # Vectorized approach:
    # 5 points per polygon
    
    # Simple iteration is likely fast enough for <2000 rows
    t, _ = get_transformers()
    
    for idx, row in df_subset.iterrows():
        cx, cy = row["x_lv95"], row["y_lv95"]
        
        # Square corners in LV95
        corners_x = [cx - half, cx + half, cx + half, cx - half, cx - half]
        corners_y = [cy + half, cy + half, cy - half, cy - half, cy + half]
        
        lons, lats = t.transform(corners_x, corners_y)
        
        # GeoJSON expects [lon, lat]
        coords = [[lo, la] for lo, la in zip(lons, lats)]
        
        props = {
            "grid_id": row.get("grid_id", str(idx)),
            "value": row.get(value_col, 0),
        }
        for col in tooltip_cols:
            if col in row:
                props[col] = row[col]
                
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            }
        })
        
    return {"type": "FeatureCollection", "features": features}

@st.cache_data
def geocode_address(address: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Geocodes an address string to (lat, lon).
    Returns (None, None) if not found or error.
    """
    try:
        geolocator = Nominatim(user_agent="restaurant_analytics_app")
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        return None, None
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None, None
