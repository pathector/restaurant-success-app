import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Ensure local src/ is importable when running via Streamlit
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from src import app_utils

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Restaurant Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Sidebar & Global State
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
available_cities = app_utils.list_available_cities()
selected_city = st.sidebar.selectbox("Select City", ["All"] + available_cities, index=1)

# ---------------------------------------------------------------------------
# Helper UI Functions
# ---------------------------------------------------------------------------
def render_map(df: pd.DataFrame, color_col: str, tooltip_cols: List[str], title: str, color_scale="Viridis", reverse_scale=False, fixed_range=None):
    """Renders a choropleth map using Plotly."""
    if df.empty:
        st.info("No data to display on map.")
        return

    # Generate GeoJSON
    geojson = app_utils.df_to_geojson(df, color_col, tooltip_cols, limit=1000)
    
    if not geojson["features"]:
        st.warning("Could not generate map geometry.")
        return

    # Calculate center
    center_lat = 46.8
    center_lon = 8.2
    zoom = 7
    
    if "x_lv95" in df.columns and "y_lv95" in df.columns:
        mx = df["x_lv95"].mean()
        my = df["y_lv95"].mean()
        clon, clat = app_utils.lv95_to_wgs(mx, my)
        center_lat = clat
        center_lon = clon
        zoom = 11

    fig = px.choropleth_mapbox(
        df.head(1000), 
        geojson=geojson,
        locations="grid_id", 
        featureidkey="properties.grid_id",
        color=color_col,
        color_continuous_scale=color_scale,
        range_color=fixed_range if fixed_range else (df[color_col].min(), df[color_col].max()),
        mapbox_style="carto-positron",
        zoom=zoom,
        center={"lat": center_lat, "lon": center_lon},
        opacity=0.6,
        hover_data=tooltip_cols,
        title=title
    )
    
    if reverse_scale:
        fig.update_layout(coloraxis=dict(reversescale=True))
        
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Main App Structure
# ---------------------------------------------------------------------------
st.title("Restaurant Business Analytics")
st.markdown("### Data-driven insights for location scouting and concept testing.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Score My Locations", 
    "🎯 Best Areas for Concept", 
    "📉 Market Gaps", 
    "💎 Top Opportunities"
])

# ---------------------------------------------------------------------------
# Tab 1: Score My Locations
# ---------------------------------------------------------------------------
with tab1:
    st.header("Score Candidate Locations")
    st.markdown("""
    **Goal**: Check if your specific location is a winner.
    Input coordinates or an address to see the predicted success probability for your concept.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Define Concept")
        sample_city = selected_city if selected_city != "All" else available_cities[0]
        schema_df = app_utils.load_grid_concepts(sample_city)
        
        if schema_df.empty:
            st.error("No data available to determine concepts.")
        else:
            cuisines = sorted(schema_df["cuisine_slug"].dropna().unique())
            est_types = sorted(schema_df["est_flag"].dropna().unique())
            prices = sorted(schema_df["price_level_num"].dropna().unique().astype(int))
            
            # Use unique values directly to avoid mismatch
            # Display formatted names but use raw values for logic if possible, 
            # or just use raw values in UI for simplicity and correctness.
            
            # Helper to format display
            def format_est(s):
                return s.replace('is_', '').replace('_', ' ').title()
                
            sel_cuisines = st.multiselect("Cuisines", cuisines, default=cuisines[:1])
            sel_est = st.multiselect("Establishment Types", est_types, default=est_types[:1], format_func=format_est)
            sel_prices = st.multiselect("Price Levels", prices, default=prices)
            
    with col2:
        st.subheader("2. Input Locations")
        
        input_method = st.radio("Input Method", ["Coordinates", "Address"], horizontal=True)
        
        input_data = []
        
        if input_method == "Coordinates":
            default_data = pd.DataFrame([
                {"name": "Location A", "lat": 47.3769, "lon": 8.5417}, 
                {"name": "Location B", "lat": 46.2044, "lon": 6.1432}, 
            ])
            input_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
            if not input_df.empty:
                input_data = input_df.to_dict("records")
                
        else:
            st.info("Enter addresses below. We will try to find their coordinates.")
            address_text = st.text_area("Addresses (one per line)", "Bahnhofstrasse 1, Zurich\nRue de Rive 1, Geneva")
            if address_text:
                lines = address_text.strip().split("\n")
                for line in lines:
                    if line.strip():
                        input_data.append({"name": line.strip(), "address": line.strip()})
        
        # Use session state to persist results across reruns if needed, 
        # but simple button/container pattern works if results are inside the if block.
        if st.button("Calculate Scores", type="primary"):
            if not input_data:
                st.warning("Please add at least one location.")
            elif not (sel_cuisines and sel_est and sel_prices):
                st.warning("Please select at least one cuisine, establishment type, and price level.")
            else:
                # Geocode if needed
                valid_locations = []
                if input_method == "Address":
                    progress_text = "Geocoding addresses..."
                    my_bar = st.progress(0, text=progress_text)
                    for i, item in enumerate(input_data):
                        lat, lon = app_utils.geocode_address(item["address"])
                        if lat is not None:
                            item["lat"] = lat
                            item["lon"] = lon
                            valid_locations.append(item)
                        else:
                            st.warning(f"Could not find coordinates for: {item['address']}")
                        my_bar.progress((i + 1) / len(input_data), text=progress_text)
                    my_bar.empty()
                else:
                    # Validate coordinate inputs
                    for item in input_data:
                         lat = item.get("lat")
                         lon = item.get("lon")
                         
                         if pd.notnull(lat) and pd.notnull(lon):
                             # Check for WGS84 range
                             if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                                 st.warning(f"Location '{item.get('name', 'Unknown')}' has suspiciously large coordinates ({lat}, {lon}). Please use standard WGS84 coordinates (e.g., Lat 47.3, Lon 8.5 for Switzerland).")
                                 continue
                             
                             valid_locations.append(item)

                if not valid_locations:
                    st.error("No valid locations to score. Please check your coordinates.")
                else:
                    results = []
                    target_cities = available_cities if selected_city == "All" else [selected_city]
                    
                    progress_bar = st.progress(0, text="Scoring locations...")
                    
                    for i, city in enumerate(target_cities):
                        city_grid = app_utils.load_grid_concepts(city)
                        if city_grid.empty:
                             # Update progress even if skip
                            progress_bar.progress((i + 1) / len(target_cities), text="Scoring locations...")
                            continue
                        
                        concept_subset = city_grid[
                            city_grid["cuisine_slug"].isin(sel_cuisines) &
                            city_grid["est_flag"].isin(sel_est) &
                            city_grid["price_level_num"].isin(sel_prices)
                        ]
                        
                        if concept_subset.empty:
                            progress_bar.progress((i + 1) / len(target_cities), text="Scoring locations...")
                            continue
                            
                        tree, coords = app_utils.get_city_kdtree(city)
                        if tree is None:
                            progress_bar.progress((i + 1) / len(target_cities), text="Scoring locations...")
                            continue
                            
                        for row in valid_locations:
                            # Defensive access
                            lat = row.get("lat")
                            lon = row.get("lon")
                            
                            if lat is None or lon is None:
                                continue
                                
                            x, y = app_utils.wgs_to_lv95(lon, lat)
                            
                            dist, idx = tree.query([[x, y]], k=1)
                            # Distance threshold (e.g., 500m)
                            if dist[0][0] > 500:
                                continue 
                                
                            grid_id = coords.iloc[idx[0][0]]["grid_id"]
                            
                            grid_matches = concept_subset[concept_subset["grid_id"] == grid_id]
                            if not grid_matches.empty:
                                best_match = grid_matches.sort_values("success_prob", ascending=False).iloc[0]
                                results.append({
                                    "Location Name": row.get("name", "Unknown"),
                                    "City": city,
                                    "Grid ID": grid_id,
                                    "Best Concept": best_match["concept_name"],
                                    "Success Probability": best_match["success_prob"],
                                    "Latitude": lat,
                                    "Longitude": lon
                                })
                        
                        progress_bar.progress((i + 1) / len(target_cities), text="Scoring locations...")
                    
                    progress_bar.empty()
                    
                    if results:
                        res_df = pd.DataFrame(results).sort_values("Success Probability", ascending=False)
                        st.success(f"Found matches for {len(res_df)} locations.")
                        res_df = pd.DataFrame(results).sort_values("Success Probability", ascending=False)
                        st.success(f"Found matches for {len(res_df)} locations.")
                        
                        st.dataframe(
                            res_df.style.format({"Success Probability": "{:.1%}", "Latitude": "{:.4f}", "Longitude": "{:.4f}"}),
                            use_container_width=True
                        )
                        
                        fig = px.scatter_mapbox(
                            res_df,
                            lat="Latitude",
                            lon="Longitude",
                            color="Success Probability",
                            size="Success Probability",
                            hover_name="Location Name",
                            hover_data=["City", "Best Concept", "Grid ID"],
                            color_continuous_scale="RdYlGn",
                            mapbox_style="carto-positron",
                            zoom=8,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning("No matching grid cells found for these locations in the selected cities.")

# ---------------------------------------------------------------------------
# Tab 2: Best Areas for Concept
# ---------------------------------------------------------------------------
with tab2:
    st.header("Find Best Areas for Your Concept")
    st.markdown("""
    **Goal**: Identify the highest potential zones for a specific restaurant type.
    Use the filters below to refine your search.
    """)
    
    if selected_city == "All":
        st.warning("Please select a specific city in the sidebar to view heatmaps.")
    else:
        col1, col2, col3 = st.columns(3)
        
        df_concepts = app_utils.load_grid_concepts(selected_city)
        
        if df_concepts.empty:
            st.error(f"No data found for {selected_city}")
        else:
            cuisines = sorted(df_concepts["cuisine_slug"].dropna().unique())
            est_types = sorted(df_concepts["est_flag"].dropna().unique())
            prices = sorted(df_concepts["price_level_num"].dropna().unique().astype(int))
            
            with col1:
                c_cuisine = st.selectbox("Cuisine", cuisines)
            with col2:
                c_est = st.selectbox("Establishment Type", est_types)
            with col3:
                c_price = st.selectbox("Price Level", prices)
                
            mask = (
                (df_concepts["cuisine_slug"] == c_cuisine) &
                (df_concepts["est_flag"] == c_est) &
                (df_concepts["price_level_num"] == c_price)
            )
            subset = df_concepts[mask].copy()
            
            st.markdown("---")
            st.subheader("Filters")
            
            # Saturation Filter Explanation
            st.markdown("""
            **Saturation Filter**: Helps you avoid areas that are already crowded.
            - **Ratio > 1.0**: **Over-supplied**. There are already more successful restaurants than the model predicts should be there.
            - **Ratio < 1.0**: **Under-supplied**. There is room for more successful concepts.
            """)
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                use_sat = st.checkbox("Exclude Over-Supplied Areas (Ratio > 1.0)")
            with col_f2:
                exclude_lakes = st.checkbox("Exclude Zero-Restaurant Zones (Lakes/Forests)", value=False, help="Removes grid cells with 0 existing restaurants. Useful for hiding lakes, but may hide empty land plots.")
            
            # Apply Filters
            area_perf = app_utils.load_area_perf()
            if not area_perf.empty:
                # We need to merge area_perf if either filter is active
                if use_sat or exclude_lakes:
                    subset = subset.merge(area_perf[["grid_id", "ratio", "n_restaurants"]], on="grid_id", how="left")
                    
                    if use_sat:
                        subset = subset[subset["ratio"].fillna(0) <= 1.0]
                    
                    if exclude_lakes:
                        # Filter out cells with 0 restaurants (or NaN)
                        subset = subset[subset["n_restaurants"].fillna(0) > 0]
            
            if exclude_lakes:
                st.info("ℹ️ **Map Update**: You are now seeing only areas with **existing restaurants**. This effectively removes lakes and forests, but restricts the view to the current urban footprint.")
            
            if subset.empty:
                st.info("No areas match your criteria.")
            else:
                st.subheader(f"Top Areas for {c_cuisine} ({c_est})")
                
                top_n = subset.sort_values("success_prob", ascending=False).head(500)
                
                render_map(
                    top_n, 
                    color_col="success_prob", 
                    tooltip_cols=["grid_id", "success_prob", "concept_name"],
                    title=f"Success Probability Map - {selected_city}",
                    color_scale="RdYlGn",
                    fixed_range=(0, 1)
                )
                
                if exclude_lakes:
                    st.info("""
                    ℹ️ **Why did the points change?**
                    1. **New Opportunities**: The map shows the **Top 500** locations. Removing the "fake" high-scoring lake points allows new, valid urban locations (that were previously ranked #501+) to appear.
                    2. **Urban Focus**: You are now seeing the true best locations within the actual city footprint.
                    """)
                
                st.dataframe(
                    top_n[["grid_id", "success_prob", "concept_name"]].head(50).style.format({"success_prob": "{:.1%}"}),
                    use_container_width=True
                )

# ---------------------------------------------------------------------------
# Tab 3: Market Gaps (Under-supplied)
# ---------------------------------------------------------------------------
with tab3:
    st.header("Identify Under-Supplied Markets")
    st.markdown("""
    **Goal**: Find "Market Gaps" where demand exceeds supply.
    These are areas where the model predicts high success, but there are few existing competitors.
    """)
    
    if selected_city == "All":
        st.warning("Please select a specific city in the sidebar.")
    else:
        area_perf = app_utils.load_area_perf()
        if area_perf.empty:
            st.error("No saturation data available.")
        else:
            if "grid_city" in area_perf.columns:
                city_perf = area_perf[area_perf["grid_city"] == selected_city].copy()
            else:
                st.error("Saturation data missing city column.")
                city_perf = pd.DataFrame()

            if not city_perf.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**1. Max Saturation Ratio**")
                    st.caption("Keep this LOW (< 0.8) to find areas with much less supply than demand.")
                    max_ratio = st.slider("Max Ratio", 0.1, 1.5, 0.8, label_visibility="collapsed")
                    
                with col2:
                    st.markdown("**2. Min Existing Restaurants**")
                    st.caption("Set to 1 or 2 to filter out lakes, forests, and industrial zones (which usually have 0 restaurants).")
                    min_restaurants = st.slider("Min Restaurants", 0, 10, 2, label_visibility="collapsed")
                
                filtered = city_perf[
                    (city_perf["ratio"] <= max_ratio) & 
                    (city_perf["n_restaurants"] >= min_restaurants)
                ].copy()
                
                if filtered.empty:
                    st.info("No areas match criteria. Try increasing the Max Ratio or decreasing Min Restaurants.")
                else:
                    grid_best = app_utils.load_grid_best(selected_city)
                    if not grid_best.empty:
                        filtered = filtered.merge(
                            grid_best[["grid_id", "best_concept", "area_potential"]],
                            on="grid_id",
                            how="left"
                        )
                    
                    st.subheader(f"Under-supplied Areas in {selected_city}")
                    
                    if "grid_x_lv95" in filtered.columns:
                        filtered = filtered.rename(columns={"grid_x_lv95": "x_lv95", "grid_y_lv95": "y_lv95"})
                    
                    render_map(
                        filtered,
                        color_col="ratio",
                        tooltip_cols=["grid_id", "ratio", "n_restaurants", "expected_successes", "best_concept"],
                        title="Market Saturation (Blue = Under-supplied)",
                        color_scale="Spectral_r", 
                        reverse_scale=True
                    )
                    
                    st.dataframe(
                        filtered[["grid_id", "ratio", "n_restaurants", "expected_successes", "best_concept", "area_potential"]]
                        .sort_values("ratio")
                        .head(50)
                        .style.format({"ratio": "{:.2f}", "area_potential": "{:.1%}", "expected_successes": "{:.1f}"}),
                        use_container_width=True
                    )

# ---------------------------------------------------------------------------
# Tab 4: Top Opportunities
# ---------------------------------------------------------------------------
with tab4:
    st.header("Highest Potential Locations (Agnostic)")
    st.markdown("""
    **Goal**: Find the absolute best locations in the city, regardless of cuisine.
    This is the "Global Leaderboard" of grid cells.
    """)
    
    df_best = app_utils.load_grid_best(selected_city)
    
    if df_best.empty:
        st.warning("No data available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Min Success Probability**")
            st.caption("Only show locations with at least this chance of success.")
            min_prob = st.slider("Probability", 0.5, 0.99, 0.7, label_visibility="collapsed")
        
        top_opps = df_best[df_best["area_potential"] >= min_prob].copy()
        
        # Filters
        area_perf = app_utils.load_area_perf()
        if not area_perf.empty:
            with col2:
                st.markdown("**Filters**")
                filter_sat = st.checkbox("Exclude Saturated")
                exclude_lakes = st.checkbox("Exclude Zero-Restaurant Zones (Lakes)", value=False)
            
            # Merge if needed
            if filter_sat or exclude_lakes:
                top_opps = top_opps.merge(area_perf[["grid_id", "ratio", "n_restaurants"]], on="grid_id", how="left")
                
                if filter_sat:
                    top_opps = top_opps[top_opps["ratio"].fillna(0) <= 1.0]
                
                if exclude_lakes:
                    top_opps = top_opps[top_opps["n_restaurants"].fillna(0) > 0]
                    st.info("ℹ️ **Map Update**: You are now seeing only areas with **existing restaurants**. This effectively removes lakes and forests, but restricts the view to the current urban footprint.")
        
        if top_opps.empty:
            st.info("No opportunities match criteria.")
        else:
            top_opps = top_opps.sort_values("area_potential", ascending=False).head(500)
            
            render_map(
                top_opps,
                color_col="area_potential",
                tooltip_cols=["grid_id", "best_concept", "area_potential"],
                title="Top Opportunities",
                color_scale="Magma",
                fixed_range=(0, 1)
            )
            
            if exclude_lakes:
                st.info("""
                ℹ️ **Why did the points change?**
                1. **New Opportunities**: The map shows the **Top 500** locations. Removing the "fake" high-scoring lake points allows new, valid urban locations (that were previously ranked #501+) to appear.
                2. **Urban Focus**: You are now seeing the true best locations within the actual city footprint.
                """)
            
            st.dataframe(
                top_opps[["grid_id", "best_concept", "area_potential", "city"]]
                .head(100)
                .style.format({"area_potential": "{:.1%}"}),
                use_container_width=True
            )
