"""
Merge Google Maps + OSM into a master restaurants table.
- Standardize columns
- Deduplicate within-source
- Cross-source link (spatial + name similarity)
- Output: data/processed/restaurants_master.parquet
"""
