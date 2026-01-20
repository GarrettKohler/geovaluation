"""Debug GEOID matching between EPA and Census data"""
import pandas as pd
import geopandas as gpd
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "epa_walkability"

# Load cached data
print("Loading EPA data...")
epa = pd.read_csv(CACHE_DIR / "epa_sld.csv", dtype={'GEOID10': str})
print(f"EPA columns: {list(epa.columns[:20])}")
print(f"EPA sample GEOIDs:\n{epa['GEOID10'].head(10)}")
print(f"EPA GEOID10 length: {epa['GEOID10'].str.len().value_counts().head()}")

print("\nLoading Census block groups (CA)...")
bg = gpd.read_file(CACHE_DIR / "blockgroups_06.gpkg")
print(f"Census columns: {list(bg.columns)}")
print(f"Census sample GEOIDs:\n{bg['GEOID'].head(10)}")
print(f"Census GEOID length: {bg['GEOID'].str.len().value_counts().head()}")

# Check if any GEOIDs match
epa_geoids = set(epa['GEOID10'])
census_geoids = set(bg['GEOID'])
matches = epa_geoids & census_geoids
print(f"\nMatching GEOIDs: {len(matches)}")

if len(matches) == 0:
    print("\nNo matches found. Checking if it's a format issue...")
    print(f"Example EPA GEOID10: {epa['GEOID10'].iloc[0]} (type: {type(epa['GEOID10'].iloc[0])})")
    print(f"Example Census GEOID: {bg['GEOID'].iloc[0]} (type: {type(bg['GEOID'].iloc[0])})")
