"""
Calculate walkability scores for sites.

Run from project root: python -m scripts.site_walkability
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.epa_walkability import batch_walkability_scores

# Load your sites
sites = pd.read_csv(PROJECT_ROOT / "data" / "input" / "Sites - Base Data Set.csv")

# Calculate walkability scores (specify state FIPS codes)
results = batch_walkability_scores(
    sites,
    lat_col='Latitude',
    lon_col='Longitude',
    state_fips=["06", "12", "36"]  # CA, FL, NY
)

# Results include: walkability_index, intersection_density, etc.
