# Geospatial Site Analysis & Visualization

A Python library and web application for geospatial analysis of site locations, featuring:
- Interactive map visualization with 57K+ sites
- Distance calculations to US Interstate highways
- Site filtering by categorical attributes
- Revenue-based site analysis

## Features

### Web Visualization (`app.py`)
- **Interactive Map**: View all sites on a WebGL-accelerated map
- **Lasso Selection**: Draw to select multiple sites at once
- **Side Panel**: Comprehensive site details with 8 data categories
- **Filtering**: Filter sites by State, Network, Retailer, Hardware, and more
- **Highway Connections**: Visualize distance from sites to nearest Interstate

### Core Library (`src/`)
- **Interstate Distance**: Calculate distance from any coordinate to nearest Interstate
- **Nearest Site**: Find closest neighboring site using KDTree spatial indexing
- **EPA Walkability**: Look up walkability scores for locations
- **Batch Processing**: Efficiently process thousands of points

### Machine Learning (`site_scoring/`)
- **Revenue Prediction**: PyTorch model for site revenue prediction
- **M4 Optimized**: Optimized for Apple M4 MPS GPU acceleration

## Quick Start

### Run the Web Application

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py

# Open http://localhost:5000 in your browser
```

### Use the Library

```python
from src.interstate_distance import distance_to_nearest_interstate

# Single point query
result = distance_to_nearest_interstate(37.7749, -122.4194)
print(f"Distance: {result['distance_miles']:.2f} miles to {result['nearest_highway']}")
```

## Project Structure

```
geospatial/
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Core library modules
│   ├── __init__.py
│   ├── data_service.py         # Data loading and caching
│   ├── interstate_distance.py  # Highway distance calculations
│   ├── nearest_site.py         # Site-to-site distance calculations
│   └── epa_walkability.py      # EPA walkability lookups
│
├── site_scoring/               # ML model for revenue prediction
│   ├── config.py               # Model configuration
│   ├── data_loader.py          # PyTorch data loading
│   ├── model.py                # Neural network architecture
│   ├── train.py                # Training pipeline
│   └── predict.py              # Inference module
│
├── templates/                  # Flask templates
│   └── index.html              # Main visualization page
│
├── data/                       # Data files
│   ├── input/                  # Input CSV files
│   ├── output/                 # Generated results (distances, etc.)
│   └── shapefiles/             # Geographic data
│
├── scripts/                    # Utility scripts
│   ├── demo.py                 # Interstate distance demo
│   └── site_walkability.py     # Walkability calculations
│
└── docs/                       # Documentation
    ├── api.md                  # API reference
    ├── ARCHITECTURE.md         # System architecture & diagrams
    ├── src_and_site_scoring.md # How src/ and site_scoring/ work together
    ├── scripts_pipeline.md    # How scripts/ generates ML features
    ├── test_coverage.md       # Test coverage analysis and edge cases
    ├── highway_geospatial_research.md  # Highway geospatial research guide
    └── platinum_ml_web_app/    # App demo videos
        └── Train Lookalike Group.mov
```

## Web Application Features

### Map Visualization
- **WebGL Rendering**: GPU-accelerated display of 57K+ sites using Leaflet.glify
- **Grey Unselected Sites**: All sites shown as grey dots initially
- **Revenue Coloring**: Selected sites colored white→green by revenue
- **Zoom Scaling**: Site markers scale appropriately with zoom level

### Selection Methods
1. **Click**: Click any site to view details in side panel
2. **Lasso**: Click "Lasso Select" then draw around sites to select multiple
3. **Dropdown**: Use the searchable dropdown to select by site ID
4. **Filters**: Use categorical filters to find sites by attributes

### Side Panel
Opens when you:
- Hover over the right edge of the screen
- Click on any site

Contains:
- **Active Filters**: Add/remove filters by clicking field labels
- **Site Details**: 8 categories of information
  - Location (State, County, ZIP, DMA)
  - Site Info (Retailer, Network, Hardware)
  - Brands (Fuel, Restaurant, C-Store)
  - Revenue (Monthly avg, Total, Score)
  - Demographics (Income, Age, Ethnicity)
  - Performance (Impressions, Visits, Latency)
  - Capabilities (EMV, NFC, 24hr, Products)
  - Sales (Sellable, Schedule status)

### Filtering System
- Click any categorical field label to add it as a filter
- Filter defaults to the currently viewed site's value
- Change dropdown to filter for different values
- See live count of matching sites
- "Select Matching Sites" adds all matches to selection

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main visualization page |
| `/api/sites` | GET | All sites with coordinates and revenue |
| `/api/site/<id>` | GET | Basic site info with highway distance |
| `/api/site-details/<id>` | GET | Comprehensive site details |
| `/api/highway-connections` | POST | Calculate highway connections for sites |
| `/api/filter-options` | GET | Unique values for filterable fields |
| `/api/filtered-sites` | POST | Get sites matching filter criteria |

## Data Sources

### US Interstate Highway Data
- **Source**: [Census Bureau TIGER/Line Primary Roads](https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/)
- **Auto-download**: Downloaded automatically on first run (~15MB)
- **Caching**: Cached in memory for fast subsequent queries

### Site Data (Required)
Place these files in `data/input/`:
- `Sites - Base Data Set.csv` - Site locations (GTVID, Latitude, Longitude)
- `Site Scores - Site Revenue, Impressions, and Diagnostics.csv` - Site details and metrics
- `nearest_distances.csv` - each sites distance the closest GSTV site
- `interstate_distances.csv` - each sites distance to the nearest interstate
- `distance_to_mcdonalds_mi.csv` - each sites distance to the nearest mcdonalds
- `distance_to_walmart_mi.csv` - each sites distance to the nearest walmart
- `distance_to_kroger_mi.csv` - each sites distance to the nearest kroger
- `distance_to_------_mi.csv` - each sites distance to the nearest 


## Installation

### Requirements
- Python 3.8+
- See `requirements.txt` for full dependencies

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| Flask | Web application framework |
| geopandas | Geographic data handling |
| pandas | Data manipulation |
| shapely | Geometric operations |
| numpy | Numerical operations |
| Leaflet.js | Map visualization (via CDN) |
| Leaflet.glify | WebGL point rendering (via CDN) |

## Performance

- **Startup**: ~5-10 seconds to load all data
- **Site Rendering**: WebGL handles 57K+ points smoothly
- **Filtering**: Near-instant with pre-cached unique values

## License

This project uses public domain data from the US Census Bureau.
