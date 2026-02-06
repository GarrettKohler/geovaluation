# **Package Initialization**

## **Documentation: Geospatial Package Initialization (\_\_init\_\_.py)**

### **Overview**

The \_\_init\_\_.py file defines the public API for the root geospatial package. It serves as a convenience layer, aggregating and re-exporting the most frequently used functions from internal service modules (services.\*).

This structure allows consumers to import core utilities directly from the top-level package without needing to navigate the internal subdirectory structure.

### **Exposed API Components**

The package exposes functionality in three primary domains:

#### **1\. Highway Analytics (services.interstate\_distance)**

Functions for calculating proximity to major U.S. Interstate Highways.

* **distance\_to\_nearest\_interstate**: Calculates geodesic distance from a single point to the nearest highway.  
* **batch\_distance\_to\_interstate**: Optimized batch processing for DataFrames.  
* **preload\_highway\_data**: Manages the downloading and caching of Census Bureau shapefiles.

#### **2\. Site Proximity (services.nearest\_site)**

Functions for analyzing spatial relationships between sites.

* **calculate\_nearest\_site\_distances**: Computes the distance from every site in a dataset to its nearest neighbor within the same network.

#### **3\. Urban Walkability (services.epa\_walkability)**

Functions for assigning EPA National Walkability Index scores to locations.

* **get\_walkability\_score**: Retrieves the index score (1-20) for a specific coordinate.  
* **batch\_walkability\_scores**: Processes lists of coordinates against specific state FIPS codes.  
* **build\_walkability\_index**: Constructs the spatial index from raw EPA data.  
* **preload\_walkability\_data**: Handles data ingestion for the walkability engine.

### **Usage Pattern**

Developers can choose between **Direct Imports** (recommended for brevity) or **Submodule Imports** (explicit).

Python

\# OPTION 1: Direct Import (Enabled by this file)  
from geospatial import batch\_distance\_to\_interstate, get\_walkability\_score

\# OPTION 2: Submodule Import (Verbose)  
from geospatial.services.interstate\_distance import batch\_distance\_to\_interstate  
from geospatial.services.epa\_walkability import get\_walkability\_score

### **Module Structure**

Plaintext

geospatial/  
├── \_\_init\_\_.py                \<-- EXPORTS API HERE  
└── services/  
    ├── interstate\_distance.py  
    ├── nearest\_site.py  
    └── epa\_walkability.py  
