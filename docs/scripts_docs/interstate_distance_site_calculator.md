Here is technical documentation for the scripts.demo script, formatted for an engineering team (e.g., for a repository README or Confluence page).

# ---

**Documentation: Interstate Distance Calculator Demo**

## **Overview**

The scripts.demo module serves as both a demonstration of the src.interstate\_distance library and a utility for batch-processing site distance calculations.

It ingests a CSV of site locations (specifically expecting GSTV site data), calculates the geodesic distance from each site to the nearest U.S. Interstate Highway, and outputs a summary CSV with the nearest highway name and distance in miles.

## **System Requirements**

* **Python Version:** 3.8+  
* **Dependencies:**  
  * pandas  
  * src.interstate\_distance (internal module)

## **Directory Structure**

The script relies on a relative directory structure to locate input data and store results. Ensure your project root matches the following layout:

Plaintext

project\_root/  
├── data/  
│   ├── input/  
│   │   └── Sites \- Base Data Set.csv  \<-- REQUIRED INPUT  
│   └── output/                        \<-- Script creates this if missing  
├── src/  
│   └── interstate\_distance.py  
└── scripts/  
    └── demo.py

## **Input Specification**

The script expects a CSV file located at data/input/Sites \- Base Data Set.csv.

**Required Columns:**

| Column Name | Type | Description |

| :--- | :--- | :--- |

| Latitude | Float | Decimal latitude of the site. |

| Longitude | Float | Decimal longitude of the site. |

| GTVID | String/Int | GSTV Identifier (passed through to output). |

*Note: The script automatically filters out empty "Unnamed" columns often resulting from Excel exports.*

## **Usage**

To execute the script, run the following command from the **project root** directory:

Bash

python \-m scripts.demo

### **Execution Flow**

1. **Environment Setup:** Dynamically adds the project root to sys.path to allow importing src modules.  
2. **Data Preload:** Checks for cached highway shapefiles. If this is the **first run**, it will download \~15MB of highway data from the U.S. Census Bureau.  
3. **Processing:**  
   * Loads Sites \- Base Data Set.csv.  
   * Cleans column headers.  
   * Calls batch\_distance\_to\_interstate to process coordinates.  
4. **Export:** Saves results to data/output/ and prints summary statistics to the console.

## **Output Specification**

The script generates data/output/site\_interstate\_distances.csv.

**Output Schema:**

| Column Name | Description |

| :--- | :--- |

| GTVID | The unique identifier from the input file. |

| Latitude | Input latitude. |

| Longitude | Input longitude. |

| nearest\_interstate | Name of the closest interstate (e.g., "I-75"). |

| distance\_to\_interstate\_mi | Distance to that interstate in miles. |

## **Console Output Example**

The script provides verbose logging to standard out:

Plaintext

Interstate Distance Calculator  
\============================================================

Output directory: .../data/output

Pre-loading highway data (first run downloads \~15MB from Census Bureau)...

\============================================================  
SITE DISTANCE CALCULATION  
\============================================================

Loaded 1500 sites from Sites \- Base Data Set.csv  
...  
Calculating distances to nearest Interstate...

Results (first 10):  
...  
Results saved to: .../data/output/site\_interstate\_distances.csv

\============================================================  
SUMMARY STATISTICS  
\============================================================  
Total sites: 1500  
Min distance: 0.12 miles  
Max distance: 45.30 miles  
Mean distance: 5.42 miles  
Median distance: 2.10 miles

\============================================================  
Complete\!

## **Troubleshooting**

* **ModuleNotFoundError:** Ensure you are running the script from the *project root* using python \-m scripts.demo, not from inside the scripts folder.  
* **Download Latency:** On the very first run, the script fetches shapefiles from the Census Bureau. If the script hangs at Pre-loading highway data..., check your internet connection. Subsequent runs will use the cached data.