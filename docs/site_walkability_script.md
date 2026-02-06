# **Documentation: Site Walkability Calculator**

## **Overview**

The scripts.site\_walkability module calculates walkability metrics for a given list of site locations. It utilizes the EPA National Walkability Index methodology to assign scores based on location characteristics such as intersection density and proximity to transit.

**Note:** This script is currently configured to process specific US States (CA, FL, NY) via hardcoded FIPS codes.

## **System Requirements**

* **Python Version:** 3.8+  
* **Dependencies:**  
  * pandas  
  * src.epa\_walkability (internal module)

## **Directory Structure**

The script relies on a relative directory structure to locate input data. Ensure your project root matches the following layout:

Plaintext

project\_root/  
├── data/  
│   └── input/  
│       └── Sites \- Base Data Set.csv  \<-- REQUIRED INPUT  
├── src/  
│   └── epa\_walkability.py  
└── scripts/  
    └── site\_walkability.py

## **Input Specification**

The script reads a CSV file located at data/input/Sites \- Base Data Set.csv.

**Required Columns:**

| Column Name | Type | Description |

| :--- | :--- | :--- |

| Latitude | Float | Decimal latitude of the site. |

| Longitude | Float | Decimal longitude of the site. |

## **Usage**

To execute the script, run the following command from the **project root** directory:

Bash

python \-m scripts.site\_walkability

### **Configuration & Constraints**

**Critical:** The script currently enforces a filter for specific state FIPS codes in the batch\_walkability\_scores call. Only sites within the following regions will be processed:

* 06 (California)  
* 12 (Florida)  
* 36 (New York)

To process additional regions, the state\_fips list in scripts/site\_walkability.py must be updated.

## **Output Specification**

The script generates a results DataFrame (stored in the results variable) containing the original site data appended with EPA metrics.

**Generated Metrics:**

| Column Name | Description |

| :--- | :--- |

| walkability\_index | The National Walkability Index score (1-20). |

| intersection\_density | Measure of street connectivity. |

| transit\_proximity | Distance/accessibility to transit stops. |

*Note: In its current state, this script calculates these values in-memory but does not write them to disk (CSV export is not implemented in the provided snippet).*