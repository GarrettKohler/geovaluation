# **Data Transformation Pipeline**

## **Documentation: Data Transformation Pipeline (data\_transform.py)**

### **Overview**

The data\_transform module performs "Heavy Lifting" ETL (Extract, Transform, Load). It converts the raw, monthly time-series data into a **static, site-level dataset** suitable for the ML model. It uses **Polars** for high-performance data manipulation.

### **Workflow**

1. **Aggregation (aggregate\_site\_metrics):**  
   * Collapses monthly records (e.g., 24 rows for a 2-year-old site) into a single row.  
   * Calculates totals (e.g., total\_revenue) and averages (e.g., avg\_monthly\_revenue).  
   * Extracts the most recent metadata (e.g., current hardware\_type).  
2. **Geospatial Join (join\_geospatial\_features):**  
   * Merges the site data with nearest\_site\_distances.csv and site\_interstate\_distances.csv.  
3. **Feature Engineering:**  
   * **Log Transforms (add\_log\_transformations):** Applies log1p (natural log of x+1) to skewed distributions like revenue and distance.  
   * **One-Hot Encoding (one\_hot\_encode\_flags):** Converts capability flags (e.g., c\_sells\_beer) and restriction flags (e.g., r\_lottery) into machine-readable binary columns.  
4. **Cleaning (prepare\_training\_dataset):**  
   * Filters for Status \== Active.  
   * Drops rows with negative revenue (data errors).  
   * Removes explicit geographic identifiers (Zip, DMA) to prevent overfitting to specific locations.

### **Key Outputs**

* site\_aggregated\_precleaned.parquet: The full dataset (for human inspection).  
* site\_training\_data.parquet: The sanitized, ready-to-train dataset.  
* training\_data\_summary.txt: A statistical report of the processed data.

### **Usage**

Run as a standalone script to regenerate datasets:

Bash

python \-m site\_scoring.data\_transform  
