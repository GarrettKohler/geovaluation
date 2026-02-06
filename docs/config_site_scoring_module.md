Here is the technical documentation for the site\_scoring.config module, formatted in Markdown. You can copy the content below directly into a Google Doc or a README.md file.

# ---

**Documentation: Site Scoring Configuration**

## **Overview**

The site\_scoring.config module defines the Config dataclass, which serves as the central control plane for the machine learning pipeline. It manages hyperparameters, file paths, feature selection, and hardware acceleration settings.

This configuration is **specifically optimized for Apple Silicon (M4)**, utilizing the Metal Performance Shaders (MPS) backend and high-throughput batching strategies to leverage Unified Memory.

## **Class: Config**

### **Hardware & Optimization**

The configuration automatically detects Apple Silicon hardware and adjusts PyTorch settings for maximum throughput.

| Parameter | Default | Description |
| :---- | :---- | :---- |
| device | mps (if available) | Auto-detects macOS GPU backend. Falls back to cpu if unavailable. |
| batch\_size | 4096 | Aggressive batch size optimized for M4 unified memory bandwidth. |
| num\_workers | 4 | Number of subprocesses for data loading. |
| pin\_memory | True | Enables faster data transfer to the GPU. |
| use\_amp | False | **Automatic Mixed Precision.** Currently disabled as MPS support for AMP is experimental/unstable compared to CUDA. |

### **Input/Output & Data Integrity**

**CRITICAL:** This pipeline is designed to work with **Aggregated Data** (one row per site), not raw monthly time-series data. This design choice drastically reduces training time and prevents data leakage.

| Parameter | Description |
| :---- | :---- |
| data\_path | Path to the Parquet file containing aggregated site features. **Default:** /Users/home/gkdev/geospatial/data/processed/site\_training\_data.parquet |
| output\_dir | Directory where model checkpoints and logs will be saved. |
| target | The column name of the variable to predict. **Options:** avg\_monthly\_revenue (Recommended), total\_revenue, total\_monthly\_impressions. |

### **Feature Engineering**

The configuration defines three distinct feature sets. The model dynamically builds input layers based on the length of these lists.

#### **1\. Numeric Features**

Continuous variables, including:

* **Site Characteristics:** screen\_count, active\_months, dma\_rank.  
* **Aggregated Metrics:** Totals and averages for revenue, impressions, and NVIS (Network Video Impressions).  
* **Log Transforms:** Logarithmic versions of skewed metrics (e.g., log\_total\_revenue) for better regression stability.  
* **Geospatial:** Distances to nearest sites and interstates.  
* **Demographics:** Income, age, and ethnicity percentages.

#### **2\. Categorical Features**

Discrete variables that will be passed through an **Embedding Layer** (dim=16).

* **Metadata:** network, program, experience\_type, hardware\_type.  
* **Branding:** retailer, brand\_fuel, brand\_restaurant.

#### **3\. Boolean Features**

Binary flags (0/1). These include:

* **Restrictions (r\_\*):** E.g., r\_lottery\_encoded, r\_alcohol\_drink\_responsibly\_message\_encoded.  
* **Capabilities (c\_\*):** E.g., c\_emv\_enabled\_encoded, c\_open\_24\_hours\_encoded.

### **Automatic Leakage Prevention (\_\_post\_init\_\_)**

To ensure statistical validity, the Config class includes a \_\_post\_init\_\_ hook that automatically sanitizes the input features based on the selected target.

**Logic:**

If target is set to avg\_monthly\_revenue, the system removes:

* total\_revenue  
* total\_monthly\_revenue\_per\_screen  
* Any log-transformed versions of the above.

This prevents the model from "cheating" by using features that are mathematically derived from the target variable.

### **Training Hyperparameters**

| Parameter | Default | Description |
| :---- | :---- | :---- |
| epochs | 50 | Maximum training passes. |
| learning\_rate | 1e-4 | Learning rate (set lower for stability). |
| hidden\_dims | \[512, 256, 128, 64\] | The Multi-Layer Perceptron (MLP) architecture layers. |
| dropout | 0.2 | Regularization rate to prevent overfitting. |
| early\_stopping\_patience | 10 | Stops training if validation loss doesn't improve for 10 epochs. |

## **Usage Example**

Python

from site\_scoring.config import Config

\# Initialize with defaults (Auto-detects MPS)  
config \= Config()

\# Override for specific experiments  
config\_cpu \= Config(  
    device="cpu",  
    batch\_size=64,  
    target="total\_monthly\_impressions"  
)

\# Access sanitized feature lists (Leakage columns already removed)  
print(f"Training with {len(config.numeric\_features)} numeric features.")  
