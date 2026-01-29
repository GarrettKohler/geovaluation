# **Data Loading & Processing**

## **Documentation: Data Loading & Processing (data\_loader.py)**

### **Overview**

The data\_loader module is responsible for ingesting processed data and preparing it for the PyTorch training loop. It is optimized for Apple Silicon (M4) by utilizing **Polars** for multi-threaded CSV/Parquet parsing and generating contiguous memory arrays for efficient transfer to the MPS (Metal Performance Shaders) backend.

### **Key Components**

#### **1\. SiteDataset (Class)**

A custom PyTorch Dataset that holds the data in memory as tensors.

* **Optimization:** Stores data in four distinct tensor groups (numeric, categorical, boolean, target) to maximize cache locality during training.  
* **Output:** Returns a tuple of (numeric, categorical, boolean, target) for each index.

#### **2\. DataProcessor (Class)**

Handles the transformation of raw DataFrames into neural-network-ready tensors.

* **Persistence:** Can save/load its state (scalers and encoders) via pickle to ensure consistency between training and inference.  
* **Numeric Processing:** Applies StandardScaler, clips outliers (1st/99th percentile), and handles NaN values.  
* **Categorical Processing:** Uses LabelEncoder to convert string categories into integer indices for embedding layers.  
* **Boolean Processing:** Standardizes various boolean formats (True/False, 1/0, "Yes"/"No") into float tensors.  
* **Target Processing:** Scales the target variable using a separate StandardScaler to stabilize gradients.

#### **3\. create\_data\_loaders (Function)**

The primary entry point for the training pipeline.

* **Inputs:** Config object.  
* **Outputs:** Train, Validation, and Test DataLoader objects \+ the fitted DataProcessor.  
* **M4 Optimization:** Configures num\_workers, prefetch\_factor, and pin\_memory based on the hardware configuration defined in Config.

### **Usage Example**

Python

from site\_scoring.data\_loader import create\_data\_loaders  
from site\_scoring.config import Config

config \= Config()  
train\_loader, val\_loader, test\_loader, processor \= create\_data\_loaders(config)

for numeric, cat, bools, target in train\_loader:  
    \# numeric.shape \-\> \[batch\_size, n\_numeric\]  
    pass

# 

# 

