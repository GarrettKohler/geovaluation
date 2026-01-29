# **Services Package Initialization**

## **Documentation: Services Package Initialization (services/\_\_init\_\_.py)**

### **Overview**

The services/\_\_init\_\_.py file defines the internal service layer for the geospatial application. It aggregates core business logic for data management and machine learning operations, exposing them as a cohesive API for the application controller or UI layer.

### **Exposed Modules**

#### **1\. Data Management (data\_service)**

Handles all I/O operations, caching strategies, and data filtering logic.

* **Loaders:** load\_sites, load\_revenue\_metrics, load\_site\_details.  
* **Filtering:** get\_filter\_options, get\_filtered\_site\_ids.  
* **Display:** get\_site\_details\_for\_display (formats raw data for UI consumption).  
* **Optimization:** preload\_all\_data (initializes caches).  
* **Constants:** CATEGORICAL\_FIELDS (list of filterable fields).

#### **2\. Machine Learning Control (training\_service)**

Manages the lifecycle of the PyTorch training process, specifically handling the asynchronous nature of training on the GPU.

* **Lifecycle:** start\_training, stop\_training.  
* **Monitoring:** get\_training\_status, stream\_training\_progress (generator for real-time UI updates).  
* **Diagnostics:** get\_system\_info (checks for M4/MPS availability).

### **Note on Other Modules**

The docstring references interstate\_distance, nearest\_site, and epa\_walkability. While these modules reside within the services directory, they are currently **not** re-exported by this initialization file. To use them, consumers must import them directly (e.g., from services.interstate\_distance import ...) or they must be added to this file's imports.

### **Usage Example**

Python

from services import load\_sites, start\_training

\# Load data  
sites\_df \= load\_sites()

\# Trigger ML training  
start\_training(config={'target': 'revenue'})  
