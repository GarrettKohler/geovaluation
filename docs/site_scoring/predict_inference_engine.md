# **Inference Engine**

## **Documentation: Inference Engine (predict.py)**

### **Overview**

The predict module is designed for production inference. It loads a trained model artifact and generates predictions for new data. It handles the exact same preprocessing steps (scaling/encoding) as the training pipeline to ensure validity.

### **Class: SiteScorer**

#### **Initialization**

* Loads the model weights (best\_model.pt) and the preprocessor state (preprocessor.pkl).  
* Auto-detects the hardware accelerator (mps or cpu).

#### **predict(df)**

* Accepts a Polars DataFrame.  
* Applies the saved DataProcessor transforms (Scaling/Encoding).  
* Runs the forward pass on the GPU.  
* **Inverse Transforms** the output: Converts the model's scaled output back to real-world units (e.g., Dollars).

### **Function: batch\_predict**

Optimized for processing massive files (millions of rows) without running out of RAM.

* Reads the input CSV in chunks (lazy loading).  
* Predicts on each chunk sequentially.  
* Writes results to disk immediately.

### **Usage Example**

Python

from site\_scoring.predict import SiteScorer

scorer \= SiteScorer(model\_path="outputs/best\_model.pt")  
df\_new \= pl.read\_csv("new\_sites.csv")

\# Returns array of revenue predictions in dollars  
predictions \= scorer.predict(df\_new)