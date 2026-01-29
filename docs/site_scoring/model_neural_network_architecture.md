# **Neural Network Architecture**

## **Documentation: Neural Network Architecture (model.py)**

### **Overview**

The model module defines SiteScoringModel, a hybrid tabular neural network designed for high-cardinality datasets. It combines learned embeddings for categorical data with a Residual MLP (Multi-Layer Perceptron) for numeric features.

### **Architecture Rationale**

* **Embeddings:** Essential for features like retailer or dma which have too many unique values for one-hot encoding.  
* **Residual Connections:** ResidualBlock layers allow gradients to flow through deeper networks without vanishing, enabling the model to capture complex non-linear feature interactions.  
* **Batch Normalization:** Stabilizes training, allowing for the large batch sizes (e.g., 4096\) required to saturate the M4 GPU bandwidth.

### **Class Definitions**

#### **ResidualBlock**

A building block consisting of:

Linear \-\> BatchNorm \-\> ReLU \-\> Dropout \-\> Linear \-\> BatchNorm \-\> ReLU (+ Residual)

#### **CategoricalEmbedding**

Dynamically creates embedding layers based on the vocabulary size of each categorical feature.

* **Logic:** Dimension size is calculated as min(embedding\_dim, (vocab \+ 1\) // 2\) to prevent over-parameterization of low-cardinality features.

#### **SiteScoringModel**

The main container class.

1. **Forward Pass:**  
   * Embeds categorical inputs.  
   * Normalizes numeric inputs (BatchNorm1d).  
   * Concatenates Embeddings \+ Numeric \+ Boolean features.  
   * Passes the combined vector through the mlp (sequence of ResidualBlocks).  
   * Projects to a single scalar output (Prediction).  
2. **Initialization:** Uses Xavier Uniform initialization for stability.

### **Usage Example**

Python

from site\_scoring.model import SiteScoringModel

\# Instantiated automatically via Config, but can be manual:  
model \= SiteScoringModel(  
    n\_numeric=15,  
    n\_boolean=5,  
    categorical\_vocab\_sizes={'network': 4, 'retailer': 50},  
    embedding\_dim=16  
)  
