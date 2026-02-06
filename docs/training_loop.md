# **Training Loop**

## **Documentation: Training Loop (train.py)**

### **Overview**

The train module orchestrates the model optimization process. It implements a robust training loop designed for the M4 architecture, featuring adaptive learning rates and early stopping to prevent overfitting.

### **Class: Trainer**

#### **Optimization Strategy**

* **Loss Function:** HuberLoss. Chosen over MSE because it is less sensitive to outliers (extreme revenue sites) which are common in this dataset.  
* **Optimizer:** AdamW (Adam with Weight Decay).  
* **Scheduler:** ReduceLROnPlateau. If validation loss stalls, the learning rate is halved to help the model converge.

#### **Key Methods**

* train\_epoch(): Handles the forward/backward pass, gradient clipping (max\_norm=1.0), and logging.  
* evaluate(): Calculates validation metrics including **MAE** (Mean Absolute Error) and **R²** (Coefficient of Determination).  
* train(): The main loop. Handles:  
  * Iterating through epochs.  
  * Saving checkpoints (best\_model.pt) whenever validation loss improves.  
  * Triggering **Early Stopping** if no improvement is seen for patience epochs.

### **Metrics**

The trainer tracks and logs:

* **Loss:** Huber loss (optimization objective).  
* **MAE:** Average error in real dollars (interpretable business metric).  
* **R²:** Variance explained (statistical goodness-of-fit).

### **Usage (Internal)**

This module is typically called via run.py, but can be used programmatically:

Python

trainer \= Trainer(model, config, processor)  
best\_model \= trainer.train(train\_loader, val\_loader)

