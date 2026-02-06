# **Documentation: Site Scoring ML Module**

## **Overview**

The site\_scoring module is a high-performance PyTorch-based machine learning library designed to predict site-level metrics (Revenue and Impressions).

It is specifically optimized for Apple Silicon (M-series chips, targeting the M4 architecture) using Metal Performance Shaders (MPS) for hardware acceleration, offering significant speed improvements over standard CPU training on macOS devices.

## **Module Metadata**

* **Version:** 0.1.0  
* **Framework:** PyTorch  
* **Target Hardware:** Apple Silicon (M4/M3/M2/M1) via mps backend

## **Architecture & Components**

The package exposes five core components via its top-level interface:

| Component | Type | Description |
| :---- | :---- | :---- |
| Config | Class | Central configuration object for hyperparameters (batch size, learning rate), hardware flags, and file paths. |
| SiteDataset | Class | Custom PyTorch Dataset implementation for handling site feature vectors and targets. |
| create\_data\_loaders | Function | Utility to generate optimized PyTorch DataLoader instances for training and validation sets. |
| SiteScoringModel | Class | The neural network architecture. Designed to ingest site features and output regression predictions for revenue/impressions. |
| Trainer | Class | Encapsulates the training loop, validation logic, checkpointing, and MPS device management. |

## **Usage Example**

Below is a standard workflow for initializing and training the model:

Python

from site\_scoring import Config, SiteScoringModel, Trainer, create\_data\_loaders

\# 1\. Initialize Configuration  
\# automatically detects 'mps' device if available  
config \= Config(batch\_size=64, learning\_rate=0.001)

\# 2\. Prepare Data Loaders  
train\_loader, val\_loader \= create\_data\_loaders(  
    data\_path="data/processed/site\_features.csv",  
    config=config  
)

\# 3\. Initialize Model  
model \= SiteScoringModel(config)

\# 4\. Initialize Trainer  
trainer \= Trainer(  
    model=model,  
    train\_loader=train\_loader,  
    val\_loader=val\_loader,  
    config=config  
)

\# 5\. Execute Training Loop  
trainer.train()

## **Apple Silicon Optimization Notes**

To ensure the module utilizes the Apple M4 Neural Engine/GPU:

1. Ensure you have a PyTorch build with MPS support enabled (torch.backends.mps.is\_available() should return True).  
2. The Config object defaults to device='mps' on macOS environments.  
3. If debugging is required, you can force CPU execution by setting config.device \= 'cpu'.