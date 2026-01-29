# **CLI Entry Point**

## **Documentation: CLI Entry Point (run.py)**

### **Overview**

The run.py script serves as the command-line interface (CLI) for the library. It allows users to trigger training jobs with specific hyperparameters without modifying code.

### **Supported Arguments**

| Argument | Default | Description |
| :---- | :---- | :---- |
| \--target | revenue | The metric to predict. Options: revenue, monthly\_impressions. |
| \--batch-size | 4096 | Size of training batches. |
| \--epochs | 50 | Number of training passes. |
| \--lr | 0.001 | Learning rate. |
| \--device | auto | Force cpu or mps. |

### **Usage**

To train a model predicting Average Monthly Revenue:

Bash

python \-m site\_scoring.run \--target avg\_monthly\_revenue \--epochs 100 \--batch-size 4096  
