# AdvanceAI Project Submission  
### Integrated Healthcare AI Framework  
**Centralized • Federated • Noise-Augmented • Integrated Learning**

This project implements a unified Healthcare AI system combining:
- **Centralized Learning (Accuracy)**
- **Federated Learning Simulation (Privacy)**
- **Noise-Augmented Defense Training (Robustness)**
- **Integrated Ensemble Model (Majority Vote)**

The system is evaluated on a custom breast cancer dataset (`synthetic_dataset_B.csv`)
as well as the sklearn fallback dataset.

---
### **1. Code/**
Contains all executable code
  - **centralized_federated_defense_model.ipynb**
  Includes the full Google Colab notebook:  
  This notebook implements:
  - Centralized MLP Classifier  
  - Federated Learning Simulation (2 nodes)  
  - Noise-Augmented Defense Model  
  - Integrated Ensemble (Majority Voting)  
  - Plots, predictions, classification reports  
  - Saving models to `Models/` folder  

- **synthetic_dataset_B.csv**  
  Custom dataset used for training and evaluation.  
  If missing, the notebook automatically falls back to the sklearn breast cancer dataset.

---
# How to Run the Code
### **Option 1 — Google Colab (Recommended)**
1. Open Google Colab:  
   https://colab.research.google.com/
2. Upload the notebook file:  
   `Code/sourcecode/centralized_federated_defense_model.ipynb`
3. Upload the dataset when prompted:  
   `synthetic_dataset_B.csv`
4. Run all cells (Runtime ? Run All)
5. Trained models and results will automatically be generated in the `/Models` folder.

