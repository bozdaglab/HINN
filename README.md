# HINN Model: Hierarchical Integration Neural Network for Multi-Omic Data

This repository contains an implementation of the HINN (Hierarchical Integration Neural Network) model, designed to learn complex relationships across multi-omic data layers for predicting cognitive scores (e.g., MMSE). The architecture supports modular interpretation through DeepLIFT and visualizes pathway influence using Sankey diagrams.

---

## 🔧 Model Overview

The HINN model integrates the following omic layers:

- **SNPs**
- **Methylation Sites**
- **Gene Expression**
- **Demographics**

Each layer is connected through biologically-informed sparse matrices:
- SNP-to-Methylation
- Methylation-to-Gene
- Gene-to-Pathway

---

## 🧠 Architecture Flow

### 1. Primary Layer 1 (SNP → Methylation)
Custom masked transformation using the sparse SNP–Methylation connectivity matrix.

### 2. Secondary Layer 1 (Methylation)
Identity mapping with learnable weights constrained by an identity mask.

### 3. Multiplicative Fusion
SNP-based transformation and methylation data are combined via element-wise multiplication followed by a non-linear transformation.

### 4. Hierarchical Progression
- Output is concatenated with a dense-transformed SNP layer.
- Further processed through Primary/Secondary layers for methylation → expression.
- Division and nonlinear operations refine the gene layer.

### 5. Final Integration (Gene → Pathway → Output)
- Pathway transformation is appended.
- Dense layers with batch normalization and dropout extract predictive signals.
- Demographic data is fused near the output layer.

---

## 🛠 Features

- **Captum DeepLIFT Integration** – Interpretation of feature importance.
- **Plotly Sankey Visualization** – Visual flow of top omic features to pathway level.
- **Custom Layers** – Enforces biological priors through sparse matrices.

---

## 📂 File Structure

- `HINN_model_deep_lift.py` – Main training and interpretation script.
- `*.csv` – Input files: omics datasets and sparse connectivity matrices.
- `requirements.txt` – Python package dependencies.

---

## ▶️ Usage

```bash
python HINN_model_deep_lift.py
