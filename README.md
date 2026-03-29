# A Deep Ensemble Framework with Focal Loss for Robust Underwater Fish Classification

> **Official code repository for the manuscript:**
> *"A Deep Ensemble Framework with Focal Loss for Robust Underwater Fish Classification"*
> Submitted to **The Visual Computer**, Springer.
>
> ⭐ If you use this code or build upon this work, please **cite our manuscript** (see [Citation](#citation) below).

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Grad-CAM Visualizations](#grad-cam-visualizations)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository contains the complete implementation of a deep ensemble learning framework for classifying underwater fish species across 31 classes. The framework combines three complementary deep learning architectures:

| Model | Type | Key Strength |
|---|---|---|
| **EfficientNet-B0** | CNN | Compound scaling, lightweight, efficient |
| **Swin Transformer-Tiny** | Vision Transformer | Hierarchical windowed self-attention, global context |
| **ConvNeXt-Tiny** | Modernized CNN | Transformer-inspired design, strong generalization |

Each model is trained under **two loss functions**:
- ✅ **Class-weighted Cross-Entropy Loss** — for stable training
- ✅ **Focal Loss** — for improved minority-class recognition

Predictions from all three models are combined using **hard voting** and **soft voting** ensemble strategies. The best result — **soft voting with focal loss** — achieved:
- 🏆 **96.12% test accuracy**
- 🏆 **Macro-F1 score of 0.9547**

---

## Dataset

We use the publicly available [Mark Daniel Lampa Fish Dataset](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset) from Kaggle.

- **31 fish species**
- **13,323 images** (original); **8,897 images** after duplicate removal
- Split into train / validation / test sets

**Download instructions:**
1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset)
2. Download and extract the dataset
3. Place it in a folder accessible from your notebooks (update the path inside each notebook accordingly)

> A custom test dataset of **310 images** (10 per class) curated from internet sources was also used to evaluate real-world generalization. These are not included in this repo due to copyright constraints, but the preprocessing pipeline used is identical to the main dataset.

---

## Repository Structure

```
📦 A-Deep-Ensemble-Framework-with-Focal-Loss-for-Robust-Underwater-Fish-Classification/
│
├── 📓 EfficientNet(fin) (3).ipynb           # EfficientNet-B0 training with Cross-Entropy Loss
├── 📓 EfficientNet(fin)(focal loss).ipynb   # EfficientNet-B0 training with Focal Loss
│
├── 📓 Convnext(fin) (1).ipynb               # ConvNeXt-Tiny training with Cross-Entropy Loss
├── 📓 Convnext(fin)(focal).ipynb            # ConvNeXt-Tiny training with Focal Loss
│
├── 📓 swin_tiny(fin) (1).ipynb              # Swin Transformer-Tiny training with Cross-Entropy Loss
├── 📓 swin_tiny(fin)(focal).ipynb           # Swin Transformer-Tiny training with Focal Loss
│
├── 📓 emsemble.ipynb                        # Hard voting & soft voting ensemble evaluation
├── 📓 GradCam.ipynb                         # Grad-CAM visualizations for model interpretability
│
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## Requirements

- Python 3.10
- PyTorch 2.2
- torchvision 0.17
- timm
- scikit-learn
- NumPy
- Matplotlib
- Pandas
- Pillow
- tqdm
- CUDA 11.1 *(recommended for GPU acceleration)*

All dependencies are listed in `requirements.txt`.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Prithvi-Shenoy06/A-Deep-Ensemble-Framework-with-Focal-Loss-for-Robust-Underwater-Fish-Classification.git
cd A-Deep-Ensemble-Framework-with-Focal-Loss-for-Robust-Underwater-Fish-Classification

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Then launch Jupyter Notebook:
```bash
jupyter notebook
```

---

## How to Run

### Step 1 — Train Individual Models

Open and run the notebooks in this order. Each notebook includes:
- Data loading and preprocessing (resize to 224×224, ImageNet normalization)
- Data augmentation (horizontal flip, color jitter, random rotation)
- Class weight computation to handle imbalance
- 5-fold stratified cross-validation
- Final training with early stopping (patience = 5)
- Evaluation on the held-out test set (confusion matrix, ROC-AUC, F1)

| Notebook | Model | Loss Function |
|---|---|---|
| `EfficientNet(fin) (3).ipynb` | EfficientNet-B0 | Cross-Entropy |
| `EfficientNet(fin)(focal loss).ipynb` | EfficientNet-B0 | Focal Loss |
| `Convnext(fin) (1).ipynb` | ConvNeXt-Tiny | Cross-Entropy |
| `Convnext(fin)(focal).ipynb` | ConvNeXt-Tiny | Focal Loss |
| `swin_tiny(fin) (1).ipynb` | Swin-Tiny | Cross-Entropy |
| `swin_tiny(fin)(focal).ipynb` | Swin-Tiny | Focal Loss |

> ⚙️ **Training configuration used:**
> - Optimizer: Adam
> - Learning rate: 1e-4 (EfficientNet), 5e-5 (Swin-Tiny, ConvNeXt)
> - Batch size: 32 (EfficientNet, Swin-Tiny), 64 (ConvNeXt)
> - Max epochs: 30 with early stopping

### Step 2 — Run Ensemble

Open `emsemble.ipynb` to:
- Load saved model weights from all six trained models
- Run **hard voting** (majority vote) ensemble
- Run **soft voting** (average of softmax probabilities) ensemble
- Evaluate on the Kaggle test set and the custom internet-sourced test set

### Step 3 — Grad-CAM Visualizations

Open `GradCam.ipynb` to generate Grad-CAM heatmaps that highlight the image regions each model focuses on for its predictions.

---

## Results

### Individual Model Performance

| Model | Loss | CV Accuracy | Test Accuracy | Macro-F1 |
|---|---|---|---|---|
| EfficientNet-B0 | Cross-Entropy | 93.24% ± 0.21% | 93.93% | 0.9273 |
| EfficientNet-B0 | Focal Loss | 92.52% ± 0.38% | 92.54% | 0.9111 |
| Swin-Tiny | Cross-Entropy | 92.13% ± 0.30% | 94.22% | 0.9362 |
| Swin-Tiny | Focal Loss | 90.05% ± 1.46% | 92.32% | 0.9171 |
| ConvNeXt-Tiny | Cross-Entropy | 94.43% ± 1.82% | 93.42% | 0.9256 |
| **ConvNeXt-Tiny** | **Focal Loss** | 91.12% ± 4.45% | **95.32%** | **0.9470** |

### Ensemble Performance

| Ensemble Type | Loss | Test Accuracy | Macro-F1 |
|---|---|---|---|
| Hard Voting | Cross-Entropy | 95.39% | 0.9475 |
| Soft Voting | Cross-Entropy | 95.54% | 0.9471 |
| Hard Voting | Focal Loss | 95.25% | 0.9441 |
| **Soft Voting** | **Focal Loss** | **96.12%** | **0.9547** |

### Real-World Generalization (Custom Test Dataset)

The soft voting ensemble achieved **81.64% accuracy** on the custom internet-sourced dataset of 310 images (10 per class), with a macro-F1 of **0.7983**, demonstrating strong generalization to unseen real-world images.

---

## Grad-CAM Visualizations

The `GradCam.ipynb` notebook generates class activation maps to visually interpret what regions of an image each model uses to make its prediction. This is useful for verifying that the models are focusing on biologically meaningful fish features rather than background artifacts.

---

## Citation

If you use this code, please cite our manuscript:

```bibtex
@article{shenoy2025underwater,
  title     = {A Deep Ensemble Framework with Focal Loss for Robust Underwater Fish Classification},
  author    = {Shenoy, Prithvi and Ramyashree and Raghavendra, S. and Anoop, B. N.},
  journal   = {The Visual Computer},
  year      = {2025},
  publisher = {Springer},
  note      = {Manuscript submitted for publication. Code: https://doi.org/10.5281/zenodo.19315919}
}

```

> **DOI:** *(To be updated upon acceptance)*

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
