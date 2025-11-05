# Cervical Cytology Image Classification (SwinAD2Net)

This repository presents **SwinAD2Net**, a hybrid deep-learning model designed for the **automated classification of cervical cytology images**.  
The architecture combines **transformer-based global attention** with **multi-scale convolutional feature extraction**, aiming to improve the performance of medical image classification systems — particularly for **Pap smear and cervical cell analysis**.

---

## Introduction

Cervical cancer screening is one of the most effective preventive measures in women’s healthcare.  
Manual cytology inspection, however, is labor-intensive, prone to inter-observer variability, and limited by human fatigue.  
Automated classification systems can assist pathologists by providing **consistent triage**, **reducing false negatives**, and **accelerating diagnostic workflows**.

This work introduces **SwinAD2Net**, a **hybrid vision architecture** inspired by recent Vision Transformer (ViT) developments and the study by Zhang *et al.* (2025) [[DOI:10.1038/s41598-025-87953-1](https://doi.org/10.1038/s41598-025-87953-1)].  
The model integrates **Swin Transformer blocks**, **Atrous Dense Blocks (ADB)**, and **Squeeze-and-Excitation (SE)** modules to capture both fine-grained cellular morphology and broad contextual dependencies within each image.

---

## Objective

The primary goal of this project is to **build an improved deep-learning architecture** for medical image classification and to achieve **higher diagnostic accuracy** than previously reported approaches in cervical cytology.  
The specific task addressed is **binary classification** of cervical images:

- **Normal** (healthy epithelial or metaplastic cells)  
- **Abnormal** (atypical, dysplastic, or malignant cells)

---

## Datasets

Two publicly available datasets were used in this study:

- **MDE-Lab Pap Smear Collection** — [https://mde-lab.aegean.gr/index.php/downloads/](https://mde-lab.aegean.gr/index.php/downloads/)  
- **SIPaKMeD Dataset** — [https://www.cs.uoi.gr/~marina/sipakmed.html](https://www.cs.uoi.gr/~marina/sipakmed.html)

Data augmentation was extensively applied — including rotation, flipping, scaling, color jitter, contrast adjustment, random cropping, and elastic deformation — to improve model generalization and balance class distributions.  
After augmentation, the total number of training samples reached **54,792 images**.

---

## Method Overview

**Task:** Binary classification (normal vs. abnormal cervical cells)  
**Architecture:**  
- **Patch Embedding** — converts images into non-overlapping patches  
- **Swin Transformer Stages** — hierarchical attention over shifted windows for global context  
- **Atrous Dense Blocks (ADB)** — dilated convolutions for multi-scale feature extraction  
- **Squeeze-and-Excitation (SE)** — adaptive channel reweighting  
- **Transition Layers** — down-sampling between stages  
- **Global Average Pooling + Linear Classifier**

This hybrid design merges **transformer-level contextual reasoning** with **CNN-style dense connectivity**, enabling rich spatial and semantic representation learning.

---

## Implementation

The repository is implemented in **PyTorch** and includes:

- `src/models/model.py` — SwinAD2Net architecture definition  
- `src/models/layers.py` — supporting layers (ADB, SE, transitions, Swin blocks)  
- `src/models/train.py` — training utilities with TensorBoard logging and checkpointing  
- `src/models/script_train.py` — example K-Fold cross-validation pipeline  
- `src/models/dataset.py` — dataset utilities and augmentation helpers  
- `tests/test_model.py` — minimal verification tests  

Training supports **mixed precision (AMP)** and **gradient checkpointing** for GPU memory optimization.  
Experiments were conducted using an **NVIDIA RTX A4500 (20 GB)** GPU.

---

## ⚡ Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
