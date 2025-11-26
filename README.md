# Cervical Cytology Image Classification (SwinAD2Net)

This repository presents **SwinAD2Net**, a hybrid deep-learning model designed for the **automated classification of cervical cytology images**.  
The architecture combines **transformer-based global attention** with **multi-scale convolutional feature extraction**, aiming to improve the performance of medical image classification systems, particularly for **Pap smear and cervical cell analysis**.

---

## Introduction

Cervical cancer screening is one of the most effective preventive measures in women’s healthcare.  
Manual cytology inspection, however, is labor-intensive, prone to inter-observer variability, and limited by human fatigue.  
Automated classification systems can assist pathologists by providing **consistent triage**, **reducing false negatives**, and **accelerating diagnostic workflows**.

This work introduces **SwinAD2Net**, a **hybrid vision architecture** inspired by recent Vision Transformer (ViT) developments and the study by Zhang *et al.* (2025) [[DOI:10.1038/s41598-025-87953-1](https://doi.org/10.1038/s41598-025-87953-1)]. The main objective is to fill the gaps in the work produced by Zhang *et al.* by improving global lattent representation hidden layers.
The model integrates **Swin Transformer blocks**, **Atrous Dense Blocks (ADB)**, and **Squeeze-and-Excitation (SE)** modules to capture both fine-grained cellular morphology and broad contextual dependencies within each image.

---

## Objective

The primary goal of this project is to build an improved deep-learning architecture for medical image classification and to achieve higher diagnostic accuracy than previously reported approaches in cervical cytology.  
We addressed two two tasks: binary and seven classification of cervical images. The binary classification consisted of two classes:

- **Normal** (healthy epithelial or metaplastic cells)  
- **Abnormal** (atypical, dysplastic, or malignant cells)

and in the seven-class classification we had seven types of cells in the Herlev dataset:
- **Normal superficial**
- **Normal intermediate**,
- **Normal columnar**
- **Lightweight dysplastic**
- **Moderate dysplastic**
- **Server dysplastic**
- **Carcinoma**

and five class in the SipaKMed dataset:
- **Superficial intermediate**
- **Parabasal**
- **koilocytotic**
- **dyskeratotic**
- **metaplastic**

In the Herlev dataset the first three can
be further classified as normal cells and the last four as abnormal cells. And, in the SipakMed, the first two types are normal cells, the middle two are abnormal cells and the last type is benign, but belong to the abnormal group.

---

## Datasets

Two publicly available datasets were used in this study:

- **MDE-Lab Pap Smear Collection** — [https://mde-lab.aegean.gr/index.php/downloads/](https://mde-lab.aegean.gr/index.php/downloads/) (917 images)
- **SIPaKMeD Dataset** — [https://www.cs.uoi.gr/~marina/sipakmed.html](https://www.cs.uoi.gr/~marina/sipakmed.html) (4096 images)

Data augmentation was extensively applied (from each image was generated other 7) by randomly applying rotation, flipping, scaling, color jitter, contrast adjustment, random cropping, and elastic deformation (possibly composing more than one transformation) to improve model generalization and balance class distributions.  
Data augmentation were used only on the training data. In the training-validation, we used k-fold cross validation to separate the train and validation the data. In each fold, the indexes of the data were sampled only considering the original data. The images of the training set was augmented while the validation set remained the same.

---

## Method

The model was built to perform image classification. We adress the problem by proposing a new hybrid archtecture based mainly on Atrous Dense Block, and Swin Tranformers. 
![schme of SwinAD2Net](./img/main_scheme_AD2Net.png)

The figure above shows the architecture proposed in this work.  We aim to maintain the high capability for local feature extraction and recognition of DenseNet121 (proposed by Zhang et al. (2025)) while attaining global feature relationship with swin tranformers proposed by [Ze liu](https://arxiv.org/pdf/2103.14030) et al. (2021). 

**Detailing Architecture:** 
- **Image** - as input, it is used (224, 224) RGB images. The tensor size for the input is (batch, channels, height, width).
- **Patch Embedding** — converts images into non-overlapping patches and projects to the embeding space. The image is decomposed into patches of size `patch_size` and each patch is embedded into space of dimension `embed_dim`. For an input of dimension (batch, channels, height, width) we apply 2D convolution with kernel of size `patch_size`, stride of `patch_size` and output channels of `embed_dim` and we get a new tensor with dimensions (batch, embed_dim, height//patch_size, width//patch_size). Then, we obtain a representation of each patch independently of each other in the same linear space. It starts the process of representing the features in a latent space and helps in textures recognition by taking local relationships.
- **Swin Transformer Stages** — hierarchical attention over shifted windows for global context. A transformer arquitecture is used as an attention mechanism in each window, crossing information between each embeded patch inside this window. The above described archtecture is called Swin Transformer Block. In each stage, some Swin tranformer blocks are stacked and a shift/roll in the window partitioning is applied in order to connect information between diferent windows. In this way, the model can learn global relationships between patches and extract features that are not only local. ![shift window](./img/shift_window.png).
- **Atrous Dense Blocks (ADB)** — dilated convolutions for multi-scale feature extraction. The ADB is composed of mutiple convolutional layers with dilated convolutions. Each layer receives as input the concatenation of the feature maps of all preceding layers, promoting feature reuse and eficient gradient flow. In each layear, we apply a dilation in the kernel, what favors the global feature extraction by expanding the receptive field.
![Dilated kernel](./img/dilated_kernel.png)  In an ADB, the resolution remains unchanged in each convolutional layer and the output resolution is the same as the input, changing only the number of channels. This block is responsible for extracting local features in multiple scales, capturing fine-grained details in the images, which is crucial for medical image analysis. ![ADB](./img/adb.png)

- **Squeeze-and-Excitation (SE)** — adaptive channel reweighting. The SE module adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels. It consists of two main operations: "squeeze", which aggregates global spatial information into a channel descriptor using global average pooling:
$$
Z_{c}^{f} = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} U_{c}^{f}(i, j)
$$
 and "excitation", which captures channel-wise dependencies through a bottleneck architecture with two fully connected layers and non-linear activations. The output is a set of weights that are applied to the original feature maps via channel-wise multiplication, enhancing informative features while suppressing less useful ones. This mechanism allows the network to focus on the most relevant features for the classification task, improving overall performance.
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
