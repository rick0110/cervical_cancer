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
![Dilated kernel](./img/dilated_kernel.png)  
In an ADB, the resolution remains unchanged in each convolutional layer and the output resolution is the same as the input, changing only the number of channels. This block is responsible for extracting local features in multiple scales, capturing fine-grained details in the images, which is crucial for medical image analysis. ![ADB](./img/adb.png)

- **Squeeze-and-Excitation (SE)** — adaptive channel reweighting. The SE module adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels. It consists of two main operations: "squeeze", which aggregates global spatial information into a channel descriptor using global average pooling:$$ Z_{c}^{f} = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} U_{c}^{f}(i, j) $$, where $ U_{c}^{f}(i,j) $ is the feature map of the \( c \) channel and (i,j) height-width coordiantes, and \( H, W \) are its height and width;
and "excitation", which captures channel-wise dependencies through a bottleneck architecture with two fully connected layers and non-linear activations. The output is a set of weights that are applied to the original feature maps via channel-wise multiplication, enhancing informative features while suppressing less useful ones. This mechanism allows the network to focus on the most relevant features for the classification task, improving overall performance.
![SE](./img/squeeze_excitation_block.png)
- **Transition Layers** — down-sampling between stages. Transition layers are used to reduce the spatial and channel dimensions of feature maps between ADB and Transformer-blocks stages of this network. This is achieved through a convolutional layer with kernel size 1 and an average pooling layer with stride greater than one. A parameter $\theta$ is set to be the decrease rate of the number of channels and sptial resolution ($\theta = 0.5$ in our model). A Batch Normalization followed by ReLU activation is applied and finally it is passed through a convolutional layer with kernel size 1 that maps the channel feature map to $\theta$ times the input size. Then, an average pooling with kernel and stride sizes $\dfrac{1}{\theta}$ is apllied. By down-sampling the feature maps, transition layers help to decrease computational complexity and memory usage, while also allowing the network to learn more abstract representations at deeper levels. In SwinAD2Net, transition layers are placed after certain stages to effectively manage the resolution of feature maps as they progress through the network.
![Transition layer](./img/Transition_layer.png)

- **Global Average Pooling + Linear Classifier** - final prediction layer. After the feature extraction stages, a global average pooling layer is applied to reduce each spatial feature map to a single value by averaging over all spatial locations so that $Z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} U_{c}(i, j)$. This results in a fixed-size feature vector regardless of the input image size. This vector is then passed through a fully connected linear layer that maps it to the desired number of output classes for classification. The linear classifier produces the final predictions by applying a softmax activation (for multi-class classification) to obtain class probabilities.

This hybrid design merges **transformer-level contextual reasoning** with **CNN-style dense connectivity**, enabling rich spatial and semantic representation learning. 

---
## Evaluation

The model was evaluated using standard classification metrics. We performed experiments on both datasets using **5-fold cross-validation** to ensure robustness and generalizability. The first experiment was binary classification between normal and abnormal cells using the two datasets. The primary metrics used for evaluation included:
- **Accuracy (Acc)**: $Acc = \frac{TP + TN}{TP + FN + TN + FP}$
- **Precision (Pre)**: $Pre = \frac{TP}{TP + FP}$
- **Specificity (Spe)**: $Spe = \frac{TN}{TN + FP}$
- **Recall (Rec)**: $Rec = \frac{TP}{TP + FN}$
- **F1-Score**: $F1\text{-}Score = 2 \times \frac{Pre \times Rec}{Pre + Rec} = \frac{2TP}{2TP + FN + FP}$

Accuracy measures the overall correctness of the model, precision quantifies the accuracy of positive predictions, specificity assesses the model's ability to identify negative cases, recall evaluates the model's sensitivity to positive cases, and F1-Score provides a balance between precision and recall.
These metrics provide a comprehensive assessment of the model's performance in correctly identifying both normal and abnormal cervical cells. For multi-class classification, we used the same metrics calculated for each class and then averaged (macro average) to get an overall performance measure.

### Results
The experiments were conducted using an **NVIDIA RTX A4500 (20 GB)** GPU and in each fold, we used **batch size of 32** and trained for **700 epochs**. The learning rate was initialized at **0.001** and decayed using a cosine annealing schedule. The optimizer used was **AdamW** with weight decay of **0.01**. The loss function employed was **Cross-Entropy Loss**. 


## Implementation

The repository is implemented in **PyTorch** and includes:

- `src/swinad2net/models/model.py` - **SwinAD2Net Architecture**: Defines the core `SwinAD2Net` and `SwinAD2Net_ASPP_like` (for more paralelization) classes, which integrates Swin Transformer blocks with Atrous Dense Blocks (ADB) and Squeeze-and-Excitation (SE) modules.
- `src/swinad2net/models/layers.py` - **Custom Layers**: Implements the building blocks of the network, including:
    - `PatchEmbedding`: Converts input images into patch embeddings.
    - `AtrousDenseBlock and AtrousDenseBlock_ASPP_like`: A dense block variant using dilated convolutions for multi-scale feature extraction.
    - `TransitionLayer`: Handles downsampling and channel reduction between stages.
    - `ADB_SE_Transition and ADB_SE_Transition_ASPP_like`: Combines Atrous Dense Block with Squeeze-and-Excitation and TransitionLayer for enhanced feature recalibration.
    - `SwinTransformerBlock`: Wrappers or implementations for the Swin Transformer attention mechanism.
    
- `src/swinad2net/models/script_compare_paper_vs_hyperband.py` - **Main Training Script**: Runs a single stratified split experiment (train/validation/test) to compare `A2SDNet121`, `SwinAD2Net_ASPP_like`, and `SwinAD2Net_ASPP_like_SwinResidual` in parallel. It logs metrics, histograms, confusion matrix and detailed Lipschitz rates in TensorBoard, and only evaluates the test split at the end using the best validation checkpoint.
- `src/swinad2net/models/dataset.py` - **Data Loading**: Implements custom PyTorch datasets that load images from DataFrames and apply `torchvision.transforms` pipelines.
- `src/swinad2net/models/lipschitz_regularization.py` - **Lipschitz Utilities**: Provides per-layer spectral norm estimation and network Lipschitz bound helpers used for TensorBoard diagnostics.
- `tests/` - **Unit Tests**:
    - `test_dataset.py`: unittest for utilitaries classes and functions related to dataset handling.
    - `test_model_full.py`: Verifies the `SwinAD2Net and SwinAD2Net_ASPP_like` model instantiation and forward pass with expected input shapes.
    - `test_layers.py`: Tests individual layers like `AtrousDenseBlock` and `TransitionLayer` to ensure correct output dimensions.

---

## References:

 - Zhang, Y., et al. (2025). An automatic cervical cell
classification model based on
improved DenseNet121. [https://www.nature.com/articles/s41598-025-87953-1](https://www.nature.com/articles/s41598-025-87953-1).
 - Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. [https://arxiv.org/pdf/2103.14030](https://arxiv.org/pdf/2103.14030).
- Huang, G., et al. (2016). Densely Connected Convolutional Networks. [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993).
