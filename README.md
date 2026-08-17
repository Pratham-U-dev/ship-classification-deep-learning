# 🚢 Ship Classification using Deep Learning (FGSC-23)

> Fine-grained classification of commercial and civilian ships from remote sensing imagery using transfer learning — MobileNetV2 vs ResNet50V2.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-red?logo=keras)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Table of Contents

- [Result](#result)
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Getting Started](#getting-started)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Performance Comparison](#performance-comparison)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [License](#license)

---
## Result
<img width="914" height="506" alt="image" src="https://github.com/user-attachments/assets/d469bf65-63bd-407d-8e19-e852e46e4a53" />


---

## Overview

This project tackles **fine-grained ship classification** from optical remote sensing satellite images. Using the FGSC-23 dataset, two deep learning architectures are trained and evaluated on **10 selected civilian and commercial ship categories**, excluding all military vessel types.

The project was developed as an academic assignment and demonstrates:

- Dataset loading directly from Kaggle within Google Colab
- Transfer learning with frozen backbone feature extractors
- Comparative evaluation of MobileNetV2 and ResNet50V2
- Classification reports, confusion matrices, and per-model inference visualization

---

## Dataset

**Source:** [FGSC-23 — Fine-Grained Ship Classification Dataset](https://www.kaggle.com/datasets/mrkk8565/ship-classification)

The FGSC-23 dataset contains 4,080 remote sensing images cropped from Google Earth and GF-2 satellite imagery, covering 23 ship categories annotated by domain experts.

### Selected Classes (Civilian & Commercial Only)

This project intentionally excludes military and defence vessel types (IDs 1–10). Only the following 10 categories are used:

| Ship ID | Category           | Train Samples | Test Samples |
|---------|--------------------|:-------------:|:------------:|
| 11      | Medical Ship       | ~93           | ~10          |
| 14      | Container Ship     | ~186          | ~19          |
| 15      | Car Carrier        | ~130          | ~14          |
| 16      | Hovercraft         | ~230          | ~24          |
| 17      | Bulk Carrier       | ~672          | ~69          |
| 18      | Oil Tanker         | ~319          | ~33          |
| 19      | Fishing Boat       | ~195          | ~20          |
| 20      | Passenger Ship     | ~174          | ~18          |
| 21      | Liquefied Gas Ship | ~194          | ~20          |
| 22      | Barge              | ~107          | ~11          |

**Total:** ~927 training images / ~238 test images across 10 classes

> Images are RGB (colour) satellite imagery resized to 224×224 pixels for model input.

---

## Project Structure

```
ship-classification/
│
├── 4SO23AI070.ipynb          # Main Jupyter/Colab notebook
├── README.md                 # This file
└── LICENSE                   # MIT License
```

> The dataset is fetched automatically from Kaggle via `kagglehub` — no manual download required.

---

## Models

Two pretrained architectures were fine-tuned for this classification task:

### 1. MobileNetV2 (Feature Extractor)

- **Base:** MobileNetV2 pretrained on ImageNet, frozen
- **Head:** GlobalAveragePooling2D → Dense(128, ReLU) → BatchNorm → Dropout(0.5) → Dense(10, Softmax)
- **Preprocessing:** MobileNetV2 native preprocessing (scales pixel values to [-1, 1])
- **Optimizer:** Adam, LR = 1e-3 with ReduceLROnPlateau (factor 0.5, patience 3)
- **Callbacks:** EarlyStopping (patience 8, restore best weights)

### 2. ResNet50V2 (Feature Extractor)

- **Base:** ResNet50V2 pretrained on ImageNet, all layers frozen (~23.5M non-trainable params)
- **Head:** GlobalAveragePooling2D → Dropout(0.5) → Dense(512, ReLU) → BatchNorm → Dropout(0.3) → Dense(10, Softmax)
- **Trainable params:** ~1.05M (head only)
- **Optimizer:** Adam, LR = 1e-3 with ReduceLROnPlateau
- **Callbacks:** EarlyStopping (patience 8, restore best weights)

Both models are trained for up to 30 epochs with `categorical_crossentropy` loss.

---

## Results

### MobileNetV2 — Classification Report

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|:---------:|:------:|:--------:|:-------:|
| Medical Ship       | 0.83      | 1.00   | 0.91     | 10      |
| Container Ship     | 0.82      | 0.74   | 0.78     | 19      |
| Car Carrier        | 0.93      | 1.00   | 0.97     | 14      |
| Hovercraft         | 0.96      | 0.96   | 0.96     | 24      |
| Bulk Carrier       | 0.84      | 0.77   | 0.80     | 69      |
| Oil Tanker         | 0.81      | 0.88   | 0.84     | 33      |
| Fishing Boat       | 0.62      | 0.75   | 0.68     | 20      |
| Passenger Ship     | 0.62      | 0.56   | 0.59     | 18      |
| Liquefied Gas Ship | 0.95      | 0.95   | 0.95     | 20      |
| Barge              | 0.91      | 0.91   | 0.91     | 11      |
| **Accuracy**       |           |        | **0.83** | 238     |
| Macro Avg          | 0.83      | 0.85   | 0.84     | 238     |
| Weighted Avg       | 0.83      | 0.83   | 0.83     | 238     |

### ResNet50V2 — Classification Report

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|:---------:|:------:|:--------:|:-------:|
| Medical Ship       | 1.00      | 0.80   | 0.89     | 10      |
| Container Ship     | 0.71      | 0.63   | 0.67     | 19      |
| Car Carrier        | 1.00      | 0.79   | 0.88     | 14      |
| Hovercraft         | 0.89      | 1.00   | 0.94     | 24      |
| Bulk Carrier       | 0.75      | 0.81   | 0.78     | 69      |
| Oil Tanker         | 0.71      | 0.73   | 0.72     | 33      |
| Fishing Boat       | 0.67      | 0.60   | 0.63     | 20      |
| Passenger Ship     | 0.65      | 0.61   | 0.63     | 18      |
| Liquefied Gas Ship | 1.00      | 1.00   | 1.00     | 20      |
| Barge              | 0.91      | 0.91   | 0.91     | 11      |
| **Accuracy**       |           |        | **0.79** | 238     |
| Macro Avg          | 0.83      | 0.79   | 0.80     | 238     |
| Weighted Avg       | 0.79      | 0.79   | 0.79     | 238     |

### Summary Comparison

| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|:--------:|:---------:|:------:|:--------:|
| MobileNetV2 | **0.8277** | **0.8295** | **0.8277** | **0.8267** |
| ResNet50V2  | 0.7899   | 0.7922    | 0.7899 | 0.7887   |

> **MobileNetV2 outperforms ResNet50V2** on this dataset across all metrics, likely due to the small training set size favouring the lighter, more efficient MobileNetV2 architecture.

---

## Getting Started

### Prerequisites

This notebook is designed to run on **Google Colab**. No local environment setup is required beyond a Kaggle API token.

### Step 1 — Set up Kaggle API credentials in Colab

Before running the notebook, upload your `kaggle.json` API token:

```python
from google.colab import files
files.upload()  # Upload kaggle.json

import os
os.makedirs('/root/.kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 600)
```

You can download your `kaggle.json` from [kaggle.com → Settings → API → Create New Token](https://www.kaggle.com/settings).

### Step 2 — Open the notebook in Colab

Click the badge below or upload `4SO23AI070.ipynb` to your Google Drive and open it with Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Step 3 — Run all cells

The notebook will automatically:

1. Install `kagglehub`
2. Download the FGSC-23 dataset
3. Preprocess and load images via `ImageDataGenerator`
4. Train MobileNetV2 (feature extraction mode)
5. Train ResNet50V2 (feature extraction mode)
6. Evaluate both models with classification reports and confusion matrices
7. Display a side-by-side test image inference for each model

### Dependencies

All dependencies are pre-installed in Google Colab. The notebook uses:

```
tensorflow >= 2.20
kagglehub
numpy
matplotlib
scikit-learn
pandas
```

---

## Notebook Walkthrough

The notebook (`4SO23AI070.ipynb`) is structured as follows:

**1. Setup & Dataset Download**
Installs `kagglehub` and downloads the FGSC-23 dataset directly into the Colab runtime using the Kaggle API.

**2. Data Preprocessing & Loading**
Configures `ImageDataGenerator` with `rescale=1.0/255` normalization. Images are loaded at 224×224 resolution in batches of 32, filtered to the 10 civilian ship classes only.

**3. Architecture 1 — MobileNetV2**
Loads MobileNetV2 with frozen ImageNet weights. A custom classification head is appended and trained for up to 30 epochs with adaptive learning rate scheduling.

**4. Architecture 2 — ResNet50V2**
Loads ResNet50V2 with frozen ImageNet weights (~23.5M non-trainable parameters). A deeper classification head (Dense 512) is trained similarly.

**5. Evaluation**
Both models are evaluated on the held-out test set. Outputs include full classification reports, confusion matrices, and training history plots (accuracy & loss curves).

**6. Performance Comparison**
A side-by-side bar chart compares Accuracy, Precision, Recall, and F1-Score for both architectures.

**7. Inference — Test Image Prediction**
A random test image is passed through both models. The predicted class and confidence score are displayed alongside the ground truth label.

---

## Performance Comparison

The key takeaway from this experiment is that **a lighter model can outperform a deeper one** when data is limited:

- MobileNetV2 converged faster (best weights at epoch 20, early stopped at 28) and achieved a higher test accuracy of **82.77%**
- ResNet50V2 with a much larger backbone (24.6M total params) converged earlier (best epoch 6, stopped at 14) and plateaued at **78.99%**
- Both models struggled most with **Fishing Boat** and **Passenger Ship** — likely due to visual similarity with Bulk Carriers and other large vessels
- **Liquefied Gas Ship** was the easiest class for ResNet50V2, achieving a perfect 1.00 F1-score, likely due to its highly distinctive spherical tank structure
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/5095e8b5-cbab-47d0-935d-43c2eff7b6ae" />


---

## Limitations & Future Work

**Current Limitations:**
- Small training set (~927 images, 10 classes) — susceptible to overfitting on training data
- No data augmentation was applied, leaving room for improved generalization
- Only feature extraction (frozen backbone) was used; fine-tuning deeper layers may improve accuracy

**Potential Improvements:**
- Add data augmentation (horizontal flip, rotation, zoom, brightness shift)
- Unfreeze top layers of the backbone for fine-tuning in a second training phase
- Experiment with EfficientNetB0/B3 or Vision Transformers (ViT)
- Apply class-weighting or oversampling to handle class imbalance (Bulk Carrier dominates with 69 test samples vs Medical Ship with 10)
- Grad-CAM visualization to interpret model attention regions on ship images

---

## References

1. **Zhang, X., Lv, Y., Yao, L., Xiong, W., & Fu, C.** (2020). *A New Benchmark and an Attribute-Guided Multilevel Feature Representation Network for Fine-Grained Ship Classification in Optical Remote Sensing Images.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

2. **Liu, Z., Yuan, L., Weng, L., et al.** (2017). *A high resolution optical satellite image dataset for ship recognition and some new baselines.* Proceedings of the 6th International Conference on Pattern Recognition Applications and Methods.

3. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C.** (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.

4. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Identity Mappings in Deep Residual Networks.* ECCV 2016. (ResNet50V2)

5. Kaggle Dataset: [mrkk8565/ship-classification](https://www.kaggle.com/datasets/mrkk8565/ship-classification)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The FGSC-23 dataset is intended for **research purposes only**. Please cite the original papers (listed above) when using this dataset in any publication.

---

<p align="center">
  Made with ❤️ for academic purposes · TensorFlow · Google Colab
</p>
