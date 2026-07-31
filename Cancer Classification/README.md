

A deep learning project that classifies brain MRI scans into **4 categories** — Glioma, Meningioma, No Tumor, and Pituitary — using a custom **Convolutional Neural Network (CNN)** built with TensorFlow/Keras, targeting **90%+ accuracy**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

---

## 🔍 Overview

Brain tumors are one of the most critical conditions requiring early and accurate diagnosis. This project leverages **deep learning** to automate the classification of brain MRI images into four distinct categories. A custom CNN architecture with data augmentation, batch normalization, and learning rate scheduling is used to achieve high classification accuracy on unseen test data.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| **Training Images** | ~5,600 |
| **Testing Images** | ~1,600 |
| **Image Type** | Grayscale MRI scans |
| **Input Size** | Resized to 64×64 pixels |
| **Classes** | 4 |

### Classes

| Class | Description |
|---|---|
| **Glioma** | Tumor arising from glial cells |
| **Meningioma** | Tumor arising from the meninges |
| **No Tumor** | Healthy brain MRI (no tumor present) |
| **Pituitary** | Tumor in the pituitary gland |

---

## 🛠️ Project Workflow

```
1. Environment Setup
   └─ Install TensorFlow, Matplotlib, NumPy, Scikit-learn, Seaborn

2. Dataset Acquisition
   └─ Download Brain Tumor MRI Dataset from Kaggle via opendatasets

3. Data Exploration
   ├─ Inspect folder structure and per-class image counts
   └─ Visualize sample MRI images from each category

4. Data Preprocessing
   ├─ Resize all images to 64×64 (grayscale)
   ├─ Apply data augmentation on training set
   │   ├─ Rotation (±10°)
   │   ├─ Width/Height shift (5%)
   │   ├─ Horizontal flip
   │   └─ Zoom (5%)
   └─ Rescale pixel values to [0, 1]

5. Model Building
   └─ Design a 3-block CNN with BatchNorm, Dropout & GlobalAveragePooling

6. Model Training
   ├─ Train for up to 40 epochs
   ├─ ReduceLROnPlateau callback (factor=0.5, patience=4)
   └─ EarlyStopping callback (patience=12, restore best weights)

7. Evaluation
   ├─ Test accuracy and loss
   ├─ Prediction visualization on test samples
   ├─ Classification report (Precision, Recall, F1 per class)
   └─ Confusion matrix heatmap
```

---

## 🏗️ Model Architecture

A custom **Sequential CNN** with 3 convolutional blocks:

| Block | Layers | Filters | Details |
|---|---|---|---|
| **Block 1** | 2 × Conv2D + BatchNorm + MaxPool + Dropout | 32 | 3×3 kernel, same padding, 25% dropout |
| **Block 2** | 2 × Conv2D + BatchNorm + MaxPool + Dropout | 64 | 3×3 kernel, same padding, 25% dropout |
| **Block 3** | 2 × Conv2D + BatchNorm + MaxPool + Dropout | 128 | 3×3 kernel, same padding, 25% dropout |
| **Head** | GlobalAveragePooling2D → Dense → Softmax | — | Reduces parameters vs. Flatten |

### Training Configuration

| Parameter | Value |
|---|---|
| Input Shape | 64 × 64 × 1 (grayscale) |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Max Epochs | 40 |
| LR Scheduler | ReduceLROnPlateau (min LR = 1e-6) |
| Early Stopping | Patience = 12, monitors val_accuracy |

---

## 📈 Results

The model is evaluated on the **test set** using:

| Metric | Description |
|---|---|
| **Test Accuracy** | Overall classification accuracy on unseen data |
| **Test Loss** | Categorical cross-entropy loss on test set |
| **Precision** | Per-class ratio of true positives to predicted positives |
| **Recall** | Per-class ratio of true positives to actual positives |
| **F1-Score** | Per-class harmonic mean of Precision and Recall |
| **Confusion Matrix** | Visual heatmap of predicted vs actual class labels |

Additional visualizations:
- 📉 Training vs Validation **Accuracy** curve
- 📉 Training vs Validation **Loss** curve
- 🖼️ Predicted vs Actual labels on sample MRI images

> Detailed results, curves, and confusion matrix are available inside the Jupyter Notebook.

---

## 🧰 Tech Stack

| Tool / Library | Purpose |
|---|---|
| **Python 3** | Programming language |
| **TensorFlow / Keras** | Deep learning framework for CNN |
| **NumPy** | Numerical computing |
| **Matplotlib** | Plotting training curves and sample images |
| **Seaborn** | Confusion matrix heatmap |
| **Scikit-learn** | Classification report and confusion matrix |
| **Pillow (PIL)** | Image loading and inspection |
| **OpenDatasets** | Kaggle dataset download |
| **Jupyter Notebook** | Interactive development environment |

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.8
- pip / conda package manager
- Kaggle account (for dataset download)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd "Cancer Classification"

# 2. Install dependencies
pip install tensorflow matplotlib numpy scipy scikit-learn seaborn opendatasets

# 3. Launch the notebook
jupyter notebook "Cancer Classification AKSHAT GARG 23BCE10641.ipynb"
```

> **Note:** The notebook downloads the dataset automatically using `opendatasets`. You will be prompted for your Kaggle username and API key on first run.

---

## 📁 Project Structure

```
Cancer Classification/
├── Cancer Classification AKSHAT GARG 23BCE10641.ipynb   # Main notebook
├── README.md                                             # Project documentation
└── brain-tumor-mri-dataset/                              # MRI image dataset
    ├── Training/                                         # ~5,600 training images
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/                                          # ~1,600 test images
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

---

<p align="center">
  <i>Built with ❤️ using Python & TensorFlow</i>
</p>
