# 🖼️ CIFAR-10 Image Classification using CNN

A deep learning project that classifies **32×32 RGB images** into **10 categories** using a custom **Convolutional Neural Network (CNN)** built with TensorFlow/Keras, targeting **85%+ accuracy**.

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

CIFAR-10 is a widely-used benchmark dataset in computer vision. This project builds a custom **CNN with batch normalization, dropout, and data augmentation** to classify small images into 10 real-world object categories. The model is trained with a learning rate scheduler and evaluated using accuracy curves, a classification report, and a confusion matrix.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (loaded via `tf.keras.datasets`) |
| **Training Images** | 50,000 |
| **Test Images** | 10,000 |
| **Image Size** | 32 × 32 × 3 (RGB) |
| **Classes** | 10 |

### Classes

| # | Class | # | Class |
|---|---|---|---|
| 0 | Airplane | 5 | Dog |
| 1 | Automobile | 6 | Frog |
| 2 | Bird | 7 | Horse |
| 3 | Cat | 8 | Ship |
| 4 | Deer | 9 | Truck |

---

## 🛠️ Project Workflow

```
1. Environment Setup
   └─ Install TensorFlow, Matplotlib, NumPy, Scikit-learn, Seaborn

2. Data Loading
   └─ Load CIFAR-10 directly from tf.keras.datasets (pre-split)

3. Data Exploration
   └─ Visualize 25 sample images with their class labels

4. Preprocessing
   ├─ Normalize pixel values from [0, 255] → [0, 1]
   └─ Apply data augmentation on training set
       ├─ Rotation (±15°)
       ├─ Width/Height shift (10%)
       └─ Horizontal flip

5. Model Building
   └─ Design a 3-block CNN with BatchNorm, Dropout & Dense head

6. Model Training
   ├─ Train for 30 epochs with batch size 64
   └─ ReduceLROnPlateau callback (factor=0.5, patience=3)

7. Evaluation
   ├─ Test accuracy and loss
   ├─ Training vs Validation accuracy/loss curves
   ├─ Prediction visualization (correct = green, wrong = red)
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
| **Head** | Flatten → Dense → Softmax | — | 10-class output layer |

### Training Configuration

| Parameter | Value |
|---|---|
| Input Shape | 32 × 32 × 3 (RGB) |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Cross-Entropy |
| Batch Size | 64 |
| Max Epochs | 30 |
| LR Scheduler | ReduceLROnPlateau (min LR = 1e-6) |
| Data Augmentation | Rotation, shift, horizontal flip |

---

## 📈 Results

The model is evaluated on the **10,000 test images** using:

| Metric | Description |
|---|---|
| **Test Accuracy** | Overall classification accuracy on unseen data |
| **Test Loss** | Sparse categorical cross-entropy loss on test set |
| **Precision** | Per-class ratio of true positives to predicted positives |
| **Recall** | Per-class ratio of true positives to actual positives |
| **F1-Score** | Per-class harmonic mean of Precision and Recall |
| **Confusion Matrix** | 10×10 heatmap of predicted vs actual labels |

Additional visualizations:
- 📉 Training vs Validation **Accuracy** curve
- 📉 Training vs Validation **Loss** curve
- 🖼️ Predicted vs Actual labels on sample test images (color-coded)

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
| **Jupyter Notebook** | Interactive development environment |

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.8
- pip / conda package manager

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd "CIFAR 10"

# 2. Install dependencies
pip install tensorflow matplotlib numpy scipy scikit-learn seaborn

# 3. Launch the notebook
jupyter notebook "CIFAR-10 AKSHAT GARG 23BCE10641.ipynb"
```

> **Note:** The CIFAR-10 dataset is downloaded automatically by TensorFlow on first run (~170 MB). No Kaggle account needed.

---

## 📁 Project Structure

```
CIFAR 10/
├── CIFAR-10 AKSHAT GARG 23BCE10641.ipynb   # Main notebook (full pipeline)
└── README.md                                # Project documentation
```

---

<p align="center">
  <i>Built with ❤️ using Python & TensorFlow</i>
</p>
