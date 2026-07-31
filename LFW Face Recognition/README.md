# Face Recognition using CNN — Labeled Faces in the Wild (LFW)
 

A deep learning project that recognizes **faces of 7 public figures** from the **LFW (Labeled Faces in the Wild)** dataset using a custom **Convolutional Neural Network (CNN)** built with TensorFlow/Keras, targeting **90%+ accuracy**.

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

Face recognition is a fundamental task in computer vision with applications in security, authentication, and photo management. This project uses the **LFW benchmark dataset** — a collection of real-world face photographs — to train a CNN that identifies individuals. The model handles the challenge of **limited training data** through data augmentation, batch normalization, and learning rate scheduling.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [LFW – Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) (loaded via `sklearn.datasets.fetch_lfw_people`) |
| **Total Images** | 1,288 |
| **Image Size** | 50 × 37 pixels (grayscale) |
| **Filter** | Minimum 70 images per person |
| **Classes** | 7 individuals |
| **Train/Test Split** | 80/20 (stratified) |

### People in Dataset

| Person | Description |
|---|---|
| Ariel Sharon | Former Prime Minister of Israel |
| Colin Powell | Former U.S. Secretary of State |
| Donald Rumsfeld | Former U.S. Secretary of Defense |
| George W. Bush | 43rd President of the United States |
| Gerhard Schröder | Former Chancellor of Germany |
| Hugo Chávez | Former President of Venezuela |
| Tony Blair | Former Prime Minister of the United Kingdom |

> The dataset is imbalanced — George W. Bush has significantly more images than others.

---

## 🛠️ Project Workflow

```
1. Environment Setup
   └─ Install TensorFlow, Matplotlib, NumPy, Scikit-learn, Seaborn

2. Data Loading
   └─ Fetch LFW dataset via sklearn (min_faces_per_person=70, resize=0.4)

3. Data Exploration
   ├─ Inspect dataset dimensions and class count
   ├─ Count images per person
   └─ Visualize sample faces from each individual

4. Preprocessing
   ├─ Pixels already normalized to [0, 1] by sklearn
   ├─ Reshape to (50, 37, 1) for grayscale CNN input
   └─ Stratified train/test split (80/20)

5. Model Building
   └─ Design a 3-block CNN with BatchNorm, Dropout & Dense head

6. Data Augmentation & Training
   ├─ Augmentation: rotation (±10°), shift (10%), flip, zoom (10%)
   ├─ Compile with Adam + sparse categorical cross-entropy
   ├─ ReduceLROnPlateau (factor=0.5, patience=8)
   └─ EarlyStopping (patience=20, restore best weights)

7. Evaluation
   ├─ Test accuracy and loss
   ├─ Training vs Validation accuracy/loss curves
   ├─ Prediction visualization (correct = green, wrong = red)
   ├─ Classification report (Precision, Recall, F1 per person)
   └─ Confusion matrix heatmap
```

---

## 🏗️ Model Architecture

A custom **Sequential CNN** with 3 convolutional blocks:

| Block | Layers | Filters | Details |
|---|---|---|---|
| **Block 1** | Conv2D + BatchNorm + MaxPool + Dropout | 32 | 3×3 kernel, same padding, 25% dropout |
| **Block 2** | Conv2D + BatchNorm + MaxPool + Dropout | 64 | 3×3 kernel, same padding, 25% dropout |
| **Block 3** | Conv2D + BatchNorm + MaxPool + Dropout | 128 | 3×3 kernel, same padding, 25% dropout |
| **Head** | Flatten → Dense → Dropout → Softmax | — | 7-class output layer |

### Training Configuration

| Parameter | Value |
|---|---|
| Input Shape | 50 × 37 × 1 (grayscale) |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Cross-Entropy |
| LR Scheduler | ReduceLROnPlateau (min LR = 1e-6) |
| Early Stopping | Patience = 20, monitors val_accuracy |
| Data Augmentation | Rotation, shift, horizontal flip, zoom |

---

## 📈 Results

The model is evaluated on the **stratified test set** using:

| Metric | Description |
|---|---|
| **Test Accuracy** | Overall classification accuracy on unseen faces |
| **Test Loss** | Sparse categorical cross-entropy loss on test set |
| **Precision** | Per-person ratio of true positives to predicted positives |
| **Recall** | Per-person ratio of true positives to actual positives |
| **F1-Score** | Per-person harmonic mean of Precision and Recall |
| **Confusion Matrix** | 7×7 heatmap of predicted vs actual identities |

Additional visualizations:
- 📉 Training vs Validation **Accuracy** curve
- 📉 Training vs Validation **Loss** curve
- 🖼️ Predicted vs Actual labels on test faces (color-coded: green ✓ / red ✗)

> Detailed results, curves, and confusion matrix are available inside the Jupyter Notebook.

---

## 🧰 Tech Stack

| Tool / Library | Purpose |
|---|---|
| **Python 3** | Programming language |
| **TensorFlow / Keras** | Deep learning framework for CNN |
| **Scikit-learn** | LFW dataset loading, train-test split, metrics |
| **NumPy** | Numerical computing |
| **Matplotlib** | Plotting training curves and sample faces |
| **Seaborn** | Confusion matrix heatmap |
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
cd "LFW face recognition"

# 2. Install dependencies
pip install tensorflow matplotlib numpy scipy scikit-learn seaborn

# 3. Launch the notebook
jupyter notebook "LFW AKSHAT GARG 23BCE10641.ipynb"
```

> **Note:** The LFW dataset is downloaded automatically by scikit-learn on first run (~200 MB). No Kaggle account needed.

---

## 📁 Project Structure

```
LFW face recognition/
├── LFW AKSHAT GARG 23BCE10641.ipynb   # Main notebook (full pipeline)
└── README.md                          # Project documentation
```

---

<p align="center">
  <i>Built with ❤️ using Python & TensorFlow</i>
</p>
