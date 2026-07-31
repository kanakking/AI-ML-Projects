# AI-ML Assignments

A collection of **9 projects** spanning Machine Learning, Deep Learning, Computer Vision, Natural Language Processing, and Reinforcement Learning — built with Python, TensorFlow, Scikit-learn, Flask, and more.

---

## 📑 Projects at a Glance

| # | Project | Domain | Tech Highlights | Live Demo |
|---|---|---|---|---|
| 1 | [Adult Census Income Classification](./Adult%20Census%20Income%20Classification/) | ML — Classification | Logistic Regression, Decision Tree, Random Forest, KNN, SVM | — |
| 2 | [Cancer Classification (Brain Tumor MRI)](./Cancer%20Classification/) | DL — Medical Imaging | CNN, TensorFlow, ImageDataGenerator, 4-class MRI | — |
| 3 | [Car Price Prediction](./car%20price%20prediction/) | ML — Regression | Random Forest, Flask, Pickle | [🌍 Live on Render](https://car-price-prediction-app-00au.onrender.com) |
| 4 | [CIFAR-10 Image Classification](./CIFAR%2010/) | DL — Computer Vision | CNN, Data Augmentation, BatchNorm | — |
| 5 | [LFW Face Recognition](./LFW%20face%20recognition/) | DL — Face Recognition | CNN, LFW Dataset, 7-class face ID | — |
| 6 | [Movie Recommendation System](./Movie-Recommendation-System/) | ML — NLP / Recommender | TF-IDF, Cosine Similarity, Flask | — |
| 7 | [RAG Chatbot (Amazon 10-Q)](./Rag_Chatbot/) | NLP — RAG | FastAPI, FAISS, Gemini AI, Sentence-Transformers | — |
| 8 | [CartPole (PPO)](./cartpole/) | RL — Control | Stable-Baselines3, PPO, Gymnasium | — |
| 9 | [Lunar Lander (DQN)](./lunar%20lander/) | RL — Control | Stable-Baselines3, DQN, Box2D | — |

---

## 🔬 Project Details

### 1. 💰 [Adult Census Income Classification](./Adult%20Census%20Income%20Classification/)

Predicts whether an individual's annual income exceeds **$50K** based on census data. Compares **5 classifiers** (Logistic Regression, Decision Tree, Random Forest, KNN, SVM) on metrics like Accuracy, Precision, Recall, F1, and ROC-AUC.

**Dataset:** Adult Census Income Dataset (Kaggle) — ~32,561 records, 14 features  
**Pipeline:** Data Cleaning → Feature Engineering (One-Hot, StandardScaler) → Model Training → Evaluation

---

### 2. 🧠 [Cancer Classification — Brain Tumor MRI](./Cancer%20Classification/)

Classifies brain MRI images into **4 categories** (Glioma, Meningioma, No Tumor, Pituitary) using a custom **3-block CNN** with BatchNorm, Dropout, and data augmentation. Targets **90%+ accuracy**.

**Dataset:** Brain Tumor MRI Dataset (Kaggle) — 5,600 training + 1,600 test images  
**Architecture:** 3 × Conv blocks (32→64→128) + GlobalAveragePooling + EarlyStopping

---

### 3. 🚗 [Car Price Prediction](./car%20price%20prediction/)

Predicts the selling price of used cars using a **Random Forest Regressor**, deployed as a **Flask web app** on Render.

**Dataset:** Vehicle Dataset from CarDekho (Kaggle) — ~301 records  
**Features:** Present Price, Kms Driven, Fuel Type, Transmission, Car Age, etc.  
**Deployment:** Flask + Pickle → Render

---

### 4. 🖼️ [CIFAR-10 Image Classification](./CIFAR%2010/)

Classifies **32×32 RGB images** into **10 object categories** (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck) using a custom CNN with data augmentation. Targets **85%+ accuracy**.

**Dataset:** CIFAR-10 — 50,000 training + 10,000 test images (loaded via TensorFlow)  
**Architecture:** 3 × Conv blocks (32→64→128) + Flatten + Dense + ReduceLROnPlateau

---

### 5. 👤 [LFW Face Recognition](./LFW%20face%20recognition/)

Recognizes **faces of 7 public figures** from the Labeled Faces in the Wild dataset using a custom CNN. Handles limited training data through augmentation and stratified splitting.

**Dataset:** LFW — 1,288 grayscale images (50×37), 7 classes (loaded via scikit-learn)  
**Architecture:** 3 × Conv blocks (32→64→128) + EarlyStopping (patience=20)

---

### 6. 🎬 [Movie Recommendation System](./Movie-Recommendation-System/)

A content-based movie recommendation engine using **TF-IDF Vectorization** and **Cosine Similarity** on movie genres. Served through a clean **Flask web interface**.

**Dataset:** TMDB Movie Dataset  
**Pipeline:** TF-IDF on genres → Cosine Similarity matrix → Top-N recommendations

---

### 7. 🤖 [RAG Chatbot — Amazon Quarterly Report](./Rag_Chatbot/)

A Retrieval-Augmented Generation chatbot that answers questions about Amazon's 10-Q quarterly report, grounded in the actual filing text with **citations**.

**Stack:** FastAPI + Sentence-Transformers (MiniLM) + FAISS (vector search) + Gemini AI  
**Flow:** Embed query → FAISS similarity search → Top-k excerpts → Gemini generates grounded answer

---

### 8. 🎮 [CartPole — PPO Reinforcement Learning](./cartpole/)

Trains an agent to balance a pole on a cart using **Proximal Policy Optimization (PPO)** with Stable-Baselines3 in the Gymnasium CartPole-v1 environment.

**Framework:** Stable-Baselines3, Gymnasium  
**Components:** `train.py`, `evaluate.py`, `test.py`, `record_video.py`, `plot_training.py`

---

### 9. 🚀 [Lunar Lander — DQN Reinforcement Learning](./lunar%20lander/)

Trains an autonomous spacecraft agent to safely land on a designated pad using **Deep Q-Network (DQN)** with the Box2D physics simulator.

**Framework:** Stable-Baselines3, Gymnasium (LunarLander-v3)  
**Components:** `train.py`, `evaluate.py`, `test.py`, `record_video.py`, `plot_training.py`

---

## 🧰 Technologies Used

| Category | Tools & Libraries |
|---|---|
| **Languages** | Python 3 |
| **ML / DL** | Scikit-learn, TensorFlow / Keras |
| **RL** | Stable-Baselines3, Gymnasium |
| **NLP / RAG** | TF-IDF, Sentence-Transformers, FAISS, Gemini AI |
| **Web** | Flask, FastAPI |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Render, Pickle |
| **Environment** | Jupyter Notebook, VS Code |

---

## 📁 Repository Structure

```
├── Adult Census Income Classification/   # ML classification (5 models)
├── Cancer Classification/                # CNN brain tumor MRI (4-class)
├── car price prediction/                 # Random Forest + Flask web app
├── CIFAR 10/                             # CNN image classification (10-class)
├── LFW face recognition/                 # CNN face recognition (7-class)
├── Movie-Recommendation-System/          # TF-IDF + Flask recommender
├── Rag Chatbot/                          # RAG chatbot with FAISS + Gemini
├── cartpole/                             # PPO reinforcement learning
├── lunar lander/                         # DQN reinforcement learning
└── README.md                             # This file
```

---

<p align="center">
  <i>Built with ❤️ by Akshat Garg</i>
</p>
