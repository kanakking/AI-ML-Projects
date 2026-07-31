# Car Price Prediction System

A machine learning project that predicts the **selling price of used cars** using a **Random Forest Regressor**, saved with Pickle and deployed as a **Flask web application**.

### 🌍 [Live Demo → car-price-prediction-app-00au.onrender.com](https://car-price-prediction-app-00au.onrender.com)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Details](#model-details)
- [Web Application](#web-application)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

---

## 🔍 Overview

Buying or selling a used car often involves guesswork around fair pricing. This project builds a **supervised regression model** that estimates a car's selling price based on features like present price, kilometers driven, fuel type, transmission, and age. The trained model is serialized using Pickle and served through a clean **Flask web interface** where users can input car details and get an instant price prediction in ₹ Lakhs.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Vehicle Dataset from CarDekho – Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) |
| **File Used** | `car data.csv` |
| **Records** | ~301 |
| **Target Variable** | `Selling_Price` (in Lakhs ₹) |

### Features

| Feature | Type | Description |
|---|---|---|
| `Present_Price` | Numeric | Current ex-showroom price of the car (Lakhs) |
| `Kms_Driven` | Numeric | Total kilometers driven |
| `Fuel_Type` | Categorical | Petrol (0), Diesel (1), CNG (2) |
| `Seller_Type` | Categorical | Dealer (0), Individual (1) |
| `Transmission` | Categorical | Manual (0), Automatic (1) |
| `Owner` | Numeric | Number of previous owners |
| `Car_Age` | Engineered | Age of the car (2025 − Year) |

> **Dropped columns:** `Car_Name` (too many unique values) and `Year` (replaced by `Car_Age`).

---

## 🛠️ Project Workflow

```
1. Environment Setup
   └─ Install pandas, numpy, scikit-learn, flask, seaborn, matplotlib

2. Dataset Acquisition
   └─ Download CarDekho dataset from Kaggle via opendatasets

3. Exploratory Data Analysis
   ├─ Inspect shape, info, describe
   ├─ Check for null values (none found)
   └─ Review column types and distributions

4. Data Cleaning & Feature Engineering
   ├─ Create 'Car_Age' feature from 'Year'
   ├─ Drop 'Car_Name' and 'Year'
   └─ Manually encode categorical variables (Fuel, Seller, Transmission)

5. Model Training
   ├─ 80/20 train-test split
   └─ Train RandomForestRegressor (n_estimators=100)

6. Model Evaluation
   ├─ R² Score
   └─ RMSE (Root Mean Squared Error)

7. Model Serialization
   └─ Save trained model as 'car_price_model.pkl' using Pickle

8. Web App Deployment
   ├─ Build Flask backend (app.py)
   ├─ Create HTML form interface (templates/index.html)
   └─ Serve predictions on localhost
```

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| **Algorithm** | Random Forest Regressor |
| **n_estimators** | 100 |
| **random_state** | 42 |
| **Train/Test Split** | 80/20 |
| **Serialization** | Pickle (`.pkl`) |

### Why Random Forest?

- Handles non-linear relationships well
- Robust to outliers and noisy data
- Captures feature interactions automatically
- Generally outperforms Linear Regression on tabular data

---

## 🌐 Web Application

The project includes a full **Flask web app** with a user-friendly form interface:

### Input Fields

| Field | Input Type | Options |
|---|---|---|
| Present Price (Lakhs) | Number | Free input |
| Kilometers Driven | Number | Free input |
| Fuel Type | Dropdown | Petrol / Diesel / CNG |
| Seller Type | Dropdown | Dealer / Individual |
| Transmission | Dropdown | Manual / Automatic |
| Number of Owners | Number | Free input |
| Car Age (Years) | Number | Free input |

### Output
- Predicted selling price displayed as: **₹ X.XX Lakhs**

### Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the prediction form |
| `/predict` | POST | Accepts form data, returns predicted price |

---

## 📈 Results

| Metric | Description |
|---|---|
| **R² Score** | Proportion of variance in selling price explained by the model |
| **RMSE** | Root Mean Squared Error — average prediction deviation in Lakhs |

> Detailed metric values are printed in the Jupyter Notebook after training.

---

## 🧰 Tech Stack

| Tool / Library | Purpose |
|---|---|
| **Python 3** | Programming language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Random Forest model, metrics, train-test split |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical plots |
| **Flask** | Web framework for model deployment |
| **Pickle** | Model serialization / deserialization |
| **OpenDatasets** | Kaggle dataset download |
| **Jupyter Notebook** | Interactive development environment |

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.8
- pip / conda package manager
- Kaggle account (for dataset download)

### Installation & Run

**Try it live:** [https://car-price-prediction-app-00au.onrender.com](https://car-price-prediction-app-00au.onrender.com)

Or run locally:

```bash
# 1. Clone the repository
git clone <repository-url>
cd "car price prediction"

# 2. Install dependencies
pip install pandas numpy scikit-learn flask seaborn matplotlib opendatasets

# 3. Option A: Run the notebook (trains model + saves pickle + creates app)
jupyter notebook "CAR_PRICE_PREDICTOR_AKSHAT_GARG_23BCE10641.ipynb"

# 3. Option B: Run the Flask app directly (if model pickle already exists)
python app.py
```

Then open **http://127.0.0.1:5000** in your browser to use the predictor.

> **Note:** The notebook downloads the dataset using `opendatasets`. You will be prompted for your Kaggle username and API key on first run.

---

## 📁 Project Structure

```
car price prediction/
├── CAR_PRICE_PREDICTOR_AKSHAT_GARG_23BCE10641.ipynb   # Main notebook (EDA + training + app creation)
├── app.py                                              # Flask web application
├── car_price_model.pkl                                 # Serialized Random Forest model
├── templates/
│   └── index.html                                      # Web form UI
├── vehicle-dataset-from-cardekho/                      # Dataset directory
│   ├── car data.csv                                    # Primary dataset used
│   ├── CAR DETAILS FROM CAR DEKHO.csv
│   ├── Car details v3.csv
│   └── car details v4.csv
└── README.md                                           # Project documentation
```

---

<p align="center">
  <i>Built with ❤️ using Python, Scikit-learn & Flask</i>
</p>
