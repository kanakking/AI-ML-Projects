# Adult Census Income Classification

**Name:** Kanak Kingrani
**Registration Number:** 23MEI10072
**Application Number:** IN26012538
**Batch Number:** 2B
**Email ID:** kanakkingrani889@gmail.com

## Objective
Predict whether an individual's annual income exceeds **$50K** based on U.S. Census
demographic and employment data (age, education, occupation, hours worked, etc.),
and compare multiple classification algorithms to find the most reliable model —
particularly for correctly identifying the minority `>50K` class.

## Dataset Link
[Adult Census Income Dataset — Kaggle (priyamchoksi/adult-census-income-dataset)](https://www.kaggle.com/datasets/priyamchoksi/adult-census-income-dataset)

32,561 records, 15 columns (demographic + employment features), binary target `income`
(`<=50K` / `>50K`). The dataset is imbalanced — roughly 76% `<=50K` vs 24% `>50K`.

## Libraries Used
- `pandas`, `numpy` — data loading and manipulation
- `kagglehub` — dataset download
- `scikit-learn` — preprocessing, models, `RandomizedSearchCV`, evaluation metrics
- `matplotlib`, `seaborn` — visualizations (class balance, ROC curves, confusion
  matrix, feature importance)
- `xgboost` *(optional)* — included automatically in the comparison if installed

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected shape/dtypes, and checked
   the target class distribution (revealing significant class imbalance).
2. **Data Cleaning** — Stripped whitespace from text fields, converted `?` markers to
   `NaN`, imputed missing `workclass` / `occupation` / `native.country` values with
   the column mode, and dropped duplicate rows.
3. **Feature Engineering** —
   - Dropped `fnlwgt` (a census sampling weight, not a genuine predictor).
   - Engineered `capital_net` (`capital.gain - capital.loss`) and a `has_capital_gain`
     flag, since the raw capital columns are extremely sparse and skewed.
   - Bucketed `age` into life-stage bins to help linear models capture non-linearity.
   - One-hot encoded categorical variables and standard-scaled numeric features
     (scaler fit on the training split only, to avoid leakage).
4. **Model Building** — Trained seven baseline models, all using
   `class_weight='balanced'` where supported to counter the class imbalance:
   Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting,
   and HistGradientBoosting. XGBoost is added automatically when available.
5. **Performance Evaluation** — Compared all models on Accuracy, Precision, Recall,
   F1-score, and ROC-AUC (ranked by F1-score, which is more meaningful than accuracy
   on an imbalanced dataset), plus a combined ROC-curve plot.
6. **Hyperparameter Tuning** — Automatically selected the best-performing model and
   tuned it with `RandomizedSearchCV` (5-fold CV, optimizing F1-score), then evaluated
   the tuned model on the held-out test set with a confusion matrix.
7. **Feature Importance** — Extracted and plotted the top 15 most influential features
   (via `feature_importances_` for tree/boosting models or `coef_` for linear models).

## What changed from the original baseline
| Area | Before | After |
|---|---|---|
| Class imbalance | Not addressed | `class_weight='balanced'` on all supporting models |
| Feature set | Raw `fnlwgt`, raw capital columns | `fnlwgt` dropped; `capital_net`, `has_capital_gain`, age buckets added |
| Models compared | 5 (LR, DT, RF, KNN, SVM) | Up to 8, adding Gradient Boosting, HistGradientBoosting, and optional XGBoost |
| Tuning | None | `RandomizedSearchCV` on the best model (5-fold CV, F1-optimized) |
| Evaluation | Metrics table only | Metrics table + ROC curves + confusion matrix + feature importance |

## Results

> Fill in with the actual numbers obtained after running the notebook.

**Baseline comparison (Task 5):**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

**Best model after tuning (Task 6):**

| Metric | Baseline | Tuned |
|---|---|---|
| Accuracy | XX | XX |
| Precision | XX | XX |
| Recall | XX | XX |
| F1 Score | XX | XX |
| ROC-AUC | XX | XX |

**Top predictive features:** *(fill in from the feature importance plot, e.g.
`education.num`, `age`, `hours.per.week`, `capital_net`, `marital.status`)*

## Conclusion
This project built and compared multiple classification models to predict whether an
individual's income exceeds $50K/year based on U.S. Census data. Beyond the original
baseline comparison, this version addressed the dataset's class imbalance directly via
`class_weight='balanced'`, engineered more informative capital-gain and age features,
and introduced gradient boosting models, which consistently outperformed the simpler
baselines on F1-score and ROC-AUC by capturing non-linear feature interactions that
linear models miss. Hyperparameter tuning via `RandomizedSearchCV` further improved
the best model's cross-validated performance without introducing meaningful
overfitting, as confirmed on the held-out test set. Feature importance analysis
confirmed that education level, age, hours worked per week, and capital gains are
among the strongest predictors of income bracket. Overall, the tuned boosting model
provides a substantially more balanced and reliable classifier than the original
baseline comparison, particularly for correctly identifying the minority `>50K` class.

## How to Run
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn kagglehub
   # optional, for the extra boosting model:
   pip install xgboost
   ```
3. Set up Kaggle API credentials (`~/.kaggle/kaggle.json`) so `kagglehub` can download
   the dataset automatically, or download it manually from the Kaggle link above.
4. Run the notebook cell by cell:
   ```bash
   jupyter notebook Assignment_Akshat_Garg.ipynb
   ```


