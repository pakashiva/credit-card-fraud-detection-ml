# Credit Card Fraud Detection using Machine Learning

This project builds a machine learning system to detect fraudulent credit card transactions using an imbalanced financial dataset.
The workflow includes data preprocessing, exploratory data analysis, handling class imbalance, model training, evaluation, and interpretability.

---

# Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small fraction of the dataset.

The goal of this project is to build a model that can accurately identify fraudulent transactions while minimizing costly missed fraud cases.

The project demonstrates a complete machine learning workflow from raw data to model evaluation.

---

# Dataset

Dataset used: **Credit Card Fraud Detection Dataset**

Characteristics of the dataset:

* Highly imbalanced dataset
* Fraud transactions are extremely rare
* Features are PCA transformed (V1–V28) for privacy
* Includes transaction amount and class label

Target variable:

```
Class
0 → Normal transaction
1 → Fraud transaction
```

---

# Project Workflow

## 1. Data Preprocessing

* Removed unnecessary columns
* Scaled numerical features
* Split dataset into training and test sets
* Used stratified sampling to maintain class distribution

---

## 2. Exploratory Data Analysis (EDA)

Performed exploratory analysis using:

* Histogram plots
* Frequency plots
* Class distribution visualization
* Fraud vs non-fraud comparisons

Libraries used:

* Matplotlib
* Seaborn

---

## 3. Handling Class Imbalance

Since fraud datasets are extremely imbalanced, several resampling techniques were explored:

* Random Undersampling
* Random Oversampling
* SMOTE
* SMOTE + Tomek Links

These techniques help the model better learn patterns associated with fraudulent transactions.

---

## 4. Model Training

Machine learning models were trained and evaluated using:

* Logistic Regression
* Tree-based models
* Cross-validation
* GridSearchCV for hyperparameter tuning

---

## 5. Model Evaluation

Multiple evaluation metrics were used to properly assess model performance:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Cross-Validation Scores

Precision-Recall curves are particularly useful for imbalanced classification problems.

---

## 6. Threshold Optimization

A cost-sensitive threshold tuning approach was implemented to reduce the impact of:

* False Negatives (missed fraud)
* False Positives (unnecessary investigation)

This improves the practical usefulness of the model in real-world fraud detection systems.

---

## 7. Model Interpretability

SHAP (SHapley Additive exPlanations) was used to understand feature contributions to predictions.

This helps explain how different features influence fraud detection decisions.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* SHAP

---

# Key Highlights

* End-to-end machine learning pipeline
* Handling highly imbalanced datasets
* Advanced evaluation metrics (ROC & PR curves)
* Cost-based threshold optimization
* Model explainability using SHAP

---

## Dataset Credit
The dataset used in this project is publicly available on **Kaggle** and is utilized here strictly for educational and analytical purposes.
[[https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)]

---

# Future Improvements

Possible improvements include:

* Testing advanced models such as XGBoost and LightGBM
* Building a real-time fraud detection pipeline
* Deploying the model using a web API
* Adding automated monitoring for model drift

---

# Author

**Shiva Prasad**

Machine Learning & Data Science Enthusiast
