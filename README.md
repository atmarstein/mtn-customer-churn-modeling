# Telco Customer Churn Prediction

This project applies business analytics techniques to predict customer churn using a cleaned Telco dataset.

## 🔍 Project Overview
This is part of a Master's-level capstone for BAN6800 (Business Analytics Capstone). The particular work is divided into two major phases:

- **Milestone 1**: Data cleaning, preparation, and feature engineering
- **Module 4 Assignment**: Model development, evaluation, and results reporting

---

## 📁 Files in This Repository

| File Name | Description |
|-----------|-------------|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Raw dataset before cleaning, encoding, and PCA |
| `cleaned_telco_final.csv` | Final dataset after cleaning, encoding, and PCA |
| `churn_cleaning_script.py` | Python script used for cleaning and preparing the data |
| `churn_modeling_script.py` | Python script for model training and evaluation |
| `churn_modeling_notebook.ipynb` | Jupyter version of the model development process |
| `roc_curve_comparison.png` | ROC curve comparing Logistic Regression and Random Forest performance |
| `README.md` | This documentation |

---

## 📊 Models Used

1. **Logistic Regression**
2. **Random Forest Classifier**

---

## 🧪 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- ROC Curve

All evaluation metrics were computed and printed. ROC curve is saved as `roc_curve_comparison.png`.

---

## 💡 Instructions

To reproduce results:

1. Clone the repo
2. Run either `churn_modeling_script.py` in a Python environment, or open the notebook
3. Ensure `cleaned_telco_final.csv` is in the same directory

---

