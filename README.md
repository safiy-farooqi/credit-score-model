
# 🏦 Credit Scoring Model with Python

This project implements a machine learning pipeline for **credit risk assessment**, built using real-world loan application data. The goal is to predict whether a borrower will default on a loan using financial and historical credit features.

## 📌 Overview

- **Tools Used:** `Python`, `pandas`, `NumPy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Models Applied:** `Logistic Regression`, `Decision Tree`, `Random Forest`
- **Target Variable:** `defaulted` (binary classification)
- **Key Techniques:** Data cleaning, feature engineering, log transformations, scaling, model training & evaluation

## 🔍 Project Workflow

### 1. Data Preprocessing
- Loaded loan data and explored shape, types, and missing values.
- Imputed missing values:
  - Median for `income` and `loan_amount`
  - Mode for `credit_history`

### 2. Exploratory Data Analysis
- Visualized:
  - Income and loan distributions
  - Loan term distribution
  - Relationship between credit history and default rate
- Correlation heatmap used to identify feature relationships

### 3. Feature Engineering
- Binary encoding of loan term (`60 months` → 1, `36 months` → 0)
- Log-transformed `income` and `loan_amount` to normalize skewed data

### 4. Feature Scaling
- Applied `StandardScaler` to normalized features for better model convergence

### 5. Model Training & Evaluation
- Split data into training and test sets (80/20)
- Trained 3 classifiers using `Pipeline`
- Evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**

## 📈 Results & Insights
- Compared the performance of Logistic Regression, Decision Tree, and Random Forest
- Gained insights into model interpretability vs performance trade-offs
- Focused on minimizing false negatives to reduce loan default risk

## 🧠 Key Takeaways
- Learned how to implement binary classification for financial applications
- Practiced robust data cleaning and feature engineering techniques
- Built job-ready machine learning models for credit risk scoring

## 🔗 Source Code
[GitHub Repository](https://github.com/safiy-farooqi/credit-score-model.git)
