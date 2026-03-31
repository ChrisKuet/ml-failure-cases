# Failure Cases in Machine Learning Models

## Motivation
Machine learning models are often evaluated based on average predictive performance. However, strong average performance can mask **systematic failures** under realistic but challenging data conditions.

This repository investigates **when and why machine learning models break**, using controlled simulation studies and diagnostic tools grounded in statistical principles.

---

## Project Goal
To study how common machine learning models behave under **distributional stress**, including:

- Heavy-tailed noise  
- Outliers and contamination  
- Class imbalance  
- Covariate shift (train-test mismatch)

We focus not only on prediction accuracy, but also on:
- robustness  
- calibration  
- uncertainty  
- interpretability of failure  

---

## Models Studied
- Linear Regression  
- Ridge / Lasso  
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- (Optional) Bayesian Additive Regression Trees (BART)

---

## Failure Scenarios

### 1. Heavy-Tailed Noise
Examines how models behave when errors deviate from normality (e.g., t-distributions, skewed noise).

### 2. Outliers and Contamination
Studies sensitivity to extreme or corrupted observations.

### 3. Class Imbalance
Analyzes performance when one class is rare (e.g., 95/5 split).

### 4. Covariate Shift
Evaluates model performance when training and testing distributions differ.

---

## Evaluation Metrics
We go beyond standard metrics to highlight failure modes:

- RMSE / MAE  
- Prediction interval coverage  
- ROC-AUC / PR-AUC  
- Calibration curves  
- Residual diagnostics  
- Sensitivity to perturbations  

---

## Key Questions
- When does high accuracy become misleading?  
- Which models are robust to heavy tails and outliers?  
- How does class imbalance distort evaluation metrics?  
- What diagnostics reveal hidden model failure?  
- How does distribution shift affect generalization?  

