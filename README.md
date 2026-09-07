# Credit-Score-Classification

## Overview

This project aims to build a machine learning pipeline to predict the credit score of customers for a European bank in 2022. The objective is not only to train a model, but to understand how different data cleaning decisions, feature engineering steps, and modeling choices affect the final performance.

The project follows an iterative approach, analyzing how small changes in the pipeline impact the results.

## Data

The dataset is obtained from Kaggle:

https://www.kaggle.com/datasets/parisrohan/credit-score-classification

The data contains customer-level financial and behavioral information observed over multiple months, forming a panel structure.

Input: financial attributes such as income, debt, payment behavior, credit utilization, etc.
Output: credit score category for each customer.

The setup simulates a production scenario where the model is trained on historical data and then applied to future observations.

## Structure

```
project/
├── notebooks/      # exploration, cleaning validation, EDA, modeling
├── src/            # reusable code (data cleaning, feature engineering)
├── tests/          # scripts to validate functions and transformations
```

## Experimental Backlog


### Data Quality and Preprocessing effects on model

- [] Compare missing value imputation vs row deletion
- [] Measure the impact of removing observations with invalid or nonsensical data
- [] COmpare different numerical imputation strategies
- [] Analyze whether missingness itself contains predictive information
- [] Test feature scaling and empirically verify its effect on tree-based models

### Categorical features

- [] One-hot enconding vs target encoding
- [] native categorical handling vs manually encoded categorical features
- [] Study the effect of high-cardinality categorical varialbes

### Feature engineering

- Feature selection experiments (test different framewors such as BORUTA, SHAP, boosted tree's elimination, etc)

### Modeling

- [] Boosted trees family (XGBOOST, LIGHTGBM, CATBOOST, RANDOMFORSEST, ETC)

- [] New models such as tableFM, NGBoost, etc

### Credit-Risk specific methods

- [] Weight of Evidence (WoE)
- [] Information Value
- [] Evaluation metrics (KS, GIni, etc)
- [] Probability calibration
- [] Scorecard construction

### Validation

- [] Measure the effect of random split and temporal split on the results and metric (variance and bias)
- [] Test feature stability and drift on simulated test data

### Mlops / Engineering

- [] Experiment tracking with MLFLOW, Wights and Bias or SKORE
- [] Try different feature store libraries
- [] Api deployment
- [] Docker

### Agentic ML

- [] Test agent framework for automated data-quality checks
- [] Agent assisted feature discovery
- [] Automated experiment analysis
- [] Test skore agent 
