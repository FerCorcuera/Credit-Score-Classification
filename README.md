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

