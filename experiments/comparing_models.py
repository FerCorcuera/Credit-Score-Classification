import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.classification_model_comparison import (
    evaluate_classification_model,
    compare_classification_models,
)

# 0. Declaring parameteres

target = "Credit_Score_binomial"

# 1. reading clean data

df = pd.read_csv("data/df_cleaned.csv")

# prepare data

X = df.select_dtypes(include=["number"])
X = X.drop(columns=["num_month", target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print(f"Shape of the train data: {X_train.shape}")
print(f"Shape of the test data: {X_test.shape}")
print(f"Shape of the train target data: {y_train.shape}")
print(f"Shape of the test target data: {y_test.shape} \n")


# train a single model

xgb_model = XGBClassifier()

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]


report, threshold_report = evaluate_classification_model(
    y_pred, y_proba, y_test, model_name="XGBoost", threshold_analysis=True
)

print(report.to_string(index=False))
print(threshold_report.to_string(index=False))

# train several classification models:

models = {
    "XGBClassifier": XGBClassifier(),
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
}

df_report = compare_classification_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
)

print(
    f"\n Comparing clasification models > \n {df_report.head(10).to_string(index=False)}"
)
