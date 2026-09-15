import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# 1. Import data (only two features for now)

df = pd.read_csv(
    "data/df_cleaned.csv", usecols=["Age", "Interest_Rate", "Credit_Score_binomial"]
)

print(df.head().to_string(index=False))

target = "Credit_Score_binomial"

# 2. Prepare data

X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.25,
    random_state=42,
)

# fit models
model = LogisticRegression()
model_2 = LogisticRegression(class_weight="balanced", C=0.1)
model.fit(X_train, y_train)
model_2.fit(X_train, y_train)

for feature, coef in zip(X.columns, model.coef_[0]):
    print("Simple model: \n")
    print(f"{feature}: {np.round(coef,3)} \n")

print(f"Intercept: {np.round(model.intercept_[0], 3)}")

for feature, coef in zip(X.columns, model_2.coef_[0]):
    print(" Penalized model: \n")
    print(f"{feature}: {np.round(coef,3)} \n")

print(f"Intercept: {np.round(model_2.intercept_[0], 3)}")


# 4. evaluate models

for m in (model, model_2):
    y_pred = m.predict(X_test)
    y_proba = m.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(
        f" \n Precision: {precision} \n Recall: {recall} \n roc_auc: {roc_auc} \n pr_auc: {pr_auc} \n"
    )

    print(classification_report(y_test, y_pred))
