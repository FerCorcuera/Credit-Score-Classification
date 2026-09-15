import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

models_to_test = {
    "Logistic": LogisticRegression(),
    "XGB": XGBClassifier(),
    "RandomForest": RandomForestClassifier(),
}


def evaluate_classification_model(
    y_pred: pd.Series,
    y_proba: pd.Series,
    y_true: pd.Series,
    model_name: str,
    threshold_analysis: bool = False,
) -> dict[str, pd.DataFrame | None]:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()

    report = {
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Real_Positives": tp + fn,
        "Total_Positive_Preds": tp + fp,
        "True Positives": tp,
    }

    if threshold_analysis:
        report_2 = []
        thresholds = np.arange(0.1, 1, 0.1)

        for thresh in thresholds:
            y_tpred = y_proba >= thresh
            t_precision = precision_score(y_true, y_tpred)
            t_recall = recall_score(y_true, y_tpred)

            tn, fp, fn, tp = confusion_matrix(y_true, y_tpred).ravel().tolist()

            report_2.append(
                {
                    "threshold": thresh,
                    "Precision": t_precision,
                    "Recall": t_recall,
                    "Real_Positives": tp + fn,
                    "Total_Positive_Preds": tp + fp,
                    "True Positives": tp,
                }
            )
        return pd.DataFrame([report]), pd.DataFrame(report_2)

    return pd.DataFrame([report])


def compare_classification_models(
    models: dict,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    result = []

    for model in models.values():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        df_report = evaluate_classification_model(
            y_pred, y_proba, y_test, model_name=model.__class__.__name__
        )

        result.append(df_report)

    return pd.concat(result, ignore_index=True)


def test_compare_classification_models(
    models: dict,
):
    y_train = pd.Series([1, 0, 1, 1, 1, 0])
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6]})
    y_test = pd.Series([1, 1, 0, 0])

    X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})

    df_report = compare_classification_models(models, X_train, y_train, X_test, y_test)

    print(df_report.to_string(index=False))


test_compare_classification_models(models_to_test)
