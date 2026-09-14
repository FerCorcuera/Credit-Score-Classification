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
