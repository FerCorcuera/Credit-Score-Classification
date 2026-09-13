import pandas as pd
import numpy as np


def correlation_analysis(feature: str, df: pd.DataFrame) -> pd.DataFrame:
    associations = []

    for feat in df.columns:
        if feat == feature:
            continue

        value = df[feature].corr(df[feat], method="spearman")

        associations.append(
            {"feature": feat, "correlation": value, "method": "spearman"}
        )

    return pd.DataFrame(associations).sort_values(
        by="correlation", ascending=False, key=lambda x: np.abs(x)
    )
