import pandas as pd
import numpy as np


def cleaning_int_variables(feature, char="_", new_char="", new_type=float):
    new_feature = feature.str.replace(char, new_char)

    return new_feature.astype(new_type)


def clean_credit_history(credit_history: pd.Series):
    if pd.isna(credit_history):
        return np.nan

    try:
        parts = credit_history.split()
        years = int(parts[0])
        months = int(parts[3])

        return years + months / 12

    except:
        return np.nan
