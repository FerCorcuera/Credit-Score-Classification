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


def clean_type_of_loan(type_of_loan: pd.Series):
    if pd.isna(type_of_loan):
        return []

    return [loan.strip() for loan in type_of_loan.replace("and", "").split(",")]


def fill_credit_age(customer_history: pd.Series):
    customer_history = customer_history.copy()
    step = 1 / 12

    first_valid = customer_history.first_valid_index()

    if first_valid is not None:
        first_pos = customer_history.index.get_loc(first_valid)

        for i in range(first_pos - 1, -1, -1):
            customer_history.iloc[i] = customer_history.iloc[i + 1] - step

    previous_value = None

    for index, value in enumerate(customer_history):
        if not pd.isna(value):
            previous_value = value

        else:
            if previous_value is not None:
                customer_history.iloc[index] = previous_value + 1 / 12
                previous_value = customer_history.iloc[index]

    return customer_history
