import pandas as pd
from src.data_cleaning import (
    cleaning_int_variables,
    clean_credit_history,
    clean_type_of_loan,
)
import numpy as np


def test_clean_numeric():
    data = pd.Series(["123_", "45", "_2"])
    data_2 = pd.Series(["123", "_", "5"])

    result = cleaning_int_variables(data, "_", "", float)
    result_2 = cleaning_int_variables(data_2, "_", "0", int)
    expected = pd.Series([123.0, 45.0, 2.0])

    expected_2 = pd.Series([123, 0, 5])

    assert result.equals(expected)
    assert result_2.equals(expected_2)


def test_clean_credit_age():
    data = "3 Years and 9 Months"
    result = clean_credit_history(data)

    expected = 3 + 9 / 12
    assert result == expected


def test_clean_type_of_loan():
    data = "Auto Loan, Credit-Builder Loan, Personal Loan, and Home Equity Loan"

    result = clean_type_of_loan(data)
    expected = ["Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan"]

    assert result == expected
