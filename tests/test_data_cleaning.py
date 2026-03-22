import pandas as pd
from src.data_cleaning import cleaning_int_variables
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
