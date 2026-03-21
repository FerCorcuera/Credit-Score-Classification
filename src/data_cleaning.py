import pandas as pd
import numpy as np


def remove_string(feature, char="_", new_char=""):
    if pd.isna(feature):
        return feature

    return feature.replace(char, new_char)


def cleaning_int_variables(feature, char="_", new_char="", new_type=int):
    new_feature = remove_string(feature, char, new_char)

    new_feature = new_feature.astype(new_type)

    return new_feature
