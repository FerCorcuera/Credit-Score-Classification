import pandas as pd
import numpy as np


def cleaning_int_variables(feature, char="_", new_char="", new_type=float):
    new_feature = feature.str.replace(char, new_char)

    return new_feature.astype(new_type)
