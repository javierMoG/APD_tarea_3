import pandas as pd
from utils import fill_na, onehot, empty_df, load_data, write_data
import pytest

"""
Fixture - La función test_load_data() va a utilizar el retorno del path()
como un argumento.
"""

@pytest.fixture(scope="module")
def path():
    return "./data/train.csv"

def test_load_data(path):
    df = load_data(path)
    assert isinstance(df, pd.DataFrame)

def test_fill_na(path):
    df = load_data(path)
    filled_df = fill_na(df)
    # Check for missing values in the DataFrame
    missing_values = filled_df.isna().sum().sum()
    assert missing_values == 0

def test_one_hot(path):
    df = load_data(path)
    result = onehot(df)
    assert all(not pd.api.types.is_string_dtype(result[column]) for column in result.columns)