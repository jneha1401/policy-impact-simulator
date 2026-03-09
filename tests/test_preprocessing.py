import pytest
import pandas as pd
from src.preprocessing import clean_data, filter_by_year_range

def test_filter_by_year_range():
    df = pd.DataFrame({'year': [2000, 2005, 2010], 'val': [1, 2, 3]})
    filtered = filter_by_year_range(df, 2005, 2015)
    assert len(filtered) == 2
