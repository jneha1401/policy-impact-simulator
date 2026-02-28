import pandas as pd

def validate_schema(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return True

def check_data_bounds(df):
    warnings = []
    if df['unemployment_rate'].max() > 50:
        warnings.append("Extreme unemployment values detected (>50%)")
    if df['gdp_growth'].min() < -30:
        warnings.append("Severe economic contraction detected in data (<-30%)")
    return warnings
