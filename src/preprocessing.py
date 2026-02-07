"""
Preprocessing pipeline for the Policy Impact Simulation Platform.
Handles cleaning, imputation, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with excessive missingness and fill minor gaps."""
    df = df.copy()
    df = df.dropna(thresh=int(len(df.columns) * 0.6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived policy indicators."""
    df = df.copy()

    # Fiscal efficiency: tax revenue relative to education spending
    df["fiscal_efficiency"] = df["tax_revenue_pct_gdp"] / (
        df["education_spending_pct_gdp"] + 1e-6
    )

    # GDP per capita proxy
    df["gdp_per_capita"] = (df["gdp_billion_usd"] * 1e9) / (
        df["population_millions"] * 1e6 + 1e-6
    )

    # Year-over-year poverty change per country
    df = df.sort_values(["country", "year"])
    df["poverty_change_yoy"] = df.groupby("country")["poverty_rate"].diff()

    # Rolling 3-year average unemployment
    df["unem_rolling3"] = (
        df.groupby("country")["unemployment_rate"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Social investment score (education + HDI combined)
    df["social_investment"] = (
        0.6 * df["education_spending_pct_gdp"]
        + 0.4 * df["human_development_index"] * 10
    )

    return df


def scale_features(X_train, X_test=None):
    """Standard-scale feature matrices."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def prepare_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    df = clean_data(df)
    df = engineer_features(df)
    return df
# updated
