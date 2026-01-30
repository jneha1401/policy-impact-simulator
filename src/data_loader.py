"""
Data loading utilities for the Policy Impact Simulation Platform.
Loads the synthetic socioeconomic dataset and provides helper functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_COLS = [
    "gdp_growth_pct",
    "unemployment_rate",
    "education_spending_pct_gdp",
    "tax_revenue_pct_gdp",
    "human_development_index",
    "policy_score",
    "fiscal_efficiency",
    "gdp_per_capita",
    "poverty_change_yoy",
    "unem_rolling3",
    "social_investment",
    "social_investment_ratio",
    "tax_edu_ratio",
]

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "policy_data.csv"


def load_data(path=None) -> pd.DataFrame:
    """Load and return the raw policy dataset."""
    p = Path(path) if path else DATA_PATH
    df = pd.read_csv(p)
    df["year"] = df["year"].astype(int)
    return df


def get_countries(df: pd.DataFrame) -> list:
    """Return sorted list of unique country codes."""
    return sorted(df["country"].unique().tolist())


def get_latest_row(df: pd.DataFrame, country: str) -> pd.Series:
    """Return the most recent row for a given country."""
    subset = df[df["country"] == country]
    if subset.empty:
        raise ValueError(f"Country '{country}' not found in dataset.")
    return subset.loc[subset["year"].idxmax()]


def get_feature_matrix(df: pd.DataFrame):
    """Extract feature matrix X and target y for model training."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df.dropna(subset=available + ["poverty_rate"])
    X = df_clean[available].values
    y = df_clean["poverty_rate"].values
    return X, y, available
# cache data
# Handle edge cases
# Refactor
