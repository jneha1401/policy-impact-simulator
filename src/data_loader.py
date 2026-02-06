"""
Dataset loading utilities for the Policy Impact Simulation Platform.
Handles loading, filtering, and slicing the socioeconomic dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "policy_data.csv"

FEATURE_COLS = [
    "gdp_growth_pct",
    "unemployment_rate",
    "education_spending_pct_gdp",
    "tax_revenue_pct_gdp",
    "human_development_index",
    "policy_score",
]

TARGET_COL = "poverty_rate"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the policy dataset and return a clean DataFrame."""
    df = pd.read_csv(path)
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    return df


def get_countries(df: pd.DataFrame) -> list:
    return sorted(df["country"].unique().tolist())


def get_years(df: pd.DataFrame) -> list:
    return sorted(df["year"].unique().tolist())


def filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    return df[df["country"] == country].copy()


def filter_year_range(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df[(df["year"] >= start) & (df["year"] <= end)].copy()


def get_latest_row(df: pd.DataFrame, country: str) -> pd.Series:
    country_df = filter_country(df, country)
    return country_df.sort_values("year").iloc[-1]


def get_feature_matrix(df: pd.DataFrame):
    """Return X (features) and y (target) for model training."""
    clean = df[FEATURE_COLS + [TARGET_COL]].dropna()
    X = clean[FEATURE_COLS].values
    y = clean[TARGET_COL].values
    return X, y, FEATURE_COLS
