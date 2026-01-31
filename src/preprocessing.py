"""
Preprocessing pipeline for the Policy Impact Simulation Platform.
Handles cleaning, imputation, and feature engineering.
Includes year-range filter helper and outlier diagnostics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(thresh=int(len(df.columns) * 0.6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fiscal_efficiency"] = df["tax_revenue_pct_gdp"] / (
        df["education_spending_pct_gdp"] + 1e-6)
    df["gdp_per_capita"] = (df["gdp_billion_usd"] * 1e9) / (
        df["population_millions"] * 1e6 + 1e-6)
    df = df.sort_values(["country", "year"])
    df["poverty_change_yoy"] = df.groupby("country")["poverty_rate"].diff()
    df["unem_rolling3"] = (
        df.groupby("country")["unemployment_rate"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean()))
    df["social_investment"] = (
        0.6 * df["education_spending_pct_gdp"]
        + 0.4 * df["human_development_index"] * 10)
    df["social_investment_ratio"] = df["social_investment"] / (
        df["gdp_billion_usd"].clip(lower=1) ** 0.25)
    df["tax_edu_ratio"] = df["tax_revenue_pct_gdp"] / (
        df["education_spending_pct_gdp"] + 1e-6)
    return df


def scale_features(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        return X_train_scaled, scaler.transform(X_test), scaler
    return X_train_scaled, scaler


def prepare_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(clean_data(df))


def filter_by_year_range(df: pd.DataFrame, year_range: Tuple[int, int]) -> pd.DataFrame:
    """Filter dataset to [year_range[0], year_range[1]] inclusive."""
    lo, hi = int(year_range[0]), int(year_range[1])
    if lo > hi:
        raise ValueError(f"year_range start ({lo}) must be <= end ({hi})")
    return df[(df["year"] >= lo) & (df["year"] <= hi)].copy().reset_index(drop=True)


def outlier_report(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Return per-column outlier counts using z-score threshold."""
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["year"], errors="ignore")
    rows = []
    for col in numeric.columns:
        z = np.abs((numeric[col] - numeric[col].mean()) / (numeric[col].std() + 1e-9))
        n_out = int((z > z_threshold).sum())
        rows.append({
            "feature":     col,
            "outliers":    n_out,
            "pct_outlier": round(n_out / len(df) * 100, 2),
            "mean":        round(float(numeric[col].mean()), 4),
            "std":         round(float(numeric[col].std()), 4),
        })
    return pd.DataFrame(rows).sort_values("outliers", ascending=False).reset_index(drop=True)
# stats update
# Handle NAs
# Engineered features
# NA handling
