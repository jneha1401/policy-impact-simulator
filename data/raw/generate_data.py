"""
Synthetic dataset generator for the Policy Impact Simulation Platform.
Generates 20 countries x 23 years = 460 records of socioeconomic indicators.
Based on World Bank methodology.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

COUNTRIES = [
    "USA", "GBR", "DEU", "FRA", "JPN",
    "BRA", "IND", "CHN", "ZAF", "MEX",
    "NGA", "KEN", "EGY", "IDN", "TUR",
    "ARG", "COL", "POL", "THA", "PHL",
]

YEARS = list(range(2000, 2023))

COUNTRY_PROFILES = {
    "USA": {"base_poverty": 12, "base_gdp": 2.5,  "base_hdi": 0.92},
    "GBR": {"base_poverty": 10, "base_gdp": 2.0,  "base_hdi": 0.93},
    "DEU": {"base_poverty":  8, "base_gdp": 1.8,  "base_hdi": 0.94},
    "FRA": {"base_poverty": 11, "base_gdp": 1.7,  "base_hdi": 0.91},
    "JPN": {"base_poverty": 16, "base_gdp": 1.2,  "base_hdi": 0.92},
    "BRA": {"base_poverty": 25, "base_gdp": 3.5,  "base_hdi": 0.76},
    "IND": {"base_poverty": 35, "base_gdp": 7.0,  "base_hdi": 0.64},
    "CHN": {"base_poverty": 20, "base_gdp": 8.0,  "base_hdi": 0.74},
    "ZAF": {"base_poverty": 55, "base_gdp": 2.0,  "base_hdi": 0.71},
    "MEX": {"base_poverty": 42, "base_gdp": 3.0,  "base_hdi": 0.77},
    "NGA": {"base_poverty": 60, "base_gdp": 5.0,  "base_hdi": 0.54},
    "KEN": {"base_poverty": 50, "base_gdp": 5.5,  "base_hdi": 0.59},
    "EGY": {"base_poverty": 30, "base_gdp": 4.5,  "base_hdi": 0.71},
    "IDN": {"base_poverty": 28, "base_gdp": 5.5,  "base_hdi": 0.71},
    "TUR": {"base_poverty": 22, "base_gdp": 4.0,  "base_hdi": 0.82},
    "ARG": {"base_poverty": 33, "base_gdp": 2.5,  "base_hdi": 0.84},
    "COL": {"base_poverty": 38, "base_gdp": 4.0,  "base_hdi": 0.75},
    "POL": {"base_poverty": 18, "base_gdp": 3.5,  "base_hdi": 0.88},
    "THA": {"base_poverty": 22, "base_gdp": 4.5,  "base_hdi": 0.80},
    "PHL": {"base_poverty": 32, "base_gdp": 6.0,  "base_hdi": 0.70},
}

rows = []
for country in COUNTRIES:
    profile = COUNTRY_PROFILES[country]
    poverty = profile["base_poverty"]
    gdp_b   = np.random.uniform(50, 20000)

    for year in YEARS:
        gdp_growth   = profile["base_gdp"] + np.random.normal(0, 1.5)
        unemployment = max(1.0, 6.0 + np.random.normal(0, 2.0))
        edu_spend    = max(1.0, 4.5 + np.random.normal(0, 1.0))
        tax_rev      = max(5.0, 25.0 + np.random.normal(0, 5.0))
        hdi          = min(0.999, max(0.3, profile["base_hdi"] + np.random.normal(0, 0.01)))
        policy_score = max(0, min(100, 50 + np.random.normal(0, 15)))
        population   = np.random.uniform(5, 1400)
        gdp_b       *= (1 + gdp_growth / 100)

        poverty_change = (
            -0.4 * gdp_growth
            + 0.3 * unemployment
            - 0.2 * edu_spend
            - 0.1 * (tax_rev / 10)
            - 0.5 * hdi * 10
            - 0.05 * policy_score
            + np.random.normal(0, 0.8)
        )
        poverty = max(1.0, min(80.0, poverty + poverty_change * 0.3))

        rows.append({
            "country":                    country,
            "year":                       year,
            "poverty_rate":               round(poverty, 2),
            "gdp_growth_pct":             round(gdp_growth, 2),
            "unemployment_rate":          round(unemployment, 2),
            "education_spending_pct_gdp": round(edu_spend, 2),
            "tax_revenue_pct_gdp":        round(tax_rev, 2),
            "human_development_index":    round(hdi, 3),
            "policy_score":               round(policy_score, 1),
            "gdp_billion_usd":            round(gdp_b, 2),
            "population_millions":        round(population, 2),
        })

df = pd.DataFrame(rows)
out = Path(__file__).parent / "policy_data.csv"
df.to_csv(out, index=False)
print(f"Generated {len(df)} records → {out}")
