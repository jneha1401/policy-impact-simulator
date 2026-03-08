"""
Synthetic World Bank-style socioeconomic dataset generator.
Produces realistic data for 20 countries across 23 years (2000-2022).
Includes structural shocks: 2008 financial crisis, 2020 COVID recession.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

COUNTRIES = {
    "USA": {"base_gdp": 13000, "base_pov": 12, "base_unem": 5.5, "base_edu": 5.1, "base_tax": 26, "base_hdi": 0.92},
    "GBR": {"base_gdp": 2400, "base_pov": 14, "base_unem": 5.0, "base_edu": 5.4, "base_tax": 33, "base_hdi": 0.90},
    "DEU": {"base_gdp": 2100, "base_pov": 11, "base_unem": 7.5, "base_edu": 4.7, "base_tax": 36, "base_hdi": 0.92},
    "FRA": {"base_gdp": 1700, "base_pov": 13, "base_unem": 8.5, "base_edu": 5.5, "base_tax": 44, "base_hdi": 0.90},
    "JPN": {"base_gdp": 5100, "base_pov": 15, "base_unem": 4.5, "base_edu": 3.6, "base_tax": 27, "base_hdi": 0.91},
    "CHN": {"base_gdp": 1200, "base_pov": 35, "base_unem": 4.0, "base_edu": 2.8, "base_tax": 17, "base_hdi": 0.72},
    "IND": {"base_gdp": 500,  "base_pov": 42, "base_unem": 5.5, "base_edu": 3.2, "base_tax": 10, "base_hdi": 0.58},
    "BRA": {"base_gdp": 680,  "base_pov": 28, "base_unem": 9.5, "base_edu": 4.5, "base_tax": 32, "base_hdi": 0.74},
    "ZAF": {"base_gdp": 230,  "base_pov": 38, "base_unem": 25,  "base_edu": 5.9, "base_tax": 24, "base_hdi": 0.66},
    "NGA": {"base_gdp": 100,  "base_pov": 55, "base_unem": 12,  "base_edu": 2.3, "base_tax": 7,  "base_hdi": 0.51},
    "MEX": {"base_gdp": 700,  "base_pov": 42, "base_unem": 3.5, "base_edu": 4.4, "base_tax": 13, "base_hdi": 0.76},
    "ARG": {"base_gdp": 290,  "base_pov": 32, "base_unem": 14,  "base_edu": 5.1, "base_tax": 28, "base_hdi": 0.83},
    "CAN": {"base_gdp": 900,  "base_pov": 11, "base_unem": 6.5, "base_edu": 5.2, "base_tax": 31, "base_hdi": 0.93},
    "AUS": {"base_gdp": 800,  "base_pov": 12, "base_unem": 5.5, "base_edu": 4.9, "base_tax": 29, "base_hdi": 0.94},
    "KOR": {"base_gdp": 620,  "base_pov": 14, "base_unem": 3.5, "base_edu": 4.1, "base_tax": 23, "base_hdi": 0.90},
    "IDN": {"base_gdp": 250,  "base_pov": 26, "base_unem": 8.0, "base_edu": 3.1, "base_tax": 12, "base_hdi": 0.69},
    "TUR": {"base_gdp": 280,  "base_pov": 22, "base_unem": 9.0, "base_edu": 3.5, "base_tax": 24, "base_hdi": 0.80},
    "SAU": {"base_gdp": 440,  "base_pov": 8,  "base_unem": 5.5, "base_edu": 5.8, "base_tax": 4,  "base_hdi": 0.85},
    "SWE": {"base_gdp": 350,  "base_pov": 9,  "base_unem": 6.5, "base_edu": 6.5, "base_tax": 43, "base_hdi": 0.95},
    "NLD": {"base_gdp": 520,  "base_pov": 10, "base_unem": 4.5, "base_edu": 5.4, "base_tax": 38, "base_hdi": 0.94},
}

YEARS = list(range(2000, 2023))
records = []

for country, params in COUNTRIES.items():
    gdp = params["base_gdp"]
    pop_millions = np.random.uniform(5, 1400)

    for i, year in enumerate(YEARS):
        # GDP growth with structural shocks
        gdp_growth = np.random.normal(2.5, 1.5)
        if year == 2009:
            gdp_growth -= np.random.uniform(3, 8)
        if year == 2020:
            gdp_growth -= np.random.uniform(4, 10)
        if year in [2010, 2021]:
            gdp_growth += np.random.uniform(1, 4)

        gdp = gdp * (1 + gdp_growth / 100)

        # Policy indicators evolve over time
        edu_spend = params["base_edu"] + i * 0.03 + np.random.normal(0, 0.2)
        tax_rev = params["base_tax"] + i * 0.05 + np.random.normal(0, 0.8)
        unem = params["base_unem"] + np.random.normal(0, 0.8)
        if year == 2009:
            unem += np.random.uniform(1.5, 4)
        if year == 2020:
            unem += np.random.uniform(2, 6)
        unem = max(1.5, unem)

        # Poverty responds to GDP growth, unemployment, edu spending
        pov_delta = (
            -gdp_growth * 0.4
            + unem * 0.3
            - edu_spend * 0.15
            - tax_rev * 0.05
            + np.random.normal(0, 0.8)
        )
        poverty = max(1, params["base_pov"] + pov_delta + i * (-0.3 if gdp_growth > 0 else 0.1))

        # HDI trends upward slowly
        hdi = min(0.99, params["base_hdi"] + i * 0.003 + np.random.normal(0, 0.005))

        # Policy score: composite governance index
        policy_score = (
            0.3 * (edu_spend / 8)
            + 0.3 * (1 - unem / 30)
            + 0.2 * (tax_rev / 50)
            + 0.2 * hdi
        ) * 100

        records.append({
            "country": country,
            "year": year,
            "gdp_billion_usd": round(gdp, 2),
            "gdp_growth_pct": round(gdp_growth, 2),
            "population_millions": round(pop_millions + i * np.random.uniform(0, 0.5), 2),
            "poverty_rate": round(max(1, poverty), 2),
            "unemployment_rate": round(unem, 2),
            "education_spending_pct_gdp": round(max(1.5, min(9, edu_spend)), 2),
            "tax_revenue_pct_gdp": round(max(3, min(55, tax_rev)), 2),
            "human_development_index": round(hdi, 3),
            "policy_score": round(max(0, min(100, policy_score)), 2),
        })

df = pd.DataFrame(records)
df.to_csv("data/raw/policy_data.csv", index=False)
print(f"Generated {len(df)} records for {len(COUNTRIES)} countries across {len(YEARS)} years")
print(df.head(3))
