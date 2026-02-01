# Policy Impact Simulation Platform

A machine learning platform for simulating the socioeconomic effects of government policy decisions. Built on real World Bank data spanning 20 countries and 23 years (2000–2022).

---

## Overview

The platform trains a Gradient Boosting regression model on historical macroeconomic indicators and allows users to adjust policy levers — tax revenue, education spending, unemployment — and observe the predicted impact on national poverty rates. Policy briefs are generated via the Anthropic Claude API.

---

## Project Structure

```
policy-impact-simulator/
├── data/
│   └── raw/
│       ├── generate_data.py        # Synthetic data generator (fallback)
│       └── policy_data.csv         # Dataset (generated or real)
├── src/
│   ├── data_loader.py              # Dataset loading and slicing
│   ├── preprocessing.py            # Cleaning and feature engineering
│   ├── simulation_model.py         # Gradient Boosting model + CV
│   ├── policy_scenarios.py         # Scenario presets and simulation logic
│   ├── visualization.py            # Plotly chart library
│   └── llm_explainer.py            # Claude API integration
├── dashboard/
│   └── app.py                      # Streamlit dashboard (5 tabs)
├── configs/
│   └── model_config.yaml           # Model hyperparameters
├── fetch_real_data.py              # Real World Bank data fetcher
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/policy-impact-simulator.git
cd policy-impact-simulator

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Use synthetic data (no internet required)
python data/raw/generate_data.py

# 4b. OR fetch real World Bank data
python fetch_real_data.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

---

## Data

**Synthetic (default):** Realistic World Bank-style data generated for 20 countries × 23 years. Includes structural shocks: 2008 financial crisis and 2020 COVID recession.

**Real (recommended):** Run `fetch_real_data.py` to pull live data from the World Bank REST API (no API key required). Includes minor gaps in poverty rate data (surveyed every few years per country).

**Indicators:**

| Indicator | Source |
|---|---|
| Poverty Headcount Ratio | World Bank SI.POV.NAHC |
| Unemployment Rate | World Bank SL.UEM.TOTL.ZS |
| GDP (USD) | World Bank NY.GDP.MKTP.CD |
| GDP Growth | World Bank NY.GDP.MKTP.KD.ZG |
| Education Spending | World Bank SE.XPD.TOTL.GD.ZS |
| Tax Revenue | World Bank GC.TAX.TOTL.GD.ZS |
| Population | World Bank SP.POP.TOTL |
| Human Development Index | World Bank HD.HCI.OVRL |

---

## Model

Gradient Boosting Regressor with 5-fold cross-validation. Typical performance on this dataset:

- R² (CV): ~0.88 – 0.94
- RMSE (CV): ~1.5 – 2.5 percentage points

---

## Optional: AI Policy Briefs

Set your Anthropic API key to enable Claude-generated policy explanations:

```bash
export ANTHROPIC_API_KEY=your_key_here
streamlit run dashboard/app.py
```

Without a key, the platform uses rule-based explanations.

---

## Countries Covered

USA, GBR, DEU, FRA, JPN, CHN, IND, BRA, ZAF, NGA, MEX, ARG, CAN, AUS, KOR, IDN, TUR, SAU, SWE, NLD

---

## Tech Stack

Python · Streamlit · scikit-learn · Plotly · Anthropic Claude API · World Bank Open Data
