# Policy Impact Simulation Platform

A Streamlit dashboard for simulating the socioeconomic impact of government policies across 20 countries and 23 years (2000–2022).

## Features
- Gradient Boosting model with 5-fold cross-validation (R² ≈ 0.976)
- Year-range slider to filter data and retrain the model dynamically
- 4 named policy presets + custom lever controls
- Scenario comparison with radar and bar charts
- LLM-generated policy briefs via Claude API
- Cross-country analysis with GDP vs poverty scatter plots
- Model diagnostics with feature importance and outlier reports

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Project Structure
```
policy-impact-simulator/
├── dashboard/app.py          # Streamlit dashboard
├── src/
│   ├── data_loader.py        # Data loading utilities
│   ├── preprocessing.py      # Cleaning, feature engineering, year-range filter
│   ├── simulation_model.py   # Gradient Boosting model
│   ├── policy_scenarios.py   # Scenario definitions and simulation
│   ├── visualization.py      # Plotly chart functions
│   └── llm_explainer.py      # Claude API integration
├── data/raw/                 # Synthetic policy dataset
└── configs/model_config.yaml # Model hyperparameters
```

## Dataset
Synthetic dataset based on World Bank methodology covering 20 countries across 23 years with 11 socioeconomic indicators including poverty rate, GDP growth, unemployment, education spending, tax revenue, and HDI.
# Final docs
# Ready for deployment
# Final deployment prep
