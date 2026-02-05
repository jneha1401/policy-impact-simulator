"""
Policy scenario definitions and simulation logic.
Includes 4 named presets, custom slider support, and per-year range simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

SCENARIO_COLORS = {
    "Austerity":         "#F87171",
    "Social Investment": "#34D399",
    "Education Push":    "#60A5FA",
    "Tax Cut":           "#FBBF24",
    "Custom":            "#A78BFA",
}

PRESETS = {
    "Austerity": {
        "gdp_growth_pct": -0.5, "unemployment_rate": +1.5,
        "education_spending_pct_gdp": -1.2, "tax_revenue_pct_gdp": -2.0,
        "human_development_index": -0.01, "policy_score": -5.0,
        "description": "Government spending cuts and tax reductions. Short-term fiscal balance but can increase inequality.",
    },
    "Social Investment": {
        "gdp_growth_pct": +0.8, "unemployment_rate": -1.0,
        "education_spending_pct_gdp": +2.0, "tax_revenue_pct_gdp": +3.0,
        "human_development_index": +0.02, "policy_score": +8.0,
        "description": "Increased social spending funded by moderate tax increases targeting higher income brackets.",
    },
    "Education Push": {
        "gdp_growth_pct": +0.5, "unemployment_rate": -0.5,
        "education_spending_pct_gdp": +3.5, "tax_revenue_pct_gdp": +1.0,
        "human_development_index": +0.03, "policy_score": +6.0,
        "description": "Aggressive education investment. Long-run poverty reduction driver.",
    },
    "Tax Cut": {
        "gdp_growth_pct": +1.2, "unemployment_rate": -0.8,
        "education_spending_pct_gdp": -0.3, "tax_revenue_pct_gdp": -4.0,
        "human_development_index": 0.0, "policy_score": -2.0,
        "description": "Broad-based tax cuts to stimulate private investment. Higher short-run growth, fiscal risk.",
    },
}

BOUNDS = {
    "gdp_growth_pct":             (-15.0, 20.0),
    "unemployment_rate":          (0.5,   40.0),
    "education_spending_pct_gdp": (0.5,   15.0),
    "tax_revenue_pct_gdp":        (2.0,   60.0),
    "human_development_index":    (0.2,    1.0),
    "policy_score":               (0.0,  100.0),
}


@dataclass
class PolicyScenario:
    name: str
    deltas: Dict[str, float] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_preset(cls, preset_name: str) -> "PolicyScenario":
        data = PRESETS[preset_name].copy()
        desc = data.pop("description", "")
        return cls(name=preset_name, deltas=data, description=desc)

    @classmethod
    def custom(cls, deltas: Dict[str, float]) -> "PolicyScenario":
        return cls(name="Custom", deltas=deltas, description="User-defined policy levers.")

    def apply(self, baseline: dict) -> dict:
        result = baseline.copy()
        for feature, delta in self.deltas.items():
            if feature in result:
                result[feature] = result[feature] + delta
        return result

    def apply_clamped(self, baseline: dict) -> dict:
        result = self.apply(baseline)
        for feat, (lo, hi) in BOUNDS.items():
            if feat in result:
                result[feat] = max(lo, min(hi, result[feat]))
        return result


def validate_baseline(baseline: dict) -> Optional[str]:
    required = list(BOUNDS.keys())
    missing = [k for k in required if k not in baseline]
    if missing:
        return f"Baseline missing keys: {', '.join(missing)}"
    if baseline.get("unemployment_rate", 0) < 0:
        return "Unemployment rate cannot be negative."
    return None


def run_scenario(model, baseline: dict, scenario: PolicyScenario) -> dict:
    err = validate_baseline(baseline)
    if err:
        raise ValueError(f"Invalid baseline: {err}")
    modified       = scenario.apply_clamped(baseline)
    baseline_pred  = model.predict(baseline)
    scenario_pred  = model.predict(modified)
    delta          = round(scenario_pred - baseline_pred, 2)
    return {
        "baseline_poverty":  round(baseline_pred, 2),
        "scenario_poverty":  round(scenario_pred, 2),
        "delta":             delta,
        "pct_change":        round(delta / max(baseline_pred, 0.01) * 100, 2),
        "scenario_name":     scenario.name,
        "modified_features": modified,
        "color":             SCENARIO_COLORS.get(scenario.name, "#94A3B8"),
    }


def run_all_presets(model, baseline: dict) -> list:
    results = []
    for name in PRESETS:
        scenario = PolicyScenario.from_preset(name)
        result   = run_scenario(model, baseline, scenario)
        result["description"] = scenario.description
        results.append(result)
    return results


def run_scenario_range(model, df, country: str, scenario: PolicyScenario, year_range: tuple) -> List[dict]:
    """Run scenario for every year in year_range for a given country."""
    rows = df[(df["country"] == country) &
              (df["year"] >= year_range[0]) &
              (df["year"] <= year_range[1])].sort_values("year")
    results = []
    for _, row in rows.iterrows():
        baseline = {
            "gdp_growth_pct":             float(row["gdp_growth_pct"]),
            "unemployment_rate":          float(row["unemployment_rate"]),
            "education_spending_pct_gdp": float(row["education_spending_pct_gdp"]),
            "tax_revenue_pct_gdp":        float(row["tax_revenue_pct_gdp"]),
            "human_development_index":    float(row["human_development_index"]),
            "policy_score":               float(row["policy_score"]),
        }
        r = run_scenario(model, baseline, scenario)
        r["year"] = int(row["year"])
        results.append(r)
    return results
# bounds check
# Setup bounds
# Custom levers
# Update presets
# Bounds check
