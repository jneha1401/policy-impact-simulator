"""
Policy scenario definitions and simulation logic.
Includes 4 named presets and custom slider support.
"""

from dataclasses import dataclass, field
from typing import Dict

PRESETS = {
    "Austerity": {
        "gdp_growth_pct": -0.5,
        "unemployment_rate": +1.5,
        "education_spending_pct_gdp": -1.2,
        "tax_revenue_pct_gdp": -2.0,
        "human_development_index": -0.01,
        "policy_score": -5.0,
        "description": "Government spending cuts, reduced public services, tax reductions. Often leads to short-term fiscal balance but can increase inequality.",
    },
    "Social Investment": {
        "gdp_growth_pct": +0.8,
        "unemployment_rate": -1.0,
        "education_spending_pct_gdp": +2.0,
        "tax_revenue_pct_gdp": +3.0,
        "human_development_index": +0.02,
        "policy_score": +8.0,
        "description": "Increase social spending, healthcare, and education. Funded by moderate tax increases targeting higher income brackets.",
    },
    "Education Push": {
        "gdp_growth_pct": +0.5,
        "unemployment_rate": -0.5,
        "education_spending_pct_gdp": +3.5,
        "tax_revenue_pct_gdp": +1.0,
        "human_development_index": +0.03,
        "policy_score": +6.0,
        "description": "Aggressive investment in education infrastructure, teacher pay, and tertiary enrollment. Long-run poverty reduction driver.",
    },
    "Tax Cut": {
        "gdp_growth_pct": +1.2,
        "unemployment_rate": -0.8,
        "education_spending_pct_gdp": -0.3,
        "tax_revenue_pct_gdp": -4.0,
        "human_development_index": 0.0,
        "policy_score": -2.0,
        "description": "Broad-based tax cuts to stimulate private investment and consumption. Higher short-run growth, but fiscal risk and potential service reduction.",
    },
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
        """Apply scenario deltas to a baseline feature dict."""
        result = baseline.copy()
        for feature, delta in self.deltas.items():
            if feature in result:
                result[feature] = result[feature] + delta
        return result


def run_scenario(model, baseline: dict, scenario: PolicyScenario) -> dict:
    """Run a single scenario and return prediction + delta."""
    modified = scenario.apply(baseline)
    baseline_pred = model.predict(baseline)
    scenario_pred = model.predict(modified)
    return {
        "baseline_poverty": round(baseline_pred, 2),
        "scenario_poverty": round(scenario_pred, 2),
        "delta": round(scenario_pred - baseline_pred, 2),
        "pct_change": round((scenario_pred - baseline_pred) / max(baseline_pred, 0.01) * 100, 2),
        "scenario_name": scenario.name,
        "modified_features": modified,
    }


def run_all_presets(model, baseline: dict) -> list:
    """Run all 4 presets and return list of result dicts."""
    results = []
    for name in PRESETS:
        scenario = PolicyScenario.from_preset(name)
        result = run_scenario(model, baseline, scenario)
        result["description"] = scenario.description
        results.append(result)
    return results
