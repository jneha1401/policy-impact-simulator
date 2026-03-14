"""
LLM-based policy explanation generator.
Uses the Anthropic Claude API. Falls back to rule-based HTML if no API key.
year_range is threaded through all code paths so briefs surface the analysis period.
"""

import os
from typing import Optional, Tuple


def _build_prompt(result, country, baseline, year_range=None):
    yr_str = f"{year_range[0]}–{year_range[1]}" if year_range else "full dataset period"
    return f"""You are an expert development economist writing a concise policy brief.

Country: {country}
Analysis period: {yr_str}
Scenario: {result['scenario_name']}
Baseline poverty rate: {result['baseline_poverty']:.1f}%
Projected poverty rate: {result['scenario_poverty']:.1f}%
Change: {result['delta']:+.2f} percentage points ({result['pct_change']:+.1f}%)

Policy changes applied:
- GDP growth: {result['modified_features'].get('gdp_growth_pct', 0):.1f}%
- Unemployment: {result['modified_features'].get('unemployment_rate', 0):.1f}pp
- Education spending: {result['modified_features'].get('education_spending_pct_gdp', 0):.1f}% GDP
- Tax revenue: {result['modified_features'].get('tax_revenue_pct_gdp', 0):.1f}% GDP

Write 3 short paragraphs (150–200 words total):
1. Predicted outcome for the selected period
2. Economic transmission mechanisms
3. Risks, caveats, and implementation considerations

No bullet points. No markdown headers. Plain prose only."""


def _rule_based_explanation(result, country, year_range=None):
    delta     = result["delta"]
    scenario  = result["scenario_name"]
    pred      = result["scenario_poverty"]
    base      = result["baseline_poverty"]
    pct       = result["pct_change"]
    is_good   = delta < 0
    direction = "reduce" if is_good else "increase"
    magnitude = "significantly" if abs(delta) > 2 else "modestly"
    yr_str    = f" over {year_range[0]}–{year_range[1]}" if year_range else " across the full dataset period"
    uncertainty = max(0.1, abs(delta) * 0.15)

    edu  = result["modified_features"].get("education_spending_pct_gdp", 0)
    tax  = result["modified_features"].get("tax_revenue_pct_gdp", 0)
    unem = result["modified_features"].get("unemployment_rate", 0)
    gdp  = result["modified_features"].get("gdp_growth_pct", 0)

    mechs = []
    if edu > 4:   mechs.append("elevated education investment building long-run human capital")
    elif edu < 3: mechs.append("reduced education spending constraining workforce productivity")
    if tax > 30:  mechs.append("higher tax revenue enabling redistributive public expenditure")
    elif tax < 15:mechs.append("lower tax revenue limiting the scope for poverty-reducing programmes")
    if unem < 5:  mechs.append("tight labour markets driving wage growth at lower income brackets")
    elif unem > 9:mechs.append("elevated unemployment concentrating income shocks among vulnerable households")
    if gdp > 2:   mechs.append("robust GDP growth generating broad-based employment opportunities")
    mech_text = (". ".join(m.capitalize() for m in mechs[:3]) + "." if mechs
                 else "Changes in fiscal policy, labour conditions, and social investment interact to produce this outcome.")

    rec = ("Favourable trajectory. Monitor distributional effects and ensure fiscal sustainability."
           if is_good else
           "Poverty-increasing risk. Compensatory transfers should be considered to shield vulnerable households.")

    return f"""
<div style="background:#0C1220;border:1px solid #1E293B;border-radius:10px;overflow:hidden;margin-bottom:0.5rem;">
  <div style="background:#0F172A;padding:0.9rem 1.2rem;border-bottom:1px solid #1E293B;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;">Policy Brief</div>
    <div style="font-size:0.92rem;font-weight:600;color:#F1F5F9;margin-top:0.15rem;">{scenario} — {country}{(' · ' + str(year_range[0]) + '–' + str(year_range[1])) if year_range else ''}</div>
  </div>
  <div style="padding:1rem 1.2rem;display:flex;flex-direction:column;gap:0.85rem;">
    <div style="border-left:3px solid #22C55E;padding-left:0.85rem;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.56rem;font-weight:600;color:#22C55E;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.25rem;">Predicted Outcome</div>
      <div style="font-size:0.82rem;color:#CBD5E1;line-height:1.7;">Under the <strong style="color:#F1F5F9;">{scenario}</strong> scenario{yr_str}, poverty is projected to {direction} {magnitude} from <strong style="color:#94A3B8;">{base:.1f}%</strong> to <strong style="color:{'#22C55E' if is_good else '#F87171'};">{pred:.1f}%</strong> ({pct:+.1f}%).</div>
    </div>
    <div style="border-left:3px solid #F59E0B;padding-left:0.85rem;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.56rem;font-weight:600;color:#F59E0B;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.25rem;">Economic Mechanism</div>
      <div style="font-size:0.82rem;color:#CBD5E1;line-height:1.7;">{mech_text} Transmission lags: 2–4 years for labour markets, 5–10 years for human capital channels.</div>
    </div>
    <div style="border-left:3px solid #A78BFA;padding-left:0.85rem;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.56rem;font-weight:600;color:#A78BFA;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.25rem;">Recommendation</div>
      <div style="font-size:0.82rem;color:#CBD5E1;line-height:1.7;">{rec} Model uncertainty ±{uncertainty:.1f}pp should be factored into planning horizons.</div>
    </div>
  </div>
  <div style="background:#070B14;padding:0.45rem 1.2rem;border-top:1px solid #1E293B;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;color:#1E293B;">Set ANTHROPIC_API_KEY in sidebar for Claude-generated briefs</span>
  </div>
</div>"""


def generate_explanation(result, country, baseline, year_range=None):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _rule_based_explanation(result, country, year_range)
    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=api_key)
        prompt   = _build_prompt(result, country, baseline, year_range)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=400,
            messages=[{"role": "user", "content": prompt}])
        raw        = response.content[0].text.strip()
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        labels     = [("Predicted Outcome","#22C55E"),("Economic Mechanism","#F59E0B"),("Recommendation","#A78BFA")]
        yr_str     = f"{year_range[0]}–{year_range[1]}" if year_range else "full period"
        html = f'<div style="background:#0C1220;border:1px solid #1E293B;border-radius:10px;overflow:hidden;padding:1rem 1.2rem;">'
        html += f'<div style="font-size:0.92rem;font-weight:600;color:#F1F5F9;margin-bottom:0.75rem;">{result["scenario_name"]} — {country} · {yr_str}</div>'
        for i, para in enumerate(paragraphs[:3]):
            lbl, clr = labels[i] if i < len(labels) else ("Analysis","#38BDF8")
            html += f'<div style="border-left:3px solid {clr};padding-left:0.85rem;margin-bottom:0.75rem;">'
            html += f'<div style="font-family:JetBrains Mono,monospace;font-size:0.56rem;color:{clr};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.2rem;">{lbl}</div>'
            html += f'<div style="font-size:0.82rem;color:#CBD5E1;line-height:1.7;">{para}</div></div>'
        html += '</div>'
        return html
    except Exception as e:
        return _rule_based_explanation(result, country, year_range) + f'\n<div style="color:#F87171;font-size:0.7rem;padding:0.5rem;">API error: {e}</div>'
# error handling
# prompt engineering

def call_api_with_retry(prompt, max_retries=3):
    """Wrapper to handle intermittent API failures gracefully."""
    import time
    for attempt in range(max_retries):
        try:
            # Simulate API call execution
            return generate_explanation(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                return "API Error: Unable to generate brief at this time."
            time.sleep(2 ** attempt) # Exponential backoff
# Prompt tuning
# Fallback HTML
# API retry
