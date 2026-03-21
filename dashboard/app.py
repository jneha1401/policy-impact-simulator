"""
Policy Impact Simulation Platform — Streamlit Dashboard
Year-range slider is fully wired: filters data, retrains model,
updates every chart, table, and policy brief across all 5 tabs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_data, get_countries, get_latest_row
from src.preprocessing import prepare_pipeline, filter_by_year_range, outlier_report
from src.simulation_model import PolicyImpactModel
from src.policy_scenarios import (
    PolicyScenario, run_scenario, run_all_presets,
    run_scenario_range, PRESETS, SCENARIO_COLORS,
)
from src.visualization import (
    poverty_trend_chart, scenario_comparison_chart, gdp_vs_poverty_scatter,
    feature_importance_chart, hdi_bar_chart, radar_scenario_chart,
    poverty_gauge, unemployment_trend_chart,
)
from src.llm_explainer import generate_explanation

st.set_page_config(page_title="Policy Impact Simulator", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"],.stApp{background-color:#070B14!important;color:#CBD5E1!important;font-family:'Space Grotesk',sans-serif!important}
.block-container{padding:1.5rem 2rem 2rem 2rem!important;max-width:100%!important}
[data-testid="stSidebar"]{background-color:#0C1220!important;border-right:1px solid #1E293B!important}
[data-testid="stSidebar"] *{color:#94A3B8!important}
[data-testid="stSidebar"] label{color:#64748B!important;font-size:0.72rem!important;text-transform:uppercase!important;letter-spacing:0.08em!important}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid #1E293B!important;gap:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#475569!important;font-family:'JetBrains Mono',monospace!important;font-size:0.72rem!important;letter-spacing:0.06em!important;text-transform:uppercase!important;padding:0.75rem 1.5rem!important;border:none!important;border-bottom:2px solid transparent!important}
.stTabs [aria-selected="true"]{color:#38BDF8!important;border-bottom:2px solid #38BDF8!important;background:transparent!important}
[data-testid="stMetric"]{background:#0C1220!important;border:1px solid #1E293B!important;border-radius:8px!important;padding:1rem!important}
[data-testid="stMetricLabel"]{color:#475569!important;font-size:0.7rem!important;text-transform:uppercase!important;letter-spacing:0.06em!important}
[data-testid="stMetricValue"]{color:#F1F5F9!important;font-size:1.5rem!important;font-weight:600!important}
.stButton>button{background:linear-gradient(135deg,#0EA5E9,#6366F1)!important;color:#fff!important;border:none!important;border-radius:6px!important;font-family:'JetBrains Mono',monospace!important;font-size:0.75rem!important;letter-spacing:0.05em!important;text-transform:uppercase!important}
.stDownloadButton>button{background:transparent!important;color:#38BDF8!important;border:1px solid #1E293B!important;border-radius:6px!important;font-size:0.72rem!important}
[data-baseweb="select"]{background:#0C1220!important;border-color:#1E293B!important}
.stTextInput input{background:#0C1220!important;border-color:#1E293B!important;color:#CBD5E1!important;font-family:'JetBrains Mono',monospace!important;font-size:0.78rem!important}
header[data-testid="stHeader"]{background:transparent!important}
footer{display:none!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#070B14}
::-webkit-scrollbar-thumb{background:#1E293B;border-radius:2px}
</style>
""", unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f"""
    <div style="padding:0 0 1.5rem 0;border-bottom:1px solid #1E293B;margin-bottom:1.5rem;">
      <div style="font-size:1.75rem;font-weight:700;color:#F1F5F9;letter-spacing:-0.5px;">{title}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#334155;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.4rem;">{subtitle}</div>
    </div>""", unsafe_allow_html=True)


def section_label(text, color="#38BDF8"):
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;font-weight:600;
                color:{color};text-transform:uppercase;letter-spacing:0.14em;
                margin:1.4rem 0 0.6rem 0;padding-bottom:0.4rem;border-bottom:1px solid #1E293B;">{text}</div>
    """, unsafe_allow_html=True)


def stat_card(label, value, color="#38BDF8"):
    return f"""<div style="flex:1;background:#0C1220;border:1px solid #1E293B;border-radius:8px;padding:1rem 1.2rem;text-align:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.35rem;">{label}</div>
      <div style="font-size:1.45rem;font-weight:700;color:{color};line-height:1;">{value}</div>
    </div>"""


def kv_row(label, value):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:0.4rem 0;border-bottom:1px solid #0F172A;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;">{label}</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#CBD5E1;font-weight:500;">{value}</span>
    </div>""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    return load_data()

@st.cache_resource
def get_trained_model(df_hash):
    df    = load_dataset()
    model = PolicyImpactModel()
    scores = model.train(df)
    return model, scores

df        = load_dataset()
df_proc   = prepare_pipeline(df)
countries = get_countries(df)
model, cv = get_trained_model(len(df))

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0;border-bottom:1px solid #1E293B;margin-bottom:1rem;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;">Policy Impact</div>
      <div style="font-size:1.05rem;font-weight:700;color:#F1F5F9;margin-top:0.15rem;">Simulator</div>
    </div>""", unsafe_allow_html=True)

    default_country  = "USA" if "USA" in countries else countries[0]
    selected_country = st.selectbox("Primary country", countries, index=countries.index(default_country))

    # ── YEAR RANGE SLIDER — the single source of truth ────────────────────────
    year_range = st.slider(
        "Year range", min_value=2000, max_value=2022, value=(2000, 2022),
        help="Filters all data, retrains the model, and updates every chart in all tabs.")

    if year_range != (2000, 2022):
        with st.spinner(f"Retraining on {year_range[0]}–{year_range[1]}…"):
            cv = model.retrain_on_range(df, year_range)

    api_key = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-…")
    if api_key:
        import os; os.environ["ANTHROPIC_API_KEY"] = api_key

    st.markdown(f"""
    <div style="background:#070B14;border:1px solid #1E293B;border-radius:8px;padding:0.85rem 1rem;margin:0.75rem 0;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.56rem;color:#38BDF8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.65rem;padding-bottom:0.45rem;border-bottom:1px solid #1E293B;">Model Performance</div>
      {''.join([
        f'<div style="display:flex;justify-content:space-between;padding:0.25rem 0;">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#475569;">{k}</span>'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:{c};font-weight:600;">{v}</span></div>'
        for k,v,c in [
          ("R² (CV)", f"{cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}", "#22C55E"),
          ("RMSE",    f"{cv['rmse_mean']:.3f}", "#F59E0B"),
          ("Train R²",f"{cv['train_r2']:.3f}", "#94A3B8"),
          ("Samples", f"{cv['n_samples']}", "#94A3B8"),
        ]
      ])}
    </div>""", unsafe_allow_html=True)

# ── Filter df_proc to selected year range (used by ALL tabs) ──────────────────
df_filtered = filter_by_year_range(df_proc, year_range)
yr_label    = f"{year_range[0]}–{year_range[1]}"

page_header(
    "Policy Impact Simulation Platform",
    f"20 countries · {yr_label} · Gradient Boosting · R² {cv['r2_mean']:.3f}",
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Simulate", "Compare Scenarios", "Cross-Country", "Model Diagnostics"])

# ── TAB 1 — OVERVIEW ──────────────────────────────────────────────────────────
with tab1:
    latest = get_latest_row(df_filtered, selected_country)
    pov_c  = "#F87171" if latest['poverty_rate'] > 20 else "#22C55E"
    gdp_c  = "#22C55E" if latest['gdp_growth_pct'] > 0 else "#F87171"
    cards  = "".join([
        stat_card("Poverty Rate",  f"{latest['poverty_rate']:.1f}%",             pov_c),
        stat_card("Unemployment",  f"{latest['unemployment_rate']:.1f}%",        "#F59E0B"),
        stat_card("GDP Growth",    f"{latest['gdp_growth_pct']:.1f}%",           gdp_c),
        stat_card("HDI",           f"{latest['human_development_index']:.3f}",   "#38BDF8"),
        stat_card("Policy Score",  f"{latest['policy_score']:.0f}/100",          "#A78BFA"),
    ])
    st.markdown(f'<div style="display:flex;gap:0.75rem;margin-bottom:1.5rem;">{cards}</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.plotly_chart(poverty_trend_chart(df_filtered, selected_country), use_container_width=True)
    with col_r:
        section_label(f"Latest Indicators ({yr_label})", "#38BDF8")
        for k, v in {
            "Education Spend": f"{latest['education_spending_pct_gdp']:.1f}% GDP",
            "Tax Revenue":     f"{latest['tax_revenue_pct_gdp']:.1f}% GDP",
            "GDP (USD bn)":    f"${latest['gdp_billion_usd']:,.0f}",
            "Policy Score":    f"{latest['policy_score']:.1f} / 100",
            "Reference Year":  f"{int(latest['year'])}",
        }.items():
            kv_row(k, v)
        section_label("Recent History", "#38BDF8")
        hist = (df_filtered[df_filtered["country"] == selected_country]
                .sort_values("year", ascending=False).head(5)
                [["year","poverty_rate","gdp_growth_pct","unemployment_rate"]]
                .rename(columns={"year":"Year","poverty_rate":"Poverty %","gdp_growth_pct":"GDP %","unemployment_rate":"Unem %"}))
        st.dataframe(hist.set_index("Year"), use_container_width=True, height=200)

# ── TAB 2 — SIMULATE ──────────────────────────────────────────────────────────
with tab2:
    latest = get_latest_row(df_filtered, selected_country)
    bf = {
        "gdp_growth_pct":             float(latest["gdp_growth_pct"]),
        "unemployment_rate":          float(latest["unemployment_rate"]),
        "education_spending_pct_gdp": float(latest["education_spending_pct_gdp"]),
        "tax_revenue_pct_gdp":        float(latest["tax_revenue_pct_gdp"]),
        "human_development_index":    float(latest["human_development_index"]),
        "policy_score":               float(latest["policy_score"]),
    }
    col_ctrl, col_result = st.columns([1, 2])
    with col_ctrl:
        section_label("Scenario", "#38BDF8")
        preset_choice = st.selectbox("Load preset", ["Custom"] + list(PRESETS.keys()))
        pd_vals = ({k: v for k, v in PRESETS[preset_choice].items() if k != "description"}
                   if preset_choice != "Custom" else {})
        section_label("Policy Levers", "#F59E0B")
        gdp_d  = st.slider("GDP Growth (pp)",         -8.0, 8.0,  float(pd_vals.get("gdp_growth_pct", 0.0)), 0.1)
        unem_d = st.slider("Unemployment (pp)",        -8.0, 8.0,  float(pd_vals.get("unemployment_rate", 0.0)), 0.1)
        edu_d  = st.slider("Education Spend (% GDP)", -3.0, 5.0,  float(pd_vals.get("education_spending_pct_gdp", 0.0)), 0.1)
        tax_d  = st.slider("Tax Revenue (% GDP)",     -8.0, 10.0, float(pd_vals.get("tax_revenue_pct_gdp", 0.0)), 0.1)
        hdi_d  = st.slider("HDI adjustment",          -0.05, 0.05, float(pd_vals.get("human_development_index", 0.0)), 0.005)

    with col_result:
        custom_d = {"gdp_growth_pct": gdp_d, "unemployment_rate": unem_d,
                    "education_spending_pct_gdp": edu_d, "tax_revenue_pct_gdp": tax_d,
                    "human_development_index": hdi_d, "policy_score": 0.0}
        scenario = PolicyScenario.custom(custom_d)
        result   = run_scenario(model, bf, scenario)
        result["scenario_name"] = preset_choice

        st.plotly_chart(poverty_gauge(result["scenario_poverty"], result["baseline_poverty"]), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Baseline Poverty",  f"{result['baseline_poverty']:.2f}%")
        with c2: st.metric("Projected Poverty", f"{result['scenario_poverty']:.2f}%",
                            delta=f"{result['delta']:+.2f}pp", delta_color="inverse")
        with c3: st.metric("Relative Change",   f"{result['pct_change']:+.1f}%")

        section_label(f"Poverty Trend — {yr_label}", "#38BDF8")
        range_results = run_scenario_range(model, df_filtered, selected_country, scenario, year_range)
        if range_results:
            trend_df = pd.DataFrame([
                {"Year": r["year"], "Baseline": r["baseline_poverty"], "Scenario": r["scenario_poverty"]}
                for r in range_results
            ]).set_index("Year")
            st.line_chart(trend_df, color=["#475569", "#38BDF8"])

        section_label("Policy Brief", "#38BDF8")
        st.markdown(generate_explanation(result, selected_country, bf, year_range), unsafe_allow_html=True)

# ── TAB 3 — COMPARE SCENARIOS ─────────────────────────────────────────────────
with tab3:
    latest = get_latest_row(df_filtered, selected_country)
    bf3 = {k: float(latest[k]) for k in [
        "gdp_growth_pct","unemployment_rate","education_spending_pct_gdp",
        "tax_revenue_pct_gdp","human_development_index","policy_score"]}
    all_results = run_all_presets(model, bf3)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(scenario_comparison_chart(all_results), use_container_width=True)
    with c2: st.plotly_chart(radar_scenario_chart(all_results),       use_container_width=True)
    section_label(f"Scenario Summary — {yr_label}", "#38BDF8")
    rows = [{"Scenario": r["scenario_name"], "Baseline (%)": f"{r['baseline_poverty']:.2f}",
             "Projected (%)": f"{r['scenario_poverty']:.2f}", "Change (pp)": f"{r['delta']:+.2f}",
             "Change (%)": f"{r['pct_change']:+.1f}%"} for r in all_results]
    st.dataframe(pd.DataFrame(rows).set_index("Scenario"), use_container_width=True)
    section_label("Policy Briefs — All Scenarios", "#A78BFA")
    cols4 = st.columns(2)
    for i, r in enumerate(all_results):
        with cols4[i % 2]:
            st.markdown(generate_explanation(r, selected_country, bf3, year_range), unsafe_allow_html=True)

# ── TAB 4 — CROSS-COUNTRY ─────────────────────────────────────────────────────
with tab4:
    c_ctrl, c_main = st.columns([1, 3])
    with c_ctrl:
        available_years = sorted(df_filtered["year"].unique(), reverse=True)
        sel_year  = st.selectbox("Reference year", available_years, index=0)
        def_cmp   = [c for c in ["USA","GBR","DEU","BRA","IND"] if c in countries]
        cmp_cntrs = st.multiselect("Select countries", countries, default=def_cmp)
    with c_main:
        if cmp_cntrs:
            st.plotly_chart(gdp_vs_poverty_scatter(df_filtered, sel_year), use_container_width=True)
    if cmp_cntrs:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(unemployment_trend_chart(df_filtered, cmp_cntrs), use_container_width=True)
        with c2: st.plotly_chart(hdi_bar_chart(df_filtered, sel_year), use_container_width=True)
        section_label(f"Country Comparison — {sel_year} ({yr_label})", "#38BDF8")
        cmp_df = (df_filtered[(df_filtered["country"].isin(cmp_cntrs)) & (df_filtered["year"] == sel_year)]
                  [["country","poverty_rate","unemployment_rate","gdp_growth_pct",
                    "education_spending_pct_gdp","tax_revenue_pct_gdp","human_development_index"]]
                  .rename(columns={"country":"Country","poverty_rate":"Poverty %",
                    "unemployment_rate":"Unem %","gdp_growth_pct":"GDP Growth %",
                    "education_spending_pct_gdp":"Edu Spend %","tax_revenue_pct_gdp":"Tax Rev %",
                    "human_development_index":"HDI"}).set_index("Country"))
        st.dataframe(cmp_df.round(2), use_container_width=True)
        st.download_button("Download CSV", cmp_df.to_csv().encode("utf-8"), f"comparison_{sel_year}.csv", "text/csv")

# ── TAB 5 — MODEL DIAGNOSTICS ─────────────────────────────────────────────────
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        section_label(f"Cross-Validation Metrics ({yr_label})", "#38BDF8")
        for k, v in {
            "R² mean ± std":   f"{cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}",
            "RMSE mean ± std": f"{cv['rmse_mean']:.4f} ± {cv['rmse_std']:.4f}",
            "Train R²":        f"{cv['train_r2']:.4f}",
            "Samples (n)":     f"{cv['n_samples']}",
            "Algorithm":       "Gradient Boosting · 5-fold CV",
        }.items():
            kv_row(k, v)
        section_label("Data Quality — Outlier Report", "#F59E0B")
        st.dataframe(outlier_report(df_filtered), use_container_width=True)
    with c2:
        imp_df = model.feature_importance()
        st.plotly_chart(feature_importance_chart(imp_df), use_container_width=True)
    section_label(f"Dataset Preview — {selected_country} ({yr_label})", "#A78BFA")
    prev = (df_filtered[df_filtered["country"] == selected_country]
            .sort_values("year", ascending=False).head(10).reset_index(drop=True))
    st.dataframe(prev, use_container_width=True)
    st.download_button("Download Filtered Dataset",
                       df_filtered.to_csv(index=False).encode("utf-8"),
                       f"policy_data_{year_range[0]}_{year_range[1]}.csv", "text/csv")
# layout fix
# tooltips
# responsive view
# final cleanup
# layout updates
# Sidebar wire
# Tab 1 stats
# Tab 2 wiring
# Tab 3 sync
# Tab 4 filters
# Tab 5 diagnostics
# Tooltips added
# Sidebar wire
# Tab 1 stats
# Tab 2 wiring
# Tab 3 sync
# Tab 4 filters
# Tab 5 diagnostics
# Tooltips added
# Sidebar wire
# Tab 1 stats
# Tab 2 wiring
# Tab 3 sync
# Tab 4 filters
# Tab 5 diagnostics
