"""
Visualization utilities for the Policy Impact Simulation Platform.
All charts use Plotly with a consistent dark professional theme.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Design tokens ──────────────────────────────────────────────────────────────
THEME = {
    "bg": "#0D1117",
    "surface": "#161B22",
    "border": "#30363D",
    "text": "#E6EDF3",
    "subtext": "#8B949E",
    "accent": "#58A6FF",
    "accent2": "#3FB950",
    "accent3": "#F78166",
    "accent4": "#D2A8FF",
    "warn": "#E3B341",
}

PALETTE = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"], THEME["warn"]]

LAYOUT_BASE = dict(
    paper_bgcolor=THEME["bg"],
    plot_bgcolor=THEME["surface"],
    font=dict(family="IBM Plex Mono, monospace", color=THEME["text"], size=12),
    margin=dict(l=40, r=30, t=50, b=40),
    legend=dict(bgcolor=THEME["surface"], bordercolor=THEME["border"], borderwidth=1),
)


def _apply_base(fig) -> go.Figure:
    fig.update_layout(**LAYOUT_BASE)
    fig.update_xaxes(
        gridcolor=THEME["border"], linecolor=THEME["border"],
        zerolinecolor=THEME["border"],
    )
    fig.update_yaxes(
        gridcolor=THEME["border"], linecolor=THEME["border"],
        zerolinecolor=THEME["border"],
    )
    return fig


# ── 1. Historical poverty trend ────────────────────────────────────────────────
def poverty_trend_chart(df: pd.DataFrame, country: str) -> go.Figure:
    cdf = df[df["country"] == country].sort_values("year")
    fig = go.Figure()

    # Shaded fill under the line
    fig.add_trace(go.Scatter(
        x=cdf["year"], y=cdf["poverty_rate"],
        mode="lines+markers",
        name="Poverty Rate",
        line=dict(color=THEME["accent"], width=2.5),
        marker=dict(size=5, color=THEME["accent"]),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.10)",
    ))

    # Annotations for crisis years
    for year, label, color in [(2009, "2008 Crisis", THEME["accent3"]), (2020, "COVID-19", THEME["warn"])]:
        row = cdf[cdf["year"] == year]
        if not row.empty:
            fig.add_vline(x=year, line_dash="dash", line_color=color, opacity=0.6)
            fig.add_annotation(x=year, y=row["poverty_rate"].values[0],
                               text=label, showarrow=True, arrowhead=2,
                               font=dict(color=color, size=10), ax=30, ay=-25)

    fig.update_layout(
        title=dict(text=f"Poverty Rate Trend — {country}", font=dict(size=14)),
        xaxis_title="Year", yaxis_title="Poverty Rate (%)",
        **LAYOUT_BASE,
    )
    return _apply_base(fig)


# ── 2. Scenario comparison bar chart ──────────────────────────────────────────
def scenario_comparison_chart(results: list) -> go.Figure:
    names = [r["scenario_name"] for r in results]
    deltas = [r["delta"] for r in results]
    colors = [THEME["accent2"] if d <= 0 else THEME["accent3"] for d in deltas]

    fig = go.Figure(go.Bar(
        x=names, y=deltas,
        marker_color=colors,
        text=[f"{d:+.2f}pp" for d in deltas],
        textposition="outside",
        textfont=dict(size=11, color=THEME["text"]),
    ))
    fig.update_layout(
        title="Poverty Rate Change by Policy Scenario (percentage points)",
        xaxis_title="Scenario", yaxis_title="Change in Poverty Rate (pp)",
        **LAYOUT_BASE,
    )
    return _apply_base(fig)


# ── 3. GDP vs Poverty scatter ──────────────────────────────────────────────────
def gdp_vs_poverty_scatter(df: pd.DataFrame, year: int) -> go.Figure:
    ydf = df[df["year"] == year].dropna(subset=["gdp_billion_usd", "poverty_rate"])
    fig = px.scatter(
        ydf, x="gdp_billion_usd", y="poverty_rate",
        size="population_millions", color="country",
        hover_name="country",
        hover_data={"unemployment_rate": True, "education_spending_pct_gdp": True},
        size_max=55,
        color_discrete_sequence=px.colors.qualitative.Safe,
        trendline="ols",
    )
    fig.update_layout(
        title=f"GDP vs Poverty Rate — {year}  (bubble = population)",
        xaxis_title="GDP (USD billion)", yaxis_title="Poverty Rate (%)",
        **LAYOUT_BASE,
    )
    return _apply_base(fig)


# ── 4. Feature importance lollipop ────────────────────────────────────────────
def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    df = importance_df.sort_values("importance")
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_shape(type="line",
                      x0=0, x1=row["importance"], y0=row["feature"], y1=row["feature"],
                      line=dict(color=THEME["border"], width=2))
    fig.add_trace(go.Scatter(
        x=df["importance"], y=df["feature"],
        mode="markers+text",
        marker=dict(size=10, color=THEME["accent"], line=dict(color=THEME["bg"], width=1.5)),
        text=[f"{v:.3f}" for v in df["importance"]],
        textposition="middle right",
        textfont=dict(size=10),
        name="Importance",
    ))
    fig.update_layout(
        title="Model Feature Importances",
        xaxis_title="Importance Score", yaxis_title="",
        height=320,
        **LAYOUT_BASE,
    )
    return _apply_base(fig)


# ── 5. HDI horizontal bar ──────────────────────────────────────────────────────
def hdi_bar_chart(df: pd.DataFrame, year: int) -> go.Figure:
    ydf = df[df["year"] == year].dropna(subset=["human_development_index"])
    ydf = ydf.sort_values("human_development_index", ascending=True)

    def hdi_color(v):
        if v >= 0.80:
            return THEME["accent2"]
        elif v >= 0.60:
            return THEME["warn"]
        return THEME["accent3"]

    colors = [hdi_color(v) for v in ydf["human_development_index"]]
    fig = go.Figure(go.Bar(
        x=ydf["human_development_index"], y=ydf["country"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in ydf["human_development_index"]],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title=f"Human Development Index — {year}",
        xaxis_title="HDI", yaxis_title="",
        height=500,
        **LAYOUT_BASE,
    )
    return _apply_base(fig)


# ── 6. Radar / spider chart for scenario comparison ───────────────────────────
def radar_scenario_chart(results: list) -> go.Figure:
    dims = ["unemployment_rate", "education_spending_pct_gdp",
            "tax_revenue_pct_gdp", "human_development_index", "gdp_growth_pct"]
    labels = ["Unemployment", "Education Spend", "Tax Revenue", "HDI", "GDP Growth"]

    fig = go.Figure()
    for i, res in enumerate(results):
        mf = res.get("modified_features", {})
        values = [mf.get(d, 0) for d in dims]
        # normalise 0-1 for radar
        normed = [(v - 0) / (max(abs(v) + 1, 1)) for v in values]
        normed.append(normed[0])
        fig.add_trace(go.Scatterpolar(
            r=normed, theta=labels + [labels[0]],
            fill="toself", name=res["scenario_name"],
            line=dict(color=PALETTE[i % len(PALETTE)]),
            fillcolor=PALETTE[i % len(PALETTE)].replace("FF", "33"),
            opacity=0.8,
        ))
    fig.update_layout(
        title="Policy Scenario Radar Comparison",
        polar=dict(bgcolor=THEME["surface"],
                   radialaxis=dict(visible=True, color=THEME["subtext"]),
                   angularaxis=dict(color=THEME["text"])),
        **LAYOUT_BASE,
    )
    return fig


# ── 7. Poverty gauge ──────────────────────────────────────────────────────────
def poverty_gauge(predicted: float, baseline: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted,
        delta=dict(reference=baseline, decreasing=dict(color=THEME["accent2"]),
                   increasing=dict(color=THEME["accent3"])),
        gauge=dict(
            axis=dict(range=[0, 60], tickcolor=THEME["subtext"]),
            bar=dict(color=THEME["accent"]),
            bgcolor=THEME["surface"],
            bordercolor=THEME["border"],
            steps=[
                dict(range=[0, 15], color="#1C2A1C"),
                dict(range=[15, 35], color="#2A2A1C"),
                dict(range=[35, 60], color="#2A1C1C"),
            ],
            threshold=dict(line=dict(color=THEME["warn"], width=3), thickness=0.8, value=baseline),
        ),
        number=dict(suffix="%", font=dict(size=28, color=THEME["text"])),
        title=dict(text="Predicted Poverty Rate", font=dict(size=14, color=THEME["subtext"])),
    ))
    fig.update_layout(height=260, **LAYOUT_BASE)
    return fig


# ── 8. Unemployment trend ─────────────────────────────────────────────────────
def unemployment_trend_chart(df: pd.DataFrame, countries: list) -> go.Figure:
    fig = go.Figure()
    for i, country in enumerate(countries):
        cdf = df[df["country"] == country].sort_values("year")
        fig.add_trace(go.Scatter(
            x=cdf["year"], y=cdf["unemployment_rate"],
            name=country,
            mode="lines",
            line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
        ))
    fig.update_layout(
        title="Unemployment Rate — Multi-Country Comparison",
        xaxis_title="Year", yaxis_title="Unemployment Rate (%)",
        **LAYOUT_BASE,
    )
    return _apply_base(fig)
# updated
