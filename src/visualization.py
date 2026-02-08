"""
Plotly visualization functions for the Policy Impact Simulation Platform.
All charts use a consistent dark theme matching the dashboard design system.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

DARK_BG    = "#070B14"
CARD_BG    = "#0C1220"
BORDER     = "#1E293B"
TEXT_DIM   = "#475569"
TEXT_MAIN  = "#CBD5E1"
ACCENT     = "#38BDF8"
GREEN      = "#22C55E"
RED        = "#F87171"
AMBER      = "#F59E0B"
PURPLE     = "#A78BFA"

LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
    font=dict(family="Space Grotesk, sans-serif", color=TEXT_MAIN, size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_DIM)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_DIM)),
)


def poverty_trend_chart(df: pd.DataFrame, country: str) -> go.Figure:
    sub = df[df["country"] == country].sort_values("year")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["year"], y=sub["poverty_rate"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=5, color=ACCENT),
        name="Poverty Rate",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text=f"Poverty Rate Trend — {country}",
        font=dict(size=13, color=TEXT_MAIN), x=0.0))
    return fig


def scenario_comparison_chart(results: list) -> go.Figure:
    names  = [r["scenario_name"] for r in results]
    deltas = [r["delta"] for r in results]
    colors = [GREEN if d < 0 else RED for d in deltas]
    fig = go.Figure(go.Bar(
        x=names, y=deltas,
        marker_color=colors,
        hovertemplate="%{x}: %{y:+.2f}pp<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text="Poverty Change by Scenario (pp)",
        font=dict(size=13, color=TEXT_MAIN), x=0.0),
        showlegend=False)
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    return fig


def gdp_vs_poverty_scatter(df: pd.DataFrame, year: int) -> go.Figure:
    sub = df[df["year"] == year]
    fig = px.scatter(
        sub, x="gdp_growth_pct", y="poverty_rate",
        hover_name="country", color="human_development_index",
        color_continuous_scale=["#F87171", "#FBBF24", "#22C55E"],
        size="population_millions", size_max=35,
        labels={"gdp_growth_pct": "GDP Growth (%)", "poverty_rate": "Poverty Rate (%)"},
    )
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text=f"GDP Growth vs Poverty Rate — {year}",
        font=dict(size=13, color=TEXT_MAIN), x=0.0))
    return fig


def feature_importance_chart(imp_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=imp_df["importance"].head(10),
        y=imp_df["feature"].head(10),
        orientation="h",
        marker_color=ACCENT,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE,
        title=dict(text="Feature Importance (Top 10)", font=dict(size=13, color=TEXT_MAIN), x=0.0),
        yaxis=dict(autorange="reversed", gridcolor=BORDER, tickfont=dict(color=TEXT_DIM)),
    )
    return fig


def hdi_bar_chart(df: pd.DataFrame, year: int) -> go.Figure:
    sub = df[df["year"] == year].sort_values("human_development_index", ascending=False)
    fig = go.Figure(go.Bar(
        x=sub["country"], y=sub["human_development_index"],
        marker_color=PURPLE,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text=f"Human Development Index — {year}",
        font=dict(size=13, color=TEXT_MAIN), x=0.0))
    return fig


def radar_scenario_chart(results: list) -> go.Figure:
    categories = ["GDP Effect", "Employment", "Education", "Fiscal", "HDI"]
    fig = go.Figure()
    for r in results:
        mf = r["modified_features"]
        vals = [
            mf.get("gdp_growth_pct", 0) * 5,
            -mf.get("unemployment_rate", 0) * 5,
            mf.get("education_spending_pct_gdp", 0) * 8,
            mf.get("tax_revenue_pct_gdp", 0) * 2,
            mf.get("human_development_index", 0) * 200,
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", name=r["scenario_name"],
            line=dict(color=r.get("color", ACCENT)),
        ))
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(family="Space Grotesk, sans-serif", color=TEXT_MAIN),
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, gridcolor=BORDER, tickfont=dict(color=TEXT_DIM)),
            angularaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT_DIM)),
        ),
        title=dict(text="Scenario Radar Comparison", font=dict(size=13, color=TEXT_MAIN), x=0.0),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def poverty_gauge(value: float, baseline: float) -> go.Figure:
    color = GREEN if value < baseline else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta=dict(reference=baseline, valueformat=".2f", suffix="pp"),
        gauge=dict(
            axis=dict(range=[0, 80], tickfont=dict(color=TEXT_DIM)),
            bar=dict(color=color),
            bgcolor=CARD_BG,
            bordercolor=BORDER,
            steps=[
                dict(range=[0, 20],  color="#0F2A1A"),
                dict(range=[20, 40], color="#1A1A0F"),
                dict(range=[40, 80], color="#2A0F0F"),
            ],
        ),
        number=dict(suffix="%", font=dict(color=TEXT_MAIN, size=28)),
        title=dict(text="Projected Poverty Rate", font=dict(color=TEXT_DIM, size=12)),
    ))
    fig.update_layout(paper_bgcolor=DARK_BG, margin=dict(l=20, r=20, t=40, b=20), height=220)
    return fig


def unemployment_trend_chart(df: pd.DataFrame, countries: list) -> go.Figure:
    COLORS = [ACCENT, GREEN, AMBER, RED, PURPLE, "#F472B6", "#34D399", "#FB923C"]
    fig = go.Figure()
    for i, country in enumerate(countries):
        sub = df[df["country"] == country].sort_values("year")
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["unemployment_rate"],
            mode="lines", name=country,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"{country} %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text="Unemployment Rate Trends",
        font=dict(size=13, color=TEXT_MAIN), x=0.0))
    return fig
# a11y colors
# formatting
# Docstring enhancements applied to visualization tools
# Color contrast
# Radar chart
# Remove unused
# Radar chart
