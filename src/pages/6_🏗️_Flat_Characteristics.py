"""
Page 6 -- Flat Characteristics
================================
Analysis of HDB flat physical attributes: flat type, floor area,
storey level, and flat model — and how they relate to resale prices.
"""
import sys
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eda_helpers import load_clean, storey_band

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Flat Characteristics",
    page_icon="\U0001f3d7\ufe0f",
    layout="wide",
)

st.title("\U0001f3d7\ufe0f Flat Characteristics")
st.caption(
    "How flat type, floor area, storey level, and flat model affect "
    "HDB resale prices across Singapore."
)

# ── Load data ────────────────────────────────────────────────────
df = load_clean()

# ── Town filter ──────────────────────────────────────────────────
all_towns = sorted(df["town"].dropna().unique())
selected_towns = st.multiselect(
    "Filter by town (leave empty for all towns)",
    options=all_towns,
    default=[],
    key="flat_char_town_filter",
)
if selected_towns:
    df = df[df["town"].isin(selected_towns)].copy()

# ── Metric cards ─────────────────────────────────────────────────
most_common_flat_type = df["flat_type"].mode().iloc[0] if len(df) else "N/A"
avg_floor_area = df["floor_area_sqm"].mean() if len(df) else 0
most_common_flat_model = df["flat_model"].mode().iloc[0] if len(df) else "N/A"
avg_storey = df["storey_mid"].mean() if len(df) else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Most Common Flat Type", most_common_flat_type)
m2.metric("Avg Floor Area", f"{avg_floor_area:.1f} sqm")
m3.metric("Most Common Flat Model", most_common_flat_model)
m4.metric("Avg Storey", f"{avg_storey:.1f}")

st.divider()

# =====================================================================
# 1. Price by flat type over time
# =====================================================================
st.subheader("1. Median Resale Price by Flat Type Over Time")
yearly_type = (
    df.groupby(["year", "flat_type"])["resale_price"]
    .median()
    .reset_index()
)
fig1 = px.line(
    yearly_type,
    x="year",
    y="resale_price",
    color="flat_type",
    markers=True,
    labels={
        "resale_price": "Median Resale Price ($)",
        "year": "Year",
        "flat_type": "Flat Type",
    },
    title="Yearly Median Resale Price by Flat Type",
)
fig1.update_layout(legend_title_text="Flat Type", hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

# =====================================================================
# 2. Floor area vs price scatter
# =====================================================================
st.subheader("2. Floor Area vs Resale Price")
st.caption("Sampled to 10,000 points for performance. Each dot coloured by flat type.")

sample_n = min(10_000, len(df))
scatter_df = df.sample(n=sample_n, random_state=42)
fig2 = px.scatter(
    scatter_df,
    x="floor_area_sqm",
    y="resale_price",
    color="flat_type",
    opacity=0.4,
    trendline="ols",
    labels={
        "floor_area_sqm": "Floor Area (sqm)",
        "resale_price": "Resale Price ($)",
        "flat_type": "Flat Type",
    },
    title="Floor Area vs Resale Price (with OLS trendline per flat type)",
)
fig2.update_layout(legend_title_text="Flat Type")
st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
# 3. Storey premium analysis
# =====================================================================
st.subheader("3. Storey Premium Analysis")
st.caption(
    "Median price per sqm by storey band, with percentage premium "
    "relative to the lowest band."
)

df["storey_band"] = df["storey_mid"].apply(storey_band)
band_order = ["01-03", "04-06", "07-09", "10-12", "13-15", "16-21", "22-30", "31+"]
storey_agg = (
    df.groupby("storey_band")["price_per_sqm"]
    .median()
    .reindex(band_order)
    .dropna()
    .reset_index()
)
storey_agg.columns = ["storey_band", "median_psqm"]
base_price = storey_agg["median_psqm"].iloc[0]
storey_agg["premium_pct"] = (
    (storey_agg["median_psqm"] - base_price) / base_price * 100
)

fig3 = go.Figure()
fig3.add_trace(
    go.Bar(
        x=storey_agg["storey_band"],
        y=storey_agg["median_psqm"],
        text=storey_agg["premium_pct"].apply(lambda v: f"+{v:.1f}%"),
        textposition="outside",
        marker_color=px.colors.sequential.Teal,
        name="Median $/sqm",
    )
)
fig3.update_layout(
    title="Median Price per sqm by Storey Band (% premium vs lowest band)",
    xaxis_title="Storey Band",
    yaxis_title="Median Price per sqm ($)",
    showlegend=False,
)
st.plotly_chart(fig3, use_container_width=True)

# =====================================================================
# 4. Flat model ranking (recent 5 years)
# =====================================================================
st.subheader("4. Top 15 Flat Models by Median Price per sqm (Recent 5 Years)")
max_year = int(df["year"].max())
recent = df[df["year"] >= max_year - 4]
model_rank = (
    recent.groupby("flat_model")["price_per_sqm"]
    .median()
    .sort_values(ascending=True)
    .tail(15)
    .reset_index()
)
model_rank.columns = ["flat_model", "median_psqm"]

fig4 = px.bar(
    model_rank,
    x="median_psqm",
    y="flat_model",
    orientation="h",
    text="median_psqm",
    labels={
        "median_psqm": "Median Price per sqm ($)",
        "flat_model": "Flat Model",
    },
    title=f"Top 15 Flat Models by Median $/sqm ({max_year - 4}\u2013{max_year})",
    color="median_psqm",
    color_continuous_scale="Viridis",
)
fig4.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
fig4.update_layout(yaxis_categoryorder="total ascending", coloraxis_showscale=False)
st.plotly_chart(fig4, use_container_width=True)

# =====================================================================
# 5. Flat type mix evolution
# =====================================================================
st.subheader("5. Flat Type Mix Evolution")
st.caption(
    "Percentage of transactions by flat type per year — how the "
    "market composition has changed over time."
)

type_year = df.groupby(["year", "flat_type"]).size().reset_index(name="count")
type_year_total = type_year.groupby("year")["count"].transform("sum")
type_year["pct"] = type_year["count"] / type_year_total * 100

fig5 = px.area(
    type_year,
    x="year",
    y="pct",
    color="flat_type",
    groupnorm="",  # already computed percentages
    labels={
        "pct": "% of Transactions",
        "year": "Year",
        "flat_type": "Flat Type",
    },
    title="Market Composition: % of Transactions by Flat Type per Year",
)
fig5.update_layout(
    legend_title_text="Flat Type",
    yaxis_title="% of Transactions",
    hovermode="x unified",
)
st.plotly_chart(fig5, use_container_width=True)

# =====================================================================
# 6. Floor area trends by flat type over time
# =====================================================================
st.subheader("6. Floor Area Trends by Flat Type Over Time")
st.caption(
    "Median floor area per flat type per year — are newer flats "
    "getting smaller?"
)

area_trend = (
    df.groupby(["year", "flat_type"])["floor_area_sqm"]
    .median()
    .reset_index()
)
fig6 = px.line(
    area_trend,
    x="year",
    y="floor_area_sqm",
    color="flat_type",
    markers=True,
    labels={
        "floor_area_sqm": "Median Floor Area (sqm)",
        "year": "Year",
        "flat_type": "Flat Type",
    },
    title="Median Floor Area by Flat Type Over Time",
)
fig6.update_layout(legend_title_text="Flat Type", hovermode="x unified")
st.plotly_chart(fig6, use_container_width=True)

# =====================================================================
# 7. Storey distribution by era
# =====================================================================
st.subheader("7. Storey Distribution by Era")
st.caption(
    "How the distribution of storey levels in resale transactions has "
    "shifted across decades."
)

df["era"] = df["year"].apply(
    lambda y: "1990s" if y < 2000
    else "2000s" if y < 2010
    else "2010s" if y < 2020
    else "2020s"
)
era_order = ["1990s", "2000s", "2010s", "2020s"]
eras_present = [e for e in era_order if e in df["era"].unique()]

fig7 = px.histogram(
    df[df["era"].isin(eras_present)],
    x="storey_mid",
    color="era",
    barmode="group",
    nbins=30,
    category_orders={"era": eras_present},
    labels={
        "storey_mid": "Storey Midpoint",
        "era": "Era",
        "count": "Transactions",
    },
    title="Storey-Level Distribution by Era",
    opacity=0.75,
)
fig7.update_layout(
    xaxis_title="Storey Midpoint",
    yaxis_title="Number of Transactions",
    legend_title_text="Era",
    bargap=0.1,
)
st.plotly_chart(fig7, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Data: {len(df):,} resale transactions "
    f"({int(df['year'].min())}\u2013{int(df['year'].max())}). "
    "Source: data.gov.sg."
)
