"""
Page 3 — Price Deep Dive
=========================
Comprehensive price analysis for Singapore HDB resale transactions.
Includes distribution plots, million-dollar analysis, percentile bands,
volatility heatmaps, and town × flat-type median-price matrices.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from eda_helpers import load_clean, fmt_price

# ── page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Price Deep Dive", page_icon="💰", layout="wide")
st.title("💰 Price Deep Dive")
st.caption("Distribution, segmentation, and outlier analysis of HDB resale prices.")

# ── load data ────────────────────────────────────────────────────
df_all = load_clean()

# ── global year-range filter ─────────────────────────────────────
year_min, year_max = int(df_all["year"].min()), int(df_all["year"].max())
yr_lo, yr_hi = st.slider(
    "Filter by year range (applies to all charts below)",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1,
)
df = df_all[(df_all["year"] >= yr_lo) & (df_all["year"] <= yr_hi)].copy()

# ── metric cards ─────────────────────────────────────────────────
highest_price = df["resale_price"].max()
median_price = df["resale_price"].median()
most_expensive_town = df.groupby("town")["resale_price"].median().idxmax()
most_expensive_type = df.groupby("flat_type")["resale_price"].median().idxmax()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Highest Price Ever", fmt_price(highest_price))
m2.metric("Median Price (All Time)", fmt_price(median_price))
m3.metric("Most Expensive Town", most_expensive_town)
m4.metric("Most Expensive Flat Type", most_expensive_type)

st.divider()

# =====================================================================
# 1. Violin plots — price by flat type
# =====================================================================
st.subheader("1. Price Distribution by Flat Type")
st.markdown("Violin plots reveal the full distribution shape, median, and quartiles for each flat type.")

fig_violin = px.violin(
    df,
    x="flat_type",
    y="resale_price",
    color="flat_type",
    box=True,
    points=False,
    labels={"resale_price": "Resale Price ($)", "flat_type": "Flat Type"},
    title="Resale Price Distribution by Flat Type",
    category_orders={
        "flat_type": sorted(df["flat_type"].unique(), key=lambda t: df.loc[df["flat_type"] == t, "resale_price"].median())
    },
)
fig_violin.update_layout(showlegend=False, height=550)
st.plotly_chart(fig_violin, use_container_width=True)

# =====================================================================
# 2. Box plots — price by town (top 15 by median)
# =====================================================================
st.subheader("2. Price Distribution by Town (Top 15)")
st.markdown("Horizontal box plots for the 15 towns with the highest median resale price.")

town_medians = df.groupby("town")["resale_price"].median().nlargest(15)
top15_towns = town_medians.index.tolist()
df_top15 = df[df["town"].isin(top15_towns)].copy()

# Sort towns by median for the chart
town_order = town_medians.sort_values(ascending=True).index.tolist()

fig_box = px.box(
    df_top15,
    x="resale_price",
    y="town",
    orientation="h",
    color="town",
    labels={"resale_price": "Resale Price ($)", "town": "Town"},
    title="Resale Price by Town (Top 15 by Median)",
    category_orders={"town": town_order},
)
fig_box.update_layout(showlegend=False, height=600)
st.plotly_chart(fig_box, use_container_width=True)

# =====================================================================
# 3. Price per sqm by town — bar chart (normalised comparison)
# =====================================================================
st.subheader("3. Price per sqm by Town")
st.markdown(
    "Median price per square metre removes size bias, enabling a fair comparison "
    "of location premiums across towns."
)

town_psqm = (
    df.groupby("town")["price_per_sqm"]
    .median()
    .sort_values(ascending=False)
    .reset_index()
)
town_psqm.columns = ["town", "median_price_per_sqm"]

fig_psqm = px.bar(
    town_psqm,
    x="median_price_per_sqm",
    y="town",
    orientation="h",
    labels={"median_price_per_sqm": "Median Price per sqm ($)", "town": "Town"},
    title="Median Price per sqm by Town (sorted descending)",
    color="median_price_per_sqm",
    color_continuous_scale="Oranges",
)
fig_psqm.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=650,
    coloraxis_showscale=False,
)
st.plotly_chart(fig_psqm, use_container_width=True)

# =====================================================================
# 4. Million-dollar flat analysis
# =====================================================================
st.subheader("4. Million-Dollar Flat Analysis")
st.markdown("Transactions where the resale price reached **$1,000,000** or above.")

df_mil = df[df["resale_price"] >= 1_000_000].copy()

if len(df_mil) == 0:
    st.info("No million-dollar transactions in the selected year range.")
else:
    st.metric("Million-Dollar Transactions", f"{len(df_mil):,}")

    col_a, col_b = st.columns(2)

    # 4a — count per year (line chart)
    mil_year = df_mil.groupby("year").size().reset_index(name="count")
    fig_mil_yr = px.line(
        mil_year,
        x="year",
        y="count",
        markers=True,
        labels={"count": "Transactions", "year": "Year"},
        title="Million-Dollar Transactions per Year",
    )
    fig_mil_yr.update_layout(height=400)
    col_a.plotly_chart(fig_mil_yr, use_container_width=True)

    # 4b — towns with the most million-dollar flats
    mil_town = (
        df_mil.groupby("town").size().reset_index(name="count").sort_values("count", ascending=False)
    )
    fig_mil_town = px.bar(
        mil_town,
        x="count",
        y="town",
        orientation="h",
        labels={"count": "Transactions", "town": "Town"},
        title="Million-Dollar Flats by Town",
    )
    fig_mil_town.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    col_b.plotly_chart(fig_mil_town, use_container_width=True)

    # 4c — which flat types reach $1M
    mil_type = (
        df_mil.groupby("flat_type").size().reset_index(name="count").sort_values("count", ascending=False)
    )
    fig_mil_type = px.bar(
        mil_type,
        x="flat_type",
        y="count",
        color="flat_type",
        labels={"count": "Transactions", "flat_type": "Flat Type"},
        title="Million-Dollar Flats by Flat Type",
    )
    fig_mil_type.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_mil_type, use_container_width=True)

st.divider()

# =====================================================================
# 5. Price percentile bands over time
# =====================================================================
st.subheader("5. Price Percentile Bands Over Time")
st.markdown(
    "Annual percentile lines (p10, p25, p50, p75, p90) show how the price "
    "distribution has widened or narrowed over time. Shaded bands highlight "
    "the inter-decile and interquartile ranges."
)

pct_year = (
    df.groupby("year")["resale_price"]
    .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    .unstack()
    .reset_index()
)
pct_year.columns = ["year", "p10", "p25", "p50", "p75", "p90"]

fig_pct = go.Figure()

# Shaded band: p10 → p90
fig_pct.add_trace(go.Scatter(
    x=pct_year["year"], y=pct_year["p90"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig_pct.add_trace(go.Scatter(
    x=pct_year["year"], y=pct_year["p10"],
    mode="lines", line=dict(width=0), fill="tonexty",
    fillcolor="rgba(255,165,0,0.12)", showlegend=False, hoverinfo="skip",
))

# Shaded band: p25 → p75
fig_pct.add_trace(go.Scatter(
    x=pct_year["year"], y=pct_year["p75"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig_pct.add_trace(go.Scatter(
    x=pct_year["year"], y=pct_year["p25"],
    mode="lines", line=dict(width=0), fill="tonexty",
    fillcolor="rgba(255,165,0,0.25)", showlegend=False, hoverinfo="skip",
))

# Lines for each percentile
for col, name, color, dash in [
    ("p10", "10th percentile", "#1f77b4", "dot"),
    ("p25", "25th percentile", "#ff7f0e", "dash"),
    ("p50", "Median (50th)", "#2ca02c", "solid"),
    ("p75", "75th percentile", "#ff7f0e", "dash"),
    ("p90", "90th percentile", "#1f77b4", "dot"),
]:
    fig_pct.add_trace(go.Scatter(
        x=pct_year["year"], y=pct_year[col],
        mode="lines+markers", name=name,
        line=dict(color=color, dash=dash, width=2),
        marker=dict(size=4),
    ))

fig_pct.update_layout(
    title="Resale Price Percentile Bands (Yearly)",
    xaxis_title="Year",
    yaxis_title="Resale Price ($)",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
st.plotly_chart(fig_pct, use_container_width=True)

# =====================================================================
# 6. Price range (p90 − p10) by town per year — heatmap (volatility)
# =====================================================================
st.subheader("6. Price Spread by Town Over Time")
st.markdown(
    "The inter-decile range (p90 − p10) per town per year captures price "
    "volatility / dispersion within each location."
)


def _iqr_range(s):
    """Return p90 − p10 for a series."""
    if len(s) < 10:
        return np.nan
    return s.quantile(0.90) - s.quantile(0.10)


spread = (
    df.groupby(["year", "town"])["resale_price"]
    .apply(_iqr_range)
    .reset_index(name="price_range")
)
spread_pivot = spread.pivot(index="town", columns="year", values="price_range")
# Keep only towns with enough data (at least 5 years present)
spread_pivot = spread_pivot.dropna(thresh=5)
spread_pivot = spread_pivot.loc[spread_pivot.mean(axis=1).sort_values(ascending=False).index]

fig_spread = px.imshow(
    spread_pivot,
    labels=dict(x="Year", y="Town", color="P90 − P10 ($)"),
    title="Price Spread (P90 − P10) by Town per Year",
    aspect="auto",
    color_continuous_scale="YlOrRd",
)
fig_spread.update_layout(height=700)
st.plotly_chart(fig_spread, use_container_width=True)

# =====================================================================
# 7. Median price heatmap — town × flat_type (recent 3 years)
# =====================================================================
st.subheader("7. Median Price Heatmap — Town × Flat Type (Recent 3 Years)")
st.markdown(
    "Using only the **most recent 3 years** of data for relevance. "
    "Each cell shows the median resale price for that town / flat-type combination."
)

recent_cutoff = year_max - 2  # last 3 calendar years
df_recent = df[df["year"] >= recent_cutoff]

heat = (
    df_recent.groupby(["town", "flat_type"])["resale_price"]
    .median()
    .reset_index()
)
heat_pivot = heat.pivot(index="town", columns="flat_type", values="resale_price")

# Sort rows by overall median descending
row_order = heat_pivot.median(axis=1).sort_values(ascending=False).index
heat_pivot = heat_pivot.loc[row_order]

# Sort columns logically
type_order = [t for t in ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM",
                           "EXECUTIVE", "MULTI-GENERATION"] if t in heat_pivot.columns]
remaining = [c for c in heat_pivot.columns if c not in type_order]
heat_pivot = heat_pivot[type_order + remaining]

fig_heat = px.imshow(
    heat_pivot,
    labels=dict(x="Flat Type", y="Town", color="Median Price ($)"),
    title=f"Median Resale Price by Town & Flat Type ({recent_cutoff}–{year_max})",
    aspect="auto",
    color_continuous_scale="Viridis",
    text_auto=".0f",
)
fig_heat.update_layout(height=750)
st.plotly_chart(fig_heat, use_container_width=True)

# ── footer ───────────────────────────────────────────────────────
st.divider()
st.caption(f"Data range after filter: {yr_lo}–{yr_hi} · {len(df):,} transactions")
