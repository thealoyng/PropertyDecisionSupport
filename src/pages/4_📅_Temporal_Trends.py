"""
Page 4 — Temporal Trends
========================
Time-series and temporal analysis of Singapore HDB resale prices.
Includes monthly/quarterly trends, seasonal decomposition, policy
event annotations, and a rebased price index.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from eda_helpers import load_clean, POLICY_EVENTS, fmt_pct

# ── page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Temporal Trends",
    page_icon="📅",
    layout="wide",
)

st.title("📅 Temporal Trends")
st.caption(
    "Time-series analysis of HDB resale prices — trends, seasonality, "
    "policy impacts, and relative performance across towns."
)

# ── load data ────────────────────────────────────────────────────
df = load_clean()

# ── pre-compute aggregates used by multiple charts ───────────────
monthly = (
    df.groupby(df["month"].dt.to_period("M"))
    .agg(
        median_price=("resale_price", "median"),
        txn_count=("resale_price", "size"),
    )
    .reset_index()
)
monthly["month"] = monthly["month"].dt.to_timestamp()
monthly = monthly.sort_values("month")
monthly["rolling_12m_price"] = monthly["median_price"].rolling(12, min_periods=1).mean()
monthly["rolling_12m_vol"] = monthly["txn_count"].rolling(12, min_periods=1).mean()

yearly = (
    df.groupby("year")["resale_price"]
    .median()
    .reset_index()
    .rename(columns={"resale_price": "median_price"})
    .sort_values("year")
)
yearly["yoy_growth"] = yearly["median_price"].pct_change() * 100

# ── KPI metric cards ────────────────────────────────────────────
first_year = int(yearly["year"].min())
last_year = int(yearly["year"].max())
n_years = last_year - first_year

price_first = yearly.loc[yearly["year"] == first_year, "median_price"].iloc[0]
price_last = yearly.loc[yearly["year"] == last_year, "median_price"].iloc[0]

cagr = ((price_last / price_first) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
latest_growth = yearly.loc[yearly["year"] == last_year, "yoy_growth"].iloc[0]
peak_row = yearly.loc[yearly["median_price"].idxmax()]
trough_row = yearly.loc[yearly["median_price"].idxmin()]

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    f"CAGR since {first_year}",
    fmt_pct(cagr),
    help="Compound annual growth rate of the yearly median resale price.",
)
k2.metric(
    f"{last_year} YoY growth",
    fmt_pct(latest_growth),
    help="Year-over-year change in yearly median resale price.",
)
k3.metric(
    "Peak year",
    f"{int(peak_row['year'])}",
    f"${peak_row['median_price']:,.0f}",
    help="Year with the highest median resale price.",
)
k4.metric(
    "Trough year",
    f"{int(trough_row['year'])}",
    f"${trough_row['median_price']:,.0f}",
    help="Year with the lowest median resale price.",
)

st.divider()

# ── tabs ─────────────────────────────────────────────────────────
tab_price, tab_volume, tab_seasonal, tab_policy, tab_heatmap, tab_flat, tab_index = st.tabs(
    [
        "📈 Price trend",
        "📊 Volume",
        "🔄 Seasonality",
        "🏛️ Policy timeline",
        "🗺️ Growth heatmap",
        "🏢 By flat type",
        "📐 Price index",
    ]
)

# ================================================================
# 1. Monthly median price with 12-month rolling average
# ================================================================
with tab_price:
    st.subheader("Monthly median resale price")

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["median_price"],
            mode="lines",
            name="Monthly median",
            line=dict(color="#93c5fd", width=1),
            opacity=0.6,
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["rolling_12m_price"],
            mode="lines",
            name="12-month rolling avg",
            line=dict(color="#2563eb", width=2.5),
        )
    )
    fig1.update_layout(
        yaxis_title="Median price ($)",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ────────────────────────────────────────────────────────────
    # 2. Year-over-year price growth rate
    # ────────────────────────────────────────────────────────────
    st.subheader("Year-over-year price growth rate")

    yoy = yearly.dropna(subset=["yoy_growth"]).copy()
    yoy["colour"] = np.where(yoy["yoy_growth"] >= 0, "#16a34a", "#dc2626")

    fig2 = go.Figure(
        go.Bar(
            x=yoy["year"],
            y=yoy["yoy_growth"],
            marker_color=yoy["colour"],
            hovertemplate="Year %{x}<br>Growth: %{y:+.1f}%<extra></extra>",
        )
    )
    fig2.update_layout(
        yaxis_title="YoY growth (%)",
        xaxis_title="",
        margin=dict(t=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ================================================================
# 4. Transaction volume by month
# ================================================================
with tab_volume:
    st.subheader("Monthly transaction volume")

    fig4 = go.Figure()
    fig4.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["txn_count"],
            name="Monthly count",
            marker_color="#a5b4fc",
            opacity=0.6,
        )
    )
    fig4.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["rolling_12m_vol"],
            mode="lines",
            name="12-month rolling avg",
            line=dict(color="#4f46e5", width=2.5),
        )
    )
    fig4.update_layout(
        yaxis_title="Transaction count",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ================================================================
# 3. Seasonal decomposition (month-of-year pattern, post-2000)
# ================================================================
with tab_seasonal:
    st.subheader("Seasonal price pattern (post-2000)")
    st.caption(
        "Average median price by calendar month across all years since 2000. "
        "Helps reveal recurring intra-year patterns."
    )

    post2k = df[df["year"] >= 2000].copy()
    post2k["cal_month"] = post2k["month"].dt.month
    seasonal = (
        post2k.groupby("cal_month")["resale_price"]
        .median()
        .reset_index()
        .rename(columns={"resale_price": "median_price"})
    )
    seasonal["month_name"] = seasonal["cal_month"].apply(
        lambda m: pd.Timestamp(2000, m, 1).strftime("%b")
    )

    fig3 = px.line(
        seasonal,
        x="month_name",
        y="median_price",
        markers=True,
        labels={"median_price": "Median price ($)", "month_name": "Month"},
    )
    fig3.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=[
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ],
        ),
        margin=dict(t=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # deviation from annual mean
    overall_median = seasonal["median_price"].mean()
    seasonal["pct_dev"] = (seasonal["median_price"] / overall_median - 1) * 100
    fig3b = px.bar(
        seasonal,
        x="month_name",
        y="pct_dev",
        labels={"pct_dev": "% deviation from mean", "month_name": "Month"},
        color="pct_dev",
        color_continuous_scale=["#dc2626", "#f5f5f5", "#16a34a"],
        color_continuous_midpoint=0,
    )
    fig3b.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=[
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ],
        ),
        showlegend=False,
        margin=dict(t=20),
    )
    st.plotly_chart(fig3b, use_container_width=True)

# ================================================================
# 6. Annotated timeline with policy events
# ================================================================
with tab_policy:
    st.subheader("Price timeline with policy event annotations")

    fig6 = go.Figure()
    fig6.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["rolling_12m_price"],
            mode="lines",
            name="12-month rolling median",
            line=dict(color="#2563eb", width=2),
        )
    )

    # alternate annotation y-positions to reduce overlap
    for idx, (date_str, label) in enumerate(POLICY_EVENTS):
        fig6.add_vline(
            x=pd.Timestamp(date_str),
            line_width=1,
            line_dash="dot",
            line_color="#94a3b8",
        )
        y_shift = -40 if idx % 2 == 0 else 20
        fig6.add_annotation(
            x=pd.Timestamp(date_str),
            y=1.0 if idx % 2 == 0 else 0.92,
            yref="paper",
            text=label,
            showarrow=False,
            textangle=-90,
            font=dict(size=9, color="#475569"),
            xanchor="left",
            yanchor="top",
        )

    fig6.update_layout(
        yaxis_title="12-month rolling median ($)",
        xaxis_title="",
        hovermode="x unified",
        margin=dict(t=30, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig6, use_container_width=True)

    with st.expander("Policy events reference"):
        for date_str, desc in POLICY_EVENTS:
            st.markdown(f"- **{date_str}** — {desc}")

# ================================================================
# 5. Price growth heatmap (town × year)
# ================================================================
with tab_heatmap:
    st.subheader("YoY price growth heatmap — town × year")
    st.caption(
        "Cells show year-over-year median price growth (%). "
        "Green = appreciation, red = decline. "
        "Only town-years with ≥ 30 transactions shown."
    )

    town_yr = (
        df.groupby(["town", "year"])
        .agg(median_price=("resale_price", "median"), count=("resale_price", "size"))
        .reset_index()
    )
    # require minimum 30 transactions for reliability
    town_yr = town_yr[town_yr["count"] >= 30].copy()
    town_yr = town_yr.sort_values(["town", "year"])
    town_yr["yoy"] = town_yr.groupby("town")["median_price"].pct_change() * 100

    pivot = town_yr.pivot(index="town", columns="year", values="yoy")
    # drop years where most towns have NaN (first year per town)
    pivot = pivot.dropna(axis=1, thresh=5)

    fig5 = px.imshow(
        pivot.values,
        x=[str(int(c)) for c in pivot.columns],
        y=pivot.index.tolist(),
        color_continuous_scale=["#dc2626", "#fef9c3", "#16a34a"],
        color_continuous_midpoint=0,
        aspect="auto",
        labels=dict(x="Year", y="Town", color="YoY %"),
        zmin=-20,
        zmax=20,
    )
    fig5.update_layout(
        height=max(500, len(pivot) * 22),
        margin=dict(t=30),
    )
    st.plotly_chart(fig5, use_container_width=True)

# ================================================================
# 7. Quarterly median price by flat type
# ================================================================
with tab_flat:
    st.subheader("Quarterly median price by flat type")

    df_q = df.copy()
    df_q["quarter"] = df_q["month"].dt.to_period("Q").dt.to_timestamp()
    qt = (
        df_q.groupby(["quarter", "flat_type"])["resale_price"]
        .median()
        .reset_index()
        .rename(columns={"resale_price": "median_price"})
    )

    fig7 = px.line(
        qt,
        x="quarter",
        y="median_price",
        color="flat_type",
        labels={
            "median_price": "Median price ($)",
            "quarter": "",
            "flat_type": "Flat type",
        },
    )
    fig7.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30),
    )
    st.plotly_chart(fig7, use_container_width=True)

# ================================================================
# 8. Price index (rebased to 100)
# ================================================================
with tab_index:
    st.subheader("Rebased price index by town")
    st.caption(
        "Shows relative price performance: index = (yearly median / base-year median) × 100. "
        "Select a base year and compare how different towns have diverged."
    )

    # town-year medians (reuse town_yr from heatmap, but recompute without count filter)
    ty = (
        df.groupby(["town", "year"])["resale_price"]
        .median()
        .reset_index()
        .rename(columns={"resale_price": "median_price"})
    )

    valid_years = sorted(ty["year"].unique())
    base_year = st.slider(
        "Base year (index = 100)",
        min_value=int(valid_years[0]),
        max_value=int(valid_years[-1]),
        value=2000 if 2000 in valid_years else int(valid_years[0]),
    )

    # compute index
    base_prices = ty[ty["year"] == base_year][["town", "median_price"]].rename(
        columns={"median_price": "base_price"}
    )
    ty_idx = ty.merge(base_prices, on="town", how="inner")
    ty_idx["index"] = (ty_idx["median_price"] / ty_idx["base_price"]) * 100

    # pick top 8 towns by latest index value
    latest_yr = ty_idx["year"].max()
    top_towns = (
        ty_idx[ty_idx["year"] == latest_yr]
        .nlargest(8, "index")["town"]
        .tolist()
    )

    selected_towns = st.multiselect(
        "Towns to show (default: top 8 by latest index)",
        sorted(ty_idx["town"].unique()),
        default=top_towns,
    )

    if selected_towns:
        plot_data = ty_idx[ty_idx["town"].isin(selected_towns)]
        fig8 = px.line(
            plot_data,
            x="year",
            y="index",
            color="town",
            labels={"index": f"Price index (base {base_year} = 100)", "year": ""},
        )
        fig8.add_hline(
            y=100,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Base ({base_year})",
        )
        fig8.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=30),
        )
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("Select at least one town to display the index chart.")
