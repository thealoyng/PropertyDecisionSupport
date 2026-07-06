"""
Page 8 — Market Dynamics
========================
Market health and momentum signals for Singapore HDB resale data.
Covers price-volume relationships, market cycles, transaction
concentration, buyer age preferences, price momentum, supply
heatmaps, and high-value segment growth.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eda_helpers import load_clean, fmt_price, fmt_pct, trend_arrow

# ── page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Dynamics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Market Dynamics")
st.caption(
    "Market health and momentum signals — price-volume relationships, "
    "cycle identification, transaction concentration, buyer preferences, "
    "and premium segment growth."
)

# ── load data ────────────────────────────────────────────────────
df = load_clean()

# ── pre-compute aggregates ───────────────────────────────────────
latest_year = int(df["year"].max())
prev_year = latest_year - 1

txn_latest = int(df[df["year"] == latest_year].shape[0])
txn_prev = int(df[df["year"] == prev_year].shape[0])
vol_change = ((txn_latest / txn_prev) - 1) * 100 if txn_prev > 0 else 0.0

median_latest = df[df["year"] == latest_year]["resale_price"].median()
median_prev = df[df["year"] == prev_year]["resale_price"].median()
price_growth = ((median_latest / median_prev) - 1) * 100 if median_prev > 0 else 0.0

# ── KPI metric cards ────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    f"Transactions ({latest_year})",
    f"{txn_latest:,}",
    help="Total number of HDB resale transactions in the latest year.",
)
k2.metric(
    "YoY volume change",
    fmt_pct(vol_change),
    delta=f"{trend_arrow(vol_change)} {fmt_pct(vol_change)}",
    help="Year-over-year change in transaction volume.",
)
k3.metric(
    f"Median price ({latest_year})",
    fmt_price(median_latest),
    help="Median resale price for the latest year.",
)
k4.metric(
    "YoY price growth",
    fmt_pct(price_growth),
    delta=f"{trend_arrow(price_growth)} {fmt_pct(price_growth)}",
    help="Year-over-year change in median resale price.",
)

st.divider()

# ── tabs ─────────────────────────────────────────────────────────
(
    tab_scatter,
    tab_cycle,
    tab_conc,
    tab_age,
    tab_momentum,
    tab_supply,
    tab_premium,
) = st.tabs(
    [
        "🔵 Price-Volume",
        "🔄 Market Cycle",
        "📍 Concentration",
        "🏠 Flat Age Pref.",
        "📈 Momentum",
        "🗺️ Supply Heatmap",
        "💎 Premium Growth",
    ]
)

# ================================================================
# 1. Price-volume scatter (annual)
# ================================================================
with tab_scatter:
    st.subheader("Price vs. volume scatter (annual)")
    st.caption(
        "Each dot is one year. The X-axis shows total transaction count "
        "and the Y-axis shows the median resale price. A positive "
        "correlation suggests that higher market activity accompanies "
        "rising prices — a classic sign of a healthy bull market."
    )

    annual = (
        df.groupby("year")
        .agg(
            txn_count=("resale_price", "size"),
            median_price=("resale_price", "median"),
        )
        .reset_index()
    )

    fig1 = px.scatter(
        annual,
        x="txn_count",
        y="median_price",
        text="year",
        labels={
            "txn_count": "Total transactions",
            "median_price": "Median resale price ($)",
        },
        color_discrete_sequence=["#2563eb"],
    )
    fig1.update_traces(
        textposition="top center",
        marker=dict(size=10),
        textfont=dict(size=10),
    )
    # add OLS trendline
    if len(annual) > 2:
        z = np.polyfit(annual["txn_count"], annual["median_price"], 1)
        x_line = np.linspace(annual["txn_count"].min(), annual["txn_count"].max(), 50)
        y_line = np.polyval(z, x_line)
        fig1.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trend",
                line=dict(color="#94a3b8", dash="dash", width=1.5),
            )
        )
    fig1.update_layout(
        hovermode="closest",
        margin=dict(t=30),
        showlegend=False,
    )
    st.plotly_chart(fig1, use_container_width=True)

    corr = annual["txn_count"].corr(annual["median_price"])
    st.info(
        f"**Correlation coefficient: {corr:.2f}** — "
        + (
            "Strong positive: prices and volumes tend to move together."
            if corr > 0.5
            else (
                "Weak or negative: prices and volumes may diverge — "
                "watch for potential market turning points."
                if corr < 0.2
                else "Moderate relationship between price and volume."
            )
        )
    )

# ================================================================
# 2. Market cycle identification (dual-axis)
# ================================================================
with tab_cycle:
    st.subheader("Market cycle — price trend & volume")
    st.caption(
        "The blue line is the 12-month rolling median price (left axis) "
        "and the grey bars show monthly transaction volume (right axis). "
        "Rising price + rising volume = expansion phase; falling price + "
        "falling volume = contraction phase."
    )

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
    monthly["rolling_12m_price"] = (
        monthly["median_price"].rolling(12, min_periods=1).mean()
    )

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    fig2.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["txn_count"],
            name="Monthly volume",
            marker_color="#cbd5e1",
            opacity=0.5,
        ),
        secondary_y=True,
    )
    fig2.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["rolling_12m_price"],
            mode="lines",
            name="12-month rolling median price",
            line=dict(color="#2563eb", width=2.5),
        ),
        secondary_y=False,
    )

    fig2.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
        ),
        margin=dict(t=30),
    )
    fig2.update_yaxes(title_text="Median price ($)", secondary_y=False)
    fig2.update_yaxes(title_text="Transaction count", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

# ================================================================
# 3. Transaction concentration — top 5 towns share over time
# ================================================================
with tab_conc:
    st.subheader("Transaction concentration — top 5 towns")
    st.caption(
        "Shows the percentage of total monthly transactions contributed "
        "by the 5 most active towns each year. A rising share means "
        "market activity is becoming more geographically concentrated."
    )

    yearly_town = (
        df.groupby(["year", "town"])
        .agg(txn=("resale_price", "size"))
        .reset_index()
    )
    yearly_total = (
        yearly_town.groupby("year")["txn"].sum().reset_index(name="total")
    )
    yearly_town = yearly_town.merge(yearly_total, on="year")
    yearly_town["share"] = yearly_town["txn"] / yearly_town["total"] * 100

    # identify overall top 5 towns by total transactions
    top5 = (
        yearly_town.groupby("town")["txn"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )

    conc = yearly_town[yearly_town["town"].isin(top5)].copy()
    conc = conc.sort_values(["year", "town"])

    fig3 = px.area(
        conc,
        x="year",
        y="share",
        color="town",
        labels={"share": "Share of transactions (%)", "year": ""},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig3.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
        ),
        margin=dict(t=30),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "**For buyers:** Towns with consistently high transaction share "
        "tend to have better liquidity — easier to sell in the future. "
        "However, less concentrated towns may offer better value."
    )

# ================================================================
# 4. New vs old flat preference — median flat age over time
# ================================================================
with tab_age:
    st.subheader("Buyer preference — median flat age at transaction")
    st.caption(
        "Tracks the median age (in years) of flats being transacted each "
        "year. A rising trend means buyers are increasingly purchasing "
        "older flats; a falling trend indicates a shift toward newer stock."
    )

    if "flat_age" in df.columns:
        age_annual = (
            df.dropna(subset=["flat_age"])
            .groupby("year")["flat_age"]
            .median()
            .reset_index()
            .rename(columns={"flat_age": "median_age"})
            .sort_values("year")
        )

        fig4 = px.line(
            age_annual,
            x="year",
            y="median_age",
            markers=True,
            labels={"median_age": "Median flat age (years)", "year": ""},
            color_discrete_sequence=["#0891b2"],
        )
        fig4.update_traces(
            line=dict(width=2.5),
            marker=dict(size=6),
        )
        fig4.update_layout(margin=dict(t=30))
        st.plotly_chart(fig4, use_container_width=True)

        latest_age = age_annual[age_annual["year"] == latest_year][
            "median_age"
        ].iloc[0]
        prev_age = age_annual[age_annual["year"] == prev_year]["median_age"]
        if len(prev_age) > 0:
            age_delta = latest_age - prev_age.iloc[0]
            st.info(
                f"**{latest_year} median flat age: {latest_age:.1f} years** "
                f"({'↑' if age_delta > 0 else '↓'} {abs(age_delta):.1f} yrs "
                f"vs {prev_year}). "
                + (
                    "Buyers are trending toward older, potentially more "
                    "affordable flats. Check remaining lease carefully."
                    if age_delta > 0
                    else "Buyers prefer newer flats — expect higher prices "
                    "but longer remaining leases."
                )
            )
    else:
        st.warning(
            "The `flat_age` column is not available in the dataset. "
            "Ensure your cleaning pipeline computes it."
        )

# ================================================================
# 5. Price momentum — 3-month vs 12-month rolling median
# ================================================================
with tab_momentum:
    st.subheader("Price momentum — short vs. long-term trend")
    st.caption(
        "The 3-month rolling median (fast signal) vs. the 12-month rolling "
        "median (slow signal). When the fast line crosses **above** the "
        "slow line, it signals bullish momentum; crossing **below** "
        "signals bearish momentum."
    )

    mom = (
        df.groupby(df["month"].dt.to_period("M"))["resale_price"]
        .median()
        .reset_index()
    )
    mom.columns = ["period", "median_price"]
    mom["month"] = mom["period"].dt.to_timestamp()
    mom = mom.sort_values("month")
    mom["fast"] = mom["median_price"].rolling(3, min_periods=1).mean()
    mom["slow"] = mom["median_price"].rolling(12, min_periods=1).mean()

    fig5 = go.Figure()

    # shade areas between fast and slow
    fig5.add_trace(
        go.Scatter(
            x=mom["month"],
            y=mom["fast"],
            mode="lines",
            name="3-month rolling median",
            line=dict(color="#16a34a", width=2),
        )
    )
    fig5.add_trace(
        go.Scatter(
            x=mom["month"],
            y=mom["slow"],
            mode="lines",
            name="12-month rolling median",
            line=dict(color="#dc2626", width=2),
            fill="tonexty",
            fillcolor="rgba(220, 38, 38, 0.08)",
        )
    )

    # re-add bullish shading where fast > slow
    bullish = mom[mom["fast"] >= mom["slow"]].copy()
    if not bullish.empty:
        fig5.add_trace(
            go.Scatter(
                x=bullish["month"],
                y=bullish["fast"],
                mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig5.add_trace(
            go.Scatter(
                x=bullish["month"],
                y=bullish["slow"],
                mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                fill="tonexty",
                fillcolor="rgba(22, 163, 74, 0.15)",
                name="Bullish (3m > 12m)",
            )
        )

    fig5.update_layout(
        yaxis_title="Median price ($)",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
        ),
        margin=dict(t=30),
    )
    st.plotly_chart(fig5, use_container_width=True)

    # current signal
    if len(mom) >= 12:
        current_fast = mom["fast"].iloc[-1]
        current_slow = mom["slow"].iloc[-1]
        signal = "🟢 **Bullish**" if current_fast >= current_slow else "🔴 **Bearish**"
        gap_pct = ((current_fast / current_slow) - 1) * 100
        st.info(
            f"**Current signal: {signal}** — "
            f"3-month median is {fmt_pct(gap_pct)} "
            f"{'above' if gap_pct >= 0 else 'below'} the 12-month median. "
            "A widening gap reinforces the trend; convergence suggests a "
            "potential reversal."
        )

# ================================================================
# 6. Supply heatmap — transactions per town per quarter (recent 5yr)
# ================================================================
with tab_supply:
    st.subheader("Transaction supply heatmap (recent 5 years)")
    st.caption(
        "Shows the number of transactions per town per quarter for the "
        "most recent 5 years. Brighter cells = higher liquidity. "
        "Useful for identifying where supply is abundant or scarce."
    )

    cutoff_year = latest_year - 4  # 5 years inclusive
    df_recent = df[df["year"] >= cutoff_year].copy()
    df_recent["quarter"] = (
        df_recent["month"].dt.to_period("Q").astype(str)
    )

    supply = (
        df_recent.groupby(["town", "quarter"])
        .agg(txn=("resale_price", "size"))
        .reset_index()
    )
    supply_pivot = supply.pivot(index="town", columns="quarter", values="txn").fillna(0)
    supply_pivot = supply_pivot.sort_index()

    fig6 = px.imshow(
        supply_pivot.values,
        x=supply_pivot.columns.tolist(),
        y=supply_pivot.index.tolist(),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels=dict(x="Quarter", y="Town", color="Transactions"),
    )
    fig6.update_layout(
        height=max(550, len(supply_pivot) * 24),
        margin=dict(t=30),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig6, use_container_width=True)

    # summary insight
    top_supply = supply.groupby("town")["txn"].sum().nlargest(3)
    towns_str = ", ".join(top_supply.index.tolist())
    st.info(
        f"**Highest liquidity towns (last 5 years):** {towns_str}. "
        "These towns offer the widest selection and fastest turnover — "
        "good for buyers who want options and sellers who want speed."
    )

# ================================================================
# 7. High-value segment growth (premiumization)
# ================================================================
with tab_premium:
    st.subheader("High-value segment growth")
    st.caption(
        "Percentage of annual transactions exceeding key price thresholds. "
        "A rising share of $500K+, $750K+, and $1M+ sales indicates "
        "market premiumization."
    )

    thresholds = [
        (500_000, "$500K+", "#93c5fd"),
        (750_000, "$750K+", "#3b82f6"),
        (1_000_000, "$1M+", "#1e3a5f"),
    ]

    annual_total = df.groupby("year").size().reset_index(name="total")
    premium_frames = []
    for thresh, label, _color in thresholds:
        above = (
            df[df["resale_price"] >= thresh]
            .groupby("year")
            .size()
            .reset_index(name="count")
        )
        above = above.merge(annual_total, on="year")
        above["pct"] = above["count"] / above["total"] * 100
        above["segment"] = label
        premium_frames.append(above[["year", "pct", "segment"]])

    premium = pd.concat(premium_frames, ignore_index=True)
    premium = premium.sort_values(["year", "segment"])

    color_map = {label: color for _, label, color in thresholds}

    fig7 = px.area(
        premium,
        x="year",
        y="pct",
        color="segment",
        labels={"pct": "% of transactions", "year": "", "segment": "Segment"},
        color_discrete_map=color_map,
    )
    fig7.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
        ),
        margin=dict(t=30),
    )
    st.plotly_chart(fig7, use_container_width=True)

    # latest year breakdown
    latest_prem = premium[premium["year"] == latest_year]
    if not latest_prem.empty:
        parts = []
        for _, row in latest_prem.iterrows():
            parts.append(f"{row['segment']}: {row['pct']:.1f}%")
        st.info(
            f"**{latest_year} breakdown:** " + " · ".join(parts) + ". "
            "The steady rise in high-value transactions reflects "
            "upgrading demand, newer premium flats entering the resale "
            "market, and overall price appreciation."
        )
    else:
        st.info("Insufficient data for the latest year breakdown.")

# ── footer ───────────────────────────────────────────────────────
st.divider()
st.caption(
    "💡 **Reading guide:** Momentum signals are best used alongside "
    "volume trends. A bullish price crossover on declining volume "
    "may lack conviction. Combine these indicators with town-level "
    "analysis (Pages 3 & 5) for a complete picture before making "
    "buying decisions."
)
