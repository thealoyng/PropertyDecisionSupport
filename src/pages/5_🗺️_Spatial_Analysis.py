"""
Page 5 -- Spatial Analysis
===========================
Geographic and town-level analysis of Singapore HDB resale data.
Includes bubble maps, town rankings, price convergence/divergence,
MRT proximity analysis, and price variation along MRT lines.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from eda_helpers import (
    load_clean,
    load_mrt,
    TOWN_CENTROIDS,
    LINE_COLORS,
    LINE_NAMES,
    fmt_price,
)

# -- page config ------------------------------------------------------
st.set_page_config(
    page_title="Spatial Analysis",
    page_icon="\U0001f5fa\ufe0f",
    layout="wide",
)

st.title("\U0001f5fa\ufe0f Spatial Analysis")
st.caption(
    "Geographic and town-level analysis of HDB resale prices -- "
    "bubble maps, town rankings, MRT proximity, and price variation along rail lines."
)

# -- load data ---------------------------------------------------------
df = load_clean()
mrt_df = load_mrt()


# -- helper: haversine distance (km) ----------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Return distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# -- pre-compute town-level aggregates --------------------------------
town_year = (
    df.groupby(["town", "year"])
    .agg(
        median_price=("resale_price", "median"),
        median_psm=("price_per_sqm", "median"),
        txn_count=("resale_price", "size"),
    )
    .reset_index()
)

town_all = (
    df.groupby("town")
    .agg(
        median_price=("resale_price", "median"),
        median_psm=("price_per_sqm", "median"),
        txn_count=("resale_price", "size"),
    )
    .reset_index()
)

latest_year = int(df["year"].max())

# Recent 3-year growth per town
recent_years = sorted(df["year"].unique())
recent_3 = [y for y in recent_years if y >= latest_year - 2]
if len(recent_3) >= 2:
    first_yr, last_yr = min(recent_3), max(recent_3)
    growth_start = town_year[town_year["year"] == first_yr][["town", "median_price"]].rename(
        columns={"median_price": "price_start"}
    )
    growth_end = town_year[town_year["year"] == last_yr][["town", "median_price"]].rename(
        columns={"median_price": "price_end"}
    )
    growth_df = growth_start.merge(growth_end, on="town", how="inner")
    growth_df["growth_pct"] = (growth_df["price_end"] / growth_df["price_start"] - 1) * 100
else:
    growth_df = pd.DataFrame(columns=["town", "growth_pct"])

# -- KPI metric cards -------------------------------------------------
most_expensive = town_all.loc[town_all["median_psm"].idxmax()]
cheapest = town_all.loc[town_all["median_psm"].idxmin()]
most_active = town_all.loc[town_all["txn_count"].idxmax()]

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Most expensive town",
    most_expensive["town"].title(),
    f"{fmt_price(most_expensive['median_psm'])}/sqm",
    help="Town with the highest overall median price per sqm.",
)
k2.metric(
    "Cheapest town",
    cheapest["town"].title(),
    f"{fmt_price(cheapest['median_psm'])}/sqm",
    help="Town with the lowest overall median price per sqm.",
)
if not growth_df.empty:
    top_growth = growth_df.loc[growth_df["growth_pct"].idxmax()]
    k3.metric(
        f"Highest growth ({first_yr}-{last_yr})",
        top_growth["town"].title(),
        f"{top_growth['growth_pct']:+.1f}%",
        help="Town with the largest 3-year median price appreciation.",
    )
else:
    k3.metric("Highest growth", "N/A", "")

k4.metric(
    "Most active town",
    most_active["town"].title(),
    f"{most_active['txn_count']:,.0f} txns",
    help="Town with the highest total transaction count across all years.",
)

st.divider()

# -- tabs --------------------------------------------------------------
(
    tab_bubble,
    tab_bump,
    tab_converge,
    tab_heatmap,
    tab_hotcold,
    tab_mrt_prox,
    tab_mrt_line,
) = st.tabs(
    [
        "\U0001f4cd Bubble map",
        "\U0001f3c6 Town ranking",
        "\U0001f4c9 Convergence",
        "\U0001f525 Volume heatmap",
        "\U0001f321\ufe0f Hot vs Cold",
        "\U0001f687 MRT proximity",
        "\U0001f6e4\ufe0f Price along lines",
    ]
)

# ================================================================
# 1. Bubble map: median price per sqm by town (pydeck)
# ================================================================
with tab_bubble:
    st.subheader("Median price per sqm by town")
    st.caption(
        "Bubble **size** = transaction volume; **colour intensity** = median price/sqm. "
        "Use the year slider to filter."
    )

    year_range = st.slider(
        "Year range",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=(int(df["year"].min()), int(df["year"].max())),
        key="bubble_yr",
    )

    mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    bubble_data = (
        df[mask]
        .groupby("town")
        .agg(median_psm=("price_per_sqm", "median"), txn_count=("resale_price", "size"))
        .reset_index()
    )

    # attach coordinates
    bubble_data["lat"] = bubble_data["town"].map(lambda t: TOWN_CENTROIDS.get(t, (None, None))[0])
    bubble_data["lon"] = bubble_data["town"].map(lambda t: TOWN_CENTROIDS.get(t, (None, None))[1])
    bubble_data = bubble_data.dropna(subset=["lat", "lon"])

    # normalise for sizing and colour
    max_txn = bubble_data["txn_count"].max() if len(bubble_data) else 1
    min_psm = bubble_data["median_psm"].min() if len(bubble_data) else 0
    max_psm = bubble_data["median_psm"].max() if len(bubble_data) else 1
    psm_range = max_psm - min_psm if max_psm != min_psm else 1

    bubble_data["radius"] = (bubble_data["txn_count"] / max_txn) * 1800 + 200
    bubble_data["color_val"] = ((bubble_data["median_psm"] - min_psm) / psm_range).clip(0, 1)

    # colour: blue (cold) -> red (hot)
    bubble_data["r"] = (bubble_data["color_val"] * 220 + 35).astype(int)
    bubble_data["g"] = ((1 - bubble_data["color_val"].abs()) * 80 + 40).astype(int)
    bubble_data["b"] = ((1 - bubble_data["color_val"]) * 220 + 35).astype(int)
    bubble_data["a"] = 180

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=bubble_data,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color=["r", "g", "b", "a"],
        pickable=True,
        auto_highlight=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=bubble_data,
        get_position=["lon", "lat"],
        get_text="town",
        get_size=11,
        get_color=[0, 0, 0, 200],
        get_angle=0,
        get_text_anchor='"middle"',
        get_alignment_baseline='"bottom"',
        get_pixel_offset=[0, -20],
    )

    view = pdk.ViewState(
        latitude=1.3521, longitude=103.8198, zoom=10.5, pitch=0
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer, text_layer],
            initial_view_state=view,
            tooltip={
                "html": (
                    "<b>{town}</b><br/>"
                    "Median $/sqm: ${median_psm}<br/>"
                    "Transactions: {txn_count}"
                ),
                "style": {"backgroundColor": "#1e293b", "color": "white"},
            },
        )
    )

# ================================================================
# 2. Town ranking bump chart
# ================================================================
with tab_bump:
    st.subheader("Town ranking by median resale price over time")
    st.caption(
        "Tracks how the top 10 towns (by latest-year median price) shift in rank each year. "
        "Lower rank number = more expensive."
    )

    # rank towns within each year
    rank_df = town_year.copy()
    rank_df["rank"] = rank_df.groupby("year")["median_price"].rank(
        ascending=False, method="min"
    )

    # pick top 10 towns by latest year rank
    latest_ranks = rank_df[rank_df["year"] == latest_year].nsmallest(10, "rank")
    top10_towns = latest_ranks["town"].tolist()

    bump = rank_df[rank_df["town"].isin(top10_towns)].copy()
    bump = bump.sort_values(["town", "year"])

    # assign consistent colours
    palette = px.colors.qualitative.D3[:10]
    town_color = {t: palette[i % len(palette)] for i, t in enumerate(top10_towns)}

    fig_bump = go.Figure()
    for town in top10_towns:
        td = bump[bump["town"] == town]
        fig_bump.add_trace(
            go.Scatter(
                x=td["year"],
                y=td["rank"],
                mode="lines+markers+text",
                name=town.title(),
                text=td["rank"].astype(int).astype(str),
                textposition="top center",
                textfont=dict(size=9),
                line=dict(color=town_color[town], width=2),
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{town.title()}</b><br>"
                    "Year: %{x}<br>"
                    "Rank: %{y}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig_bump.update_layout(
        yaxis=dict(
            title="Rank (1 = most expensive)",
            autorange="reversed",
            dtick=1,
        ),
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40),
        height=550,
    )
    st.plotly_chart(fig_bump, use_container_width=True)

# ================================================================
# 3. Price convergence / divergence
# ================================================================
with tab_converge:
    st.subheader("Price convergence / divergence across towns")
    st.caption(
        "Shows the spread between the 90th-percentile town and 10th-percentile town "
        "(by yearly median price). A widening gap signals divergence."
    )

    yearly_town_medians = town_year.groupby("year")["median_price"].agg(
        p90=lambda x: np.percentile(x, 90),
        p10=lambda x: np.percentile(x, 10),
        median="median",
    ).reset_index()
    yearly_town_medians["spread"] = yearly_town_medians["p90"] - yearly_town_medians["p10"]

    fig_conv = go.Figure()
    fig_conv.add_trace(
        go.Scatter(
            x=yearly_town_medians["year"],
            y=yearly_town_medians["p90"],
            mode="lines",
            name="P90 town median",
            line=dict(color="#ef4444", width=1.5, dash="dash"),
        )
    )
    fig_conv.add_trace(
        go.Scatter(
            x=yearly_town_medians["year"],
            y=yearly_town_medians["p10"],
            mode="lines",
            name="P10 town median",
            line=dict(color="#3b82f6", width=1.5, dash="dash"),
        )
    )
    fig_conv.add_trace(
        go.Scatter(
            x=yearly_town_medians["year"],
            y=yearly_town_medians["spread"],
            mode="lines+markers",
            name="P90 - P10 spread",
            line=dict(color="#8b5cf6", width=2.5),
            marker=dict(size=4),
            yaxis="y2",
        )
    )
    fig_conv.update_layout(
        yaxis=dict(title="Median price ($)"),
        yaxis2=dict(
            title="Spread ($)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40),
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # interpretation
    first_spread = yearly_town_medians["spread"].iloc[0]
    last_spread = yearly_town_medians["spread"].iloc[-1]
    if last_spread > first_spread * 1.1:
        st.info(
            f"The spread has **widened** from {fmt_price(first_spread)} to "
            f"{fmt_price(last_spread)}, suggesting **divergence** -- the gap between "
            "the most and least expensive towns is growing."
        )
    elif last_spread < first_spread * 0.9:
        st.info(
            f"The spread has **narrowed** from {fmt_price(first_spread)} to "
            f"{fmt_price(last_spread)}, suggesting **convergence** -- towns are "
            "becoming more similar in price."
        )
    else:
        st.info(
            f"The spread has remained relatively stable "
            f"({fmt_price(first_spread)} -> {fmt_price(last_spread)})."
        )

# ================================================================
# 4. Town-level transaction volume heatmap (town x year)
# ================================================================
with tab_heatmap:
    st.subheader("Transaction volume heatmap -- town x year")
    st.caption("Darker cells = higher transaction count for that town-year combination.")

    vol_pivot = town_year.pivot(index="town", columns="year", values="txn_count").fillna(0)
    # sort towns by total volume descending
    vol_pivot = vol_pivot.loc[vol_pivot.sum(axis=1).sort_values(ascending=False).index]

    fig_heat = px.imshow(
        vol_pivot.values,
        x=[str(int(c)) for c in vol_pivot.columns],
        y=vol_pivot.index.tolist(),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels=dict(x="Year", y="Town", color="Transactions"),
    )
    fig_heat.update_layout(
        height=max(500, len(vol_pivot) * 22),
        margin=dict(t=30),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ================================================================
# 5. Hot vs Cold towns (recent 3-year growth)
# ================================================================
with tab_hotcold:
    st.subheader(f"Hot vs Cold towns -- median price growth ({first_yr}-{last_yr})")
    st.caption(
        "Year-over-year median price growth computed over the most recent 3 years. "
        "Green = above-average growth (hot); red = below-average growth (cold)."
    )

    if not growth_df.empty:
        growth_sorted = growth_df.sort_values("growth_pct", ascending=True).copy()
        avg_growth = growth_sorted["growth_pct"].mean()
        growth_sorted["status"] = np.where(
            growth_sorted["growth_pct"] >= avg_growth, "Hot", "Cold"
        )
        growth_sorted["color"] = np.where(
            growth_sorted["growth_pct"] >= avg_growth, "#16a34a", "#dc2626"
        )
        growth_sorted["town_title"] = growth_sorted["town"].str.title()

        fig_hc = go.Figure(
            go.Bar(
                x=growth_sorted["growth_pct"],
                y=growth_sorted["town_title"],
                orientation="h",
                marker_color=growth_sorted["color"],
                text=growth_sorted["growth_pct"].apply(lambda v: f"{v:+.1f}%"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Growth: %{x:+.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )
        fig_hc.add_vline(
            x=avg_growth,
            line_dash="dash",
            line_color="#6366f1",
            annotation_text=f"Avg: {avg_growth:+.1f}%",
            annotation_position="top right",
        )
        fig_hc.update_layout(
            xaxis_title="Median price growth (%)",
            yaxis_title="",
            height=max(450, len(growth_sorted) * 24),
            margin=dict(t=30, l=120),
        )
        st.plotly_chart(fig_hc, use_container_width=True)
    else:
        st.warning("Insufficient data to compute recent growth.")

# ================================================================
# 6. MRT proximity analysis
# ================================================================
with tab_mrt_prox:
    st.subheader("MRT proximity vs median price per sqm")
    st.caption(
        "For each town, the nearest MRT station is identified using haversine distance "
        "from the town centroid. Scatter plot shows the relationship between proximity "
        "and price."
    )

    if len(mrt_df) > 0:
        # compute nearest MRT for each town
        prox_rows = []
        for town, (tlat, tlon) in TOWN_CENTROIDS.items():
            min_dist = float("inf")
            nearest_stn = ""
            for _, stn in mrt_df.iterrows():
                d = haversine(tlat, tlon, stn["lat"], stn["lon"])
                if d < min_dist:
                    min_dist = d
                    nearest_stn = stn["name"]
            prox_rows.append(
                {"town": town, "nearest_mrt": nearest_stn, "dist_km": round(min_dist, 2)}
            )
        prox = pd.DataFrame(prox_rows)

        # merge with town-level price data
        prox = prox.merge(town_all[["town", "median_psm", "txn_count"]], on="town", how="left")
        prox = prox.dropna(subset=["median_psm"])

        fig_prox = px.scatter(
            prox,
            x="dist_km",
            y="median_psm",
            size="txn_count",
            text="town",
            hover_data=["nearest_mrt", "txn_count"],
            labels={
                "dist_km": "Distance to nearest MRT (km)",
                "median_psm": "Median price per sqm ($)",
                "txn_count": "Transactions",
                "nearest_mrt": "Nearest MRT",
            },
        )
        fig_prox.update_traces(
            textposition="top center",
            textfont_size=9,
            marker=dict(opacity=0.7, line=dict(width=1, color="white")),
        )

        # add trend line manually
        if len(prox) > 2:
            z = np.polyfit(prox["dist_km"], prox["median_psm"], 1)
            x_line = np.linspace(prox["dist_km"].min(), prox["dist_km"].max(), 50)
            y_line = np.polyval(z, x_line)
            fig_prox.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Trend",
                    line=dict(color="#94a3b8", dash="dash", width=1.5),
                    showlegend=True,
                )
            )

        fig_prox.update_layout(
            hovermode="closest",
            margin=dict(t=30),
            height=550,
        )
        st.plotly_chart(fig_prox, use_container_width=True)

        # show correlation stat
        corr = prox["dist_km"].corr(prox["median_psm"])
        if corr < -0.3:
            st.info(
                f"Correlation: **{corr:.2f}** -- towns closer to MRT stations tend to "
                "have higher prices per sqm."
            )
        elif corr > 0.3:
            st.info(
                f"Correlation: **{corr:.2f}** -- interestingly, towns farther from MRT "
                "stations show higher prices per sqm (possibly newer estates)."
            )
        else:
            st.info(
                f"Correlation: **{corr:.2f}** -- weak relationship between MRT proximity "
                "and price at the town level."
            )
    else:
        st.warning("MRT station data not available.")

# ================================================================
# 7. Price per sqm along each MRT line
# ================================================================
with tab_mrt_line:
    st.subheader("Price per sqm along MRT lines")
    st.caption(
        "For each MRT station, the nearest HDB town is identified and its median "
        "price/sqm is plotted. Follow the line to see how prices vary along the route."
    )

    if len(mrt_df) > 0:
        # for each MRT station, find nearest town
        stn_town_rows = []
        for _, stn in mrt_df.iterrows():
            min_dist = float("inf")
            nearest_town = ""
            for town, (tlat, tlon) in TOWN_CENTROIDS.items():
                d = haversine(stn["lat"], stn["lon"], tlat, tlon)
                if d < min_dist:
                    min_dist = d
                    nearest_town = town
            stn_town_rows.append(
                {
                    "station": stn["name"],
                    "line": stn["line"],
                    "lines": stn.get("lines", stn["line"]),
                    "town": nearest_town,
                    "dist_km": round(min_dist, 2),
                }
            )
        stn_town = pd.DataFrame(stn_town_rows)

        # merge median price per sqm from town_all
        stn_town = stn_town.merge(
            town_all[["town", "median_psm"]], on="town", how="left"
        )
        stn_town = stn_town.dropna(subset=["median_psm"])

        # one line chart per MRT line (using the primary 'line' column)
        selected_lines = st.multiselect(
            "MRT lines to show",
            sorted(stn_town["line"].unique()),
            default=sorted(stn_town["line"].unique()),
            format_func=lambda k: f"{k} ({LINE_NAMES.get(k, k)})",
            key="mrt_line_select",
        )

        if selected_lines:
            fig_line = go.Figure()
            for line_code in selected_lines:
                ld = stn_town[stn_town["line"] == line_code].copy()
                # keep station order as per CSV (roughly route order)
                ld = ld.reset_index(drop=True)
                fig_line.add_trace(
                    go.Scatter(
                        x=ld["station"],
                        y=ld["median_psm"],
                        mode="lines+markers",
                        name=f"{line_code} ({LINE_NAMES.get(line_code, line_code)})",
                        line=dict(
                            color=LINE_COLORS.get(line_code, "#888888"), width=2.5
                        ),
                        marker=dict(size=7),
                        text=ld["town"],
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Town: %{text}<br>"
                            "Median $/sqm: $%{y:,.0f}<br>"
                            "<extra>" + LINE_NAMES.get(line_code, line_code) + "</extra>"
                        ),
                    )
                )

            fig_line.update_layout(
                xaxis=dict(title="Station (in route order)", tickangle=-45),
                yaxis_title="Median price per sqm ($)",
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                ),
                margin=dict(t=40, b=120),
                height=550,
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # summary table
            with st.expander("Station-town mapping details"):
                display_cols = ["station", "line", "town", "median_psm", "dist_km"]
                st.dataframe(
                    stn_town[stn_town["line"].isin(selected_lines)][display_cols]
                    .rename(
                        columns={
                            "station": "Station",
                            "line": "Line",
                            "town": "Nearest town",
                            "median_psm": "Median $/sqm",
                            "dist_km": "Distance (km)",
                        }
                    )
                    .style.format({"Median $/sqm": "${:,.0f}", "Distance (km)": "{:.2f}"}),
                    use_container_width=True,
                )
        else:
            st.info("Select at least one MRT line to display.")
    else:
        st.warning("MRT station data not available.")
