"""
Page 11 -- Smart Money Value Finder
=====================================
Five value-analysis tools for Singapore HDB resale buyers:
  Tab 1  Undervalued Areas      -- blocks cheaper than 1km neighbourhood
  Tab 2  Comps Finder           -- recent comparable transactions
  Tab 3  Percentile Pricer      -- price distribution + percentile lookup
  Tab 4  Floor Premium Validator -- storey-band premium analysis
  Tab 5  MRT Proximity Premium  -- price lift near MRT stations
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
    fmt_price,
    fmt_pct,
    TOWN_CENTROIDS,
    storey_band,
    COORDS_CSV,
)

# ── page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Value Finder",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Smart Money Value Finder")
st.caption(
    "Five analytical lenses to identify undervalued opportunities, "
    "validate pricing, and understand floor / MRT premiums in the "
    "Singapore HDB resale market."
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── shared data loaders ───────────────────────────────────────────────

@st.cache_data
def load_resale():
    """Load full cleaned resale dataset."""
    return load_clean()


@st.cache_data
def load_coords():
    """Load block geocoordinates; returns empty DataFrame if file absent."""
    if os.path.exists(COORDS_CSV):
        df = pd.read_csv(COORDS_CSV)
        df["block"] = df["block"].astype(str)
        return df
    return pd.DataFrame(columns=["block", "street_name", "lat", "lon"])


@st.cache_data
def load_resale_with_coords():
    """Resale data joined with block lat/lon."""
    df = load_resale()
    coords = load_coords()
    if coords.empty:
        return df
    df["block"] = df["block"].astype(str)
    return df.merge(coords, on=["block", "street_name"], how="left")


# ── haversine ─────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fast_haversine_vec(lat, lon, ref_lat, ref_lon):
    """Vectorised flat-earth approximation (km); good for <50 km ranges."""
    dlat = (lat - ref_lat) * 111.0
    dlon = (lon - ref_lon) * 111.0 * math.cos(math.radians(ref_lat))
    return np.sqrt(dlat ** 2 + dlon ** 2)


# ── storey midpoint parser ────────────────────────────────────────────

def parse_storey_mid(storey_text: str) -> float:
    """Parse '07 TO 09' → 8.0; also handles plain numbers."""
    s = storey_text.strip()
    if "TO" in s.upper():
        parts = s.upper().split("TO")
        try:
            lo = float(parts[0].strip())
            hi = float(parts[1].strip())
            return (lo + hi) / 2.0
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ── storey band order ─────────────────────────────────────────────────
STOREY_BAND_ORDER = ["01-03", "04-06", "07-09", "10-12",
                     "13-15", "16-21", "22-30", "31+"]

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Undervalued Areas",
    "🔍 Comps Finder",
    "📊 Percentile Pricer",
    "📐 Floor Premium Validator",
    "🚇 MRT Proximity Premium",
])


# ═════════════════════════════════════════════════════════════════════
# TAB 1 — Undervalued Areas
# ═════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🗺️ Undervalued Areas — Blocks cheaper than their neighbourhood")

    # ── sidebar controls ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🗺️ Undervalued Areas")
        all_types = sorted(load_resale()["flat_type"].dropna().unique().tolist())
        sel_types = st.multiselect(
            "Flat type(s)",
            options=all_types,
            default=all_types,
            key="uv_flat_types",
        )
        min_txn = st.slider(
            "Min. transactions per block",
            min_value=5,
            max_value=20,
            value=5,
            step=1,
            key="uv_min_txn",
        )

    coords = load_coords()
    coords_available = not coords.empty

    if not coords_available:
        st.warning(
            "⚠️ `data/address_coords.csv` not found. "
            "Block-level coordinates are required for this analysis."
        )
    else:
        df_all = load_resale()

        # Filter to last 24 months
        max_month = df_all["month"].max()
        cutoff = max_month - pd.DateOffset(months=24)
        df_24 = df_all[df_all["month"] >= cutoff].copy()

        if sel_types:
            df_24 = df_24[df_24["flat_type"].isin(sel_types)]

        # Join coordinates
        df_24["block"] = df_24["block"].astype(str)
        df_24 = df_24.merge(coords, on=["block", "street_name"], how="left")
        df_24 = df_24.dropna(subset=["lat", "lon", "price_per_sqm"])

        if df_24.empty:
            st.info("No geocoded transactions for the selected flat types in the last 24 months.")
        else:
            # Block-level aggregates
            block_agg = (
                df_24.groupby(["block", "street_name", "town", "flat_type", "lat", "lon"])
                .agg(
                    median_psm=("price_per_sqm", "median"),
                    txn_count=("price_per_sqm", "size"),
                )
                .reset_index()
            )
            block_agg = block_agg[block_agg["txn_count"] >= min_txn].copy()

            if block_agg.empty:
                st.info("No blocks meet the minimum transaction threshold. Try lowering the slider.")
            else:
                # For each block compute neighbourhood median PSM (all OTHER blocks within 1 km)
                lats = block_agg["lat"].values
                lons = block_agg["lon"].values
                medians = block_agg["median_psm"].values
                txns = block_agg["txn_count"].values

                nbr_psm = np.full(len(block_agg), np.nan)
                for i in range(len(block_agg)):
                    ref_lat, ref_lon = lats[i], lons[i]
                    dists = fast_haversine_vec(lats, lons, ref_lat, ref_lon)
                    mask = (dists <= 1.0) & (np.arange(len(block_agg)) != i)
                    nbr_vals = medians[mask]
                    if len(nbr_vals) > 0:
                        nbr_psm[i] = np.median(nbr_vals)

                block_agg["nbr_median_psm"] = nbr_psm
                block_agg = block_agg.dropna(subset=["nbr_median_psm"])
                block_agg["discount_pct"] = (
                    (block_agg["median_psm"] - block_agg["nbr_median_psm"])
                    / block_agg["nbr_median_psm"] * 100
                )

                # Classify
                def classify(d):
                    if d < -12:
                        return "Undervalued"
                    elif d > 12:
                        return "Above-market"
                    return "Normal"

                block_agg["status"] = block_agg["discount_pct"].apply(classify)

                color_map = {
                    "Undervalued":   [0, 200, 80, 200],
                    "Normal":        [150, 150, 150, 140],
                    "Above-market":  [220, 50, 50, 200],
                }
                block_agg["color"] = block_agg["status"].map(color_map)
                block_agg["radius"] = np.clip(block_agg["txn_count"] * 6, 30, 300)

                # ── Map ──────────────────────────────────────────────
                st.markdown("#### Price Status Map (last 24 months)")
                st.caption("🟢 Undervalued (>12% below neighbourhood)  |  ⚫ Normal  |  🔴 Above-market (>12% above)")

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=block_agg,
                    get_position=["lon", "lat"],
                    get_fill_color="color",
                    get_radius="radius",
                    radius_scale=1,
                    pickable=True,
                )
                view_state = pdk.ViewState(
                    latitude=1.3521,
                    longitude=103.8198,
                    zoom=11,
                    pitch=0,
                )
                tooltip = {
                    "html": (
                        "<b>{block} {street_name}</b><br/>"
                        "Town: {town}<br/>"
                        "Median PSM: ${median_psm}<br/>"
                        "Neighbourhood PSM: ${nbr_median_psm}<br/>"
                        "Discount: {discount_pct:.1f}%<br/>"
                        "Transactions: {txn_count}"
                    ),
                    "style": {"backgroundColor": "steelblue", "color": "white"},
                }
                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                )
                st.pydeck_chart(deck)

                # ── Top 30 undervalued table ─────────────────────────
                undervalued = (
                    block_agg[block_agg["status"] == "Undervalued"]
                    .sort_values("discount_pct")
                    .head(30)
                )

                st.markdown(f"#### Top Undervalued Blocks ({len(undervalued)} shown, top 30)")
                if undervalued.empty:
                    st.info("No blocks are more than 12% below their neighbourhood median PSM.")
                else:
                    display_uv = undervalued[[
                        "block", "street_name", "town", "flat_type",
                        "median_psm", "nbr_median_psm", "discount_pct", "txn_count"
                    ]].copy()
                    display_uv.columns = [
                        "Block", "Street", "Town", "Flat Type",
                        "Median PSM ($)", "Neighbourhood PSM ($)", "Discount (%)", "Transactions"
                    ]
                    display_uv["Median PSM ($)"] = display_uv["Median PSM ($)"].round(0).astype(int)
                    display_uv["Neighbourhood PSM ($)"] = display_uv["Neighbourhood PSM ($)"].round(0).astype(int)
                    display_uv["Discount (%)"] = display_uv["Discount (%)"].round(1)
                    st.dataframe(display_uv, use_container_width=True, hide_index=True)

                st.info(
                    "**DATA CONFIDENCE: Medium** — Blocks flagged as undervalued are cheaper "
                    "than their 1 km neighbourhood median PSM for the selected flat type(s) "
                    "in the last 24 months. This does **not** mean they are mispriced — "
                    "possible explanations include: lower floor, unfavourable facing, "
                    "proximity to road/industrial noise, past incidents, older flat model, "
                    "or simply less marketing. **Always verify on the ground.**"
                )


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — Comps Finder
# ═════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔍 Comps Finder — Recent comparable transactions")

    df_all2 = load_resale()
    towns2 = sorted(df_all2["town"].dropna().unique().tolist())
    flat_types2 = sorted(df_all2["flat_type"].dropna().unique().tolist())

    col_a, col_b = st.columns(2)
    with col_a:
        sel_town2 = st.selectbox("Town", towns2, key="comps_town")
        sel_ftype2 = st.selectbox("Flat type", flat_types2, key="comps_ftype")
        storey_input = st.text_input(
            "Storey range (e.g. 07 TO 09)",
            value="07 TO 09",
            key="comps_storey",
        )
        user_area = st.number_input(
            "Floor area (sqm)",
            min_value=20.0,
            max_value=300.0,
            value=90.0,
            step=1.0,
            key="comps_area",
        )
    with col_b:
        user_block = st.text_input(
            "Block (optional — for reference only)",
            value="",
            key="comps_block",
        )
        year_range2 = st.select_slider(
            "Look-back period",
            options=[1, 2, 3, 5],
            value=2,
            key="comps_years",
            format_func=lambda x: f"Last {x} year{'s' if x > 1 else ''}",
        )
        user_price = st.number_input(
            "Your target price (optional, for percentile)",
            min_value=0,
            max_value=5_000_000,
            value=0,
            step=10_000,
            key="comps_price",
        )

    user_storey_mid = parse_storey_mid(storey_input)

    max_month2 = df_all2["month"].max()
    cutoff2 = max_month2 - pd.DateOffset(months=int(year_range2 * 12))
    df_f2 = df_all2[
        (df_all2["month"] >= cutoff2)
        & (df_all2["town"] == sel_town2)
        & (df_all2["flat_type"] == sel_ftype2)
    ].copy()

    if df_f2.empty:
        st.info("No transactions found for the selected filters.")
    else:
        # Similarity scoring
        if not math.isnan(user_storey_mid) and "storey_mid" in df_f2.columns:
            df_f2["storey_score"] = (df_f2["storey_mid"] - user_storey_mid).abs()
        else:
            df_f2["storey_score"] = 0.0

        df_f2["area_score"] = (df_f2["floor_area_sqm"] - user_area).abs()
        df_f2["similarity"] = (
            df_f2["storey_score"] * 0.4 + df_f2["area_score"] * 0.6
        )

        top20 = df_f2.nsmallest(20, "similarity").sort_values("month", ascending=False)

        st.markdown(f"#### Top 20 Most Similar Transactions ({len(df_f2):,} total matches)")
        display_cols = {
            "block": "Block",
            "street_name": "Street",
            "storey_range": "Storey",
            "floor_area_sqm": "Area (sqm)",
            "remaining_lease_yrs": "Lease Remaining (yrs)",
            "resale_price": "Price ($)",
            "price_per_sqm": "PSM ($)",
            "month": "Date",
        }
        avail_cols = [c for c in display_cols if c in top20.columns]
        top20_disp = top20[avail_cols].rename(columns=display_cols).copy()
        if "Price ($)" in top20_disp.columns:
            top20_disp["Price ($)"] = top20_disp["Price ($)"].apply(lambda x: f"${x:,.0f}")
        if "PSM ($)" in top20_disp.columns:
            top20_disp["PSM ($)"] = top20_disp["PSM ($)"].apply(lambda x: f"${x:,.0f}")
        if "Date" in top20_disp.columns:
            top20_disp["Date"] = top20_disp["Date"].dt.strftime("%Y-%m")
        st.dataframe(top20_disp, use_container_width=True, hide_index=True)

        # Histogram
        st.markdown("#### Price Distribution of Filtered Comparables")
        fig_hist = px.histogram(
            df_f2,
            x="resale_price",
            nbins=50,
            labels={"resale_price": "Resale Price ($)"},
            color_discrete_sequence=["#4C78A8"],
        )
        fig_hist.update_layout(showlegend=False, yaxis_title="Count")

        if user_price > 0:
            fig_hist.add_vline(
                x=user_price,
                line_dash="dash",
                line_color="crimson",
                annotation_text="Your price",
                annotation_position="top right",
            )
            pct = (df_f2["resale_price"] < user_price).mean() * 100
            st.plotly_chart(fig_hist, use_container_width=True)
            st.metric(
                label="Your price percentile",
                value=f"{pct:.1f}th percentile",
                help=f"Based on {len(df_f2):,} recent comparables.",
            )
            st.caption(
                f"Based on **{len(df_f2):,}** recent comparables, "
                f"your price of **{fmt_price(user_price)}** is at the "
                f"**{pct:.1f}th percentile**."
            )
        else:
            st.plotly_chart(fig_hist, use_container_width=True)

        st.success(
            "**DATA CONFIDENCE: High** — Results are drawn directly from "
            "official HDB resale transaction records."
        )


# ═════════════════════════════════════════════════════════════════════
# TAB 3 — Percentile Pricer
# ═════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Percentile Pricer — Full price distribution")

    df_all3 = load_resale()
    towns3 = sorted(df_all3["town"].dropna().unique().tolist())
    flat_types3 = sorted(df_all3["flat_type"].dropna().unique().tolist())

    col3a, col3b, col3c = st.columns(3)
    with col3a:
        sel_town3 = st.selectbox("Town", towns3, key="pp_town")
    with col3b:
        sel_ftype3 = st.selectbox("Flat type", flat_types3, key="pp_ftype")
    with col3c:
        yr_range3 = st.slider(
            "Years of history",
            min_value=1,
            max_value=10,
            value=3,
            key="pp_years",
        )

    max_month3 = df_all3["month"].max()
    cutoff3 = max_month3 - pd.DateOffset(months=int(yr_range3 * 12))
    df_f3 = df_all3[
        (df_all3["month"] >= cutoff3)
        & (df_all3["town"] == sel_town3)
        & (df_all3["flat_type"] == sel_ftype3)
    ].copy()

    if df_f3.empty:
        st.info("No data for the selected filters.")
    else:
        q1 = df_f3["resale_price"].quantile(0.25)
        q2 = df_f3["resale_price"].quantile(0.50)
        q3 = df_f3["resale_price"].quantile(0.75)
        iqr = q3 - q1

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median Price", fmt_price(q2))
        m2.metric("25th Pct", fmt_price(q1))
        m3.metric("75th Pct", fmt_price(q3))
        m4.metric("IQR", fmt_price(iqr))

        # Violin + box
        fig_viol = go.Figure()
        fig_viol.add_trace(go.Violin(
            y=df_f3["resale_price"],
            name=f"{sel_town3} – {sel_ftype3}",
            box_visible=True,
            meanline_visible=True,
            fillcolor="#4C78A8",
            opacity=0.7,
            line_color="#2c4f7c",
        ))
        fig_viol.update_layout(
            yaxis_title="Resale Price ($)",
            xaxis_title="",
            showlegend=False,
            height=450,
        )
        st.plotly_chart(fig_viol, use_container_width=True)

        # Percentile lookup
        st.markdown("#### Percentile Lookup")
        lookup_price = st.number_input(
            "Enter a price to see its percentile",
            min_value=int(df_f3["resale_price"].min()),
            max_value=int(df_f3["resale_price"].max()),
            value=int(q2),
            step=10_000,
            key="pp_lookup",
        )
        pct3 = (df_f3["resale_price"] < lookup_price).mean() * 100
        st.info(
            f"A price of **{fmt_price(lookup_price)}** falls at the "
            f"**{pct3:.1f}th percentile** of {len(df_f3):,} transactions "
            f"for {sel_ftype3} flats in {sel_town3} over the last {yr_range3} year(s)."
        )


# ═════════════════════════════════════════════════════════════════════
# TAB 4 — Floor Premium Validator
# ═════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📐 Floor Premium Validator — Storey-band PSM analysis")

    df_all4 = load_resale()
    towns4 = sorted(df_all4["town"].dropna().unique().tolist())
    flat_types4 = sorted(df_all4["flat_type"].dropna().unique().tolist())

    col4a, col4b = st.columns(2)
    with col4a:
        sel_town4 = st.selectbox("Town", towns4, key="fp_town")
    with col4b:
        sel_ftype4 = st.selectbox("Flat type", flat_types4, key="fp_ftype")

    df_f4 = df_all4[
        (df_all4["town"] == sel_town4)
        & (df_all4["flat_type"] == sel_ftype4)
    ].copy()

    if df_f4.empty:
        st.info("No data for the selected filters.")
    else:
        # Assign storey bands
        if "storey_mid" in df_f4.columns:
            df_f4["storey_band"] = df_f4["storey_mid"].apply(storey_band)
        else:
            st.warning("storey_mid column not found in dataset.")
            st.stop()

        band_agg = (
            df_f4.groupby("storey_band")
            .agg(median_psm=("price_per_sqm", "median"), count=("price_per_sqm", "size"))
            .reindex(STOREY_BAND_ORDER)
            .dropna(subset=["median_psm"])
            .reset_index()
        )

        if band_agg.empty or "01-03" not in band_agg["storey_band"].values:
            st.info("Insufficient data across storey bands for this selection.")
        else:
            ground_psm = band_agg.loc[
                band_agg["storey_band"] == "01-03", "median_psm"
            ].values[0]

            band_agg["premium_pct"] = (
                (band_agg["median_psm"] - ground_psm) / ground_psm * 100
            )

            # Bar + line chart
            fig_fp = go.Figure()
            fig_fp.add_trace(go.Bar(
                x=band_agg["storey_band"],
                y=band_agg["median_psm"],
                name="Median PSM ($)",
                marker_color="#4C78A8",
                yaxis="y1",
            ))
            fig_fp.add_trace(go.Scatter(
                x=band_agg["storey_band"],
                y=band_agg["premium_pct"],
                name="Premium vs 01-03 (%)",
                mode="lines+markers",
                line=dict(color="crimson", width=2),
                marker=dict(size=7),
                yaxis="y2",
            ))
            fig_fp.update_layout(
                xaxis_title="Storey Band",
                yaxis=dict(title="Median PSM ($)", side="left"),
                yaxis2=dict(
                    title="Premium vs Ground Floor (%)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend=dict(orientation="h", y=1.1),
                height=420,
            )
            st.plotly_chart(fig_fp, use_container_width=True)

            # Callout
            top_band = band_agg.loc[band_agg["premium_pct"].idxmax()]
            st.info(
                f"**Data-justified floor premium** in **{sel_town4}** for "
                f"**{sel_ftype4}**: the highest storey band in the data "
                f"(**{top_band['storey_band']}**) commands a premium of "
                f"**{top_band['premium_pct']:.1f}%** over the ground floor (01-03)."
            )

            # User's own storey lookup
            st.markdown("#### Your flat's storey premium")
            user_storey4 = st.number_input(
                "Enter your flat's storey number",
                min_value=1,
                max_value=60,
                value=8,
                key="fp_storey",
            )
            user_band4 = storey_band(float(user_storey4))
            row4 = band_agg[band_agg["storey_band"] == user_band4]
            if row4.empty:
                st.warning(f"No data for storey band **{user_band4}** in this selection.")
            else:
                prem4 = row4["premium_pct"].values[0]
                psm4 = row4["median_psm"].values[0]
                sign = "+" if prem4 >= 0 else ""
                st.success(
                    f"Storey **{user_storey4}** → band **{user_band4}** → "
                    f"median PSM **{fmt_price(psm4)}** "
                    f"({sign}{prem4:.1f}% vs ground floor)."
                )


# ═════════════════════════════════════════════════════════════════════
# TAB 5 — MRT Proximity Premium
# ═════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🚇 MRT Proximity Premium — Price lift near MRT stations")

    coords5 = load_coords()
    mrt5 = load_mrt()

    if coords5.empty:
        st.warning(
            "⚠️ `data/address_coords.csv` not found. "
            "Block coordinates are required for MRT proximity analysis."
        )
    elif mrt5.empty:
        st.warning("⚠️ `data/mrt_stations.csv` not found.")
    else:
        df_all5 = load_resale()

        # Filter to last 5 years
        max_month5 = df_all5["month"].max()
        cutoff5 = max_month5 - pd.DateOffset(months=60)
        df_5y = df_all5[df_all5["month"] >= cutoff5].copy()

        # MRT line filter
        all_lines = []
        if "lines" in mrt5.columns:
            for val in mrt5["lines"].dropna():
                all_lines.extend([x.strip() for x in str(val).split(",")])
        elif "line" in mrt5.columns:
            all_lines = mrt5["line"].dropna().unique().tolist()
        all_lines = sorted(set(all_lines))

        sel_line5 = st.selectbox(
            "Filter by MRT line (optional)",
            options=["All lines"] + all_lines,
            key="mrt_line",
        )

        if sel_line5 != "All lines":
            if "lines" in mrt5.columns:
                mrt_filt = mrt5[mrt5["lines"].str.contains(sel_line5, na=False)]
            else:
                mrt_filt = mrt5[mrt5["line"] == sel_line5]
        else:
            mrt_filt = mrt5

        if mrt_filt.empty:
            st.info("No MRT stations for the selected line.")
        else:
            # Build block-level lookup: nearest MRT distance
            @st.cache_data
            def compute_block_mrt_dist(mrt_subset_hash: str):
                """Compute nearest MRT distance per block; keyed by MRT subset hash."""
                mrt_arr = mrt_filt[["lat", "lon"]].values
                coords_arr = coords5[["lat", "lon"]].values
                min_dists = []
                for i in range(len(coords5)):
                    blat, blon = coords_arr[i]
                    best = float("inf")
                    for j in range(len(mrt_arr)):
                        d = haversine_km(blat, blon, mrt_arr[j, 0], mrt_arr[j, 1])
                        if d < best:
                            best = d
                    min_dists.append(best)
                result = coords5[["block", "street_name", "lat", "lon"]].copy()
                result["dist_mrt_km"] = min_dists
                return result

            mrt_hash = sel_line5 + str(len(mrt_filt))
            block_dist = compute_block_mrt_dist(mrt_hash)

            # Join to resale
            df_5y["block"] = df_5y["block"].astype(str)
            df_mrt = df_5y.merge(block_dist[["block", "street_name", "dist_mrt_km"]],
                                  on=["block", "street_name"], how="left")
            df_mrt = df_mrt.dropna(subset=["dist_mrt_km", "price_per_sqm"])

            if df_mrt.empty:
                st.info("No geocoded transactions available for MRT proximity analysis.")
            else:
                # Distance bins
                bins = [0, 0.2, 0.5, 1.0, 2.0, float("inf")]
                labels5 = ["<200m", "200-500m", "500m-1km", "1-2km", ">2km"]
                df_mrt["dist_band"] = pd.cut(
                    df_mrt["dist_mrt_km"],
                    bins=bins,
                    labels=labels5,
                    right=True,
                )

                band_agg5 = (
                    df_mrt.groupby("dist_band", observed=True)
                    .agg(
                        median_psm=("price_per_sqm", "median"),
                        count=("price_per_sqm", "size"),
                    )
                    .reindex(labels5)
                    .reset_index()
                )

                base_psm5 = band_agg5.loc[
                    band_agg5["dist_band"] == ">2km", "median_psm"
                ].values
                base_val5 = base_psm5[0] if len(base_psm5) > 0 and not np.isnan(base_psm5[0]) else None

                if base_val5 is not None:
                    band_agg5["premium_pct"] = (
                        (band_agg5["median_psm"] - base_val5) / base_val5 * 100
                    )
                else:
                    band_agg5["premium_pct"] = np.nan

                # Chart
                fig_mrt = go.Figure()
                fig_mrt.add_trace(go.Bar(
                    x=band_agg5["dist_band"].astype(str),
                    y=band_agg5["median_psm"],
                    name="Median PSM ($)",
                    marker_color="#4C78A8",
                    yaxis="y1",
                ))
                if base_val5 is not None:
                    fig_mrt.add_trace(go.Scatter(
                        x=band_agg5["dist_band"].astype(str),
                        y=band_agg5["premium_pct"],
                        name="Premium vs >2km (%)",
                        mode="lines+markers",
                        line=dict(color="orange", width=2),
                        marker=dict(size=8),
                        yaxis="y2",
                    ))
                fig_mrt.update_layout(
                    xaxis_title="Distance to Nearest MRT",
                    yaxis=dict(title="Median PSM ($)", side="left"),
                    yaxis2=dict(
                        title="Premium vs >2km (%)",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                    legend=dict(orientation="h", y=1.1),
                    height=420,
                )
                line_label = sel_line5 if sel_line5 != "All lines" else "all lines"
                st.markdown(f"#### PSM by MRT Distance Band — {line_label} (last 5 years)")
                st.plotly_chart(fig_mrt, use_container_width=True)

                # Summary table
                st.markdown("#### Summary Table")
                disp5 = band_agg5.copy()
                disp5.columns = ["Distance Band", "Median PSM ($)", "Transactions", "Premium vs >2km (%)"]
                disp5["Median PSM ($)"] = disp5["Median PSM ($)"].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                )
                disp5["Premium vs >2km (%)"] = disp5["Premium vs >2km (%)"].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
                )
                st.dataframe(disp5, use_container_width=True, hide_index=True)

                st.warning(
                    "**DATA CONFIDENCE: Medium** — Distance is straight-line (as the crow "
                    "flies), not actual walking distance. Proximity to MRT is a proxy for "
                    "many factors including walkability, noise, and neighbourhood amenities. "
                    "Results may also reflect town-level price differences rather than "
                    "pure MRT premiums."
                )
