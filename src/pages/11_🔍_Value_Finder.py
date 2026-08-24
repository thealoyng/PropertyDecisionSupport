"""
Page 11 -- Smart Money Value Finder
=====================================
Five value-analysis tools for Singapore property buyers:
  Tab 1  Undervalued Areas      -- blocks/projects cheaper than neighbourhood/district
  Tab 2  Comps Finder           -- recent comparable transactions
  Tab 3  Percentile Pricer      -- price distribution + percentile lookup
  Tab 4  Floor Premium Validator -- storey-band premium analysis
  Tab 5  MRT Proximity Premium  -- price lift near MRT stations

Supports both HDB Resale and Private (Condo) modes.
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
    load_condo_clean,
    DISTRICT_CENTROIDS,
    floor_range_mid,
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
    "Singapore property market."
)

mode = st.radio(
    "Property type",
    ["🏘️ HDB Resale", "🏢 Private (Condo)"],
    horizontal=True,
    key="vf_mode",
)
st.divider()

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


@st.cache_data
def load_private():
    """Load cleaned private condo transaction data."""
    return load_condo_clean()


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
    if mode == "🏘️ HDB Resale":
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

    else:
        # ── Private Mode — Undervalued Areas ─────────────────────────────
        st.subheader("🗺️ Undervalued Areas — Private projects vs district median")

        df_priv1 = load_private()
        if df_priv1.empty:
            st.warning("⚠️ Private condo data not available (condo_clean.csv not found).")
        else:
            # Filter last 24 months and Condo/Apartment + EC only
            max_cd1 = df_priv1["contract_date"].max()
            cutoff_priv1 = max_cd1 - pd.DateOffset(months=24)
            df_p1 = df_priv1[
                (df_priv1["contract_date"] >= cutoff_priv1)
                & (df_priv1["property_type_broad"].isin(["Condo/Apartment", "EC"]))
            ].copy()

            if df_p1.empty:
                st.info("No private Condo/Apartment or EC transactions in the last 24 months.")
            else:
                # Project-level aggregates
                proj_agg = (
                    df_p1.groupby("project")
                    .agg(
                        median_psm=("price_psm", "median"),
                        txn_count=("price_psm", "size"),
                        lat=("lat", "first"),
                        lon=("lon", "first"),
                        district=("district", "first"),
                    )
                    .reset_index()
                )

                # District-level median PSM
                dist_med1 = (
                    df_p1.groupby("district")["price_psm"]
                    .median()
                    .rename("district_median_psm")
                    .reset_index()
                )

                proj_agg = proj_agg.merge(dist_med1, on="district", how="left")
                proj_agg["pct_vs_district"] = (
                    (proj_agg["median_psm"] - proj_agg["district_median_psm"])
                    / proj_agg["district_median_psm"] * 100
                )

                # Filter: txn_count >= 3 and lat/lon available
                proj_agg = proj_agg[
                    (proj_agg["txn_count"] >= 3)
                    & proj_agg["lat"].notna()
                    & proj_agg["lon"].notna()
                ].copy()

                if proj_agg.empty:
                    st.info("No projects meet the minimum criteria (≥3 transactions with coordinates).")
                else:
                    # Colour: blue = undervalued, red = overpriced, grey = inline
                    def priv_colour(pct):
                        if pct < -5:
                            return [30, 100, 220, 210]
                        elif pct > 5:
                            return [220, 50, 50, 210]
                        return [150, 150, 150, 150]

                    proj_agg["color"] = proj_agg["pct_vs_district"].apply(priv_colour)
                    proj_agg["radius"] = np.clip(proj_agg["txn_count"] * 15, 60, 600)
                    proj_agg["pct_vs_district_r"] = proj_agg["pct_vs_district"].round(1)
                    proj_agg["median_psm_r"] = proj_agg["median_psm"].round(0).astype(int)
                    proj_agg["district_median_psm_r"] = proj_agg["district_median_psm"].round(0).astype(int)
                    proj_agg["district_str"] = "D" + proj_agg["district"].astype(str).str.zfill(2)

                    st.markdown("#### Project Price vs District Median (last 24 months)")
                    st.caption("🔵 Undervalued (<5% below district median)  |  ⚫ Inline  |  🔴 Above-district (>5%)")

                    layer_p1 = pdk.Layer(
                        "ScatterplotLayer",
                        data=proj_agg,
                        get_position=["lon", "lat"],
                        get_fill_color="color",
                        get_radius="radius",
                        radius_scale=1,
                        pickable=True,
                    )
                    view_p1 = pdk.ViewState(
                        latitude=1.3521, longitude=103.8198, zoom=11, pitch=0,
                    )
                    tooltip_p1 = {
                        "html": (
                            "<b>{project}</b><br/>"
                            "District: {district_str}<br/>"
                            "Median PSM: ${median_psm_r}<br/>"
                            "District Median PSM: ${district_median_psm_r}<br/>"
                            "vs District: {pct_vs_district_r}%<br/>"
                            "Transactions: {txn_count}"
                        ),
                        "style": {"backgroundColor": "steelblue", "color": "white"},
                    }
                    deck_p1 = pdk.Deck(
                        layers=[layer_p1],
                        initial_view_state=view_p1,
                        tooltip=tooltip_p1,
                        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    )
                    st.pydeck_chart(deck_p1)

                    # Top-15 most undervalued projects
                    top15_uv = proj_agg.sort_values("pct_vs_district").head(15)
                    st.markdown("#### Top 15 Most Undervalued Projects vs District Median")
                    disp_p1 = top15_uv[[
                        "project", "district_str", "median_psm", "district_median_psm",
                        "pct_vs_district", "txn_count"
                    ]].copy()
                    disp_p1.columns = [
                        "Project", "District", "Median PSM ($/sqm)",
                        "District Median PSM ($/sqm)", "vs District (%)", "Transactions",
                    ]
                    disp_p1["Median PSM ($/sqm)"] = disp_p1["Median PSM ($/sqm)"].round(0).astype(int)
                    disp_p1["District Median PSM ($/sqm)"] = disp_p1["District Median PSM ($/sqm)"].round(0).astype(int)
                    disp_p1["vs District (%)"] = disp_p1["vs District (%)"].round(1)
                    st.dataframe(disp_p1, use_container_width=True, hide_index=True)

                    st.warning(
                        "⚠️ Private data covers Aug 2021–2026 only. "
                        "Projects with fewer than 3 transactions are excluded."
                    )


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — Comps Finder
# ═════════════════════════════════════════════════════════════════════
with tab2:
    if mode == "🏘️ HDB Resale":
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

    else:
        # ── Private Comps Finder ──────────────────────────────────────────
        st.subheader("🔍 Comps Finder — Private condo comparable transactions")

        df_priv2 = load_private()
        if df_priv2.empty:
            st.warning("⚠️ Private condo data not available.")
        else:
            # Build select options
            district_opts2 = sorted(df_priv2["district"].dropna().unique().tolist())
            ptypes2 = sorted(df_priv2["property_type_broad"].dropna().unique().tolist())
            floor_bands2 = sorted(
                df_priv2["floor_range"].dropna().unique().tolist(),
                key=lambda b: floor_range_mid(b) if not math.isnan(floor_range_mid(b)) else 999,
            )

            with st.form("vf_priv_comps_form"):
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    sel_dist2 = st.selectbox(
                        "District",
                        options=district_opts2,
                        format_func=lambda d: f"D{int(d):02d}",
                        key="vf_priv_comps_district",
                    )
                with fc2:
                    sel_ptype2 = st.selectbox(
                        "Property type",
                        options=ptypes2,
                        key="vf_priv_comps_ptype",
                    )
                with fc3:
                    sel_band2 = st.selectbox(
                        "Floor range band",
                        options=floor_bands2,
                        key="vf_priv_comps_band",
                    )
                with fc4:
                    sel_area2 = st.number_input(
                        "Area (sqm)",
                        min_value=20.0,
                        max_value=600.0,
                        value=100.0,
                        step=5.0,
                        key="vf_priv_comps_area",
                    )
                submitted2 = st.form_submit_button("🔍 Find Comps")

            if submitted2:
                user_floor_mid2 = floor_range_mid(sel_band2)

                # Filter: district + property_type_broad + area ±30 sqm
                df_c2 = df_priv2[
                    (df_priv2["district"] == sel_dist2)
                    & (df_priv2["property_type_broad"] == sel_ptype2)
                    & (df_priv2["area_sqm"] >= sel_area2 - 30)
                    & (df_priv2["area_sqm"] <= sel_area2 + 30)
                ].copy()

                if df_c2.empty:
                    st.info("No transactions match the selected filters (±30 sqm area tolerance).")
                else:
                    # Floor midpoint
                    df_c2["floor_mid_val"] = df_c2["floor_range"].apply(floor_range_mid)

                    # Tenure similarity: 0 if matches majority tenure, 0.5 otherwise
                    majority_tenure = df_c2["tenure_clean"].mode()
                    if not majority_tenure.empty:
                        df_c2["tenure_score"] = (df_c2["tenure_clean"] != majority_tenure.iloc[0]).astype(float) * 0.5
                    else:
                        df_c2["tenure_score"] = 0.0

                    # Recency score: days since transaction / 365
                    max_date2 = df_c2["contract_date"].max()
                    df_c2["recency_score"] = (max_date2 - df_c2["contract_date"]).dt.days / 365.0

                    # Area score: abs diff / area
                    df_c2["area_score_n"] = (df_c2["area_sqm"] - sel_area2).abs() / max(sel_area2, 1)

                    # Floor score: abs diff / 10
                    if not math.isnan(user_floor_mid2):
                        df_c2["floor_score_n"] = (df_c2["floor_mid_val"] - user_floor_mid2).abs() / 10.0
                    else:
                        df_c2["floor_score_n"] = 0.0

                    # Composite similarity: area 35%, floor 20%, tenure 20%, recency 25%
                    df_c2["similarity_score"] = (
                        df_c2["area_score_n"] * 0.35
                        + df_c2["floor_score_n"] * 0.20
                        + df_c2["tenure_score"] * 0.20
                        + df_c2["recency_score"] * 0.25
                    )

                    top20_priv = df_c2.nsmallest(20, "similarity_score")

                    st.markdown(f"#### Top 20 Most Similar Transactions ({len(df_c2):,} total matches)")
                    cols_show = [
                        "project", "floor_range", "area_sqm", "price", "price_psm",
                        "contract_date", "type_of_sale", "tenure_clean", "similarity_score",
                    ]
                    avail2 = [c for c in cols_show if c in top20_priv.columns]
                    disp_c2 = top20_priv[avail2].copy()
                    disp_c2 = disp_c2.rename(columns={
                        "project": "Project",
                        "floor_range": "Floor Range",
                        "area_sqm": "Area (sqm)",
                        "price": "Price ($)",
                        "price_psm": "PSM ($/sqm)",
                        "contract_date": "Date",
                        "type_of_sale": "Sale Type",
                        "tenure_clean": "Tenure",
                        "similarity_score": "Similarity Score",
                    })
                    if "Price ($)" in disp_c2.columns:
                        disp_c2["Price ($)"] = disp_c2["Price ($)"].apply(
                            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                        )
                    if "PSM ($/sqm)" in disp_c2.columns:
                        disp_c2["PSM ($/sqm)"] = disp_c2["PSM ($/sqm)"].apply(
                            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                        )
                    if "Date" in disp_c2.columns:
                        disp_c2["Date"] = pd.to_datetime(disp_c2["Date"]).dt.strftime("%Y-%m")
                    if "Similarity Score" in disp_c2.columns:
                        disp_c2["Similarity Score"] = disp_c2["Similarity Score"].round(3)

                    st.dataframe(disp_c2, use_container_width=True, hide_index=True)

                    # CSV export
                    csv_data2 = top20_priv[avail2].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download comps as CSV",
                        data=csv_data2,
                        file_name="private_comps.csv",
                        mime="text/csv",
                        key="vf_priv_comps_dl",
                    )


# ═════════════════════════════════════════════════════════════════════
# TAB 3 — Percentile Pricer
# ═════════════════════════════════════════════════════════════════════
with tab3:
    if mode == "🏘️ HDB Resale":
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

    else:
        # ── Private Percentile Pricer ─────────────────────────────────────
        st.subheader("📊 Percentile Pricer — Private condo PSM distribution")

        df_priv3 = load_private()
        if df_priv3.empty:
            st.warning("⚠️ Private condo data not available.")
        else:
            district_opts3 = sorted(df_priv3["district"].dropna().unique().tolist())
            ptypes3 = sorted(df_priv3["property_type_broad"].dropna().unique().tolist())

            c3a, c3b = st.columns(2)
            with c3a:
                sel_dist3 = st.selectbox(
                    "District",
                    options=district_opts3,
                    format_func=lambda d: f"D{int(d):02d}",
                    key="vf_priv_pp_district",
                )
            with c3b:
                sel_ptype3 = st.selectbox(
                    "Property type",
                    options=ptypes3,
                    key="vf_priv_pp_ptype",
                )

            # Filter to last 2 years for distribution
            max_cd3 = df_priv3["contract_date"].max()
            cutoff3p = max_cd3 - pd.DateOffset(months=24)
            df_f3p = df_priv3[
                (df_priv3["contract_date"] >= cutoff3p)
                & (df_priv3["district"] == sel_dist3)
                & (df_priv3["property_type_broad"] == sel_ptype3)
            ].dropna(subset=["price_psm"]).copy()

            if df_f3p.empty:
                st.info("No data for the selected district and property type in the last 2 years.")
            else:
                dist_med3 = df_f3p["price_psm"].median()

                asking_psm3 = st.number_input(
                    "Asking PSM ($/sqm)",
                    min_value=1_000,
                    max_value=100_000,
                    value=int(dist_med3),
                    step=100,
                    key="vf_priv_pp_asking",
                )

                # Histogram with asking-PSM line
                fig3p = px.histogram(
                    df_f3p,
                    x="price_psm",
                    nbins=50,
                    labels={"price_psm": "Price PSM ($/sqm)"},
                    color_discrete_sequence=["#4C78A8"],
                    title=(
                        f"PSM Distribution — D{int(sel_dist3):02d} {sel_ptype3} "
                        f"(last 24 months, n={len(df_f3p):,})"
                    ),
                )
                fig3p.update_layout(showlegend=False, yaxis_title="Count")
                fig3p.add_vline(
                    x=asking_psm3,
                    line_dash="dash",
                    line_color="crimson",
                    annotation_text="Your PSM",
                    annotation_position="top right",
                )
                st.plotly_chart(fig3p, use_container_width=True)

                # Percentile of asking PSM
                pct3p = (df_f3p["price_psm"] < asking_psm3).mean() * 100
                st.metric(
                    "Your PSM Percentile",
                    f"{pct3p:.1f}th",
                    help="Lower percentile = relatively cheaper vs recent transactions.",
                )

                # P10 / P25 / P50 / P75 / P90 table
                percs = [10, 25, 50, 75, 90]
                perc_vals = [df_f3p["price_psm"].quantile(p / 100) for p in percs]
                perc_df = pd.DataFrame({
                    "Percentile": [f"P{p}" for p in percs],
                    "PSM ($/sqm)": [f"${v:,.0f}" for v in perc_vals],
                })
                st.table(perc_df.set_index("Percentile"))

                st.caption(
                    f"**Note:** Lower PSM percentile = relatively cheaper vs recent "
                    f"transactions in D{int(sel_dist3):02d}."
                )


# ═════════════════════════════════════════════════════════════════════
# TAB 4 — Floor Premium Validator
# ═════════════════════════════════════════════════════════════════════
with tab4:
    if mode == "🏘️ HDB Resale":
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

    else:
        # ── Private Floor Premium Validator ───────────────────────────────
        st.subheader("📐 Floor Premium Validator — Private condo floor-band PSM")

        df_priv4 = load_private()
        if df_priv4.empty:
            st.warning("⚠️ Private condo data not available.")
        else:
            district_opts4 = sorted(df_priv4["district"].dropna().unique().tolist())
            ptypes4 = sorted(df_priv4["property_type_broad"].dropna().unique().tolist())

            c4a, c4b = st.columns(2)
            with c4a:
                sel_dist4 = st.selectbox(
                    "District",
                    options=district_opts4,
                    format_func=lambda d: f"D{int(d):02d}",
                    key="vf_priv_fp_district",
                )
            with c4b:
                sel_ptype4 = st.selectbox(
                    "Property type",
                    options=ptypes4,
                    key="vf_priv_fp_ptype",
                )

            df_f4p = df_priv4[
                (df_priv4["district"] == sel_dist4)
                & (df_priv4["property_type_broad"] == sel_ptype4)
            ].dropna(subset=["price_psm", "floor_range"]).copy()

            if df_f4p.empty:
                st.info("No data for the selected district and property type.")
            else:
                # Compute floor midpoint for ordering bands
                df_f4p["floor_mid_val"] = df_f4p["floor_range"].apply(floor_range_mid)

                band_agg4p = (
                    df_f4p.groupby("floor_range")
                    .agg(
                        median_psm=("price_psm", "median"),
                        txn_count=("price_psm", "size"),
                        floor_mid=("floor_mid_val", "first"),
                    )
                    .reset_index()
                    .dropna(subset=["median_psm", "floor_mid"])
                    .sort_values("floor_mid")
                )

                if band_agg4p.empty:
                    st.info("Insufficient data across floor bands for this selection.")
                else:
                    # Reference = lowest floor-mid band
                    ground_psm4p = band_agg4p.iloc[0]["median_psm"]
                    band_agg4p["premium_vs_ground_pct"] = (
                        (band_agg4p["median_psm"] - ground_psm4p) / ground_psm4p * 100
                    )

                    # Bar + premium overlay
                    fig4p = go.Figure()
                    fig4p.add_trace(go.Bar(
                        x=band_agg4p["floor_range"],
                        y=band_agg4p["median_psm"],
                        name="Median PSM ($/sqm)",
                        marker_color="#4C78A8",
                        yaxis="y1",
                    ))
                    fig4p.add_trace(go.Scatter(
                        x=band_agg4p["floor_range"],
                        y=band_agg4p["premium_vs_ground_pct"],
                        name="Premium vs Lowest Band (%)",
                        mode="lines+markers",
                        line=dict(color="crimson", width=2),
                        marker=dict(size=7),
                        yaxis="y2",
                    ))
                    fig4p.update_layout(
                        title=f"Floor Premium — D{int(sel_dist4):02d} {sel_ptype4}",
                        xaxis_title="Floor Range Band",
                        yaxis=dict(title="Median PSM ($/sqm)", side="left"),
                        yaxis2=dict(
                            title="Premium vs Lowest Band (%)",
                            overlaying="y",
                            side="right",
                            showgrid=False,
                        ),
                        legend=dict(orientation="h", y=1.1),
                        height=420,
                    )
                    st.plotly_chart(fig4p, use_container_width=True)

                    # Floor premium table
                    st.markdown("#### Floor Premium Table")
                    tbl4p = band_agg4p[[
                        "floor_range", "median_psm", "txn_count", "premium_vs_ground_pct"
                    ]].copy()
                    tbl4p.columns = [
                        "Floor Range", "Median PSM ($/sqm)", "Transactions", "Premium vs Ground (%)"
                    ]
                    tbl4p["Median PSM ($/sqm)"] = tbl4p["Median PSM ($/sqm)"].round(0).astype(int)
                    tbl4p["Premium vs Ground (%)"] = tbl4p["Premium vs Ground (%)"].round(1)
                    st.dataframe(tbl4p, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 5 — MRT Proximity Premium
# ═════════════════════════════════════════════════════════════════════
with tab5:
    if mode == "🏘️ HDB Resale":
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

    else:
        # ── Private MRT Proximity Premium ─────────────────────────────────
        st.subheader("🚇 MRT Proximity Premium — Private condo price lift near MRT")

        mrt5p = load_mrt()
        df_priv5 = load_private()

        if df_priv5.empty:
            st.warning("⚠️ Private condo data not available.")
        elif mrt5p.empty:
            st.warning("⚠️ MRT station data not available.")
        else:
            # Use only rows with lat/lon coordinates
            df_p5 = df_priv5.dropna(subset=["lat", "lon", "price_psm"]).copy()

            if df_p5.empty:
                st.info("No private condo transactions with geocoordinates available.")
            else:
                # MRT line filter
                all_lines5p = []
                if "lines" in mrt5p.columns:
                    for val in mrt5p["lines"].dropna():
                        all_lines5p.extend([x.strip() for x in str(val).split(",")])
                elif "line" in mrt5p.columns:
                    all_lines5p = mrt5p["line"].dropna().unique().tolist()
                all_lines5p = sorted(set(all_lines5p))

                sel_line5p = st.selectbox(
                    "Filter by MRT line (optional)",
                    options=["All lines"] + all_lines5p,
                    key="vf_priv_mrt_line",
                )

                if sel_line5p != "All lines":
                    if "lines" in mrt5p.columns:
                        mrt_filt5p = mrt5p[mrt5p["lines"].str.contains(sel_line5p, na=False)]
                    else:
                        mrt_filt5p = mrt5p[mrt5p["line"] == sel_line5p]
                else:
                    mrt_filt5p = mrt5p

                if mrt_filt5p.empty:
                    st.info("No MRT stations for the selected line.")
                else:
                    # Vectorised nearest-MRT distance for all private condo rows
                    @st.cache_data
                    def compute_priv_mrt_dist(n_rows: int, mrt_hash_str: str):
                        """Return array of nearest-MRT distances (km) for df_p5."""
                        _ref_lat = 1.3521  # Singapore centroid for cos correction
                        mrt_lats = mrt_filt5p["lat"].values
                        mrt_lons = mrt_filt5p["lon"].values
                        condo_lats = df_p5["lat"].values
                        condo_lons = df_p5["lon"].values
                        dist_matrix = np.zeros((len(df_p5), len(mrt_filt5p)))
                        for j in range(len(mrt_filt5p)):
                            dlat = (condo_lats - mrt_lats[j]) * 111.0
                            dlon = (condo_lons - mrt_lons[j]) * 111.0 * math.cos(
                                math.radians(_ref_lat)
                            )
                            dist_matrix[:, j] = np.sqrt(dlat ** 2 + dlon ** 2)
                        return dist_matrix.min(axis=1)

                    mrt_hash5p = sel_line5p + str(len(mrt_filt5p))
                    dist_arr5p = compute_priv_mrt_dist(len(df_p5), mrt_hash5p)
                    df_p5 = df_p5.copy()
                    df_p5["dist_mrt_km"] = dist_arr5p

                    # Bin into distance bands
                    bins5p = [0, 0.5, 1.0, 2.0, float("inf")]
                    labels5p = ["<0.5km", "0.5-1km", "1-2km", ">2km"]
                    df_p5["dist_band"] = pd.cut(
                        df_p5["dist_mrt_km"],
                        bins=bins5p,
                        labels=labels5p,
                        right=True,
                    )

                    # Box plot: price_psm by distance band
                    fig5p_box = px.box(
                        df_p5,
                        x="dist_band",
                        y="price_psm",
                        category_orders={"dist_band": labels5p},
                        labels={
                            "dist_band": "Distance to Nearest MRT",
                            "price_psm": "PSM ($/sqm)",
                        },
                        color="dist_band",
                        color_discrete_sequence=["#1f4e79", "#2e75b6", "#9dc3e6", "#bdd7ee"],
                        title="PSM Distribution by MRT Distance Band",
                    )
                    fig5p_box.update_layout(showlegend=False, height=420)
                    st.plotly_chart(fig5p_box, use_container_width=True)

                    # Aggregate: median PSM per band
                    band_agg5p = (
                        df_p5.groupby("dist_band", observed=True)
                        .agg(
                            median_psm=("price_psm", "median"),
                            txn_count=("price_psm", "size"),
                        )
                        .reindex(labels5p)
                        .reset_index()
                    )

                    # Bar chart: median PSM by distance band
                    fig5p_bar = px.bar(
                        band_agg5p,
                        x="dist_band",
                        y="median_psm",
                        labels={
                            "dist_band": "Distance to Nearest MRT",
                            "median_psm": "Median PSM ($/sqm)",
                        },
                        color_discrete_sequence=["#4C78A8"],
                        title="Median PSM by MRT Distance Band",
                        text_auto=".0f",
                    )
                    fig5p_bar.update_traces(textposition="outside")
                    fig5p_bar.update_layout(height=380)
                    st.plotly_chart(fig5p_bar, use_container_width=True)

                    # PSM premium: <500m vs >1km
                    row_lt500 = band_agg5p.loc[band_agg5p["dist_band"] == "<0.5km", "median_psm"]
                    rows_gt1km = band_agg5p[
                        band_agg5p["dist_band"].isin(["1-2km", ">2km"])
                    ]["median_psm"].dropna()
                    if not row_lt500.empty and not rows_gt1km.empty:
                        val_lt500 = row_lt500.values[0]
                        val_gt1km = rows_gt1km.median()
                        if pd.notna(val_lt500) and pd.notna(val_gt1km) and val_gt1km > 0:
                            premium5p = (val_lt500 - val_gt1km) / val_gt1km * 100
                            sign5p = "+" if premium5p >= 0 else ""
                            st.metric(
                                "PSM Premium: <500m vs >1km",
                                f"{sign5p}{premium5p:.1f}%",
                                help="Positive = properties within 500 m of MRT command higher median PSM.",
                            )

                    # Summary table
                    st.markdown("#### Summary Table")
                    tbl5p = band_agg5p.copy()
                    tbl5p["Median PSM ($/sqm)"] = tbl5p["median_psm"].apply(
                        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                    )
                    tbl5p = tbl5p.rename(columns={
                        "dist_band": "Distance Band",
                        "txn_count": "Transactions",
                    })[["Distance Band", "Median PSM ($/sqm)", "Transactions"]]
                    st.dataframe(tbl5p, use_container_width=True, hide_index=True)

                    st.warning(
                        "**DATA CONFIDENCE: Medium** — Distance is straight-line (as the crow flies), "
                        "not actual walking distance. Coordinates cover ~79% of private transactions; "
                        "landed properties are excluded due to missing coordinates."
                    )
