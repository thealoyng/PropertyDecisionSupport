"""
Page 19 -- Smart Money One-Click Comps Report (G16)
=====================================================
Highest-ROI agent deliverable: enter a subject property, get a shareable,
client-ready comparables summary in seconds.

Sections:
  1. Executive Summary        -- 4 metric cards
  2. Comparable Transactions  -- top-20 table with PSM colour gradient
  3. Price Distribution       -- histogram with subject / FV lines
  4. Floor Premium Analysis   -- storey-band bar + premium overlay
  5. Market Trend Context     -- 24-month monthly median PSM line chart
  6. Data Confidence          -- assumptions & confidence levels
  7. Share / Export           -- CSV download + copy-ready text summary
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from eda_helpers import (
    load_clean,
    fmt_price,
    fmt_pct,
    storey_band,
    TOWN_CENTROIDS,
    load_condo_clean,
    DISTRICT_CENTROIDS,
    floor_range_mid,
)

# ── page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Comps Report",
    page_icon="📋",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

STOREY_BAND_ORDER = [
    "01-03", "04-06", "07-09", "10-12",
    "13-15", "16-21", "22-30", "31+",
]

# ── cached data loader ────────────────────────────────────────────────

@st.cache_data
def load_resale():
    """Load and cache the full cleaned resale dataset."""
    return load_clean()


# ── helper functions ──────────────────────────────────────────────────

def weighted_median(values, weights):
    """Weighted median: lower similarity score → higher weight."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~(np.isnan(values) | np.isnan(weights))
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return np.nan
    idx = np.argsort(values)
    vs = values[idx]
    ws = weights[idx]
    cumw = np.cumsum(ws)
    return float(vs[np.searchsorted(cumw, cumw[-1] / 2.0)])


def compute_similarity(df, subj_area, subj_storey, subj_lease, cur_year):
    """
    Composite similarity score (lower = more similar to subject).
      Physical area   35 %  |  Storey    20 %
      Lease           25 %  |  Recency   20 %
    """
    s_area    = (df["floor_area_sqm"]     - subj_area).abs()   / max(subj_area, 1)   * 0.35
    s_storey  = (df["storey_mid"]         - subj_storey).abs() / 10.0                * 0.20
    s_lease   = (df["remaining_lease_yrs"]- subj_lease).abs()  / 10.0                * 0.25
    s_recency = (cur_year - df["year"])                         / 3.0                 * 0.20
    return s_area + s_storey + s_lease + s_recency


def pct_rank(value, series):
    """Return the percentile (0–100) of value within series."""
    arr = series.dropna().values
    if len(arr) == 0:
        return np.nan
    return float(np.sum(arr < value) / len(arr) * 100)


def psm_color(norm_val):
    """Map a normalised value [0,1] to green→red RGB string."""
    r = int(50  + 180 * norm_val)
    g = int(200 - 150 * norm_val)
    return f"rgb({r},{g},80)"


# ── page header ───────────────────────────────────────────────────────
st.title("📋 One-Click Comps Report")
st.caption(
    "Enter subject property details to generate a shareable, client-ready "
    "comparables report — implied fair value, price distribution, floor premium "
    "analysis, and market trend context in one page."
)

mode_comps = st.radio(
    "Property type",
    ["🏘️ HDB Resale", "🏢 Private (Condo)"],
    horizontal=True,
    key="comps_mode",
)
st.divider()

# ── load data ─────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df_all = load_resale()

# ════════════════════════════════════════════════════════════════════════════
# PRIVATE (CONDO) MODE helpers & runner
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _load_condo_for_comps():
    df = load_condo_clean()
    if df.empty:
        return df
    df = df[df["property_type_broad"].isin(["Condo/Apartment", "EC"])].copy()
    df["contract_date"] = pd.to_datetime(df["contract_date"], errors="coerce")
    df["floor_mid"] = df["floor_range"].apply(floor_range_mid)
    return df


def _priv_similarity(df, subj_area, subj_floor_mid, subj_tenure, cur_year):
    s_area    = (df["area_sqm"]  - subj_area).abs()      / max(subj_area, 1)   * 0.35
    s_floor   = (df["floor_mid"] - subj_floor_mid).abs() / 10.0                * 0.20
    s_tenure  = (df["tenure_clean"] != subj_tenure).astype(float)               * 0.20
    s_recency = (cur_year - df["contract_date"].dt.year)                        / 3.0 * 0.25
    return s_area + s_floor + s_tenure + s_recency


def _run_private_comps():
    """All private comps report logic — called then st.stop() exits HDB path."""
    st.warning("⚠️ Private data covers Aug 2021–2026 (~5 years). "
               "Smaller comp pools than HDB — widen filters if fewer than 5 results.")

    with st.spinner("Loading private transaction data…"):
        cdf_all = _load_condo_for_comps()

    if cdf_all.empty:
        st.error("Private transaction data not found. Run combine_clean_condo.py first.")
        st.stop()

    PRIV_DIST_OPTS  = {d: f"D{d:02d} — {name}" for d, (name, _, _) in DISTRICT_CENTROIDS.items()}
    PRIV_PTYPES     = sorted(cdf_all["property_type_broad"].dropna().unique().tolist())
    PRIV_FLOOR_RNGS = sorted(
        cdf_all["floor_range"].dropna().unique().tolist(),
        key=lambda x: floor_range_mid(x) if not np.isnan(floor_range_mid(x)) else 99,
    )
    PRIV_TENURES    = sorted(cdf_all["tenure_clean"].dropna().unique().tolist())
    PRIV_SALE_TYPES = ["All"] + sorted(cdf_all["type_of_sale"].dropna().unique().tolist())

    with st.form("comps_form_priv"):
        st.subheader("📝 Subject Property Details")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            priv_district = st.selectbox(
                "District", list(PRIV_DIST_OPTS.keys()), index=8,
                format_func=lambda d: PRIV_DIST_OPTS[d], key="comps_priv_dist",
            )
            priv_prop_type = st.selectbox("Property Type", PRIV_PTYPES, key="comps_priv_ptype")
        with pc2:
            priv_floor_rng = st.selectbox(
                "Subject Floor Range", PRIV_FLOOR_RNGS,
                index=min(2, len(PRIV_FLOOR_RNGS) - 1), key="comps_priv_floor",
            )
            priv_tenure = st.selectbox("Tenure", PRIV_TENURES, key="comps_priv_tenure")
        with pc3:
            priv_area = st.number_input(
                "Floor Area (sqm)", 30.0, 500.0, 90.0, step=5.0, key="comps_priv_area"
            )
            priv_asking_price = st.number_input(
                "Asking Price ($ — 0 = not provided)",
                0, 20_000_000, 0, step=10_000, key="comps_priv_price",
            )
        pc4, pc5 = st.columns(2)
        with pc4:
            priv_sale_type = st.selectbox(
                "Sale Type Filter", PRIV_SALE_TYPES, index=0, key="comps_priv_sale"
            )
        with pc5:
            priv_range = st.radio(
                "Year Range for Comps",
                ["Last 12 months", "Last 2 years", "Last 3 years"],
                index=1, key="comps_priv_range",
            )
        priv_submitted = st.form_submit_button(
            "🔍 Generate Private Comps Report", use_container_width=True
        )

    if not priv_submitted:
        st.info("Fill in subject property details and click Generate Comps Report.")
        st.stop()

    priv_today    = datetime.date.today()
    priv_cur_year = priv_today.year
    priv_months   = {"Last 12 months": 12, "Last 2 years": 24, "Last 3 years": 36}[priv_range]
    priv_cutoff   = pd.Timestamp(priv_today) - pd.DateOffset(months=priv_months)

    priv_pool = cdf_all[
        (cdf_all["district"]              == priv_district)
        & (cdf_all["property_type_broad"] == priv_prop_type)
        & (cdf_all["contract_date"]       >= priv_cutoff)
    ].copy()
    if priv_sale_type != "All":
        priv_pool = priv_pool[priv_pool["type_of_sale"] == priv_sale_type]
    priv_pool = priv_pool.dropna(subset=["area_sqm", "floor_mid", "price_psm", "price"])

    if priv_pool.empty:
        st.warning("No transactions found. Try widening filters.")
        st.stop()

    priv_pool["similarity_score"] = _priv_similarity(
        priv_pool, priv_area, floor_range_mid(priv_floor_rng), priv_tenure, priv_cur_year
    )
    priv_comps = priv_pool.nsmallest(20, "similarity_score").copy().reset_index(drop=True)
    priv_comps["rank"] = priv_comps.index + 1

    pw              = 1.0 / (priv_comps["similarity_score"] + 0.01)
    priv_implied_fv = weighted_median(priv_comps["price"].values, pw.values)
    priv_med_psm    = float(priv_pool["price_psm"].median())
    priv_n_comps    = len(priv_comps)
    priv_subj_psm   = priv_asking_price / priv_area if priv_asking_price > 0 else np.nan
    priv_mkt_pct    = pct_rank(priv_asking_price, priv_pool["price"]) if priv_asking_price > 0 else np.nan
    priv_psm_vs_med = (
        (priv_subj_psm - priv_med_psm) / priv_med_psm * 100
        if priv_asking_price > 0 and not np.isnan(priv_subj_psm) else np.nan
    )
    dist_name = DISTRICT_CENTROIDS[priv_district][0]

    # Section 1: Executive Summary
    st.markdown("---")
    st.subheader(f"1️⃣ Executive Summary — D{priv_district:02d} {dist_name} | {priv_prop_type}")
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Comparable Transactions", str(priv_n_comps))
    ec2.metric("Implied Fair Value", fmt_price(priv_implied_fv))
    if priv_asking_price > 0 and not np.isnan(priv_mkt_pct):
        ec3.metric("Market Percentile", f"{priv_mkt_pct:.1f}th",
                   delta=f"{'above' if priv_mkt_pct > 50 else 'below'} median")
    else:
        ec3.metric("Market Percentile", "—", help="Provide asking price to calculate")
    if priv_asking_price > 0 and not np.isnan(priv_subj_psm):
        ec4.metric("PSM vs District Median", f"${priv_subj_psm:,.0f}/sqm",
                   delta=f"{priv_psm_vs_med:+.1f}% vs ${priv_med_psm:,.0f}/sqm",
                   delta_color="inverse")
    else:
        ec4.metric("District Median PSM", f"${priv_med_psm:,.0f}/sqm")

    if priv_asking_price > 0 and not np.isnan(priv_implied_fv):
        gap = (priv_asking_price - priv_implied_fv) / priv_implied_fv * 100
        if gap > 1.0:
            st.warning(f"⚠️ Asking {fmt_price(priv_asking_price)} is {gap:.1f}% above "
                       f"implied FV {fmt_price(priv_implied_fv)} — room to negotiate.")
        elif gap < -1.0:
            st.success(f"✅ Asking {fmt_price(priv_asking_price)} is {abs(gap):.1f}% below "
                       f"implied FV {fmt_price(priv_implied_fv)} — potential opportunity.")
        else:
            st.info(f"ℹ️ Asking price broadly in line with implied FV {fmt_price(priv_implied_fv)}.")

    # Section 2: Comps Table
    st.markdown("---")
    st.subheader("2️⃣ Comparable Transactions (Top 20)")
    st.caption("Similarity: area 35%, floor 20%, tenure match 20%, recency 25%.")
    priv_disp = priv_comps[[
        "rank", "project", "floor_range", "area_sqm", "tenure_clean",
        "type_of_sale", "contract_date", "price", "price_psm", "similarity_score",
    ]].copy()
    priv_disp["contract_date"]    = priv_disp["contract_date"].dt.strftime("%Y-%m")
    priv_disp["price"]            = priv_disp["price"].apply(lambda x: f"${x:,.0f}")
    priv_disp["price_psm"]        = priv_disp["price_psm"].apply(lambda x: f"${x:,.0f}")
    priv_disp["area_sqm"]         = priv_disp["area_sqm"].apply(lambda x: f"{x:.0f}")
    priv_disp["similarity_score"] = priv_disp["similarity_score"].apply(lambda x: f"{x:.3f}")
    priv_disp.columns = ["#", "Project", "Floor Range", "Area (sqm)", "Tenure",
                         "Sale Type", "Sale Date", "Price ($)", "PSM ($/sqm)", "Similarity"]
    st.dataframe(priv_disp, use_container_width=True, hide_index=True)

    # Section 3: Price Distribution
    st.markdown("---")
    st.subheader("3️⃣ Price Distribution")
    fig_ph = go.Figure()
    fig_ph.add_trace(go.Histogram(x=priv_pool["price"], nbinsx=40,
                                  name="All Transactions", marker_color="#4C78A8", opacity=0.75))
    fig_ph.add_vline(x=priv_implied_fv, line_dash="dash", line_color="#E45756",
                     annotation_text=f"Implied FV {fmt_price(priv_implied_fv)}",
                     annotation_position="top right", annotation_font_color="#E45756")
    if priv_asking_price > 0:
        fig_ph.add_vline(x=priv_asking_price, line_dash="dot", line_color="#54A24B",
                         annotation_text=f"Asking {fmt_price(priv_asking_price)}",
                         annotation_position="top left", annotation_font_color="#54A24B")
    fig_ph.update_layout(
        xaxis_title="Price ($)", yaxis_title="Transactions",
        title=f"D{priv_district:02d} {priv_prop_type} — Price Distribution ({priv_range})",
        height=420,
    )
    st.plotly_chart(fig_ph, use_container_width=True)

    # Section 4: Floor Premium
    st.markdown("---")
    st.subheader("4️⃣ Floor Range Premium Analysis")
    priv_fp = priv_pool.dropna(subset=["floor_mid", "price_psm"]).copy()
    if not priv_fp.empty:
        ba = (priv_fp.groupby("floor_range")
              .agg(median_psm=("price_psm", "median"), count=("price_psm", "size"))
              .reset_index())
        ba["floor_order"] = ba["floor_range"].apply(floor_range_mid)
        ba = ba.sort_values("floor_order").reset_index(drop=True)
        if len(ba) >= 2:
            g0 = ba.iloc[0]["median_psm"]
            ba["premium_pct"] = (ba["median_psm"] - g0) / g0 * 100
            colors_fp = ["#E45756" if r["floor_range"] == priv_floor_rng else "#4C78A8"
                         for _, r in ba.iterrows()]
            fig_fp = go.Figure()
            fig_fp.add_trace(go.Bar(x=ba["floor_range"], y=ba["median_psm"],
                                    name="Median PSM ($)", marker_color=colors_fp,
                                    text=[f"${v:,.0f}" for v in ba["median_psm"]],
                                    textposition="outside", yaxis="y1"))
            fig_fp.add_trace(go.Scatter(x=ba["floor_range"], y=ba["premium_pct"],
                                        name="Premium vs Lowest (%)", mode="lines+markers",
                                        line=dict(color="#F58518", width=2), yaxis="y2"))
            fig_fp.update_layout(
                xaxis_title="Floor Range", height=420,
                yaxis=dict(title="Median PSM ($)"),
                yaxis2=dict(title="Premium (%)", overlaying="y", side="right", showgrid=False),
                title=f"D{priv_district:02d} {priv_prop_type} — Floor Range vs Median PSM",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_fp, use_container_width=True)
    else:
        st.info("Insufficient data for floor premium analysis.")

    # Section 5: Trend
    st.markdown("---")
    st.subheader("5️⃣ Market Trend Context")
    pt_cutoff = pd.Timestamp(priv_today) - pd.DateOffset(months=24)
    pt_df = cdf_all[
        (cdf_all["district"]              == priv_district)
        & (cdf_all["property_type_broad"] == priv_prop_type)
        & (cdf_all["contract_date"]       >= pt_cutoff)
    ].dropna(subset=["price_psm"]).copy()
    if not pt_df.empty:
        pt_df["month_ts"] = pt_df["contract_date"].dt.to_period("M").dt.to_timestamp()
        pm = (pt_df.groupby("month_ts")["price_psm"].median().reset_index()
              .rename(columns={"month_ts": "month", "price_psm": "median_psm"}))
        fig_pt = go.Figure()
        fig_pt.add_trace(go.Scatter(x=pm["month"], y=pm["median_psm"],
                                    mode="lines+markers", name="Median PSM",
                                    line=dict(color="#4C78A8", width=2.5), fill="tozeroy",
                                    fillcolor="rgba(76,120,168,0.15)"))
        fig_pt.update_layout(xaxis_title="Month", yaxis_title="Median PSM ($/sqm)",
                              title=f"D{priv_district:02d} {priv_prop_type} — Monthly Median PSM",
                              height=380)
        st.plotly_chart(fig_pt, use_container_width=True)
    else:
        st.info("Insufficient data for trend analysis.")

    # Section 6: Data Confidence
    st.markdown("---")
    st.subheader("6️⃣ Data Confidence & Assumptions")
    conf = "🟢 High" if priv_n_comps >= 10 else ("🟡 Medium" if priv_n_comps >= 5 else "🔴 Low")
    st.info(
        f"**Comparables used:** {priv_n_comps}  \n"
        f"**Year range:** {priv_range}  \n"
        f"**Similarity:** area 35%, floor 20%, tenure 20%, recency 25%  \n"
        f"**Confidence:** {conf}  \n"
        f"**Data:** URA Private Caveats (Aug 2021–2026, ~5 years)"
    )

    # Section 7: Export
    st.markdown("---")
    st.subheader("📤 Share & Export")
    priv_csv = priv_comps[[
        "project", "floor_range", "area_sqm", "tenure_clean", "type_of_sale",
        "contract_date", "price", "price_psm", "similarity_score",
    ]].copy()
    priv_csv["contract_date"] = priv_csv["contract_date"].dt.strftime("%Y-%m")
    st.download_button(
        "⬇️ Download Comps Table as CSV",
        data=priv_csv.to_csv(index=False).encode("utf-8"),
        file_name=f"priv_comps_D{priv_district:02d}_{priv_prop_type.replace('/', '_')}_{priv_today}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# DISPATCH: private mode exits early; HDB mode continues below at top level
# ════════════════════════════════════════════════════════════════════════════
if mode_comps != "🏘️ HDB Resale":
    _run_private_comps()
    st.stop()

# ── HDB RESALE MODE ───────────────────────────────────────────────────────────
towns      = sorted(df_all["town"].dropna().unique().tolist())
flat_types = sorted(df_all["flat_type"].dropna().unique().tolist())

default_town_idx = towns.index("TAMPINES") if "TAMPINES" in towns else 0
default_ft_idx   = flat_types.index("4 ROOM") if "4 ROOM" in flat_types else 0

# ── INPUT FORM ────────────────────────────────────────────────────────
with st.form("comps_form"):
    st.subheader("📝 Subject Property Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_town  = st.selectbox("Town", towns, index=default_town_idx)
        sel_block = st.text_input("Block Number (optional)", value="")
    with col2:
        sel_flat_type = st.selectbox("Flat Type", flat_types, index=default_ft_idx)
        sel_street    = st.text_input("Street Name (optional)", value="")
    with col3:
        sel_area    = st.number_input(
            "Floor Area (sqm)", min_value=20.0, max_value=300.0, value=90.0, step=1.0
        )
        sel_storey  = st.number_input(
            "Storey Midpoint", min_value=1.0, max_value=60.0, value=8.0, step=1.0
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        sel_lease = st.number_input(
            "Remaining Lease (years)", min_value=1.0, max_value=99.0,
            value=75.0, step=1.0,
        )
    with col5:
        sel_price = st.number_input(
            "Subject Price ($ — 0 = not provided)",
            min_value=0, max_value=5_000_000, value=0, step=1000,
        )
    with col6:
        sel_range = st.radio(
            "Year Range for Comps",
            ["Last 12 months", "Last 2 years", "Last 3 years"],
            index=1,
        )

    submitted = st.form_submit_button(
        "🔍 Generate Comps Report", use_container_width=True
    )

if not submitted:
    st.info(
        "Fill in the subject property details above and click "
        "**Generate Comps Report** to begin."
    )
    st.stop()

# ── COMPUTE SHARED FILTERS ────────────────────────────────────────────
today        = datetime.date.today()
current_year = today.year

range_months_map = {"Last 12 months": 12, "Last 2 years": 24, "Last 3 years": 36}
range_months = range_months_map[sel_range]
cutoff_date  = pd.Timestamp(today) - pd.DateOffset(months=range_months)

df_pool = df_all[
    (df_all["town"]      == sel_town)
    & (df_all["flat_type"] == sel_flat_type)
    & (df_all["month"]   >= cutoff_date)
].copy()

if df_pool.empty:
    st.warning(
        f"No transactions found for **{sel_town} {sel_flat_type}** in {sel_range}. "
        "Try a wider date range."
    )
    st.stop()

df_pool = df_pool.dropna(
    subset=["floor_area_sqm", "storey_mid", "remaining_lease_yrs",
            "year", "resale_price", "price_per_sqm"]
)

if df_pool.empty:
    st.warning("Insufficient clean data after filtering. Try a wider date range.")
    st.stop()

df_pool["similarity_score"] = compute_similarity(
    df_pool, sel_area, sel_storey, sel_lease, current_year
)

# Top-20 comparables
df_comps = (
    df_pool
    .nsmallest(20, "similarity_score")
    .copy()
    .reset_index(drop=True)
)
df_comps["rank"] = df_comps.index + 1

# Core metrics
weights         = 1.0 / (df_comps["similarity_score"] + 0.01)
implied_fv      = weighted_median(df_comps["resale_price"].values, weights.values)
town_median_psm = df_pool["price_per_sqm"].median()
n_comps         = len(df_comps)

subject_psm        = sel_price / sel_area if sel_price > 0 else np.nan
market_pct         = pct_rank(sel_price, df_pool["resale_price"]) if sel_price > 0 else np.nan
psm_vs_median_pct  = (
    (subject_psm - town_median_psm) / town_median_psm * 100
    if sel_price > 0 else np.nan
)

# ─────────────────────────────────────────────────────────────────────
# SECTION 1 — EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("1️⃣ Executive Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Comparable Transactions", str(n_comps))

with c2:
    st.metric("Implied Fair Value", fmt_price(implied_fv))

with c3:
    if sel_price > 0 and not np.isnan(market_pct):
        delta_dir = "above median" if market_pct > 50 else "below median"
        st.metric(
            "Market Percentile",
            f"{market_pct:.1f}th",
            delta=f"{abs(market_pct - 50):.1f}pp {delta_dir}",
        )
    else:
        st.metric(
            "Market Percentile", "—",
            help="Provide a subject price to calculate",
        )

with c4:
    if sel_price > 0 and not np.isnan(psm_vs_median_pct):
        st.metric(
            "PSM vs Town Median",
            f"${subject_psm:,.0f}/sqm",
            delta=f"{psm_vs_median_pct:+.1f}% vs ${town_median_psm:,.0f}/sqm",
            delta_color="inverse",
        )
    else:
        st.metric(
            "PSM vs Town Median",
            f"${town_median_psm:,.0f}/sqm",
            help="Town/type median PSM for selected period (no subject price provided)",
        )

# Valuation verdict banner
if sel_price > 0 and not np.isnan(implied_fv):
    gap_pct = (sel_price - implied_fv) / implied_fv * 100
    if gap_pct > 1.0:
        st.warning(
            f"⚠️ Subject price ({fmt_price(sel_price)}) is **{gap_pct:.1f}% above** "
            f"implied fair value ({fmt_price(implied_fv)}) — buyer should negotiate."
        )
    elif gap_pct < -1.0:
        st.success(
            f"✅ Subject price ({fmt_price(sel_price)}) is **{abs(gap_pct):.1f}% below** "
            f"implied fair value ({fmt_price(implied_fv)}) — potential buying opportunity."
        )
    else:
        st.info(
            f"ℹ️ Subject price ({fmt_price(sel_price)}) is broadly in line with "
            f"implied fair value ({fmt_price(implied_fv)})."
        )

# ─────────────────────────────────────────────────────────────────────
# SECTION 2 — COMPARABLE TRANSACTIONS TABLE
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("2️⃣ Comparable Transactions (Top 20)")
st.caption(
    "Ranked by composite similarity score (area 35%, storey 20%, "
    "remaining lease 25%, recency 20%). Lower score = more similar."
)

# PSM colour gradient (computed from raw float values before formatting)
raw_psm   = df_comps["price_per_sqm"].values.copy()
psm_min   = raw_psm.min()
psm_max   = raw_psm.max()
psm_range = psm_max - psm_min if psm_max > psm_min else 1.0
norm_psm  = (raw_psm - psm_min) / psm_range

# Build display DataFrame
disp = df_comps[[
    "rank", "block", "street_name", "storey_range",
    "floor_area_sqm", "remaining_lease_yrs", "month",
    "resale_price", "price_per_sqm", "similarity_score",
]].copy()

disp["block"]               = disp["block"].astype(str)
disp["month"]               = disp["month"].dt.strftime("%Y-%m")
disp["floor_area_sqm"]      = disp["floor_area_sqm"].apply(lambda x: f"{x:.0f}")
disp["remaining_lease_yrs"] = disp["remaining_lease_yrs"].apply(lambda x: f"{x:.1f}")
disp["resale_price"]        = disp["resale_price"].apply(lambda x: f"${x:,.0f}")
disp["price_per_sqm"]       = raw_psm  # temporarily numeric for cell list
disp["price_per_sqm"]       = df_comps["price_per_sqm"].apply(lambda x: f"${x:,.0f}")
disp["similarity_score"]    = disp["similarity_score"].apply(lambda x: f"{x:.3f}")

col_labels = [
    "#", "Block", "Street", "Storey Range",
    "Area (sqm)", "Rem. Lease", "Sale Date",
    "Price ($)", "PSM ($/sqm)", "Similarity",
]
n_cols = len(col_labels)   # 10
n_rows = len(disp)

# PSM is column index 8
cell_colors = []
for i in range(n_cols):
    if i == 8:
        cell_colors.append([psm_color(n) for n in norm_psm])
    else:
        cell_colors.append(["white"] * n_rows)

fig_table = go.Figure(data=[go.Table(
    columnwidth=[28, 48, 170, 88, 68, 68, 68, 90, 80, 68],
    header=dict(
        values=[f"<b>{c}</b>" for c in col_labels],
        fill_color="#1f3b6e",
        font=dict(color="white", size=12),
        align="center",
        height=32,
    ),
    cells=dict(
        values=[disp[c].tolist() for c in disp.columns],
        fill_color=cell_colors,
        font=dict(size=11),
        align=(["center", "left", "left"] + ["center"] * (n_cols - 3)),
        height=28,
    ),
)])
fig_table.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    height=min(120 + 28 * n_rows, 720),
)
st.plotly_chart(fig_table, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# SECTION 3 — PRICE DISTRIBUTION CHART
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("3️⃣ Price Distribution")

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=df_pool["resale_price"],
    nbinsx=40,
    name="All Transactions",
    marker_color="#4C78A8",
    opacity=0.75,
))

# Implied fair value line
fig_hist.add_vline(
    x=implied_fv,
    line_dash="dash",
    line_color="#E45756",
    annotation_text=f"Implied Fair Value {fmt_price(implied_fv)}",
    annotation_position="top right",
    annotation_font_color="#E45756",
    annotation_font_size=12,
)

# Subject price line + annotation
if sel_price > 0:
    fig_hist.add_vline(
        x=sel_price,
        line_dash="dot",
        line_color="#54A24B",
        annotation_text=f"Subject {fmt_price(sel_price)}",
        annotation_position="top left",
        annotation_font_color="#54A24B",
        annotation_font_size=12,
    )
    below_pct = pct_rank(sel_price, df_pool["resale_price"])
    fig_hist.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.96,
        text=f"{below_pct:.1f}% of comparable transactions were below the subject price",
        showarrow=False,
        font=dict(size=12, color="#444"),
        bgcolor="rgba(255,255,200,0.88)",
        bordercolor="#bbb",
        borderwidth=1,
        align="left",
    )

fig_hist.update_layout(
    xaxis_title="Resale Price ($)",
    yaxis_title="Number of Transactions",
    title=(
        f"{sel_town} {sel_flat_type} — Price Distribution ({sel_range})"
        f"  |  {len(df_pool):,} transactions"
    ),
    height=430,
    bargap=0.04,
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# SECTION 4 — FLOOR PREMIUM ANALYSIS
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("4️⃣ Floor Premium Analysis")

band_agg     = pd.DataFrame()   # initialised here; may be updated below
subject_band = storey_band(sel_storey)

df_band_pool = df_all[
    (df_all["town"]      == sel_town)
    & (df_all["flat_type"] == sel_flat_type)
    & (df_all["month"]   >= cutoff_date)
].dropna(subset=["storey_mid", "price_per_sqm"]).copy()

df_band_pool["storey_band"] = df_band_pool["storey_mid"].apply(storey_band)

band_agg = (
    df_band_pool
    .groupby("storey_band")
    .agg(median_psm=("price_per_sqm", "median"), count=("price_per_sqm", "size"))
    .reindex(STOREY_BAND_ORDER)
    .dropna(subset=["median_psm"])
    .reset_index()
)

if band_agg.empty:
    st.info("Insufficient data for floor premium analysis in the selected period.")
else:
    ground_rows   = band_agg[band_agg["storey_band"] == "01-03"]
    ground_psm_v  = (
        ground_rows["median_psm"].values[0]
        if not ground_rows.empty
        else band_agg["median_psm"].iloc[0]
    )
    band_agg["premium_pct"] = (
        (band_agg["median_psm"] - ground_psm_v) / ground_psm_v * 100
    )
    bar_colors = [
        "#E45756" if b == subject_band else "#4C78A8"
        for b in band_agg["storey_band"]
    ]

    fig_floor = go.Figure()
    fig_floor.add_trace(go.Bar(
        x=band_agg["storey_band"],
        y=band_agg["median_psm"],
        name="Median PSM ($)",
        marker_color=bar_colors,
        yaxis="y1",
        text=[f"${v:,.0f}" for v in band_agg["median_psm"]],
        textposition="outside",
    ))
    fig_floor.add_trace(go.Scatter(
        x=band_agg["storey_band"],
        y=band_agg["premium_pct"],
        name="Premium vs Ground (%)",
        mode="lines+markers",
        line=dict(color="#F58518", width=2),
        marker=dict(size=7),
        yaxis="y2",
    ))
    fig_floor.update_layout(
        xaxis_title="Storey Band",
        yaxis=dict(title="Median PSM ($)", side="left"),
        yaxis2=dict(
            title="Premium vs Ground Floor (%)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        title=(
            f"{sel_town} {sel_flat_type} — Storey Band vs Median PSM"
            f"  (subject band highlighted: {subject_band})"
        ),
        legend=dict(orientation="h", y=1.08),
        height=440,
    )
    st.plotly_chart(fig_floor, use_container_width=True)

    subj_band_row = band_agg[band_agg["storey_band"] == subject_band]
    if not subj_band_row.empty:
        prem_v       = subj_band_row["premium_pct"].values[0]
        subj_med_psm = subj_band_row["median_psm"].values[0]
        if prem_v >= 0:
            st.success(
                f"✅ Storey band **{subject_band}** commands a **{prem_v:+.1f}% premium** "
                f"vs ground floor (median PSM: ${subj_med_psm:,.0f}/sqm)."
            )
        else:
            st.info(
                f"ℹ️ Storey band **{subject_band}** is at a **{prem_v:.1f}% discount** "
                f"vs ground floor (median PSM: ${subj_med_psm:,.0f}/sqm)."
            )

# ─────────────────────────────────────────────────────────────────────
# SECTION 5 — MARKET TREND CONTEXT
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("5️⃣ Market Trend Context")

monthly_psm  = pd.DataFrame()   # initialised here; may be updated below
trend_label  = None
trend_color  = "#888"

trend_cutoff = pd.Timestamp(today) - pd.DateOffset(months=24)
df_trend = df_all[
    (df_all["town"]      == sel_town)
    & (df_all["flat_type"] == sel_flat_type)
    & (df_all["month"]   >= trend_cutoff)
].dropna(subset=["price_per_sqm"]).copy()

if df_trend.empty:
    st.info("Insufficient data for trend analysis.")
else:
    df_trend["month_ts"] = df_trend["month"].dt.to_period("M").dt.to_timestamp()
    monthly_psm = (
        df_trend
        .groupby("month_ts")["price_per_sqm"]
        .median()
        .reset_index()
        .sort_values("month_ts")
        .rename(columns={"month_ts": "month", "price_per_sqm": "median_psm"})
    )

    # 12-month trend annotation
    if len(monthly_psm) >= 13:
        psm_12m_ago = monthly_psm.iloc[-13]["median_psm"]
        psm_now     = monthly_psm.iloc[-1]["median_psm"]
        trend_pct   = (psm_now - psm_12m_ago) / psm_12m_ago * 100
        if trend_pct > 2.0:
            trend_label = f"UP {trend_pct:.1f}%"
            trend_color = "#54A24B"
        elif trend_pct < -2.0:
            trend_label = f"DOWN {abs(trend_pct):.1f}%"
            trend_color = "#E45756"
        else:
            trend_label = f"FLAT ({trend_pct:+.1f}%)"
            trend_color = "#F58518"

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_psm["month"],
        y=monthly_psm["median_psm"],
        mode="lines+markers",
        name="Median PSM ($/sqm)",
        line=dict(color="#4C78A8", width=2.5),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(76,120,168,0.15)",
    ))

    if trend_label:
        fig_trend.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.95,
            text=f"Market is <b>{trend_label}</b> over last 12 months",
            showarrow=False,
            font=dict(size=13, color=trend_color),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor=trend_color,
            borderwidth=1.5,
            align="right",
        )

    fig_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="Median PSM ($/sqm)",
        title=(
            f"{sel_town} {sel_flat_type} — Monthly Median PSM "
            f"(last 24 months, {len(monthly_psm)} data points)"
        ),
        height=390,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    if trend_label:
        if "UP" in trend_label:
            st.success(
                f"📈 Market trend: **{trend_label}** over the last 12 months "
                f"for {sel_town} {sel_flat_type}."
            )
        elif "DOWN" in trend_label:
            st.error(
                f"📉 Market trend: **{trend_label}** over the last 12 months "
                f"for {sel_town} {sel_flat_type}."
            )
        else:
            st.info(
                f"➡️ Market trend: **{trend_label}** over the last 12 months "
                f"for {sel_town} {sel_flat_type}."
            )

# ─────────────────────────────────────────────────────────────────────
# SECTION 6 — DATA CONFIDENCE & ASSUMPTIONS
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("6️⃣ Data Confidence & Assumptions")

comp_conf  = "🟢 High" if n_comps >= 10 else ("🟡 Medium" if n_comps >= 5 else "🔴 Low")
fv_conf    = "🟢 High" if n_comps >= 15 else "🟡 Medium"
band_conf  = (
    "🟢 High"   if (not band_agg.empty and len(band_agg) >= 4) else
    "🟡 Medium" if (not band_agg.empty) else
    "🔴 Low"
)
trend_conf = (
    "🟢 High"   if (not monthly_psm.empty and len(monthly_psm) >= 12) else
    "🟡 Medium" if (not monthly_psm.empty) else
    "🔴 Low"
)

st.info(
    f"**Comparables used:** {n_comps} transactions  \n"
    f"**Year range:** {sel_range} "
    f"(from {cutoff_date.strftime('%b %Y')} to {today.strftime('%b %Y')})  \n"
    f"**Fair value methodology:** Weighted median of the {n_comps} most similar "
    f"transactions, weighted by 1 / similarity score.  \n"
    f"It does **not** account for: renovation quality, exact facing / view, "
    f"corner or point block layout, or individual unit condition.\n\n"
    f"**Confidence levels:**  \n"
    f"- Comparable set: {comp_conf} ({n_comps} comps)  \n"
    f"- Implied fair value: {fv_conf} (weighted median)  \n"
    f"- Floor premium analysis: {band_conf}  \n"
    f"- Market trend: {trend_conf}\n\n"
    f"**Data source:** HDB Resale Transactions (data.gov.sg)"
)

# ─────────────────────────────────────────────────────────────────────
# SECTION 7 — SHARE / EXPORT
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📤 Share & Export")

# CSV download
csv_export = df_comps[[
    "block", "street_name", "storey_range", "floor_area_sqm",
    "remaining_lease_yrs", "month", "resale_price", "price_per_sqm",
    "similarity_score",
]].copy()
csv_export["block"] = csv_export["block"].astype(str)
csv_export["month"] = csv_export["month"].dt.strftime("%Y-%m")
csv_bytes = csv_export.to_csv(index=False).encode("utf-8")

safe_town = sel_town.replace(" ", "_")
safe_ft   = sel_flat_type.replace(" ", "_")

st.download_button(
    label="⬇️ Download Comps Table as CSV",
    data=csv_bytes,
    file_name=f"comps_{safe_town}_{safe_ft}_{today}.csv",
    mime="text/csv",
)

st.markdown("")

# Text summary (copy-to-clipboard style)
top5       = df_comps.head(5)
top5_lines = "\n".join(
    "  {:d}. Blk {} {} | {} | {:.0f}sqm | ${:,.0f} | {}".format(
        i + 1,
        str(row["block"]),
        row["street_name"],
        row["storey_range"],
        row["floor_area_sqm"],
        row["resale_price"],
        row["month"].strftime("%Y-%m"),
    )
    for i, (_, row) in enumerate(top5.iterrows())
)

mkt_pct_str   = f"{market_pct:.1f}th"       if sel_price > 0 and not np.isnan(market_pct)   else "N/A"
subj_psm_str  = f"${subject_psm:,.0f}/sqm"  if sel_price > 0 and not np.isnan(subject_psm)  else "N/A"
subj_price_str = fmt_price(sel_price)        if sel_price > 0 else "Not provided"

text_summary = (
    f"COMPS REPORT — {sel_town} {sel_flat_type} {sel_area:.0f}sqm\n"
    f"Date: {today.strftime('%d %b %Y')}\n"
    f"Subject Price: {subj_price_str}\n"
    f"Implied Fair Value: {fmt_price(implied_fv)} "
    f"(based on {n_comps} comparables, {sel_range})\n"
    f"Market median PSM: ${town_median_psm:,.0f}/sqm\n"
    f"Subject PSM: {subj_psm_str} ({mkt_pct_str} percentile)\n"
    f"\nTop 5 Comparable Transactions:\n"
    f"{top5_lines}\n"
    f"\nData: HDB Resale Transactions, data.gov.sg\n"
    f"Generated by PropertyDecisionSupport — Smart Money Comps Report (G16)"
)

st.markdown("**Copy-to-Clipboard Summary:**")
st.code(text_summary, language=None)

