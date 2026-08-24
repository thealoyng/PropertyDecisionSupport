"""
Page 18 -- Property Scout
==========================
Smart Money: two tools for finding the right property.

Tabs:
  1. Client Fit Shortlister (G4) -- rank all 27 HDB towns by client fit
  2. Town Comparison (G5)        -- side-by-side stats for 2-4 towns
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

from eda_helpers import (
    load_clean, load_mrt, fmt_price, TOWN_CENTROIDS,
    load_condo_clean, DISTRICT_CENTROIDS, floor_range_mid,
)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Scout",
    page_icon="\U0001f9ed",
    layout="wide",
)

st.title("\U0001f9ed Property Scout")
st.caption(
    "Smart Money: find the right area for your client, then compare shortlisted options side-by-side. "
    "Toggle between HDB Resale towns and Private (Condo) districts."
)

mode_scout = st.radio(
    "Property type",
    ["🏘️ HDB Resale", "🏢 Private (Condo)"],
    horizontal=True,
    key="scout_mode",
)
st.divider()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
COORDS_CSV = os.path.join(DATA_DIR, "address_coords.csv")

ALL_TOWNS = sorted(TOWN_CENTROIDS.keys())

FLAT_TYPE_ORDER = [
    "1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM",
    "5 ROOM", "EXECUTIVE", "MULTI GENERATION",
]

# ── helpers ────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def minmax(s: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]. Returns 0.5 if constant."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


# ── cached data loaders ────────────────────────────────────────────────────────

@st.cache_data
def load_resale():
    """Full cleaned resale dataset with parsed month."""
    df = load_clean()
    df["month"] = pd.to_datetime(df["month"])
    return df


@st.cache_data
def load_mrt_data():
    """MRT station coordinates (lat/lon)."""
    return load_mrt()


@st.cache_data
def mrt_proximity_by_town():
    """
    For each town centroid, compute distance to the nearest MRT station (km)
    and derive mrt_proximity_score = 1 / (1 + min_dist).
    Falls back to town-centroid distances if MRT data unavailable.
    """
    mrt_df = load_mrt_data()
    records = []
    for town, (t_lat, t_lon) in TOWN_CENTROIDS.items():
        if mrt_df.empty or "lat" not in mrt_df.columns or "lon" not in mrt_df.columns:
            min_dist = 1.0  # neutral fallback
        else:
            dists = mrt_df.apply(
                lambda r: haversine_km(t_lat, t_lon, r["lat"], r["lon"]), axis=1
            )
            min_dist = float(dists.min())
        records.append({
            "town": town,
            "mrt_distance_km": round(min_dist, 3),
            "mrt_proximity_score": 1.0 / (1.0 + min_dist),
        })
    return pd.DataFrame(records).set_index("town")


# ── TAB 1 helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def compute_town_metrics(max_budget: float, flat_types: tuple):
    """
    Compute per-town metrics for the shortlister (Tab 1).
    Uses last 3 years of data filtered to selected flat types.
    Returns a DataFrame indexed by town.
    """
    df = load_resale()
    max_date = df["month"].max()
    cutoff_3yr = max_date - pd.DateOffset(years=3)
    cutoff_5yr = max_date - pd.DateOffset(years=5)
    cutoff_7yr = max_date - pd.DateOffset(years=7)

    # Base: last 3 years, selected flat types
    mask_ft = df["flat_type"].isin(flat_types) if flat_types else pd.Series(True, index=df.index)
    df3 = df[mask_ft & (df["month"] >= cutoff_3yr)].copy()

    # Price-growth reference slices (all flat types for continuity)
    df_last2 = df[mask_ft & (df["month"] >= max_date - pd.DateOffset(years=2))].copy()
    df_5yrago = df[mask_ft & (df["month"] >= cutoff_7yr) & (df["month"] < cutoff_5yr)].copy()

    records = []
    for town in ALL_TOWNS:
        t3 = df3[df3["town"] == town]
        t_l2 = df_last2[df_last2["town"] == town]
        t_5a = df_5yrago[df_5yrago["town"] == town]

        if t3.empty:
            continue

        psm_vals = t3["price_per_sqm"].dropna()
        median_psm = float(psm_vals.median()) if not psm_vals.empty else np.nan
        psm_std = float(psm_vals.std()) if len(psm_vals) > 1 else 0.0
        transaction_count = len(t3)
        avg_remaining_lease = float(t3["remaining_lease_yrs"].mean()) if "remaining_lease_yrs" in t3.columns else np.nan
        affordability_count = int((t3["resale_price"] <= max_budget).sum())

        # price_growth_5yr
        psm_last2 = float(t_l2["price_per_sqm"].median()) if not t_l2.empty else np.nan
        psm_5ago = float(t_5a["price_per_sqm"].median()) if not t_5a.empty else np.nan
        if pd.notna(psm_last2) and pd.notna(psm_5ago) and psm_5ago > 0:
            price_growth_5yr = (psm_last2 - psm_5ago) / psm_5ago * 100.0
        else:
            price_growth_5yr = np.nan

        records.append({
            "town": town,
            "median_psm": median_psm,
            "psm_std": psm_std,
            "transaction_count": transaction_count,
            "avg_remaining_lease": avg_remaining_lease,
            "affordability_count": affordability_count,
            "price_growth_5yr": price_growth_5yr,
        })

    return pd.DataFrame(records).set_index("town")


@st.cache_data
def build_shortlist(max_budget: float, flat_types: tuple, priority_weights: tuple):
    """
    Compute composite scores for all towns.
    priority_weights: tuple of (priority_label, weight) pairs for selected priorities.
    Returns a scored DataFrame indexed by town.
    """
    metrics_df = compute_town_metrics(max_budget, flat_types)
    mrt_df = mrt_proximity_by_town()

    # Join MRT data
    df = metrics_df.join(mrt_df, how="left")

    # Fill any missing MRT rows with neutral values
    df["mrt_proximity_score"] = df["mrt_proximity_score"].fillna(0.5)
    df["mrt_distance_km"] = df["mrt_distance_km"].fillna(1.0)

    # -- Normalise all raw metrics to [0,1] (higher = better) --
    df["norm_mrt_proximity"] = minmax(df["mrt_proximity_score"])            # higher = closer
    df["norm_affordable_psm"] = minmax(-df["median_psm"].fillna(df["median_psm"].max()))  # lower psm = better
    df["norm_remaining_lease"] = minmax(df["avg_remaining_lease"].fillna(0))
    df["norm_transaction_count"] = minmax(df["transaction_count"])
    df["norm_low_volatility"] = minmax(-df["psm_std"].fillna(df["psm_std"].max()))        # lower std = better
    df["norm_price_growth"] = minmax(df["price_growth_5yr"].fillna(df["price_growth_5yr"].median()))

    # Mapping from priority label to normalised column
    PRIORITY_COL_MAP = {
        "Near MRT": "norm_mrt_proximity",
        "Affordable PSM": "norm_affordable_psm",
        "Long remaining lease": "norm_remaining_lease",
        "High transaction volume": "norm_transaction_count",
        "Low price volatility": "norm_low_volatility",
        "Price growth (last 5yr)": "norm_price_growth",
    }

    # Compute weighted composite score
    total_weight = 0.0
    weighted_sum = pd.Series(0.0, index=df.index)
    for label, w in priority_weights:
        col = PRIORITY_COL_MAP.get(label)
        if col and col in df.columns:
            weighted_sum += df[col] * w
            total_weight += w

    if total_weight > 0:
        df["composite_score"] = weighted_sum / total_weight
    else:
        # Equal weighting across all dimensions
        norm_cols = list(PRIORITY_COL_MAP.values())
        df["composite_score"] = df[norm_cols].mean(axis=1)

    df["composite_score"] = (df["composite_score"] * 100).round(1)
    df = df.sort_values("composite_score", ascending=False)
    return df


# ── TAB 2 helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def compute_town_comparison(towns: tuple, flat_type: str, year_range: int):
    """
    Compute comparison metrics per town for selected flat type and year window.
    Returns a dict of DataFrames / series for display.
    """
    df = load_resale()
    max_date = df["month"].max()
    cutoff = max_date - pd.DateOffset(years=year_range)
    cutoff_1yr = max_date - pd.DateOffset(years=1)
    cutoff_5yr = max_date - pd.DateOffset(years=5)
    cutoff_6yr = max_date - pd.DateOffset(years=6)

    base = df[(df["flat_type"] == flat_type) & (df["month"] >= cutoff)].copy()

    summary = {}
    for town in towns:
        t = base[base["town"] == town]
        if t.empty:
            summary[town] = None
            continue

        psm = t["price_per_sqm"].dropna()
        price = t["resale_price"].dropna()

        # 1yr change
        t_1yr = t[t["month"] >= cutoff_1yr]["price_per_sqm"].dropna()
        t_prev1yr = t[(t["month"] < cutoff_1yr)]["price_per_sqm"].dropna()
        psm_1yr_chg = (
            (t_1yr.median() - t_prev1yr.median()) / t_prev1yr.median() * 100
            if not t_1yr.empty and not t_prev1yr.empty and t_prev1yr.median() > 0
            else np.nan
        )

        # 5yr change
        t_5yr_now = t[t["month"] >= cutoff_5yr]["price_per_sqm"].dropna()
        t_5yr_before = t[(t["month"] >= cutoff_6yr) & (t["month"] < cutoff_5yr)]["price_per_sqm"].dropna()
        psm_5yr_chg = (
            (t_5yr_now.median() - t_5yr_before.median()) / t_5yr_before.median() * 100
            if year_range >= 5 and not t_5yr_now.empty and not t_5yr_before.empty and t_5yr_before.median() > 0
            else np.nan
        )

        # Most common flat model
        if "flat_model" in t.columns and not t["flat_model"].dropna().empty:
            most_common_model = t["flat_model"].value_counts().idxmax()
        else:
            most_common_model = "N/A"

        summary[town] = {
            "median_price": float(price.median()) if not price.empty else np.nan,
            "median_psm": float(psm.median()) if not psm.empty else np.nan,
            "transaction_count": len(t),
            "psm_p10": float(psm.quantile(0.10)) if not psm.empty else np.nan,
            "psm_p90": float(psm.quantile(0.90)) if not psm.empty else np.nan,
            "avg_remaining_lease": float(t["remaining_lease_yrs"].mean()) if "remaining_lease_yrs" in t.columns else np.nan,
            "psm_1yr_chg": psm_1yr_chg,
            "psm_5yr_chg": psm_5yr_chg,
            "min_price": float(price.min()) if not price.empty else np.nan,
            "max_price": float(price.max()) if not price.empty else np.nan,
            "most_common_model": most_common_model,
        }

    return summary, base


@st.cache_data
def psm_trend_data(towns: tuple, flat_type: str, year_range: int):
    """Return monthly median PSM time series for selected towns."""
    df = load_resale()
    max_date = df["month"].max()
    cutoff = max_date - pd.DateOffset(years=year_range)
    base = df[
        (df["flat_type"] == flat_type)
        & (df["month"] >= cutoff)
        & (df["town"].isin(towns))
    ].copy()
    if base.empty:
        return pd.DataFrame()
    trend = (
        base.groupby(["month", "town"])["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"price_per_sqm": "median_psm"})
    )
    return trend


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

# ── Private-mode helper functions ──────────────────────────────────────────────

@st.cache_data
def load_condo_for_scout():
    """Load and lightly pre-process condo_clean for scout analysis."""
    df = load_condo_clean()
    if df.empty:
        return df
    df = df[df["property_type_broad"].isin(["Condo/Apartment", "EC"])].copy()
    df["contract_date"] = pd.to_datetime(df["contract_date"], errors="coerce")
    return df


@st.cache_data
def compute_district_metrics(max_budget: float, prop_types: tuple):
    """Compute per-district metrics for the private shortlister (Tab 1)."""
    df = load_condo_for_scout()
    if df.empty:
        return pd.DataFrame()
    if prop_types:
        df = df[df["property_type_broad"].isin(prop_types)]
    max_date = df["contract_date"].max()
    cutoff_3yr = max_date - pd.DateOffset(years=3)
    cutoff_5yr = max_date - pd.DateOffset(years=5)
    cutoff_7yr = max_date - pd.DateOffset(years=7)
    df3 = df[df["contract_date"] >= cutoff_3yr].copy()
    df_l2 = df[df["contract_date"] >= max_date - pd.DateOffset(years=2)].copy()
    df_5a = df[(df["contract_date"] >= cutoff_7yr) & (df["contract_date"] < cutoff_5yr)].copy()

    records = []
    for dist_num, (dist_name, d_lat, d_lon) in DISTRICT_CENTROIDS.items():
        t3 = df3[df3["district"] == dist_num]
        tl2 = df_l2[df_l2["district"] == dist_num]
        t5a = df_5a[df_5a["district"] == dist_num]
        if t3.empty:
            continue
        psm = t3["price_psm"].dropna()
        median_psm = float(psm.median()) if not psm.empty else np.nan
        psm_std = float(psm.std()) if len(psm) > 1 else 0.0
        txn_count = len(t3)
        affordability_count = int((t3["price"].dropna() <= max_budget).sum())
        freehold_pct = float((t3["tenure_clean"] == "Freehold").mean() * 100) if not t3.empty else 0.0
        # price growth
        psm_l2 = float(tl2["price_psm"].median()) if not tl2.empty else np.nan
        psm_5a = float(t5a["price_psm"].median()) if not t5a.empty else np.nan
        if pd.notna(psm_l2) and pd.notna(psm_5a) and psm_5a > 0:
            price_growth = (psm_l2 - psm_5a) / psm_5a * 100
        else:
            price_growth = np.nan
        records.append({
            "district": dist_num,
            "district_name": dist_name,
            "d_lat": d_lat,
            "d_lon": d_lon,
            "median_psm": median_psm,
            "psm_std": psm_std,
            "transaction_count": txn_count,
            "affordability_count": affordability_count,
            "freehold_pct": freehold_pct,
            "price_growth_5yr": price_growth,
        })
    if not records:
        return pd.DataFrame()
    result = pd.DataFrame(records).set_index("district")
    # MRT proximity
    mrt_df = load_mrt_data()
    mrt_prox = []
    for dist_num in result.index:
        d_lat = result.loc[dist_num, "d_lat"]
        d_lon = result.loc[dist_num, "d_lon"]
        if mrt_df.empty or "lat" not in mrt_df.columns:
            mrt_prox.append(1.0)
        else:
            dists = mrt_df.apply(
                lambda r: haversine_km(d_lat, d_lon, r["lat"], r["lon"]), axis=1
            )
            mrt_prox.append(float(dists.min()))
    result["mrt_distance_km"] = mrt_prox
    result["mrt_proximity_score"] = 1.0 / (1.0 + result["mrt_distance_km"])
    return result


@st.cache_data
def build_district_shortlist(max_budget: float, prop_types: tuple, priority_weights: tuple):
    """Compute composite scores for all districts (private mode Tab 1)."""
    df = compute_district_metrics(max_budget, prop_types)
    if df.empty:
        return df
    # Normalise to [0,1]
    def _minmax(s):
        lo, hi = s.min(), s.max()
        return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)

    df["norm_mrt_proximity"]    = _minmax(df["mrt_proximity_score"])
    df["norm_affordable_psm"]   = _minmax(-df["median_psm"].fillna(df["median_psm"].max()))
    df["norm_transaction_count"]= _minmax(df["transaction_count"])
    df["norm_low_volatility"]   = _minmax(-df["psm_std"].fillna(df["psm_std"].max()))
    df["norm_price_growth"]     = _minmax(df["price_growth_5yr"].fillna(df["price_growth_5yr"].median()))
    df["norm_freehold"]         = _minmax(df["freehold_pct"])

    PRIV_PRIORITY_MAP = {
        "Near MRT":           "norm_mrt_proximity",
        "Affordable PSM":     "norm_affordable_psm",
        "High liquidity":     "norm_transaction_count",
        "Low volatility":     "norm_low_volatility",
        "Price growth (5yr)": "norm_price_growth",
        "Freehold available": "norm_freehold",
    }
    total_weight = 0.0
    weighted_sum = pd.Series(0.0, index=df.index)
    for label, w in priority_weights:
        col = PRIV_PRIORITY_MAP.get(label)
        if col and col in df.columns:
            weighted_sum += df[col] * w
            total_weight += w
    if total_weight > 0:
        df["composite_score"] = (weighted_sum / total_weight * 100).round(1)
    else:
        df["composite_score"] = df[[c for c in PRIV_PRIORITY_MAP.values() if c in df.columns]].mean(axis=1) * 100
    return df.sort_values("composite_score", ascending=False)


@st.cache_data
def compute_district_comparison(districts: tuple, prop_type_broad: str, year_range: int):
    """Compute comparison metrics per district for Tab 2 private mode."""
    df = load_condo_for_scout()
    if df.empty:
        return {}, pd.DataFrame()
    max_date = df["contract_date"].max()
    cutoff = max_date - pd.DateOffset(years=year_range)
    base = df[
        (df["property_type_broad"] == prop_type_broad)
        & (df["contract_date"] >= cutoff)
        & (df["district"].isin(districts))
    ].copy()
    summary = {}
    for d in districts:
        t = base[base["district"] == d]
        if t.empty:
            summary[d] = None
            continue
        psm = t["price_psm"].dropna()
        cutoff_1yr = max_date - pd.DateOffset(years=1)
        t_1yr  = t[t["contract_date"] >= cutoff_1yr]["price_psm"].dropna()
        t_prev = t[t["contract_date"] < cutoff_1yr]["price_psm"].dropna()
        psm_1yr_chg = (
            (t_1yr.median() - t_prev.median()) / t_prev.median() * 100
            if not t_1yr.empty and not t_prev.empty and t_prev.median() > 0
            else np.nan
        )
        tenure_mode = t["tenure_clean"].mode()[0] if not t["tenure_clean"].dropna().empty else "N/A"
        summary[d] = {
            "median_psm":        float(psm.median()) if not psm.empty else np.nan,
            "psm_p10":           float(psm.quantile(0.10)) if not psm.empty else np.nan,
            "psm_p90":           float(psm.quantile(0.90)) if not psm.empty else np.nan,
            "transaction_count": len(t),
            "psm_1yr_chg":       psm_1yr_chg,
            "new_sale_pct":      float((t["type_of_sale"] == "New Sale").mean() * 100),
            "resale_pct":        float((t["type_of_sale"] == "Resale").mean() * 100),
            "freehold_pct":      float((t["tenure_clean"] == "Freehold").mean() * 100),
            "most_common_tenure":tenure_mode,
        }
    return summary, base


@st.cache_data
def district_psm_trend(districts: tuple, prop_type_broad: str, year_range: int):
    """Monthly median PSM trend for selected districts (private mode)."""
    df = load_condo_for_scout()
    if df.empty:
        return pd.DataFrame()
    max_date = df["contract_date"].max()
    cutoff = max_date - pd.DateOffset(years=year_range)
    base = df[
        (df["property_type_broad"] == prop_type_broad)
        & (df["contract_date"] >= cutoff)
        & (df["district"].isin(districts))
    ].copy()
    if base.empty:
        return pd.DataFrame()
    base["month_ts"] = base["contract_date"].dt.to_period("M").dt.to_timestamp()
    trend = (
        base.groupby(["month_ts", "district"])["price_psm"]
        .median()
        .reset_index()
        .rename(columns={"month_ts": "month", "price_psm": "median_psm"})
    )
    return trend


# ── TABS ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "\U0001f3af Area Shortlister",
    "\U0001f3d9\ufe0f Area Comparison",
    "\U0001f306 Lifestyle Matching",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Area Shortlister
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
  if mode_scout == "🏘️ HDB Resale":
    st.subheader("\U0001f3af Client Fit Shortlister — HDB Towns")
    st.markdown(
        "Given a client's budget, flat type preference, and lifestyle priorities, "
        "this tool **ranks all 27 HDB towns** by composite fit score."
    )

    PRIORITY_OPTIONS = [
        "Near MRT",
        "Affordable PSM",
        "Long remaining lease",
        "High transaction volume",
        "Low price volatility",
        "Price growth (last 5yr)",
    ]

    with st.form("shortlist_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            max_budget = st.number_input(
                "Max Budget ($)",
                min_value=100_000,
                max_value=3_000_000,
                value=700_000,
                step=10_000,
                format="%d",
            )
            flat_types_sel = st.multiselect(
                "Preferred Flat Types",
                options=FLAT_TYPE_ORDER,
                default=["4 ROOM", "5 ROOM"],
            )
        with col_b:
            priorities_sel = st.multiselect(
                "Lifestyle Priorities",
                options=PRIORITY_OPTIONS,
                default=["Near MRT", "Affordable PSM", "Long remaining lease"],
            )

        # Per-priority sliders
        if priorities_sel:
            st.markdown("**Priority Weights** (1 = low, 5 = high)")
            weight_cols = st.columns(min(len(priorities_sel), 6))
            priority_weights_dict = {}
            for i, prio in enumerate(priorities_sel):
                with weight_cols[i % 6]:
                    priority_weights_dict[prio] = st.slider(
                        prio, min_value=1, max_value=5, value=3, key=f"w_{prio}"
                    )
        else:
            priority_weights_dict = {}

        submitted = st.form_submit_button("\U0001f50d Rank Towns", type="primary")

    if submitted or True:   # show on first load with defaults
        if not flat_types_sel:
            st.warning("Please select at least one flat type.")
            st.stop()

        priority_weights_tuple = tuple(
            (p, priority_weights_dict.get(p, 3)) for p in priorities_sel
        )

        with st.spinner("Computing town scores…"):
            scored_df = build_shortlist(
                max_budget=float(max_budget),
                flat_types=tuple(flat_types_sel),
                priority_weights=priority_weights_tuple,
            )

        if scored_df.empty:
            st.error("No data found for the selected flat types. Please broaden your selection.")
            st.stop()

        # ── Display: Ranked Table ──────────────────────────────────────────
        st.markdown("### \U0001f4cb Ranked Towns")

        display_df = scored_df[
            [
                "composite_score",
                "median_psm",
                "affordability_count",
                "avg_remaining_lease",
                "psm_std",
                "mrt_distance_km",
                "price_growth_5yr",
            ]
        ].copy()
        display_df.index.name = "Town"
        display_df.columns = [
            "Score (/100)",
            "Median PSM ($)",
            "Affordable Units (in budget)",
            "Avg Remaining Lease (yr)",
            "PSM Volatility (std)",
            "Nearest MRT (km)",
            "5yr PSM Growth (%)",
        ]
        display_df["Median PSM ($)"] = display_df["Median PSM ($)"].map(
            lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
        )
        display_df["Avg Remaining Lease (yr)"] = display_df["Avg Remaining Lease (yr)"].map(
            lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"
        )
        display_df["PSM Volatility (std)"] = display_df["PSM Volatility (std)"].map(
            lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
        )
        display_df["Nearest MRT (km)"] = display_df["Nearest MRT (km)"].map(
            lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
        )
        display_df["5yr PSM Growth (%)"] = display_df["5yr PSM Growth (%)"].map(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A"
        )
        display_df.insert(0, "Rank", range(1, len(display_df) + 1))

        st.dataframe(display_df, use_container_width=True)

        top5 = scored_df.head(5).index.tolist()

        # ── Display: Radar Chart (top 5) ───────────────────────────────────
        st.markdown("### \U0001f578\ufe0f Top-5 Town Radar")

        RADAR_DIMS = [
            ("Near MRT", "norm_mrt_proximity"),
            ("Affordable PSM", "norm_affordable_psm"),
            ("Remaining Lease", "norm_remaining_lease"),
            ("Liquidity", "norm_transaction_count"),
            ("Low Volatility", "norm_low_volatility"),
            ("5yr Growth", "norm_price_growth"),
        ]
        dim_labels = [d[0] for d in RADAR_DIMS]
        dim_cols = [d[1] for d in RADAR_DIMS]

        fig_radar = go.Figure()
        for town in top5:
            if town not in scored_df.index:
                continue
            row = scored_df.loc[town]
            vals = [float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0 for c in dim_cols]
            vals_closed = vals + [vals[0]]
            labels_closed = dim_labels + [dim_labels[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=labels_closed,
                fill="toself",
                name=town,
                opacity=0.65,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=480,
            title="Component Scores — Top 5 Towns (normalised 0-1)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Display: Bubble Chart ──────────────────────────────────────────
        st.markdown("### \U0001f4ca Affordability vs Lease Bubble Chart")

        bubble_df = scored_df.reset_index().copy()
        bubble_df = bubble_df.dropna(subset=["median_psm", "avg_remaining_lease"])
        bubble_df["affordability_count_display"] = bubble_df["affordability_count"].clip(lower=1)

        fig_bubble = px.scatter(
            bubble_df,
            x="median_psm",
            y="avg_remaining_lease",
            size="affordability_count_display",
            color="composite_score",
            hover_name="town",
            hover_data={
                "median_psm": ":,.0f",
                "avg_remaining_lease": ":.1f",
                "affordability_count": True,
                "composite_score": ":.1f",
                "affordability_count_display": False,
            },
            color_continuous_scale="RdYlGn",
            labels={
                "median_psm": "Median PSM ($)",
                "avg_remaining_lease": "Avg Remaining Lease (yr)",
                "composite_score": "Score",
                "affordability_count": "Units in Budget",
            },
            title="Towns: PSM vs Remaining Lease (size = units within budget, colour = score)",
            size_max=60,
        )
        fig_bubble.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="white")))
        st.plotly_chart(fig_bubble, use_container_width=True)

        # ── Data Confidence ────────────────────────────────────────────────
        st.info(
            "\U0001f4ca **DATA CONFIDENCE: Medium.** "
            "This shortlist scores towns on public transaction data. "
            "It does not account for: current listing availability, specific unit quality, "
            "renovation state, exact MRT walking time, school catchments, or personal lifestyle "
            "factors not captured here. Use as a starting point for targeted research."
        )

  else:
    # ── PRIVATE MODE: District Shortlister ────────────────────────────────────
    st.subheader("🏢 Area Shortlister — Private Districts (D01–D28)")
    st.markdown(
        "Given a budget, property type, and priorities, rank all **28 Singapore postal districts** "
        "by composite fit score using URA private transaction caveats."
    )
    st.warning("⚠️ Private data covers Aug 2021–2026 (~5 years). Scores based on recent market only.")

    PRIV_PRIORITY_OPTIONS = [
        "Near MRT", "Affordable PSM", "High liquidity",
        "Low volatility", "Price growth (5yr)", "Freehold available",
    ]
    PRIV_PROP_TYPES = ["Condo/Apartment", "EC"]
    PRIV_DIST_LABELS = {d: f"D{d:02d} — {name}" for d, (name, _, _) in DISTRICT_CENTROIDS.items()}

    with st.form("priv_shortlist_form"):
        pa, pb = st.columns(2)
        with pa:
            priv_budget = st.number_input(
                "Max Budget ($)", 500_000, 10_000_000, 1_500_000, step=50_000,
                format="%d", key="scout_priv_budget",
            )
            priv_prop_types = st.multiselect(
                "Property Type", PRIV_PROP_TYPES, default=PRIV_PROP_TYPES,
                key="scout_priv_type",
            )
        with pb:
            priv_priorities = st.multiselect(
                "Priorities", PRIV_PRIORITY_OPTIONS,
                default=["Near MRT", "Affordable PSM", "Price growth (5yr)"],
                key="scout_priv_pri",
            )
        if priv_priorities:
            st.markdown("**Priority Weights** (1 = low, 5 = high)")
            wcols = st.columns(min(len(priv_priorities), 6))
            priv_weights = {}
            for i, pri in enumerate(priv_priorities):
                with wcols[i % 6]:
                    priv_weights[pri] = st.slider(
                        pri, 1, 5, 3, key=f"scout_priv_w_{i}"
                    )
        else:
            priv_weights = {}
        priv_submitted = st.form_submit_button("🔍 Rank Districts", type="primary")

    if priv_submitted or True:
        if not priv_prop_types:
            st.warning("Select at least one property type.")
            st.stop()
        with st.spinner("Computing district scores…"):
            priv_pw_tuple = tuple((p, priv_weights.get(p, 3)) for p in priv_priorities)
            scored_dist = build_district_shortlist(
                max_budget=float(priv_budget),
                prop_types=tuple(priv_prop_types),
                priority_weights=priv_pw_tuple,
            )
        if scored_dist.empty:
            st.error("No private transaction data found. Ensure condo_clean.csv exists.")
        else:
            st.markdown("### 📋 Ranked Districts")
            disp_d = scored_dist[[
                "composite_score", "district_name", "median_psm",
                "affordability_count", "psm_std", "mrt_distance_km",
                "price_growth_5yr", "freehold_pct",
            ]].copy()
            disp_d.index = [f"D{i:02d}" for i in disp_d.index]
            disp_d.insert(0, "Rank", range(1, len(disp_d) + 1))
            disp_d.columns = ["Rank", "Score (/100)", "District", "Median PSM ($)",
                               "Units in Budget", "PSM Volatility", "Nearest MRT (km)",
                               "5yr PSM Growth (%)", "Freehold %"]
            disp_d["Median PSM ($)"] = disp_d["Median PSM ($)"].map(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
            )
            disp_d["PSM Volatility"] = disp_d["PSM Volatility"].map(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
            )
            disp_d["Nearest MRT (km)"] = disp_d["Nearest MRT (km)"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
            )
            disp_d["5yr PSM Growth (%)"] = disp_d["5yr PSM Growth (%)"].map(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A"
            )
            disp_d["Freehold %"] = disp_d["Freehold %"].map(
                lambda v: f"{v:.0f}%" if pd.notna(v) else "N/A"
            )
            st.dataframe(disp_d, use_container_width=True)

            top5_d = scored_dist.head(5).index.tolist()

            # Radar chart (top 5 districts)
            st.markdown("### 🕸️ Top-5 District Radar")
            PRIV_RADAR = [
                ("Near MRT",      "norm_mrt_proximity"),
                ("Affordable PSM","norm_affordable_psm"),
                ("Liquidity",     "norm_transaction_count"),
                ("Low Volatility","norm_low_volatility"),
                ("5yr Growth",    "norm_price_growth"),
                ("Freehold",      "norm_freehold"),
            ]
            radar_labels = [r[0] for r in PRIV_RADAR]
            radar_cols   = [r[1] for r in PRIV_RADAR]
            fig_r2 = go.Figure()
            for dn in top5_d:
                if dn not in scored_dist.index:
                    continue
                row = scored_dist.loc[dn]
                vals = [float(row[c]) if pd.notna(row.get(c, np.nan)) else 0.0 for c in radar_cols]
                fig_r2.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
                    fill="toself", name=f"D{dn:02d}", opacity=0.65,
                ))
            fig_r2.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=480,
                title="Component Scores — Top 5 Districts (normalised 0-1)",
            )
            st.plotly_chart(fig_r2, use_container_width=True)

            # Bubble chart
            st.markdown("### 📈 Affordability vs Score")
            bubble_d = scored_dist.reset_index()
            bubble_d = bubble_d.dropna(subset=["median_psm"])
            bubble_d["label"] = bubble_d["district"].apply(lambda d: f"D{d:02d}")
            bubble_d["bubble_size"] = bubble_d["affordability_count"].clip(lower=1)
            fig_b2 = px.scatter(
                bubble_d, x="median_psm", y="composite_score",
                size="bubble_size", color="composite_score",
                hover_name="label",
                hover_data={"median_psm": ":,.0f", "composite_score": ":.1f",
                            "affordability_count": True, "bubble_size": False,
                            "district_name": True},
                color_continuous_scale="RdYlGn",
                labels={"median_psm": "Median PSM ($)", "composite_score": "Score (/100)"},
                title="Districts: PSM vs Score (size = units within budget)",
                size_max=55,
            )
            fig_b2.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="white")))
            st.plotly_chart(fig_b2, use_container_width=True)
            st.info(
                "📊 **DATA CONFIDENCE: Medium.** Based on URA private transaction caveats "
                "(Aug 2021–2026). Does not reflect current listing availability, developer "
                "incentives, or unit-level details."
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Area Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
  if mode_scout == "🏘️ HDB Resale":
    st.subheader("\U0001f3d9\ufe0f Town Comparison")
    st.markdown(
        "Side-by-side statistical comparison of **2–4 towns** for a specific flat type and time window."
    )

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        towns_sel = st.multiselect(
            "Select Towns (2–4)",
            options=ALL_TOWNS,
            default=["TAMPINES", "JURONG WEST", "SENGKANG"],
        )
    with c2:
        flat_type_sel = st.selectbox(
            "Flat Type",
            options=FLAT_TYPE_ORDER,
            index=FLAT_TYPE_ORDER.index("4 ROOM"),
        )
    with c3:
        year_range_map = {"Last 1 year": 1, "Last 3 years": 3, "Last 5 years": 5, "Last 10 years": 10}
        year_range_label = st.selectbox("Time Window", options=list(year_range_map.keys()), index=1)
        year_range = year_range_map[year_range_label]

    if len(towns_sel) < 2:
        st.warning("Please select 2 to 4 towns for comparison.")
        st.stop()

    if len(towns_sel) > 4:
        st.warning("Please select no more than 4 towns. Showing first 4.")
        towns_sel = towns_sel[:4]

    with st.spinner("Loading comparison data…"):
        summary, base_df = compute_town_comparison(
            towns=tuple(towns_sel),
            flat_type=flat_type_sel,
            year_range=year_range,
        )

    # Compute group medians for delta reference
    group_stats = {
        metric: np.nanmedian([
            v[metric] for v in summary.values() if v is not None and pd.notna(v.get(metric, np.nan))
        ])
        for metric in [
            "median_price", "median_psm", "transaction_count",
            "avg_remaining_lease", "psm_1yr_chg", "psm_5yr_chg",
        ]
    }

    # ── KPI Card Grid ──────────────────────────────────────────────────────
    st.markdown("### \U0001f4ca Key Metrics")

    kpi_cols = st.columns(len(towns_sel))

    METRIC_ROWS = [
        ("Median Resale Price", "median_price", fmt_price, "median_price"),
        ("Median PSM ($/sqm)", "median_psm", lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A", "median_psm"),
        ("Transactions (volume)", "transaction_count", lambda v: f"{v:,}" if pd.notna(v) else "N/A", "transaction_count"),
        ("PSM Range (P10–P90)", None, None, None),
        ("Avg Remaining Lease", "avg_remaining_lease", lambda v: f"{v:.1f} yr" if pd.notna(v) else "N/A", "avg_remaining_lease"),
        ("PSM 1yr Change", "psm_1yr_chg", lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A", "psm_1yr_chg"),
        ("PSM 5yr Change", "psm_5yr_chg", lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A", "psm_5yr_chg"),
        ("Min Transaction", "min_price", fmt_price, None),
        ("Max Transaction", "max_price", fmt_price, None),
        ("Most Common Model", "most_common_model", lambda v: str(v) if pd.notna(v) else "N/A", None),
    ]

    for col_widget, town in zip(kpi_cols, towns_sel):
        with col_widget:
            st.markdown(f"**{town}**")
            s = summary.get(town)
            if s is None:
                st.warning("No data")
                continue

            # Median Price
            v = s["median_price"]
            g = group_stats.get("median_price", np.nan)
            delta_str = f"{(v - g) / g * 100:+.1f}% vs group" if pd.notna(v) and pd.notna(g) and g > 0 else None
            st.metric("Median Price", fmt_price(v), delta=delta_str,
                      delta_color="inverse")

            # Median PSM
            v = s["median_psm"]
            g = group_stats.get("median_psm", np.nan)
            delta_str = f"{(v - g) / g * 100:+.1f}% vs group" if pd.notna(v) and pd.notna(g) and g > 0 else None
            st.metric("Median PSM ($/sqm)", f"${v:,.0f}" if pd.notna(v) else "N/A",
                      delta=delta_str, delta_color="inverse")

            # Transactions
            v = s["transaction_count"]
            g = group_stats.get("transaction_count", np.nan)
            delta_str = f"{int(v - g):+,} vs group" if pd.notna(v) and pd.notna(g) else None
            st.metric("Transactions", f"{int(v):,}", delta=delta_str)

            # PSM Range
            p10 = s["psm_p10"]
            p90 = s["psm_p90"]
            rng_str = (
                f"${p10:,.0f} – ${p90:,.0f}" if pd.notna(p10) and pd.notna(p90) else "N/A"
            )
            st.metric("PSM Range (P10–P90)", rng_str)

            # Remaining Lease
            v = s["avg_remaining_lease"]
            g = group_stats.get("avg_remaining_lease", np.nan)
            delta_str = f"{v - g:+.1f} yr vs group" if pd.notna(v) and pd.notna(g) else None
            st.metric("Avg Remaining Lease", f"{v:.1f} yr" if pd.notna(v) else "N/A",
                      delta=delta_str)

            # 1yr change
            v = s["psm_1yr_chg"]
            g = group_stats.get("psm_1yr_chg", np.nan)
            delta_str = f"{v - g:+.1f}pp vs group" if pd.notna(v) and pd.notna(g) else None
            st.metric("PSM 1yr Change", f"{v:+.1f}%" if pd.notna(v) else "N/A",
                      delta=delta_str)

            # 5yr change
            if year_range >= 5:
                v = s["psm_5yr_chg"]
                g = group_stats.get("psm_5yr_chg", np.nan)
                delta_str = f"{v - g:+.1f}pp vs group" if pd.notna(v) and pd.notna(g) else None
                st.metric("PSM 5yr Change", f"{v:+.1f}%" if pd.notna(v) else "N/A",
                          delta=delta_str)

            # Min / max
            st.metric("Min Transaction", fmt_price(s["min_price"]))
            st.metric("Max Transaction", fmt_price(s["max_price"]))

            # Most common model
            st.metric("Most Common Model", str(s["most_common_model"]))

    st.divider()

    # ── Violin Plot — Price Distribution ───────────────────────────────────
    st.markdown("### \U0001f3bb PSM Distribution by Town")

    if not base_df.empty:
        violin_df = base_df[base_df["town"].isin(towns_sel)].dropna(subset=["price_per_sqm"])
        if not violin_df.empty:
            fig_violin = px.violin(
                violin_df,
                x="town",
                y="price_per_sqm",
                color="town",
                box=True,
                points=False,
                labels={"price_per_sqm": "Price per sqm ($)", "town": "Town"},
                title=f"PSM Distribution — {flat_type_sel} ({year_range_label})",
                category_orders={"town": towns_sel},
            )
            fig_violin.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("No PSM data available for the selected towns/flat type/period.")
    else:
        st.info("No data available for the selected filters.")

    # ── PSM Trend Over Time ────────────────────────────────────────────────
    st.markdown("### \U0001f4c8 PSM Trend Over Time")

    trend_df = psm_trend_data(
        towns=tuple(towns_sel),
        flat_type=flat_type_sel,
        year_range=year_range,
    )

    if not trend_df.empty:
        fig_trend = px.line(
            trend_df,
            x="month",
            y="median_psm",
            color="town",
            labels={"median_psm": "Median PSM ($/sqm)", "month": "Month", "town": "Town"},
            title=f"Monthly Median PSM — {flat_type_sel} ({year_range_label})",
            markers=False,
        )
        fig_trend.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No trend data available for the selected filters.")

    # ── Data Confidence ────────────────────────────────────────────────────
    st.success(
        "\U0001f4c4 **DATA CONFIDENCE: High.** "
        "All metrics are computed directly from HDB resale transaction records. "
        "Figures reflect completed transactions and may not capture current asking prices or recent market shifts."
    )

  else:
    # ── PRIVATE MODE: District Comparison ────────────────────────────────────
    st.subheader("🏢 District Comparison — Private (Condo)")
    st.markdown("Side-by-side comparison of **2–4 private postal districts** for a selected property type.")
    st.warning("⚠️ Private data covers Aug 2021–2026 only.")

    PRIV_DIST_OPTIONS = {d: f"D{d:02d} — {name}" for d, (name, _, _) in DISTRICT_CENTROIDS.items()}
    PRIV_PROP_BROAD = ["Condo/Apartment", "EC"]
    PRIV_YR_MAP = {"Last 1 year": 1, "Last 2 years": 2, "Last 3 years": 3}

    dc1, dc2, dc3 = st.columns([3, 2, 2])
    with dc1:
        priv_dists_sel = st.multiselect(
            "Select Districts (2–4)",
            options=list(PRIV_DIST_OPTIONS.keys()),
            default=[9, 15, 19],
            format_func=lambda d: PRIV_DIST_OPTIONS[d],
            key="scout_priv_dists",
        )
    with dc2:
        priv_prop_broad = st.selectbox(
            "Property Type", PRIV_PROP_BROAD, key="scout_priv_broad"
        )
    with dc3:
        priv_yr_label = st.selectbox(
            "Time Window", list(PRIV_YR_MAP.keys()), index=1, key="scout_priv_yr"
        )
        priv_yr = PRIV_YR_MAP[priv_yr_label]

    if len(priv_dists_sel) < 2:
        st.warning("Select 2–4 districts for comparison.")
    else:
        if len(priv_dists_sel) > 4:
            priv_dists_sel = priv_dists_sel[:4]
        with st.spinner("Loading district data…"):
            priv_summary, priv_base = compute_district_comparison(
                districts=tuple(priv_dists_sel),
                prop_type_broad=priv_prop_broad,
                year_range=priv_yr,
            )

        if not priv_summary:
            st.warning("No data found for the selected districts / property type.")
        else:
            priv_group_stats = {
                m: np.nanmedian([v[m] for v in priv_summary.values() if v and pd.notna(v.get(m, np.nan))])
                for m in ["median_psm", "transaction_count", "psm_1yr_chg"]
            }

            st.markdown("### 📊 Key Metrics")
            kpi_cols_p = st.columns(len(priv_dists_sel))
            for col_w, d in zip(kpi_cols_p, priv_dists_sel):
                with col_w:
                    label = f"**D{d:02d}**"
                    st.markdown(label)
                    s = priv_summary.get(d)
                    if s is None:
                        st.warning("No data")
                        continue
                    v = s["median_psm"]
                    g = priv_group_stats.get("median_psm", np.nan)
                    st.metric("Median PSM ($/sqm)", f"${v:,.0f}" if pd.notna(v) else "N/A",
                              delta=f"{(v-g)/g*100:+.1f}% vs group" if pd.notna(v) and pd.notna(g) and g > 0 else None,
                              delta_color="inverse")
                    v = s["transaction_count"]
                    g = priv_group_stats.get("transaction_count", np.nan)
                    st.metric("Transactions", f"{int(v):,}",
                              delta=f"{int(v-g):+,} vs group" if pd.notna(g) else None)
                    v = s["psm_1yr_chg"]
                    st.metric("PSM 1yr Change", f"{v:+.1f}%" if pd.notna(v) else "N/A")
                    p10, p90 = s["psm_p10"], s["psm_p90"]
                    st.metric("PSM Range (P10–P90)",
                              f"${p10:,.0f} – ${p90:,.0f}" if pd.notna(p10) and pd.notna(p90) else "N/A")
                    st.metric("New Sale %", f"{s['new_sale_pct']:.0f}%")
                    st.metric("Freehold %", f"{s['freehold_pct']:.0f}%")
                    st.metric("Dominant Tenure", str(s["most_common_tenure"]))

            st.divider()

            # PSM violin by district
            if not priv_base.empty:
                priv_violin = priv_base[priv_base["district"].isin(priv_dists_sel)].dropna(subset=["price_psm"])
                priv_violin["district_label"] = priv_violin["district"].apply(lambda d: f"D{d:02d}")
                if not priv_violin.empty:
                    fig_pv = px.violin(
                        priv_violin, x="district_label", y="price_psm",
                        color="district_label", box=True, points=False,
                        labels={"price_psm": "Price per sqm ($)", "district_label": "District"},
                        title=f"PSM Distribution — {priv_prop_broad} ({priv_yr_label})",
                    )
                    fig_pv.update_layout(showlegend=False, height=420)
                    st.plotly_chart(fig_pv, use_container_width=True)

            # PSM trend
            priv_trend_df = district_psm_trend(
                tuple(priv_dists_sel), priv_prop_broad, priv_yr
            )
            if not priv_trend_df.empty:
                priv_trend_df["label"] = priv_trend_df["district"].apply(lambda d: f"D{d:02d}")
                fig_pt = px.line(
                    priv_trend_df, x="month", y="median_psm", color="label",
                    labels={"median_psm": "Median PSM ($/sqm)", "month": "Month", "label": "District"},
                    title=f"Monthly Median PSM — {priv_prop_broad} ({priv_yr_label})",
                )
                fig_pt.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig_pt, use_container_width=True)

            st.success(
                "📄 **DATA CONFIDENCE: Medium.** URA private caveats (Aug 2021–2026). "
                "Figures reflect registered sale transactions only."
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Lifestyle-Weighted Area Matching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    area_label = "Town" if mode_scout == "🏘️ HDB Resale" else "District"
    st.subheader(f"\U0001f306 Lifestyle-Weighted {area_label} Matching")
    st.markdown(
        "Combine amenity density, MRT access, and resale metrics into a personalised "
        "town ranking based on **your priorities** — not a generic index."
    )

    # ── loaders ────────────────────────────────────────────────────────────────
    @st.cache_data
    def _load_amenity_csv(path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            df["lat"] = pd.to_numeric(df.get("lat", pd.Series(dtype=float)), errors="coerce")
            df["lon"] = pd.to_numeric(df.get("lon", pd.Series(dtype=float)), errors="coerce")
            return df.dropna(subset=["lat", "lon"])
        except FileNotFoundError:
            return pd.DataFrame(columns=["lat", "lon"])

    @st.cache_data
    def _compute_lifestyle_scores(
        _hawker: pd.DataFrame,
        _cc: pd.DataFrame,
        _parks: pd.DataFrame,
        _poly: pd.DataFrame,
        _schools: pd.DataFrame,
        _mrt: pd.DataFrame,
        primary_only: bool,
        radius_km: float,
        use_districts: bool = False,
    ) -> pd.DataFrame:
        """Count amenities within radius_km of each town/district centroid (vectorised)."""
        if primary_only and not _schools.empty and "mainlevel_code" in _schools.columns:
            schools_use = _schools[_schools["mainlevel_code"].str.upper() == "PRIMARY"].copy()
        else:
            schools_use = _schools.copy()

        # Rename school lat/lon columns if needed
        if not schools_use.empty:
            if "lat" not in schools_use.columns and "latitude" in schools_use.columns:
                schools_use = schools_use.rename(columns={"latitude": "lat", "longitude": "lon"})

        # Build centroids dict: {name: (lat, lon)}
        if use_districts:
            centroids_iter = {f"D{d:02d} {name}": (lat, lon)
                              for d, (name, lat, lon) in DISTRICT_CENTROIDS.items()}
        else:
            centroids_iter = TOWN_CENTROIDS

        results = []
        for town, (t_lat, t_lon) in centroids_iter.items():
            def _count(df: pd.DataFrame) -> int:
                if df.empty or "lat" not in df.columns:
                    return 0
                dlat = np.radians(df["lat"].values - t_lat)
                dlon = np.radians(df["lon"].values - t_lon)
                a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(t_lat)) * np.cos(np.radians(df["lat"].values)) * np.sin(dlon / 2) ** 2
                dist = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                return int((dist <= radius_km).sum())

            n_hawker  = _count(_hawker)
            n_cc      = _count(_cc)
            n_parks   = _count(_parks)
            n_poly    = _count(_poly)
            n_schools = _count(schools_use)

            # MRT score
            if not _mrt.empty and "lat" in _mrt.columns and "lon" in _mrt.columns:
                dlat = np.radians(_mrt["lat"].values - t_lat)
                dlon = np.radians(_mrt["lon"].values - t_lon)
                a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(t_lat)) * np.cos(np.radians(_mrt["lat"].values)) * np.sin(dlon / 2) ** 2
                mrt_dists = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                mrt_min   = float(mrt_dists.min())
            else:
                mrt_min = 1.0

            results.append({
                "town": town,
                "n_hawker": n_hawker,
                "n_cc": n_cc,
                "n_parks": n_parks,
                "n_poly": n_poly,
                "n_schools": n_schools,
                "mrt_min_km": round(mrt_min, 3),
            })

        df = pd.DataFrame(results)

        def _norm(col: str) -> pd.Series:
            lo, hi = df[col].min(), df[col].max()
            if hi == lo:
                return pd.Series(5.0, index=df.index)
            return ((df[col] - lo) / (hi - lo) * 10).round(2)

        df["food_score"]      = _norm("n_hawker")
        df["green_score"]     = _norm("n_parks")
        df["community_score"] = _norm("n_cc")
        df["health_score"]    = _norm("n_poly")
        df["edu_score"]       = _norm("n_schools")
        df["mrt_score"]       = (10 / (1 + df["mrt_min_km"])).round(2)
        return df.set_index("town")

    if mode_scout == "🏢 Private (Condo)":
        st.warning("⚠️ District centroids are used as reference points. "
                   "Amenity distances are approximate straight-line from district centroid.")

    # ── sidebar controls ───────────────────────────────────────────────────────
    st.markdown("#### Your lifestyle priorities (0 = ignore, 10 = top priority)")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        w_food    = st.slider("🍜 Food / Hawker",   0, 10, 7, key="ls_food")
        w_green   = st.slider("🌿 Green space",      0, 10, 5, key="ls_green")
    with col_s2:
        w_comm    = st.slider("🤝 Community (CCs)",  0, 10, 4, key="ls_comm")
        w_health  = st.slider("🏥 Healthcare",        0, 10, 6, key="ls_health")
    with col_s3:
        w_edu     = st.slider("🏫 Education",         0, 10, 5, key="ls_edu")
        w_mrt     = st.slider("🚇 MRT access",        0, 10, 8, key="ls_mrt")

    radius_km   = st.slider("Amenity radius (km)",  0.5, 3.0, 1.0, 0.5, key="ls_radius")
    primary_only = st.checkbox("Primary schools only (for Education score)", value=True, key="ls_pri")

    # ── load data ──────────────────────────────────────────────────────────────
    AMENITIES_DIR = os.path.join(DATA_DIR, "amenities")
    hawker_df  = _load_amenity_csv(os.path.join(AMENITIES_DIR, "hawker_centres.csv"))
    cc_df      = _load_amenity_csv(os.path.join(AMENITIES_DIR, "community_clubs.csv"))
    parks_df   = _load_amenity_csv(os.path.join(AMENITIES_DIR, "parks.csv"))
    poly_df    = _load_amenity_csv(os.path.join(AMENITIES_DIR, "polyclinics.csv"))
    schools_df = _load_amenity_csv(os.path.join(AMENITIES_DIR, "schools.csv"))
    mrt_df     = load_mrt_data()

    scores_df = _compute_lifestyle_scores(
        hawker_df, cc_df, parks_df, poly_df, schools_df, mrt_df,
        primary_only=primary_only, radius_km=radius_km,
        use_districts=(mode_scout == "🏢 Private (Condo)"),
    )

    # ── weighted composite ─────────────────────────────────────────────────────
    weights = {
        "food_score": w_food, "green_score": w_green,
        "community_score": w_comm, "health_score": w_health,
        "edu_score": w_edu, "mrt_score": w_mrt,
    }
    total_w = sum(weights.values()) or 1
    scores_df["lifestyle_score"] = sum(
        scores_df[col] * w for col, w in weights.items()
    ) / total_w
    scores_df["lifestyle_score"] = scores_df["lifestyle_score"].round(1)

    # Add affordability metric
    if mode_scout == "🏘️ HDB Resale":
        resale_df = load_resale()
        max_dt = resale_df["month"].max()
        cutoff = max_dt - pd.DateOffset(years=2)
        recent = resale_df[resale_df["month"] >= cutoff]
        town_psm = recent.groupby("town")["price_per_sqm"].median().rename("median_psm")
        town_vol = recent.groupby("town").size().rename("tx_volume")
        scores_df = scores_df.join(town_psm, how="left").join(town_vol, how="left")
    else:
        cdf = load_condo_for_scout()
        if not cdf.empty:
            max_dt_c = cdf["contract_date"].max()
            cutoff_c = max_dt_c - pd.DateOffset(years=2)
            recent_c = cdf[cdf["contract_date"] >= cutoff_c]
            dist_psm = recent_c.groupby("district")["price_psm"].median()
            dist_vol = recent_c.groupby("district").size()
            # Map district number to label used in scores_df index
            idx_map = {f"D{d:02d} {name}": d for d, (name, _, _) in DISTRICT_CENTROIDS.items()}
            scores_df["median_psm"] = scores_df.index.map(
                lambda k: dist_psm.get(idx_map.get(k, -1), np.nan)
            )
            scores_df["tx_volume"] = scores_df.index.map(
                lambda k: dist_vol.get(idx_map.get(k, -1), np.nan)
            )
        else:
            scores_df["median_psm"] = np.nan
            scores_df["tx_volume"] = np.nan

    ranked = scores_df.sort_values("lifestyle_score", ascending=False).reset_index()

    # ── top-10 table ───────────────────────────────────────────────────────────
    area_col_label = "Town" if mode_scout == "🏘️ HDB Resale" else "District"
    st.markdown(f"#### 🏆 Top {area_col_label}s ranked by your lifestyle priorities")
    top10_disp = ranked.head(10)[
        ["town", "lifestyle_score", "food_score", "green_score",
         "community_score", "health_score", "edu_score", "mrt_score",
         "median_psm", "tx_volume"]
    ].rename(columns={
        "town": area_col_label, "lifestyle_score": "Lifestyle Score",
        "food_score": "Food", "green_score": "Green",
        "community_score": "Community", "health_score": "Health",
        "edu_score": "Education", "mrt_score": "MRT",
        "median_psm": "Median PSM ($)", "tx_volume": "Tx Vol (2yr)",
    })
    st.dataframe(top10_disp, use_container_width=True, hide_index=True)

    # ── "why this town" natural-language summary ───────────────────────────────
    if not ranked.empty:
        top_row = ranked.iloc[0]
        top_town = top_row["town"]
        st.info(
            f"🥇 **{top_town}** scores highest for your priorities.\n\n"
            f"Within {radius_km:.1f} km of the town centroid: "
            f"**{int(top_row['n_hawker'])} hawker centres**, "
            f"**{int(top_row['n_parks'])} parks**, "
            f"**{int(top_row['n_cc'])} community clubs**, "
            f"**{int(top_row['n_schools'])} schools** "
            f"({'primary only' if primary_only else 'all levels'}), "
            f"**{int(top_row['n_poly'])} polyclinics**. "
            f"Nearest MRT: **{top_row['mrt_min_km']:.2f} km**."
        )

    # ── radar: selected town vs top town ───────────────────────────────────────
    col_ra, col_rb = st.columns([1, 2])
    with col_ra:
        compare_town = st.selectbox(
            "Compare to top town:", ranked["town"].tolist(), index=min(1, len(ranked) - 1),
            key="ls_compare",
        )
    dim_keys   = ["food_score", "green_score", "community_score", "health_score", "edu_score", "mrt_score"]
    dim_labels = ["Food", "Green", "Community", "Health", "Education", "MRT"]

    fig_radar = go.Figure()
    for t_name in [top_town, compare_town]:
        row = scores_df.loc[t_name] if t_name in scores_df.index else None
        if row is not None:
            vals = [float(row[d]) for d in dim_keys]
            vals_closed = vals + [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=dim_labels + [dim_labels[0]],
                fill="toself",
                name=t_name,
                opacity=0.65,
            ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10])),
        title=f"Lifestyle radar: {top_town} vs {compare_town}",
        height=440,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── bubble chart ───────────────────────────────────────────────────────────
    bubble_df = ranked.dropna(subset=["median_psm"]).copy()
    if not bubble_df.empty:
        bubble_df["affordability"] = 1 / bubble_df["median_psm"] * 1e6  # inverse PSM
        fig_bubble = px.scatter(
            bubble_df,
            x="affordability",
            y="lifestyle_score",
            size="tx_volume",
            color="lifestyle_score",
            text="town",
            color_continuous_scale="RdYlGn",
            labels={
                "affordability": "Affordability (inverse PSM — higher = cheaper)",
                "lifestyle_score": "Lifestyle Score",
                "tx_volume": "Transaction Volume (2yr)",
            },
            title="Town bubble map: Lifestyle Score vs Affordability",
            size_max=40,
        )
        fig_bubble.update_traces(textposition="top center", textfont_size=10)
        fig_bubble.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.caption(
        "🟡 **DATA CONFIDENCE: Medium.** Amenity counts use town *centroids* as reference points, "
        "not exact block locations — actual walking distances vary within the town. "
        "MRT score is proximity from centroid, not walking-route time. "
        "School priority zones (1km admission) differ from straight-line distance."
    )
