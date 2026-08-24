"""
Page 20 \u2013 Smart Money Opportunity Score
==========================================
North-star screen combining:
  \u2022 A4  \u2013 Block Desirability Fingerprint
  \u2022 S16 \u2013 Opportunity Score (6 decomposed dimensions)
  \u2022 G10 \u2013 Negotiation Leverage Report

Tabs
  1. \u2b50 Opportunity Screener (Section 16)
  2. \U0001f3c5 Block Desirability Fingerprint (A4)
  3. \U0001f4ac Negotiation Leverage Report (G10)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk

from eda_helpers import (
    load_clean,
    fmt_price,
    fmt_pct,
    TOWN_CENTROIDS,
    DATA_DIR,
    COORDS_CSV,
)

# \u2500\u2500 page config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.set_page_config(
    page_title="Opportunity Score",
    page_icon="\u2b50",
    layout="wide",
)

TODAY = pd.Timestamp.now()

SCORE_DIMS = [
    "Valuation", "Liquidity", "Supply Risk", "Lease", "Momentum", "Confidence",
]

CONFIDENCE_CAVEAT = (
    "\u26a0\ufe0f **Data Confidence Caveat** \u2014 This score is built entirely from past "
    "transaction patterns. It does **not** reflect:\n"
    "- Current listing availability or asking price\n"
    "- Unit-specific condition, renovation quality, or facing\n"
    "- Non-public information (seller motivation, lease extension plans)\n"
    "- Future policy changes\n\n"
    "A **high score** means historical patterns favour this area relative to peers "
    "\u2014 not a guarantee of future performance. Always inspect the property and verify "
    "with current market listings before making any decision."
)

# \u2500\u2500 base data \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


@st.cache_data
def get_base_df():
    """Load cleaned resale data and left-join block coordinates."""
    df = load_clean()
    df["month"] = pd.to_datetime(df["month"])
    df["block"] = df["block"].astype(str)
    if os.path.exists(COORDS_CSV):
        coords = pd.read_csv(COORDS_CSV)
        coords["block"] = coords["block"].astype(str)
        df = df.merge(coords, on=["block", "street_name"], how="left")
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan
    return df


df_base = get_base_df()
ALL_TOWNS = sorted(df_base["town"].dropna().unique().tolist())
ALL_FLAT_TYPES = sorted(df_base["flat_type"].dropna().unique().tolist())

# \u2500\u2500 small helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def _confidence(n):
    """Map transaction count to 0\u201310 confidence score."""
    if n >= 20:
        return 10.0
    if n >= 10:
        return 7.0 + (n - 10) * 0.3
    if n >= 5:
        return 4.0 + (n - 5) * 0.6
    return 0.0


def _score_rgb(score):
    """Map 0\u201310 score to [R, G, B, A]: red at 0, yellow at 5, green at 10."""
    s = float(np.clip(score, 0.0, 10.0))
    if s >= 5.0:
        t = (s - 5.0) / 5.0
        return [int(255 * (1 - t)), int(200 + 55 * t), int(50 * t), 200]
    t = s / 5.0
    return [int(200 + 20 * (1 - t)), int(30 + 170 * t), 30, 200]


def _make_radar(cats, vals, name, bench_vals=None, bench_name=None):
    """Create a Plotly spider/radar chart with optional benchmark overlay."""
    c2 = cats + [cats[0]]
    v2 = [float(v) for v in vals] + [float(vals[0])]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v2, theta=c2, fill="toself", name=name,
        line_color="#1f77b4", fillcolor="rgba(31,119,180,0.25)",
    ))
    if bench_vals is not None:
        b2 = [float(v) for v in bench_vals] + [float(bench_vals[0])]
        fig.add_trace(go.Scatterpolar(
            r=b2, theta=c2, fill="toself",
            name=bench_name or "Benchmark",
            line_color="#ff7f0e", fillcolor="rgba(255,127,14,0.15)",
            line_dash="dash",
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True, height=420,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    return fig


def _norm10(v, lo, hi):
    """Clamp and normalise v to [0, 10] given range [lo, hi]."""
    if hi == lo:
        return 5.0
    return float(np.clip((v - lo) / (hi - lo) * 10, 0.0, 10.0))


# \u2500\u2500 Tab 1: opportunity scoring engine \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


@st.cache_data(show_spinner="Computing opportunity scores\u2026")
def compute_scores(
    _df,
    years: float,
    min_txn: int,
    max_budget: float,
    flat_types: tuple,
    towns: tuple,
):
    """
    Score every block+flat_type combination across 6 dimensions.
    Returns a scored DataFrame, or empty DataFrame if no data matches.

    Scoring dimensions
    ------------------
    1. Valuation   \u2013 block PSM vs town median PSM discount
    2. Liquidity   \u2013 annual velocity relative to flat-type median
    3. Supply Risk \u2013 inverted MOP-unlock pressure over next 24 months
    4. Lease       \u2013 avg remaining lease linearly scaled (40\u201395 yrs \u2192 0\u201310)
    5. Momentum    \u2013 6-month PSM growth rate (block, town fallback)
    6. Confidence  \u2013 evidence quality from transaction count
    """
    cutoff = TODAY - pd.DateOffset(years=years)

    sub = _df[
        (_df["month"] >= cutoff)
        & (_df["flat_type"].isin(flat_types))
        & (_df["town"].isin(towns))
        & (_df["resale_price"] <= max_budget)
    ].copy()

    if len(sub) < 5:
        return pd.DataFrame()

    # \u2500 town+flat_type benchmark PSM \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    town_psm = (
        sub.groupby(["town", "flat_type"])["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"price_per_sqm": "town_psm"})
    )

    # \u2500 block-level aggregates \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    grp = (
        sub.groupby(["block", "street_name", "town", "flat_type"])
        .agg(
            txn_count=("resale_price", "size"),
            median_psm=("price_per_sqm", "median"),
            median_price=("resale_price", "median"),
            avg_remaining_lease=("remaining_lease_yrs", "mean"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        .reset_index()
    )
    grp = grp[grp["txn_count"] >= min_txn].copy()
    if len(grp) == 0:
        return pd.DataFrame()

    grp = grp.merge(town_psm, on=["town", "flat_type"], how="left")

    # per-flat-type median annual velocity (used as liquidity benchmark)
    blk_v = (
        sub.groupby(["block", "street_name", "flat_type"])
        .size()
        .reset_index(name="ttl")
    )
    blk_v["ann_v"] = blk_v["ttl"] / years
    ft_med_v = (
        blk_v.groupby("flat_type")["ann_v"]
        .median()
        .reset_index()
        .rename(columns={"ann_v": "med_v"})
    )
    grp = grp.merge(ft_med_v, on="flat_type", how="left")

    # \u2500\u2500 1. Valuation score (0\u201310) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # 10 = block is 20%+ cheaper than town median; 5 = at market; 0 = 10%+ expensive
    grp["disc_pct"] = np.where(
        grp["town_psm"] > 0,
        (grp["median_psm"] - grp["town_psm"]) / grp["town_psm"] * 100.0,
        0.0,
    )
    grp["score_valuation"] = ((-grp["disc_pct"]) / 20.0 * 10.0).clip(0, 10)

    # \u2500\u2500 2. Liquidity score (0\u201310) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # relative_velocity = (txn_count / years) / median_flat_type_velocity
    # score = clip(relative_velocity * 5, 0, 10)
    grp["ann_vel"] = grp["txn_count"] / years
    grp["score_liquidity"] = (
        (grp["ann_vel"] / grp["med_v"].replace(0, np.nan)) * 5.0
    ).clip(0, 10).fillna(5.0)

    # \u2500\u2500 3. Supply risk score (0\u201310, inverted) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # MOP-unlock wave in next 24 months vs recent annual volume (town+flat_type)
    fy0 = TODAY.year
    fy1 = (TODAY + pd.DateOffset(months=24)).year
    full = _df[_df["town"].isin(towns) & _df["flat_type"].isin(flat_types)].copy()
    full = full.dropna(subset=["lease_commence_date"])
    full["mop_yr"] = full["lease_commence_date"].astype(int) + 5
    mop_s = (
        full[full["mop_yr"].between(fy0, fy1)]
        .groupby(["town", "flat_type"])
        .size()
        .reset_index(name="mop24")
    )
    rec_v = (
        _df[
            (_df["month"] >= TODAY - pd.DateOffset(years=1))
            & _df["town"].isin(towns)
            & _df["flat_type"].isin(flat_types)
        ]
        .groupby(["town", "flat_type"])
        .size()
        .reset_index(name="rec_vol")
    )
    sup = mop_s.merge(rec_v, on=["town", "flat_type"], how="outer").fillna(0)
    sup["sup_ratio"] = np.where(
        sup["rec_vol"] > 0, sup["mop24"] / sup["rec_vol"], np.nan
    )
    sup["score_supply"] = (10.0 - (sup["sup_ratio"] * 10.0).clip(0, 10)).fillna(5.0)
    grp = grp.merge(
        sup[["town", "flat_type", "score_supply"]], on=["town", "flat_type"], how="left"
    )
    grp["score_supply"] = grp["score_supply"].fillna(5.0)

    # \u2500\u2500 4. Lease score (0\u201310) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # 95 yrs remaining \u2192 10, 40 yrs \u2192 0, linear
    grp["score_lease"] = (
        (grp["avg_remaining_lease"] - 40.0) / 55.0 * 10.0
    ).clip(0, 10).fillna(5.0)

    # \u2500\u2500 5. Momentum score (0\u201310) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # (last_6m_median \u2212 prev_6m_median) / prev_6m_median; -5% \u2192 0, 0% \u2192 5, +5% \u2192 10
    six = TODAY - pd.DateOffset(months=6)
    twelve = TODAY - pd.DateOffset(months=12)

    l6 = (
        sub[sub["month"] >= six]
        .groupby(["block", "street_name", "flat_type"])["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"price_per_sqm": "p6"})
    )
    p12 = (
        sub[(sub["month"] >= twelve) & (sub["month"] < six)]
        .groupby(["block", "street_name", "flat_type"])["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"price_per_sqm": "p12"})
    )
    mom = l6.merge(p12, on=["block", "street_name", "flat_type"], how="inner")
    mom["g"] = (mom["p6"] - mom["p12"]) / mom["p12"].replace(0, np.nan) * 100.0
    mom["score_momentum"] = ((mom["g"] + 5.0) / 10.0 * 10.0).clip(0, 10)

    grp = grp.merge(
        mom[["block", "street_name", "flat_type", "score_momentum"]],
        on=["block", "street_name", "flat_type"],
        how="left",
    )

    # fallback: town-level momentum
    tl6 = (
        sub[sub["month"] >= six]
        .groupby(["town", "flat_type"])["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"price_per_sqm": "tp6"})
    )
    tp12 = (
        sub[(sub["month"] >= twelve) & (sub["month"] < six)]
        .groupby(["town", "flat_type"])["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"price_per_sqm": "tp12"})
    )
    tmom = tl6.merge(tp12, on=["town", "flat_type"], how="inner")
    tmom["fb"] = (
        ((tmom["tp6"] - tmom["tp12"]) / tmom["tp12"].replace(0, np.nan) * 100.0 + 5.0)
        / 10.0 * 10.0
    ).clip(0, 10)

    grp = grp.merge(tmom[["town", "flat_type", "fb"]], on=["town", "flat_type"], how="left")
    grp["score_momentum"] = grp["score_momentum"].fillna(grp["fb"]).fillna(5.0)

    # \u2500\u2500 6. Confidence score (0\u201310) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    grp["score_confidence"] = grp["txn_count"].apply(_confidence)

    # clean up intermediate columns
    for _c in ["disc_pct", "ann_vel", "med_v", "town_psm", "fb"]:
        if _c in grp.columns:
            grp.drop(columns=_c, inplace=True)

    return grp.reset_index(drop=True)


def _apply_weights(df, w):
    """Compute weighted composite opportunity score from dimension scores."""
    tw = max(sum(w.values()), 1)
    df = df.copy()
    df["opportunity_score"] = (
        df["score_valuation"] * w["Valuation"]
        + df["score_liquidity"] * w["Liquidity"]
        + df["score_supply"] * w["Supply Risk"]
        + df["score_lease"] * w["Lease"]
        + df["score_momentum"] * w["Momentum"]
        + df["score_confidence"] * w["Confidence"]
    ) / tw
    return df


# \u2500\u2500 Tab 2 / Tab 3: dropdown helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


@st.cache_data
def get_streets(_df, town):
    return sorted(_df[_df["town"] == town]["street_name"].dropna().unique().tolist())


@st.cache_data
def get_blocks(_df, town, street):
    m = (_df["town"] == town) & (_df["street_name"] == street)
    return sorted(_df[m]["block"].dropna().astype(str).unique().tolist())


# \u2500\u2500 Tab 2: fingerprint engine \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


@st.cache_data(show_spinner="Computing block fingerprint\u2026")
def compute_fingerprint(_df, town, street, block):
    """
    Compute 8-dimension desirability fingerprint for a specific block.

    Returns (R, T, bdf, tdf, err)
      R   \u2013 dict of raw block metrics
      T   \u2013 dict of raw town benchmark metrics
      bdf \u2013 block-level DataFrame
      tdf \u2013 town-level DataFrame
      err \u2013 error string if computation failed, else None
    """
    bstr = str(block)
    bdf = _df[
        (_df["block"] == bstr)
        & (_df["street_name"] == street)
        & (_df["town"] == town)
    ].copy()
    tdf = _df[_df["town"] == town].copy()

    if len(bdf) < 3:
        return None, None, None, None, "Fewer than 3 transactions recorded for this block."

    R, T = {}, {}

    # 1. Price premium vs town median
    bp = bdf["price_per_sqm"].median()
    tp = tdf["price_per_sqm"].median()
    R["price_prem"] = (bp / tp - 1.0) * 100.0 if tp > 0 else 0.0
    T["price_prem"] = 0.0  # town IS the reference

    # 2. Transaction velocity (txns per year)
    span = max((bdf["month"].max() - bdf["month"].min()).days / 365.25, 1.0)
    R["velocity"] = len(bdf) / span
    rec5 = tdf[tdf["year"] >= (TODAY.year - 5)]
    bvels = rec5.groupby(["block", "street_name"]).size() / 5.0
    T["velocity"] = float(bvels.median()) if len(bvels) > 0 else R["velocity"]

    # 3. Price resilience \u2013 2021 cooling round (compare 2021 vs 2022 PSM)
    b_pre = bdf[bdf["year"] == 2021]["price_per_sqm"].median()
    b_pst = bdf[bdf["year"] == 2022]["price_per_sqm"].median()
    t_pre = tdf[tdf["year"] == 2021]["price_per_sqm"].median()
    t_pst = tdf[tdf["year"] == 2022]["price_per_sqm"].median()

    def _pct_change(pre, pst):
        if any(pd.isna(x) for x in [pre, pst]) or pre <= 0:
            return 0.0
        return float((pst - pre) / pre * 100.0)

    b_drop = _pct_change(b_pre, b_pst)
    t_drop = _pct_change(t_pre, t_pst)
    R["resilience"] = b_drop
    T["resilience"] = t_drop
    R["_t_drop"] = t_drop     # kept for display
    R["_b_pre21"] = b_pre     # kept for recovery computation

    # 4. Storey premium ($ per sqm per floor, from linear regression)
    bs = bdf.dropna(subset=["storey_mid", "price_per_sqm"])
    if len(bs) >= 3 and bs["storey_mid"].std() > 0:
        R["storey_slope"] = float(
            np.polyfit(bs["storey_mid"].values, bs["price_per_sqm"].values, 1)[0]
        )
    else:
        R["storey_slope"] = 0.0
    ts = tdf.dropna(subset=["storey_mid", "price_per_sqm"])
    if len(ts) >= 10 and ts["storey_mid"].std() > 0:
        T["storey_slope"] = float(
            np.polyfit(ts["storey_mid"].values, ts["price_per_sqm"].values, 1)[0]
        )
    else:
        T["storey_slope"] = R["storey_slope"]

    # 5. Lease\u2013price sensitivity (Pearson correlation)
    bl = bdf.dropna(subset=["remaining_lease_yrs", "price_per_sqm"])
    rc = bl["remaining_lease_yrs"].corr(bl["price_per_sqm"]) if len(bl) >= 3 else 0.0
    R["lease_corr"] = 0.0 if pd.isna(rc) else float(rc)
    tl = tdf.dropna(subset=["remaining_lease_yrs", "price_per_sqm"])
    tc = tl["remaining_lease_yrs"].corr(tl["price_per_sqm"]) if len(tl) >= 10 else 0.0
    T["lease_corr"] = 0.0 if pd.isna(tc) else float(tc)

    # 6. Price recovery speed after 2021 cooling (quarters)
    def _recov_qtrs(df_in, pre_psm):
        if pd.isna(pre_psm) or pre_psm <= 0:
            return 6
        post = df_in[df_in["month"] >= pd.Timestamp("2022-01-01")].copy()
        if len(post) < 2:
            return 8
        post["qtr"] = post["month"].dt.to_period("Q")
        qs = post.groupby("qtr")["price_per_sqm"].median().sort_index()
        for i, (_, v) in enumerate(qs.items()):
            if v >= pre_psm:
                return i + 1
        return min(len(qs) + 4, 12)

    R["recov_q"] = _recov_qtrs(bdf, b_pre)
    T["recov_q"] = _recov_qtrs(tdf, t_pre)

    # 7. Flat model mix diversity
    R["model_n"] = int(bdf["flat_model"].nunique())
    mdiv = tdf.groupby(["block", "street_name"])["flat_model"].nunique()
    T["model_n"] = float(mdiv.median()) if len(mdiv) > 0 else 1.0

    # 8. Seasonal uniformity (CV of monthly transaction counts; lower = more uniform)
    def _season_cv(df_in):
        mc = (
            df_in.groupby(df_in["month"].dt.month)
            .size()
            .reindex(range(1, 13), fill_value=0)
        )
        m = mc.mean()
        return float(mc.std() / m) if m > 0 else 0.0

    R["season_cv"] = _season_cv(bdf)
    T["season_cv"] = _season_cv(tdf)

    return R, T, bdf, tdf, None


FP_LABELS = [
    "Price Premium",
    "Velocity",
    "Resilience",
    "Storey Premium",
    "Lease Sensitivity",
    "Recovery Speed",
    "Model Diversity",
    "Uniform Activity",
]


def fp_scores(R, T):
    """Convert raw fingerprint dicts to 0\u201310 normalised scores."""
    D, B = {}, {}

    # 1 price_prem: -30% to +30%
    D["Price Premium"] = _norm10(R["price_prem"], -30, 30)
    B["Price Premium"] = _norm10(T["price_prem"], -30, 30)

    # 2 velocity: 0 to 2\xd7max
    vh = max(R["velocity"], T["velocity"]) * 2.0 + 0.01
    D["Velocity"] = _norm10(R["velocity"], 0, vh)
    B["Velocity"] = _norm10(T["velocity"], 0, vh)

    # 3 resilience: -25% (big drop) to +10% (rise) - higher is better
    D["Resilience"] = _norm10(R["resilience"], -25, 10)
    B["Resilience"] = _norm10(T["resilience"], -25, 10)

    # 4 storey_slope: 0 to 1.2\xd7max
    sh = max(max(R["storey_slope"], T["storey_slope"]) * 1.2, 10.0)
    D["Storey Premium"] = _norm10(R["storey_slope"], 0, sh)
    B["Storey Premium"] = _norm10(T["storey_slope"], 0, sh)

    # 5 lease_corr: -1 to 1
    D["Lease Sensitivity"] = _norm10(R["lease_corr"], -1, 1)
    B["Lease Sensitivity"] = _norm10(T["lease_corr"], -1, 1)

    # 6 recov_q: fewer quarters = better \u2192 invert (-12 to -1)
    D["Recovery Speed"] = _norm10(-R["recov_q"], -12, -1)
    B["Recovery Speed"] = _norm10(-T["recov_q"], -12, -1)

    # 7 model_n: 1 to max
    mh = max(R["model_n"], T["model_n"], 4)
    D["Model Diversity"] = _norm10(R["model_n"], 1, mh)
    B["Model Diversity"] = _norm10(T["model_n"], 1, mh)

    # 8 season_cv: lower = more uniform = better \u2192 invert (-2 to 0)
    D["Uniform Activity"] = _norm10(-R["season_cv"], -2, 0)
    B["Uniform Activity"] = _norm10(-T["season_cv"], -2, 0)

    return D, B


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# MAIN PAGE
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

st.title("\u2b50 Smart Money Opportunity Score")
st.caption(
    "North-star screen \u2014 "
    "A4 Block Desirability Fingerprint \u2022 "
    "S16 Opportunity Score \u2022 "
    "G10 Negotiation Leverage Report"
)

# \u2500\u2500 Sidebar (Tab 1 inputs) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.sidebar.header("\u2b50 Screener Inputs")

max_budget = st.sidebar.number_input(
    "Max budget ($)", 100_000, 3_000_000, 800_000, 10_000, key="mb",
)
sel_ft = st.sidebar.multiselect(
    "Flat types", ALL_FLAT_TYPES, default=["4 ROOM", "5 ROOM"], key="sft",
)
sel_towns = st.sidebar.multiselect(
    "Towns", ALL_TOWNS, default=ALL_TOWNS, key="stowns",
)
yrs = st.sidebar.select_slider(
    "Years of analysis", options=[1, 2, 3, 5], value=2, key="syrs",
)
min_txn = st.sidebar.slider(
    "Min transactions required", 3, 20, 5, key="smtx",
)

with st.sidebar.expander("Adjust dimension weights"):
    w_val = st.slider("Valuation %", 0, 50, 30, key="wv")
    w_liq = st.slider("Liquidity %", 0, 50, 20, key="wl")
    w_sup = st.slider("Supply Risk %", 0, 50, 15, key="ws")
    w_lea = st.slider("Lease %", 0, 50, 15, key="wls")
    w_mom = st.slider("Momentum %", 0, 50, 15, key="wm")
    w_con = st.slider("Confidence %", 0, 20, 5, key="wc")

weights = {
    "Valuation": w_val, "Liquidity": w_liq, "Supply Risk": w_sup,
    "Lease": w_lea, "Momentum": w_mom, "Confidence": w_con,
}

tab1, tab2, tab3 = st.tabs([
    "\u2b50 Opportunity Screener",
    "\U0001f3c5 Block Fingerprint (A4)",
    "\U0001f4ac Negotiation Leverage (G10)",
])


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# TAB 1 \u2013 OPPORTUNITY SCREENER
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
with tab1:
    st.subheader("\u2b50 Opportunity Screener (Section 16)")
    st.warning(CONFIDENCE_CAVEAT)

    if not sel_ft or not sel_towns:
        st.info(
            "Select at least one flat type and at least one town in the sidebar to begin scoring."
        )
    else:
        scored = compute_scores(
            df_base,
            float(yrs),
            int(min_txn),
            float(max_budget),
            tuple(sorted(sel_ft)),
            tuple(sorted(sel_towns)),
        )

        if len(scored) == 0:
            st.warning(
                "No blocks with sufficient transactions matched your criteria. "
                "Try reducing **Min transactions required** or expanding the flat type/town selection."
            )
        else:
            scored = _apply_weights(scored, weights)
            scored = scored.sort_values(
                "opportunity_score", ascending=False
            ).reset_index(drop=True)

            # \u2500 display-only filters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            cf1, cf2, cf3 = st.columns(3)
            with cf1:
                min_score = st.slider("Min score to display", 0.0, 10.0, 0.0, 0.5, key="dms")
            with cf2:
                d_towns = st.multiselect(
                    "Town filter (display only)", ALL_TOWNS, key="dtowns",
                )
            with cf3:
                d_ft = st.multiselect(
                    "Flat type filter (display only)", ALL_FLAT_TYPES, key="dft",
                )

            disp = scored[scored["opportunity_score"] >= min_score].copy()
            if d_towns:
                disp = disp[disp["town"].isin(d_towns)]
            if d_ft:
                disp = disp[disp["flat_type"].isin(d_ft)]

            st.caption(
                f"Showing **{len(disp):,}** blocks \u00b7 {len(scored):,} scored in total"
            )

            # \u2500 scored table \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            _show_cols = [
                "block", "street_name", "town", "flat_type", "opportunity_score",
                "score_valuation", "score_liquidity", "score_supply",
                "score_lease", "score_momentum", "score_confidence",
                "median_psm", "txn_count",
            ]
            _col_labels = {
                "block": "Block", "street_name": "Street", "town": "Town",
                "flat_type": "Flat Type", "opportunity_score": "Score",
                "score_valuation": "Valuation", "score_liquidity": "Liquidity",
                "score_supply": "Supply Risk", "score_lease": "Lease",
                "score_momentum": "Momentum", "score_confidence": "Confidence",
                "median_psm": "Median PSM ($)", "txn_count": "Txns",
            }
            tbl_df = (
                disp[[c for c in _show_cols if c in disp.columns]]
                .rename(columns=_col_labels)
                .head(500)
            )
            st.dataframe(
                tbl_df,
                use_container_width=True,
                height=380,
                column_config={
                    "Score": st.column_config.NumberColumn(format="%.2f"),
                    "Valuation": st.column_config.NumberColumn(format="%.1f"),
                    "Liquidity": st.column_config.NumberColumn(format="%.1f"),
                    "Supply Risk": st.column_config.NumberColumn(format="%.1f"),
                    "Lease": st.column_config.NumberColumn(format="%.1f"),
                    "Momentum": st.column_config.NumberColumn(format="%.1f"),
                    "Confidence": st.column_config.NumberColumn(format="%.1f"),
                    "Median PSM ($)": st.column_config.NumberColumn(format="$%.0f"),
                    "Txns": st.column_config.NumberColumn(format="%d"),
                },
            )

            # \u2500 radar for selected block \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            st.subheader("Dimension radar for a selected block")
            opts = [
                f"{r['block']} \u2013 {r['street_name']} ({r['flat_type']})"
                for _, r in disp.head(200).iterrows()
            ]
            if opts:
                sel_opt = st.selectbox(
                    "Select a block to inspect", opts, key="radar_sel",
                )
                sidx = opts.index(sel_opt)
                srow = disp.iloc[sidx]
                dv = [
                    float(srow["score_valuation"]),
                    float(srow["score_liquidity"]),
                    float(srow["score_supply"]),
                    float(srow["score_lease"]),
                    float(srow["score_momentum"]),
                    float(srow["score_confidence"]),
                ]
                town_mean = scored[scored["town"] == srow["town"]].mean(numeric_only=True)
                bv = [
                    float(town_mean.get("score_valuation", 5)),
                    float(town_mean.get("score_liquidity", 5)),
                    float(town_mean.get("score_supply", 5)),
                    float(town_mean.get("score_lease", 5)),
                    float(town_mean.get("score_momentum", 5)),
                    float(town_mean.get("score_confidence", 5)),
                ]
                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    st.metric("Composite Score", f"{srow['opportunity_score']:.2f} / 10")
                    st.markdown(
                        f"**Block:** {srow['block']}  \n"
                        f"**Street:** {srow['street_name']}  \n"
                        f"**Town:** {srow['town']}  \n"
                        f"**Flat Type:** {srow['flat_type']}  \n"
                        f"**Median PSM:** {fmt_price(srow['median_psm'])}  \n"
                        f"**Transactions:** {int(srow['txn_count'])}"
                    )
                    lyr = srow.get("avg_remaining_lease", np.nan)
                    if not pd.isna(lyr):
                        st.markdown(f"**Avg Remaining Lease:** {lyr:.0f} yrs")
                with rc2:
                    st.plotly_chart(
                        _make_radar(
                            SCORE_DIMS, dv,
                            f"Block {srow['block']}",
                            bv, f"{srow['town']} avg",
                        ),
                        use_container_width=True,
                    )

            # \u2500 pydeck scatter map \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            st.subheader("\U0001f5fa\ufe0f Opportunity map")
            map_df = disp.copy()

            # fill missing block coords from town centroids
            missing_lat = map_df["lat"].isna() | map_df["lon"].isna()
            for _town in map_df.loc[missing_lat, "town"].unique():
                _c = TOWN_CENTROIDS.get(_town)
                if _c:
                    _tmask = (map_df["town"] == _town) & missing_lat
                    map_df.loc[_tmask, "lat"] = _c[0]
                    map_df.loc[_tmask, "lon"] = _c[1]

            map_df = map_df.dropna(subset=["lat", "lon"]).copy()

            if len(map_df) > 0:
                map_df["color"] = map_df["opportunity_score"].apply(_score_rgb)
                map_df["tt"] = (
                    "Blk "
                    + map_df["block"].astype(str)
                    + " "
                    + map_df["street_name"]
                    + " ("
                    + map_df["flat_type"]
                    + ")\nScore: "
                    + map_df["opportunity_score"].round(2).astype(str)
                    + "\nMedian PSM: $"
                    + map_df["median_psm"].fillna(0).round(0).astype(int).astype(str)
                )

                st.pydeck_chart(pdk.Deck(
                    layers=[pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=["lon", "lat"],
                        get_color="color",
                        get_radius=100,
                        pickable=True,
                        opacity=0.85,
                    )],
                    initial_view_state=pdk.ViewState(
                        longitude=float(map_df["lon"].mean()),
                        latitude=float(map_df["lat"].mean()),
                        zoom=11,
                        pitch=0,
                    ),
                    tooltip={"text": "{tt}"},
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                ))
                st.caption(
                    "\U0001f7e2 Green = high opportunity  "
                    "\U0001f7e1 Yellow = moderate  "
                    "\U0001f534 Red = low / avoid"
                )
            else:
                st.info("No geolocation data available for the filtered blocks.")


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# TAB 2 \u2013 BLOCK DESIRABILITY FINGERPRINT
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
with tab2:
    st.subheader("\U0001f3c5 Block Desirability Fingerprint (A4)")
    st.info(
        "\U0001f4ca **Data Confidence: Medium** \u2014 Block-level behavioural signature "
        "derived from historical transactions. Blocks with fewer than 20 transactions "
        "have higher uncertainty. Interpret with caution."
    )

    t2c1, t2c2, t2c3 = st.columns(3)
    with t2c1:
        fp_town = st.selectbox("Town", ALL_TOWNS, key="fp_town")
    with t2c2:
        fp_streets = get_streets(df_base, fp_town)
        fp_street = st.selectbox(
            "Street",
            fp_streets if fp_streets else ["(no streets found)"],
            key="fp_street",
        )
    with t2c3:
        fp_blks = get_blocks(df_base, fp_town, fp_street)
        fp_block = st.selectbox(
            "Block",
            fp_blks if fp_blks else ["(no blocks)"],
            key="fp_block",
        )

    if not fp_streets or not fp_blks:
        st.warning("No transaction data found for this town / street combination.")
    else:
        _R, _T, _bdf, _tdf, _err = compute_fingerprint(
            df_base, fp_town, fp_street, fp_block,
        )
        if _err:
            st.warning(f"Cannot compute fingerprint: {_err}")
        else:
            _D, _Bench = fp_scores(_R, _T)
            _d_labels = list(_D.keys())
            _d_vals = [_D[k] for k in _d_labels]
            _b_vals = [_Bench[k] for k in _d_labels]

            col_fp1, col_fp2 = st.columns([3, 2])
            with col_fp1:
                st.plotly_chart(
                    _make_radar(
                        _d_labels, _d_vals,
                        f"Block {fp_block} \u2013 {fp_street}",
                        _b_vals, f"{fp_town} benchmark",
                    ),
                    use_container_width=True,
                )
            with col_fp2:
                st.markdown("### Fingerprint Summary")

                _b_drop = _R.get("resilience", 0.0)
                _t_drop = _R.get("_t_drop", 0.0)
                _more_resilient = _b_drop > _t_drop
                st.markdown(
                    f"**Resilience (2021 cooling):** PSM change 2021\u21922022: "
                    f"**{_b_drop:+.1f}%** (block) vs **{_t_drop:+.1f}%** (town). "
                    + (
                        "Block was **more resilient** than the town average."
                        if _more_resilient
                        else "Block was **less resilient** than the town average."
                    )
                )
                st.markdown(
                    f"**Price premium vs town:** {_R.get('price_prem', 0.0):+.1f}%"
                )
                st.markdown(
                    f"**Annual velocity:** {_R.get('velocity', 0.0):.1f} txns / yr"
                )
                _sl = _R.get("storey_slope", 0.0)
                st.markdown(
                    f"**Storey premium:** ${_sl:.0f} per sqm per additional floor"
                )
                _lc = _R.get("lease_corr", 0.0)
                st.markdown(
                    f"**Lease\u2013price sensitivity:** {_lc:.2f} "
                    f"({'market values lease strongly' if _lc > 0.4 else 'weak lease signal'})"
                )
                _rq = _R.get("recov_q", 6)
                st.markdown(
                    f"**Post-2021 recovery:** "
                    + (
                        f"recovered in **{_rq} quarter(s)**"
                        if _rq <= 8
                        else "**not recovered** within the data window"
                    )
                )
                st.markdown(
                    f"**Flat model count:** {_R.get('model_n', 1)} distinct model(s)"
                )
                _cv = _R.get("season_cv", 0.0)
                _cv_lbl = (
                    "uniform demand"
                    if _cv < 0.5
                    else ("moderately seasonal" if _cv < 1.0 else "highly seasonal")
                )
                st.markdown(
                    f"**Activity pattern:** {_cv_lbl} (CV = {_cv:.2f})"
                )

            # \u2500 Historical PSM trend \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            st.subheader(f"Historical PSM: Block {fp_block} vs {fp_town} median")

            _bq = (
                _bdf.assign(
                    quarter=_bdf["month"].dt.to_period("Q").dt.to_timestamp()
                )
                .groupby("quarter")["price_per_sqm"]
                .median()
                .reset_index()
                .rename(columns={"price_per_sqm": "Block PSM"})
            )
            _tq = (
                _tdf.assign(
                    quarter=_tdf["month"].dt.to_period("Q").dt.to_timestamp()
                )
                .groupby("quarter")["price_per_sqm"]
                .median()
                .reset_index()
                .rename(columns={"price_per_sqm": "Town Median PSM"})
            )
            _trend = _bq.merge(_tq, on="quarter", how="outer").sort_values("quarter")

            fig_tr = go.Figure()
            fig_tr.add_trace(go.Scatter(
                x=_trend["quarter"], y=_trend["Block PSM"],
                name=f"Block {fp_block}",
                mode="lines+markers",
                line_color="#1f77b4",
            ))
            fig_tr.add_trace(go.Scatter(
                x=_trend["quarter"], y=_trend["Town Median PSM"],
                name=f"{fp_town} median",
                mode="lines",
                line=dict(color="#ff7f0e", dash="dash"),
            ))
            fig_tr.update_layout(
                xaxis_title="Quarter",
                yaxis_title="Median PSM ($)",
                height=360,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_tr, use_container_width=True)


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# TAB 3 \u2013 NEGOTIATION LEVERAGE REPORT
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
with tab3:
    st.subheader("\U0001f4ac Negotiation Leverage Report (G10)")
    st.info(
        "\U0001f4ca **Data Confidence:** High for historical data presented; "
        "Medium for leverage interpretations derived from those patterns."
    )

    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1:
        g_town = st.selectbox("Town", ALL_TOWNS, key="g_town")
    with gc2:
        g_streets = get_streets(df_base, g_town)
        g_street = st.selectbox(
            "Street",
            g_streets if g_streets else ["(none)"],
            key="g_street",
        )
    with gc3:
        g_blks = get_blocks(df_base, g_town, g_street)
        g_block = st.selectbox(
            "Block",
            g_blks if g_blks else ["(none)"],
            key="g_block",
        )
    with gc4:
        g_ft = st.selectbox("Flat Type", ALL_FLAT_TYPES, key="g_ft")

    if not g_streets or not g_blks:
        st.warning("No transaction data found for this town / street combination.")
    else:
        _g_bstr = str(g_block)
        _now = TODAY
        _one_yr = _now - pd.DateOffset(years=1)
        _two_yr = _now - pd.DateOffset(years=2)
        _six_m = _now - pd.DateOffset(months=6)
        _twelve_m = _now - pd.DateOffset(months=12)

        _sel = df_base[
            (df_base["block"] == _g_bstr)
            & (df_base["street_name"] == g_street)
            & (df_base["town"] == g_town)
            & (df_base["flat_type"] == g_ft)
        ].copy().sort_values("month")

        _town_ctx = df_base[
            (df_base["town"] == g_town) & (df_base["flat_type"] == g_ft)
        ]
        _t_psm_12m = _town_ctx[_town_ctx["month"] >= _one_yr]["price_per_sqm"].median()

        st.markdown(
            f"### \U0001f4cb Brief: Block **{g_block}**, {g_street}, "
            f"{g_town} \u2014 {g_ft}"
        )
        st.markdown("---")

        if len(_sel) == 0:
            st.warning(
                "No transactions recorded for this block + flat type combination. "
                "Try selecting a different flat type or block."
            )
        else:
            # pre-compute 6-month PSM momentum
            _psm_l6 = _sel[_sel["month"] >= _six_m]["price_per_sqm"].median()
            _psm_p6 = _sel[
                (_sel["month"] >= _twelve_m) & (_sel["month"] < _six_m)
            ]["price_per_sqm"].median()

            if (
                not pd.isna(_psm_l6)
                and not pd.isna(_psm_p6)
                and _psm_p6 > 0
            ):
                _delta_pct = (_psm_l6 - _psm_p6) / _psm_p6 * 100.0
            else:
                _delta_pct = np.nan

            # \u2500\u2500 1. Market velocity \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            _r12 = _sel[_sel["month"] >= _one_yr]
            _n12 = len(_r12)
            if _n12 >= 2:
                _gaps = _r12["month"].diff().dt.days.dropna()
                _avg_gap_m = _gaps.mean() / 30.44
                _liq = (
                    "high" if _avg_gap_m < 2 else ("medium" if _avg_gap_m < 5 else "low")
                )
                st.markdown(
                    f"**1. Market Velocity** \u2014 In the last 12 months, **{_n12} "
                    f"transactions** for {g_ft} in Block {g_block}. Average gap between "
                    f"transactions: **{_avg_gap_m:.1f} months** \u2192 **{_liq} liquidity**."
                )
            elif _n12 == 1:
                st.markdown(
                    f"**1. Market Velocity** \u2014 Only **1 transaction** in the last 12 "
                    f"months. Very thin market for this block / type."
                )
            else:
                st.markdown(
                    f"**1. Market Velocity** \u2014 **No transactions** in the last 12 months "
                    f"for Block {g_block} ({g_ft}). Consider a wider time window."
                )

            # \u2500\u2500 2. Price trajectory \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            if not pd.isna(_delta_pct):
                _traj = (
                    "Rising \u2197" if _delta_pct > 2
                    else ("Falling \u2198" if _delta_pct < -2 else "Flat \u2192")
                )
                st.markdown(
                    f"**2. Price Trajectory** \u2014 **{_traj}** | Median PSM: "
                    f"**{fmt_price(_psm_l6)}** (last 6 m) vs "
                    f"**{fmt_price(_psm_p6)}** (prior 6 m) \u2192 **{_delta_pct:+.1f}%** change."
                )
            elif not pd.isna(_psm_l6):
                _psm_12m = _sel[_sel["month"] >= _one_yr]["price_per_sqm"].median()
                st.markdown(
                    f"**2. Price Trajectory** \u2014 Insufficient data for a 6-month split. "
                    f"12-month median PSM: **{fmt_price(_psm_12m)}**."
                )
            else:
                st.markdown(
                    "**2. Price Trajectory** \u2014 Insufficient recent transaction data."
                )

            # \u2500\u2500 3. Transaction recency \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            _last_r = _sel.iloc[-1]
            _last_psm = _last_r["price_per_sqm"]
            _last_date = pd.to_datetime(_last_r["month"]).strftime("%b %Y")
            if not pd.isna(_t_psm_12m) and _t_psm_12m > 0:
                _vs_pct = (_last_psm - _t_psm_12m) / _t_psm_12m * 100.0
                _vs_lbl = (
                    "above" if _vs_pct > 2 else ("below" if _vs_pct < -2 else "at")
                )
                st.markdown(
                    f"**3. Transaction Recency** \u2014 Most recent: "
                    f"**{fmt_price(_last_r['resale_price'])}** at "
                    f"**{fmt_price(_last_psm)} / sqm** ({_last_date}). "
                    f"This was **{_vs_lbl}** the {g_town} 12-month median by "
                    f"**{_vs_pct:+.1f}%**."
                )
            else:
                st.markdown(
                    f"**3. Transaction Recency** \u2014 Most recent: "
                    f"**{fmt_price(_last_r['resale_price'])}** at "
                    f"**{fmt_price(_last_psm)} / sqm** ({_last_date})."
                )

            # \u2500\u2500 4. Volume trend \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            _n_y1 = len(_sel[_sel["month"] >= _one_yr])
            _n_y2 = len(_sel[(_sel["month"] >= _two_yr) & (_sel["month"] < _one_yr)])
            if _n_y2 > 0:
                _vchg = (_n_y1 - _n_y2) / _n_y2 * 100.0
                _vlbl = (
                    "increased" if _vchg > 10
                    else ("decreased" if _vchg < -10 else "stayed flat")
                )
                st.markdown(
                    f"**4. Volume Trend** \u2014 Transactions for this block / type have "
                    f"**{_vlbl}** over the last 2 years: {_n_y2} (yr \u22122 to \u22121) "
                    f"\u2192 {_n_y1} (last 12 m), **{_vchg:+.0f}%** change."
                )
            else:
                st.markdown(
                    f"**4. Volume Trend** \u2014 {_n_y1} transaction(s) in the last 12 months. "
                    f"Insufficient history to determine trend direction."
                )

            # \u2500\u2500 5. Lease situation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            _avg_l = _sel["remaining_lease_yrs"].mean()
            if not pd.isna(_avg_l):
                _cpf_ok = _avg_l >= 60.0
                st.markdown(
                    f"**5. Lease Situation** \u2014 Average remaining lease: "
                    f"**{_avg_l:.0f} years**. "
                    f"**{'Within' if _cpf_ok else 'Outside'}** the 60-year threshold for "
                    f"full CPF usage by a 35-year-old buyer."
                )
            else:
                st.markdown(
                    "**5. Lease Situation** \u2014 Remaining lease data not available "
                    "for this block / period."
                )

            # \u2500\u2500 6. Buyer leverage signals \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            _n6 = len(_sel[_sel["month"] >= _six_m])
            _levers = []
            if _n6 < 3:
                _levers.append(
                    f"Only **{_n6} unit(s)** transacted in the last 6 months \u2014 "
                    f"**thin supply / seller\u2019s advantage**."
                )
            if not pd.isna(_delta_pct):
                if _delta_pct > 5:
                    _levers.append(
                        f"Price has **risen {_delta_pct:.1f}%** in 6 months \u2014 "
                        f"market is **heating up**; ask the seller to justify the uplift "
                        f"before committing."
                    )
                elif _delta_pct < -3:
                    _levers.append(
                        f"Price has **fallen {abs(_delta_pct):.1f}%** in 6 months \u2014 "
                        f"**buyer has leverage**; consider offering below the most recent "
                        f"comparable."
                    )

            if _levers:
                st.markdown("**6. Buyer Leverage Signals:**")
                for _lv in _levers:
                    st.markdown(f"  - {_lv}")
            else:
                st.markdown(
                    "**6. Buyer Leverage Signals** \u2014 No strong signals detected. "
                    "Market appears balanced for this block / type."
                )

            # \u2500\u2500 7. Comparable anchors (last 5 transactions) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            _last5 = _sel.sort_values("month", ascending=False).head(5).copy()
            _last5["Date"] = _last5["month"].dt.strftime("%b %Y")
            _last5["Price"] = _last5["resale_price"].map(
                lambda x: f"${x:,.0f}"
            )
            _last5["PSM"] = _last5["price_per_sqm"].map(
                lambda x: f"${x:,.0f}"
            )
            _last5["Remaining Lease"] = _last5["remaining_lease_yrs"].map(
                lambda x: f"{x:.0f} yrs" if not pd.isna(x) else "N/A"
            )
            _last5_tbl = _last5[
                ["Date", "storey_range", "floor_area_sqm", "Price", "PSM", "Remaining Lease"]
            ].copy()
            _last5_tbl.columns = [
                "Date", "Storey", "Area (sqm)", "Price ($)", "PSM ($/sqm)", "Remaining Lease",
            ]
            st.markdown(
                "**7. Comparable Anchors** \u2014 last 5 transactions for this block / type:"
            )
            st.dataframe(_last5_tbl, use_container_width=True, hide_index=True)
