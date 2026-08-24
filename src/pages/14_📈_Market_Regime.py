"""
Page 14 - Market Regime & Contrarian Intelligence
==================================================
"Smart Money" analysis: market regime detection, seasonal entry/exit timing,
cooling measure impact studies, stigma persistence detection, and recovery
identification for Singapore HDB resale market.

Tabs:
  D1 - Market Regime Detector
  D5 - Entry/Exit Seasonality
  D6 - Cooling Measure Era Analysis
  D3 - Stigma Persistence Detector
  D4 - Recovery Detector
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from eda_helpers import load_clean, fmt_price, fmt_pct, POLICY_EVENTS

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Market Regime & Contrarian Intelligence")
st.caption(
    "Smart Money analysis: regime detection, seasonal timing, cooling measure impacts, "
    "stigma persistence, and recovery signals for the Singapore HDB resale market."
)

# ── load data ─────────────────────────────────────────────────────────────────
df_raw = load_clean()
df_raw["month"] = pd.to_datetime(df_raw["month"])
df_raw = df_raw.dropna(subset=["price_per_sqm", "resale_price"])

# ── regime constants ──────────────────────────────────────────────────────────
REGIME_COLORS = {
    "Trough":     "#d62728",
    "Recovery":   "#2ca02c",
    "Rising":     "#17becf",
    "Overheated": "#ff7f0e",
    "Cooling":    "#9467bd",
    "Neutral":    "#aec7e8",
}

# Cooling measures used for era analysis (Tab 3)
COOLING_EVENTS_ERA = [
    ("2009-09", "Cooling: 1st Seller Stamp Duty"),
    ("2011-01", "Cooling: ABSD Introduced"),
    ("2013-01", "Cooling: ABSD Raised, TDSR"),
    ("2018-07", "Cooling: ABSD Raised 5%"),
    ("2021-12", "Cooling: ABSD +5%, TDSR Tightened"),
    ("2022-09", "Cooling: 15-month Wait for Private"),
]

# ── cached aggregation helpers ────────────────────────────────────────────────

@st.cache_data
def build_quarterly(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to quarterly national level; compute YoY growth rates."""
    q = (
        data.groupby(data["month"].dt.to_period("Q"))
        .agg(
            median_psm=("price_per_sqm", "median"),
            txn_volume=("resale_price", "size"),
        )
        .reset_index()
    )
    q["quarter"] = q["month"].dt.to_timestamp()
    q = q.sort_values("quarter").reset_index(drop=True)
    q["psm_growth_yoy"] = q["median_psm"].pct_change(4) * 100
    q["vol_growth_yoy"] = q["txn_volume"].pct_change(4) * 100
    return q


@st.cache_data
def build_regimes(data: pd.DataFrame) -> pd.DataFrame:
    """Classify each quarter into a market regime."""
    q = build_quarterly(data)
    regimes = []
    for i, row in q.iterrows():
        g = row["psm_growth_yoy"]
        v = row["vol_growth_yoy"]
        if pd.isna(g) or pd.isna(v):
            regimes.append("Neutral")
            continue
        if g < -5 and v < -10:
            regime = "Trough"
        elif g > 12:
            regime = "Overheated"
        elif 3 <= g <= 12:
            regime = "Rising"
        elif -5 <= g < 3 and v > 0:
            regime = "Recovery"
        elif g < 0 and regimes and regimes[-1] in ("Rising", "Overheated"):
            regime = "Cooling"
        else:
            regime = "Neutral"
        regimes.append(regime)
    q["regime"] = regimes
    return q


@st.cache_data
def build_regime_history(q: pd.DataFrame) -> pd.DataFrame:
    """Collapse sequential same-regime quarters into periods."""
    q_valid = q.dropna(subset=["psm_growth_yoy"]).copy().reset_index(drop=True)
    q_valid["regime_block"] = (q_valid["regime"] != q_valid["regime"].shift()).cumsum()
    hist = (
        q_valid.groupby(["regime_block", "regime"])
        .agg(
            Start=("quarter", "first"),
            End=("quarter", "last"),
            Duration=("quarter", "count"),
            Avg_PSM_Growth=("psm_growth_yoy", "mean"),
        )
        .reset_index()
        .drop(columns=["regime_block"])
    )
    hist["Start"] = hist["Start"].dt.to_period("Q").astype(str)
    hist["End"] = hist["End"].dt.to_period("Q").astype(str)
    hist["Avg PSM Growth (%)"] = hist["Avg_PSM_Growth"].round(1)
    hist = hist.rename(columns={"regime": "Regime", "Duration": "Duration (qtrs)"})
    return hist[["Regime", "Start", "End", "Duration (qtrs)", "Avg PSM Growth (%)"]]


@st.cache_data
def cooling_event_study(
    q: pd.DataFrame,
    cooling_dates: list,
) -> tuple:
    """
    For each cooling measure, compute pre-cooling PSM, min post-event PSM,
    depth, recovery quarters, and indexed event series.
    Returns (results_list, event_series_dict).
    """
    q = q.sort_values("quarter").reset_index(drop=True)
    results = []
    event_series = {}

    for date_str, label in cooling_dates:
        event_date = pd.to_datetime(date_str)
        if event_date < q["quarter"].min() or event_date > q["quarter"].max():
            continue

        # Nearest quarter index
        idx = int((q["quarter"] - event_date).abs().idxmin())

        # Pre-cooling median PSM (4 quarters before)
        pre_slice = q.loc[max(0, idx - 4): idx - 1, "median_psm"]
        pre_psm = float(pre_slice.median()) if not pre_slice.empty else float("nan")

        # Post-event: min PSM in up to 8 quarters after
        post_end = min(len(q) - 1, idx + 8)
        post_psm_series = q.loc[idx: post_end, "median_psm"]
        min_psm = float(post_psm_series.min())
        min_idx = int(post_psm_series.idxmin())

        # Recovery: quarters until PSM returns to pre-cooling level
        recovery_qtrs = None
        if not np.isnan(pre_psm):
            for j in range(min_idx, min(len(q), idx + 24)):
                if q.loc[j, "median_psm"] >= pre_psm:
                    recovery_qtrs = j - idx
                    break

        depth_pct = (min_psm - pre_psm) / pre_psm * 100 if pre_psm > 0 else None

        results.append({
            "Cooling Measure": label,
            "Date": date_str,
            "Pre-cooling PSM": pre_psm,
            "Min PSM (post)": min_psm,
            "Depth (%)": round(depth_pct, 1) if depth_pct is not None else None,
            "Recovery (qtrs)": recovery_qtrs if recovery_qtrs else ">24",
        })

        # Event study series: 8 quarters before to 16 quarters after, indexed to 100
        study_start = max(0, idx - 8)
        study_end = min(len(q) - 1, idx + 16)
        chunk = q.loc[study_start: study_end, ["quarter", "median_psm"]].copy().reset_index(drop=True)
        base_psm = q.loc[idx, "median_psm"]
        if base_psm and base_psm > 0:
            chunk["indexed_psm"] = chunk["median_psm"] / base_psm * 100
            event_pos = idx - study_start          # position of event within chunk
            chunk["offset_q"] = [i - event_pos for i in range(len(chunk))]
            event_series[label] = chunk[["offset_q", "indexed_psm"]].copy()

    return results, event_series


@st.cache_data
def compute_stigma(data: pd.DataFrame) -> tuple:
    """
    Detect blocks with persistently below-market PSM (2015-present).
    Returns (block_summary, block_annual).
    """
    d = data[data["month"].dt.year >= 2015].copy()
    d["year"] = d["month"].dt.year

    # Town+flat_type annual benchmark
    bench = (
        d.groupby(["year", "town", "flat_type"])["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"price_per_sqm": "benchmark_psm"})
    )

    # Block-level annual median (min 2 transactions)
    block_annual = (
        d.groupby(["year", "block", "street_name", "town", "flat_type"])["price_per_sqm"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "block_psm", "count": "txn_count"})
    )
    block_annual = block_annual[block_annual["txn_count"] >= 2]
    block_annual = block_annual.merge(bench, on=["year", "town", "flat_type"], how="left")
    block_annual["residual"] = block_annual["block_psm"] / block_annual["benchmark_psm"] - 1

    max_year = int(block_annual["year"].max())
    last5_years = range(max_year - 4, max_year + 1)
    last5 = block_annual[block_annual["year"].isin(last5_years)]

    # Summary per (block, street, town, flat_type)
    block_summary = (
        block_annual.groupby(["block", "street_name", "town", "flat_type"])
        .agg(
            n_years=("year", "nunique"),
            median_residual=("residual", "median"),
        )
        .reset_index()
    )
    block_summary = block_summary[block_summary["n_years"] >= 3]

    # Count discount years in last 5
    discount_last5 = (
        last5[last5["residual"] < -0.15]
        .groupby(["block", "street_name", "flat_type"])["year"]
        .nunique()
        .reset_index()
        .rename(columns={"year": "discount_yrs_last5"})
    )
    block_summary = block_summary.merge(
        discount_last5, on=["block", "street_name", "flat_type"], how="left"
    )
    block_summary["discount_yrs_last5"] = block_summary["discount_yrs_last5"].fillna(0).astype(int)

    # Most recent 2-year residual
    recent2 = (
        last5[last5["year"] >= max_year - 1]
        .groupby(["block", "street_name", "flat_type"])["residual"]
        .median()
        .reset_index()
        .rename(columns={"residual": "recent2_residual"})
    )
    block_summary = block_summary.merge(
        recent2, on=["block", "street_name", "flat_type"], how="left"
    )

    # Classify
    def classify_stigma(row):
        med = row["median_residual"]
        dyl5 = row["discount_yrs_last5"]
        recent = row.get("recent2_residual", np.nan)
        if med < -0.15 and dyl5 >= 3:
            if pd.notna(recent) and recent > -0.05:
                return "Recovering"
            return "Structural discount"
        if med < -0.10 and dyl5 >= 1:
            if pd.notna(recent) and recent < -0.10:
                return "Emerging discount"
        return "Normal"

    block_summary["classification"] = block_summary.apply(classify_stigma, axis=1)
    return block_summary, block_annual


@st.cache_data
def compute_recovery(data: pd.DataFrame) -> tuple:
    """
    Find blocks with historical discount episodes and track recovery.
    Returns (recovery_df, block_annual).
    """
    d = data[data["month"].dt.year >= 2010].copy()
    d["year"] = d["month"].dt.year

    bench = (
        d.groupby(["year", "town", "flat_type"])["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"price_per_sqm": "benchmark_psm"})
    )

    block_annual = (
        d.groupby(["year", "block", "street_name", "town", "flat_type"])["price_per_sqm"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "block_psm", "count": "txn_count"})
    )
    block_annual = block_annual[block_annual["txn_count"] >= 2]
    block_annual = block_annual.merge(bench, on=["year", "town", "flat_type"], how="left")
    block_annual["residual"] = block_annual["block_psm"] / block_annual["benchmark_psm"] - 1
    block_annual = block_annual.sort_values(
        ["block", "street_name", "flat_type", "year"]
    ).reset_index(drop=True)

    recovery_records = []
    for (blk, street, town, ft), grp in block_annual.groupby(
        ["block", "street_name", "town", "flat_type"]
    ):
        grp = grp.sort_values("year").reset_index(drop=True)
        i = 0
        while i < len(grp):
            if grp.loc[i, "residual"] < -0.10:
                discount_year = int(grp.loc[i, "year"])
                depth = float(grp.loc[i, "residual"]) * 100
                recovered = False
                yrs_to_recover = None
                for j in range(i + 1, min(i + 4, len(grp))):
                    if grp.loc[j, "residual"] >= -0.05:
                        recovered = True
                        yrs_to_recover = int(grp.loc[j, "year"]) - discount_year
                        break
                recovery_records.append(
                    {
                        "block": blk,
                        "street_name": street,
                        "town": town,
                        "flat_type": ft,
                        "discount_year": discount_year,
                        "discount_depth_pct": round(depth, 1),
                        "recovered_within_3yr": recovered,
                        "yrs_to_recover": yrs_to_recover,
                    }
                )
                i += yrs_to_recover if yrs_to_recover else 1
            else:
                i += 1

    recovery_df = pd.DataFrame(recovery_records)
    return recovery_df, block_annual


# ── pre-build quarterly series used across tabs ───────────────────────────────
q_national = build_regimes(df_raw)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Market Regime Detector",
        "📅 Entry/Exit Seasonality",
        "🕐 Cooling Measure Era Analysis",
        "👻 Stigma Persistence Detector",
        "🔄 Recovery Detector",
        "🔀 Price-Fundamentals Divergence",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  –  Market Regime Detector (D1)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Market Regime Detector")
    st.info(
        "**Data confidence: Medium.** Regime labels are rules-based heuristics derived from "
        "historical price and volume patterns. They describe what has happened, not a reliable "
        "prediction of what comes next. Regimes rhyme, they don't repeat exactly."
    )

    q_reg = q_national.copy()

    # ── KPI: current regime ───────────────────────────────────────────────────
    latest = q_reg.dropna(subset=["psm_growth_yoy"]).iloc[-1]
    cur_regime = latest["regime"]
    cur_psm_growth = latest["psm_growth_yoy"]
    cur_vol_growth = latest["vol_growth_yoy"]
    cur_quarter_str = str(latest["month"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Current Regime",
        cur_regime,
        help="Market phase classification for the most recent quarter.",
    )
    c2.metric(
        "PSM Growth YoY",
        fmt_pct(cur_psm_growth),
        help="Year-over-year change in median price/sqm (most recent quarter).",
    )
    c3.metric(
        "Volume Growth YoY",
        fmt_pct(cur_vol_growth),
        help="Year-over-year change in transaction volume (most recent quarter).",
    )
    c4.metric(
        "Latest Quarter",
        cur_quarter_str,
        help="Most recent quarter with sufficient data.",
    )

    st.divider()

    # ── time-series with regime shading ──────────────────────────────────────
    fig_regime = go.Figure()

    # Regime background shading
    q_plot = q_reg.dropna(subset=["regime"]).reset_index(drop=True)
    if not q_plot.empty:
        seg_start = q_plot["quarter"].iloc[0]
        seg_regime = q_plot["regime"].iloc[0]
        for i in range(1, len(q_plot)):
            if q_plot["regime"].iloc[i] != seg_regime or i == len(q_plot) - 1:
                seg_end = q_plot["quarter"].iloc[i]
                fig_regime.add_vrect(
                    x0=seg_start,
                    x1=seg_end,
                    fillcolor=REGIME_COLORS.get(seg_regime, "#cccccc"),
                    opacity=0.18,
                    line_width=0,
                    annotation_text=seg_regime,
                    annotation_position="top left",
                    annotation_font_size=8,
                    annotation_font_color="dimgray",
                )
                seg_start = q_plot["quarter"].iloc[i]
                seg_regime = q_plot["regime"].iloc[i]

    # PSM line
    fig_regime.add_trace(
        go.Scatter(
            x=q_reg["quarter"],
            y=q_reg["median_psm"],
            mode="lines",
            name="Median PSM",
            line=dict(color="#1f77b4", width=2.5),
            hovertemplate="Quarter: %{x|%Y-Q}<br>Median PSM: $%{y:,.0f}<extra></extra>",
        )
    )

    # Policy event vertical lines
    psm_max = q_reg["median_psm"].max()
    for date_str, label in POLICY_EVENTS:
        try:
            ev_date = pd.to_datetime(date_str)
            if q_reg["quarter"].min() <= ev_date <= q_reg["quarter"].max():
                fig_regime.add_vline(
                    x=ev_date.timestamp() * 1000,
                    line_dash="dot",
                    line_color="rgba(60,60,60,0.4)",
                    line_width=1,
                )
                fig_regime.add_annotation(
                    x=ev_date,
                    y=psm_max * 1.01,
                    text=label[:28],
                    showarrow=False,
                    textangle=-75,
                    font=dict(size=7.5, color="gray"),
                    xanchor="left",
                    yanchor="bottom",
                )
        except Exception:
            pass

    fig_regime.update_layout(
        title="National Median PSM by Quarter — Regime Shading & Policy Events",
        xaxis_title="Quarter",
        yaxis_title="Median PSM (S$)",
        hovermode="x unified",
        height=540,
        margin=dict(t=100, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_regime, width="stretch")

    # Regime colour legend
    st.write("**Regime colour key:**")
    leg_cols = st.columns(len(REGIME_COLORS))
    for col, (reg, color) in zip(leg_cols, REGIME_COLORS.items()):
        col.markdown(
            f"<span style='background:{color};padding:2px 10px;"
            f"border-radius:4px;color:white;font-size:12px'>{reg}</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Regime history table ──────────────────────────────────────────────────
    st.subheader("Regime History")
    hist_df = build_regime_history(q_reg)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  –  Entry / Exit Seasonality (D5)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📅 Entry/Exit Seasonality")
    st.info(
        "**Data confidence: Low–Medium.** Seasonal patterns are modest and vary by year. "
        "Use as background context, not as a primary timing strategy."
    )

    # ── filters ───────────────────────────────────────────────────────────────
    towns_list = sorted(df_raw["town"].dropna().unique().tolist())
    col_f1, col_f2 = st.columns([2, 3])
    with col_f1:
        town_sel = st.selectbox(
            "Filter by town (optional)",
            ["All Towns"] + towns_list,
            key="seas_town",
        )

    max_data_year = int(df_raw["month"].dt.year.max())
    cutoff_year = max_data_year - 10
    df_seas = df_raw[df_raw["month"].dt.year >= cutoff_year].copy()
    if town_sel != "All Towns":
        df_seas = df_seas[df_seas["town"] == town_sel]

    df_seas["month_num"] = df_seas["month"].dt.month
    df_seas["quarter_num"] = df_seas["month"].dt.quarter

    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if df_seas.empty:
        st.warning("No data for the selected filters.")
    else:
        # ── monthly aggregates ────────────────────────────────────────────────
        monthly_seas = (
            df_seas.groupby("month_num")
            .agg(
                median_psm=("price_per_sqm", "median"),
                txn_volume=("resale_price", "size"),
            )
            .reset_index()
        )
        monthly_seas["month_name"] = monthly_seas["month_num"].apply(
            lambda m: MONTH_NAMES[m - 1]
        )
        total_vol = monthly_seas["txn_volume"].sum()
        monthly_seas["vol_share_pct"] = (monthly_seas["txn_volume"] / total_vol * 100).round(1)

        monthly_seas["is_cheap"] = (
            monthly_seas["median_psm"] <= monthly_seas["median_psm"].nsmallest(3).max()
        )
        monthly_seas["is_active"] = (
            monthly_seas["txn_volume"] >= monthly_seas["txn_volume"].nlargest(3).min()
        )

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            psm_bar_colors = [
                "#2ca02c" if v else "#aec7e8" for v in monthly_seas["is_cheap"]
            ]
            fig_mpsm = go.Figure(
                go.Bar(
                    x=monthly_seas["month_name"],
                    y=monthly_seas["median_psm"],
                    marker_color=psm_bar_colors,
                    text=monthly_seas["median_psm"].apply(lambda v: f"${v:,.0f}"),
                    textposition="outside",
                    hovertemplate="%{x}: $%{y:,.0f}/sqm<extra></extra>",
                )
            )
            fig_mpsm.update_layout(
                title="Median PSM by Month  (green = lowest 3)",
                xaxis_title="Month",
                yaxis_title="Median PSM (S$)",
                height=380,
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig_mpsm, width="stretch")
            best_buy = monthly_seas[monthly_seas["is_cheap"]]["month_name"].tolist()
            st.success(f"**Best months to buy (lowest PSM):** {', '.join(best_buy)}")

        with col_m2:
            vol_bar_colors = [
                "#1f77b4" if v else "#c7dbee" for v in monthly_seas["is_active"]
            ]
            fig_mvol = go.Figure(
                go.Bar(
                    x=monthly_seas["month_name"],
                    y=monthly_seas["txn_volume"],
                    marker_color=vol_bar_colors,
                    text=monthly_seas["txn_volume"].apply(lambda v: f"{v:,}"),
                    textposition="outside",
                    hovertemplate="%{x}: %{y:,} transactions<extra></extra>",
                )
            )
            fig_mvol.update_layout(
                title="Transaction Volume by Month  (blue = most active 3)",
                xaxis_title="Month",
                yaxis_title="Transaction Count",
                height=380,
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig_mvol, width="stretch")
            most_active = monthly_seas[monthly_seas["is_active"]]["month_name"].tolist()
            st.info(f"**Most active months:** {', '.join(most_active)}")

        st.divider()

        # ── quarterly patterns ────────────────────────────────────────────────
        st.subheader("Quarterly Patterns")
        qtr_seas = (
            df_seas.groupby("quarter_num")
            .agg(
                median_psm=("price_per_sqm", "median"),
                txn_volume=("resale_price", "size"),
            )
            .reset_index()
        )
        qtr_seas["quarter_label"] = qtr_seas["quarter_num"].apply(lambda q: f"Q{q}")

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            fig_qpsm = px.bar(
                qtr_seas,
                x="quarter_label",
                y="median_psm",
                title="Median PSM by Quarter",
                labels={"quarter_label": "Quarter", "median_psm": "Median PSM (S$)"},
                color_discrete_sequence=["#17becf"],
                text_auto=True,
            )
            fig_qpsm.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
            fig_qpsm.update_layout(height=360, margin=dict(t=50))
            st.plotly_chart(fig_qpsm, width="stretch")

        with col_q2:
            fig_qvol = px.bar(
                qtr_seas,
                x="quarter_label",
                y="txn_volume",
                title="Transaction Volume by Quarter",
                labels={"quarter_label": "Quarter", "txn_volume": "Transactions"},
                color_discrete_sequence=["#ff7f0e"],
                text_auto=True,
            )
            fig_qvol.update_traces(texttemplate="%{y:,}", textposition="outside")
            fig_qvol.update_layout(height=360, margin=dict(t=50))
            st.plotly_chart(fig_qvol, width="stretch")

        st.caption(
            f"Analysis covers last 10 years ({cutoff_year}–{max_data_year}). "
            f"Town: {town_sel}."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  –  Cooling Measure Era Analysis (D6)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🕐 Cooling Measure Era Analysis")
    st.info(
        "**Data confidence: Medium.** Historical cooling measure impacts are derived from "
        "factual price series. How the market responds to future measures may differ given "
        "different macro-economic contexts."
    )

    q_era = build_quarterly(df_raw)
    era_results, era_event_series = cooling_event_study(q_era, COOLING_EVENTS_ERA)

    # ── impact summary table ──────────────────────────────────────────────────
    st.subheader("Impact Summary by Cooling Measure")
    impact_df = pd.DataFrame(era_results).copy()
    impact_df["Pre-cooling PSM"] = impact_df["Pre-cooling PSM"].apply(
        lambda v: f"${v:,.0f}" if pd.notna(v) and isinstance(v, float) else "N/A"
    )
    impact_df["Min PSM (post)"] = impact_df["Min PSM (post)"].apply(
        lambda v: f"${v:,.0f}" if pd.notna(v) and isinstance(v, float) else "N/A"
    )
    impact_df["Depth (%)"] = impact_df["Depth (%)"].apply(
        lambda v: f"{v:.1f}%" if pd.notna(v) and isinstance(v, (int, float)) else "N/A"
    )
    impact_df["Recovery (qtrs)"] = impact_df["Recovery (qtrs)"].astype(str)

    st.dataframe(
        impact_df[
            ["Cooling Measure", "Date", "Pre-cooling PSM", "Min PSM (post)",
             "Depth (%)", "Recovery (qtrs)"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── event study chart ─────────────────────────────────────────────────────
    st.subheader(
        "Event Study: Indexed PSM — 8 quarters before to 16 quarters after each measure"
    )
    st.caption("All lines normalised to 100 at the cooling measure date (quarter 0).")

    EV_COLORS = px.colors.qualitative.Set1
    fig_ev = go.Figure()
    for i, (label, _) in enumerate(COOLING_EVENTS_ERA):
        if label in era_event_series:
            chunk = era_event_series[label]
            fig_ev.add_trace(
                go.Scatter(
                    x=chunk["offset_q"],
                    y=chunk["indexed_psm"],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=EV_COLORS[i % len(EV_COLORS)], width=2),
                    marker=dict(size=4),
                    hovertemplate="Quarter offset: %{x}<br>Index: %{y:.1f}<extra>" + label + "</extra>",
                )
            )

    fig_ev.add_hline(
        y=100,
        line_dash="dash",
        line_color="black",
        line_width=1,
        annotation_text="Event baseline (100)",
        annotation_position="right",
    )
    fig_ev.add_vline(
        x=0,
        line_dash="dash",
        line_color="crimson",
        line_width=1.5,
        annotation_text="Cooling measure",
        annotation_position="top right",
    )
    fig_ev.update_layout(
        xaxis_title="Quarters from cooling measure (0 = event date)",
        yaxis_title="Indexed PSM (100 = event date level)",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        margin=dict(b=120, t=40),
    )
    st.plotly_chart(fig_ev, width="stretch")

    st.divider()

    # ── current market context ────────────────────────────────────────────────
    st.subheader("Current Market Context")
    cur_q_era = q_era.dropna(subset=["psm_growth_yoy"]).iloc[-1]
    cur_psm_era = cur_q_era["median_psm"]
    cur_growth_era = cur_q_era["psm_growth_yoy"]

    c_e1, c_e2 = st.columns(2)
    c_e1.metric("Current Median PSM", fmt_price(cur_psm_era))
    c_e2.metric("YoY PSM Growth", fmt_pct(cur_growth_era))

    if cur_growth_era < -3:
        st.markdown(
            "With meaningfully negative YoY price growth, the market most resembles the "
            "post-cooling periods following the **2013 ABSD/TDSR measures** and "
            "**2018 ABSD hike** — both featured gradual corrections over 4–8 quarters "
            "before demand rebuilt."
        )
    elif cur_growth_era < 0:
        st.markdown(
            "With mildly negative YoY growth, the market is in an early cooling phase. "
            "Historically, HDB resale prices have stabilised within 2–4 quarters of "
            "mild corrections."
        )
    elif cur_growth_era > 12:
        st.markdown(
            "With double-digit YoY growth, the market resembles the **2021–22 overheated "
            "period**, which preceded two cooling rounds in Dec 2021 and Sep 2022. "
            "Monitor policy signals closely."
        )
    else:
        st.markdown(
            "The market is in moderate positive growth territory. "
            "Watch volume trends and government policy statements for early signals "
            "of the next intervention cycle."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  –  Stigma Persistence Detector (D3)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("👻 Stigma Persistence Detector")
    st.warning(
        "⚠️ **CRITICAL DATA CONFIDENCE WARNING:** A persistent discount is a **signal for "
        "investigation, NOT a confirmed problem or stigma.** Common innocent explanations "
        "include: consistently lower floors, older flat model, less desirable block "
        "orientation, proximity to a road or industrial area, or systematically lower "
        "renovation quality. **NEVER present this to a client as confirmed stigma without "
        "independent on-the-ground verification.**"
    )

    with st.spinner("Computing block-level stigma metrics (2015–present)…"):
        block_summary, block_annual = compute_stigma(df_raw)

    # ── filters ───────────────────────────────────────────────────────────────
    col_sf1, col_sf2 = st.columns(2)
    with col_sf1:
        stigma_towns = sorted(block_summary["town"].dropna().unique().tolist())
        sel_s_town = st.selectbox(
            "Filter by town", ["All"] + stigma_towns, key="stigma_town"
        )
    with col_sf2:
        all_cls = sorted(block_summary["classification"].unique().tolist())
        default_cls = [c for c in ["Structural discount", "Recovering", "Emerging discount"]
                       if c in all_cls]
        sel_cls = st.multiselect(
            "Filter by classification",
            all_cls,
            default=default_cls,
            key="stigma_cls",
        )

    disp = block_summary.copy()
    if sel_s_town != "All":
        disp = disp[disp["town"] == sel_s_town]
    if sel_cls:
        disp = disp[disp["classification"].isin(sel_cls)]

    disp = disp.copy()
    disp["Median Discount %"] = (disp["median_residual"] * 100).round(1)
    disp["Years With Discount (last 5)"] = disp["discount_yrs_last5"]

    disp_show = (
        disp[["block", "street_name", "town", "flat_type",
              "Median Discount %", "Years With Discount (last 5)", "classification"]]
        .rename(columns={
            "block": "Block",
            "street_name": "Street",
            "town": "Town",
            "flat_type": "Flat Type",
            "classification": "Classification",
        })
        .sort_values("Median Discount %")
        .head(200)
    )

    st.write(
        f"Showing **{len(disp_show)}** blocks "
        f"(filtered from {len(block_summary)} qualifying blocks; capped at 200 for performance)."
    )
    st.dataframe(disp_show, use_container_width=True, hide_index=True)

    # ── KPI summary ───────────────────────────────────────────────────────────
    n_struct = (block_summary["classification"] == "Structural discount").sum()
    n_recov = (block_summary["classification"] == "Recovering").sum()
    n_emerg = (block_summary["classification"] == "Emerging discount").sum()
    ks1, ks2, ks3 = st.columns(3)
    ks1.metric("Structural discount blocks", f"{n_struct:,}")
    ks2.metric("Recovering blocks", f"{n_recov:,}")
    ks3.metric("Emerging discount blocks", f"{n_emerg:,}")

    st.divider()

    # ── block trend chart ─────────────────────────────────────────────────────
    st.subheader("Block vs Town Benchmark — Annual PSM Trend")
    if not disp.empty:
        block_opts = (
            disp["block"] + " " + disp["street_name"] + " (" + disp["flat_type"] + ")"
        ).tolist()
        sel_block_label = st.selectbox(
            "Select a block to inspect", block_opts, key="stigma_block_sel"
        )
        sel_idx = block_opts.index(sel_block_label)
        sel_row = disp.iloc[sel_idx]

        block_trend = block_annual[
            (block_annual["block"] == sel_row["block"])
            & (block_annual["street_name"] == sel_row["street_name"])
            & (block_annual["flat_type"] == sel_row["flat_type"])
        ].sort_values("year")

        if not block_trend.empty:
            fig_bt = go.Figure()
            fig_bt.add_trace(
                go.Scatter(
                    x=block_trend["year"],
                    y=block_trend["block_psm"],
                    mode="lines+markers",
                    name=f"{sel_row['block']} {sel_row['street_name']}",
                    line=dict(color="#d62728", width=2.5),
                    marker=dict(size=8),
                    hovertemplate="%{x}: $%{y:,.0f}/sqm<extra>Block</extra>",
                )
            )
            fig_bt.add_trace(
                go.Scatter(
                    x=block_trend["year"],
                    y=block_trend["benchmark_psm"],
                    mode="lines+markers",
                    name=f"{sel_row['town']} {sel_row['flat_type']} benchmark",
                    line=dict(color="#1f77b4", width=2.5, dash="dash"),
                    marker=dict(size=8),
                    hovertemplate="%{x}: $%{y:,.0f}/sqm<extra>Benchmark</extra>",
                )
            )
            fig_bt.update_layout(
                title=(
                    f"{sel_row['block']} {sel_row['street_name']} "
                    f"({sel_row['flat_type']}) — PSM vs Town Benchmark"
                ),
                xaxis_title="Year",
                yaxis_title="Median PSM (S$)",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_bt, width="stretch")

            # Residual mini chart
            fig_res = go.Figure(
                go.Bar(
                    x=block_trend["year"],
                    y=(block_trend["residual"] * 100).round(1),
                    marker_color=[
                        "#d62728" if v < -15 else "#ff7f0e" if v < -5 else "#2ca02c"
                        for v in block_trend["residual"]
                    ],
                    hovertemplate="%{x}: %{y:.1f}%<extra>Discount vs benchmark</extra>",
                    name="Discount %",
                )
            )
            fig_res.add_hline(y=-15, line_dash="dot", line_color="gray",
                              annotation_text="-15% threshold")
            fig_res.update_layout(
                title="Annual Discount vs Town Benchmark (%)",
                xaxis_title="Year",
                yaxis_title="Residual (%)",
                height=280,
            )
            st.plotly_chart(fig_res, width="stretch")
        else:
            st.info("No annual trend data available for this block.")
    else:
        st.info("No blocks match the current filters.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  –  Recovery Detector (D4)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔄 Recovery Detector")
    st.error(
        "**Data confidence: Low.** Past recoveries do not guarantee future performance. "
        "Each discount has its own root cause. Use as background context only."
    )

    with st.spinner("Computing historical discount-and-recovery episodes (2010–present)…"):
        recovery_df, block_annual_rec = compute_recovery(df_raw)

    if recovery_df.empty:
        st.info("No recovery data could be computed from the current dataset.")
    else:
        # ── recovery rate by town ─────────────────────────────────────────────
        town_recovery = (
            recovery_df.groupby("town")
            .agg(
                total_episodes=("recovered_within_3yr", "count"),
                recovered_episodes=("recovered_within_3yr", "sum"),
            )
            .reset_index()
        )
        town_recovery["recovery_rate_pct"] = (
            town_recovery["recovered_episodes"] / town_recovery["total_episodes"] * 100
        ).round(1)
        town_recovery = town_recovery.sort_values("recovery_rate_pct", ascending=False)

        st.subheader("Recovery Rate by Town")
        st.caption(
            "Percentage of blocks that traded >10% below their town+flat-type benchmark "
            "and fully recovered to within 5% of benchmark within 3 years."
        )

        fig_rr = px.bar(
            town_recovery,
            x="town",
            y="recovery_rate_pct",
            color="recovery_rate_pct",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            labels={"town": "Town", "recovery_rate_pct": "Recovery Rate (%)"},
            title="Discount Recovery Rate by Town (within 3 years)",
            text="recovery_rate_pct",
        )
        fig_rr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_rr.update_layout(
            height=500,
            xaxis_tickangle=-45,
            coloraxis_showscale=False,
            margin=dict(b=120, t=60),
        )
        st.plotly_chart(fig_rr, width="stretch")

        # KPI
        avg_rate = town_recovery["recovery_rate_pct"].mean()
        best = town_recovery.iloc[0]
        worst = town_recovery.iloc[-1]
        kr1, kr2, kr3 = st.columns(3)
        kr1.metric(
            "Overall average recovery rate",
            f"{avg_rate:.1f}%",
            help="Across all towns, % of discounted blocks recovering within 3 years.",
        )
        kr2.metric(f"Highest: {best['town']}", f"{best['recovery_rate_pct']:.1f}%")
        kr3.metric(f"Lowest: {worst['town']}", f"{worst['recovery_rate_pct']:.1f}%")

        st.divider()

        # ── currently discounted blocks ───────────────────────────────────────
        st.subheader("Currently Discounted Blocks — Historical Recovery Context")
        max_year_rec = int(block_annual_rec["year"].max())
        cur_disc = block_annual_rec[
            (block_annual_rec["year"] >= max_year_rec - 1)
            & (block_annual_rec["residual"] < -0.10)
        ].copy()

        if not cur_disc.empty:
            cur_disc = cur_disc.merge(
                town_recovery[["town", "recovery_rate_pct"]], on="town", how="left"
            )
            cur_disc["Discount %"] = (cur_disc["residual"] * 100).round(1)
            cur_disc = cur_disc.rename(columns={"recovery_rate_pct": "Town Recovery Rate %"})

            col_rft, col_rtown = st.columns(2)
            with col_rft:
                ft_opts = ["All"] + sorted(cur_disc["flat_type"].dropna().unique().tolist())
                sel_rft = st.selectbox("Filter by flat type", ft_opts, key="rec_ft")
            with col_rtown:
                rt_opts = ["All"] + sorted(cur_disc["town"].dropna().unique().tolist())
                sel_rtown = st.selectbox("Filter by town", rt_opts, key="rec_town")

            disp_cur = cur_disc.copy()
            if sel_rft != "All":
                disp_cur = disp_cur[disp_cur["flat_type"] == sel_rft]
            if sel_rtown != "All":
                disp_cur = disp_cur[disp_cur["town"] == sel_rtown]

            disp_cur_show = (
                disp_cur[["block", "street_name", "town", "flat_type",
                           "Discount %", "Town Recovery Rate %"]]
                .rename(columns={
                    "block": "Block",
                    "street_name": "Street",
                    "town": "Town",
                    "flat_type": "Flat Type",
                })
                .sort_values("Discount %")
                .head(100)
            )
            st.write(
                f"Showing **{len(disp_cur_show)}** currently discounted blocks "
                f"(capped at 100 for performance)."
            )
            st.dataframe(disp_cur_show, use_container_width=True, hide_index=True)

            if sel_rtown != "All":
                town_rate_row = town_recovery[town_recovery["town"] == sel_rtown]
                if not town_rate_row.empty:
                    rate = town_rate_row.iloc[0]["recovery_rate_pct"]
                    eps = int(town_rate_row.iloc[0]["total_episodes"])
                    st.info(
                        f"In **{sel_rtown}**, **{rate:.1f}%** of historically discounted "
                        f"blocks recovered to within 5% of town benchmark within 3 years "
                        f"(based on {eps} historical episodes since 2010)."
                    )
        else:
            st.success("No blocks are currently trading at a significant discount vs their town benchmark.")

        st.divider()

        # ── all historical episodes ───────────────────────────────────────────
        st.subheader("All Historical Discount Episodes")
        col_hft, col_htown = st.columns(2)
        with col_hft:
            hft_opts = ["All"] + sorted(recovery_df["flat_type"].dropna().unique().tolist())
            sel_hft = st.selectbox("Filter by flat type", hft_opts, key="hist_ft")
        with col_htown:
            htown_opts = ["All"] + sorted(recovery_df["town"].dropna().unique().tolist())
            sel_htown = st.selectbox("Filter by town", htown_opts, key="hist_town")

        hist_ep = recovery_df.copy()
        if sel_hft != "All":
            hist_ep = hist_ep[hist_ep["flat_type"] == sel_hft]
        if sel_htown != "All":
            hist_ep = hist_ep[hist_ep["town"] == sel_htown]

        hist_ep["Recovered within 3 yrs?"] = hist_ep["recovered_within_3yr"].map(
            {True: "Yes", False: "No"}
        )
        hist_ep["Discount Depth %"] = hist_ep["discount_depth_pct"]
        hist_ep["Years to Recover"] = hist_ep["yrs_to_recover"].apply(
            lambda v: str(int(v)) if pd.notna(v) else "-"
        )

        hist_show = (
            hist_ep[["block", "street_name", "town", "flat_type",
                      "discount_year", "Discount Depth %",
                      "Recovered within 3 yrs?", "Years to Recover"]]
            .rename(columns={
                "block": "Block",
                "street_name": "Street",
                "town": "Town",
                "flat_type": "Flat Type",
                "discount_year": "Year",
            })
            .sort_values("Discount Depth %")
            .head(200)
        )
        st.write(
            f"Showing **{len(hist_show)}** episodes "
            f"(capped at 200 for performance)."
        )
        st.dataframe(hist_show, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6  –  Price-Fundamentals Divergence (D2)
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🔀 Price vs Fundamentals Divergence")
    st.caption(
        "Where has price moved faster (or slower) than volume and lease fundamentals? "
        "Locations where price surged without volume support are more likely to correct."
    )

    @st.cache_data
    def compute_divergence(_df):
        # Quarterly national stats
        q = _df.copy()
        q["qtr"] = q["month"].dt.to_period("Q").astype(str)
        # Town-level: quarterly median PSM and volume
        town_q = (q.groupby(["town", "flat_type", "qtr"])
                  .agg(median_psm=("price_per_sqm", "median"),
                       volume=("resale_price", "count"),
                       avg_lease=("remaining_lease_yrs", "mean"))
                  .reset_index())
        town_q["qtr_dt"] = pd.PeriodIndex(town_q["qtr"].str.replace("-", "").str.replace("Q","Q"), freq="Q").to_timestamp()
        town_q = town_q.sort_values(["town", "flat_type", "qtr_dt"])

        # Compute 4-quarter YoY growth for PSM and volume per town+type
        town_q["psm_yoy"] = (town_q.groupby(["town", "flat_type"])["median_psm"]
                              .pct_change(4) * 100)
        town_q["vol_yoy"] = (town_q.groupby(["town", "flat_type"])["volume"]
                              .pct_change(4) * 100)

        # Divergence = PSM growth - Volume growth (large positive = price surged without volume)
        town_q["divergence"] = town_q["psm_yoy"] - town_q["vol_yoy"]
        return town_q

    div_df = compute_divergence(df_raw)

    # Filter to recent 2 years
    recent_div = div_df[div_df["qtr_dt"] >= pd.Timestamp.now() - pd.DateOffset(years=2)]

    d2_col1, d2_col2 = st.columns([1, 3])
    with d2_col1:
        d2_flat = st.selectbox("Flat type", sorted(df_raw["flat_type"].unique()),
                               index=sorted(df_raw["flat_type"].unique()).index("4 ROOM")
                               if "4 ROOM" in df_raw["flat_type"].unique() else 0,
                               key="d2_flat")
        d2_yrs = st.slider("Analysis window (years)", 1, 5, 2, key="d2_yrs")

    filt_div = div_df[(div_df["flat_type"] == d2_flat) &
                      (div_df["qtr_dt"] >= pd.Timestamp.now() - pd.DateOffset(years=d2_yrs))].dropna(subset=["divergence"])

    if len(filt_div) > 0:
        # Average divergence per town over the window
        town_avg_div = (filt_div.groupby("town")["divergence"].mean().reset_index()
                        .sort_values("divergence", ascending=False))
        town_avg_div["signal"] = town_avg_div["divergence"].apply(
            lambda x: "Price surging > volume" if x > 10 else
                      "Price lagging > volume" if x < -10 else "Broadly in sync")

        with d2_col2:
            fig_div = px.bar(
                town_avg_div,
                x="town", y="divergence",
                color="signal",
                color_discrete_map={
                    "Price surging > volume": "#d62728",
                    "Price lagging > volume": "#2ca02c",
                    "Broadly in sync": "#7f7f7f",
                },
                labels={"town": "Town", "divergence": "Avg PSM growth - Volume growth (pp)"},
                title=f"Price vs Volume Divergence by Town — {d2_flat} (last {d2_yrs} years)",
            )
            fig_div.add_hline(y=0, line_dash="dash", line_color="black")
            fig_div.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig_div, use_container_width=True)

        # Scatter: PSM growth vs volume growth
        latest_div = (filt_div.groupby("town")[["psm_yoy", "vol_yoy"]].mean().reset_index())
        fig_scat = px.scatter(
            latest_div, x="vol_yoy", y="psm_yoy", text="town",
            labels={"vol_yoy": "Volume growth YoY (avg, %)", "psm_yoy": "PSM growth YoY (avg, %)"},
            title="Price Growth vs Volume Growth (each dot = town)",
        )
        fig_scat.add_hline(y=0, line_dash="dot"); fig_scat.add_vline(x=0, line_dash="dot")
        fig_scat.update_traces(textposition="top center")
        st.plotly_chart(fig_scat, use_container_width=True)

        st.info(
            "**Interpretation:** Top-right (Q1) = price and volume both rising — healthy bull market. "
            "Top-left (Q2) = price rising while volume falling — thinner market, possible overextension. "
            "Bottom-left (Q3) = price and volume both falling — bear/cooling market. "
            "Bottom-right (Q4) = volume rising while price lags — potential accumulation zone."
        )

    st.warning(
        "DATA CONFIDENCE: Medium. Divergence uses YoY quarterly medians as proxies for fundamentals. "
        "Volume changes can reflect genuine demand shifts, new flat releases, or seasonal noise. "
        "This is a directional indicator — not a precise market signal."
    )
