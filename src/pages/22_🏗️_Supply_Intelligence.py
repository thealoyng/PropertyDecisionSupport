"""
Page 22 — Supply Intelligence
===============================
Supply-side signals for Singapore property market:

C3: Supply Pressure Score  — MOP unlock waves + URA private pipeline by town
C5: SERS Watch List        — pattern-match of HDB blocks matching historical SERS profile
I2: Private Pipeline       — URA developer launched / sold / unsold quarterly data
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from eda_helpers import (
    load_clean, fmt_price, TOWN_CENTROIDS, POLICY_EVENTS,
    load_ura_launched, load_ura_sold, load_ura_unsold,
)

st.set_page_config(page_title="Supply Intelligence", page_icon="🏗️", layout="wide")

st.title("🏗️ Supply Intelligence")
st.caption(
    "Supply-side signals: MOP unlock waves, SERS pattern watch, and URA developer pipeline. "
    "All metrics derived from official public datasets — see confidence notes on each section."
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TODAY = pd.Timestamp.now()
CURRENT_YEAR = TODAY.year

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pressure, tab_sers, tab_pipeline = st.tabs([
    "📊 Supply Pressure Score",
    "🏢 SERS Watch List",
    "🏗️ Private Pipeline (I2)",
])


# ═══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_resale_base():
    df = load_clean()
    df["month"] = pd.to_datetime(df["month"])
    df["year"]  = df["year"].astype(int)
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    df["mop_unlock_year"] = df["lease_commence_date"] + 5
    return df


@st.cache_data
def compute_mop_pressure(_df: pd.DataFrame, horizon_months: int = 24) -> pd.DataFrame:
    """
    For each town, estimate MOP-unlock pressure over the next horizon_months.
    Returns a DataFrame with columns: town, wave_units, avg_annual_vol, mop_ratio, mop_score.
    mop_score is normalised 0-10 across towns.
    """
    horizon_end = CURRENT_YEAR + (horizon_months / 12)

    # Units with MOP unlocking in the next horizon (using lease_commence_date+5 proxy)
    # Use only data from 2017+ for better coverage
    recent = _df[_df["year"] >= 2017].copy()
    wave = recent[
        (recent["mop_unlock_year"] >= CURRENT_YEAR) &
        (recent["mop_unlock_year"] <= horizon_end)
    ]
    wave_by_town = wave.groupby("town").size().reset_index(name="wave_units")

    # Average annual volume per town (last 3 years)
    max_date = _df["month"].max()
    cutoff3yr = max_date - pd.DateOffset(years=3)
    vol3yr = (
        _df[_df["month"] >= cutoff3yr]
        .groupby("town")
        .size()
        .reset_index(name="total_3yr")
    )
    vol3yr["avg_annual_vol"] = vol3yr["total_3yr"] / 3

    merged = pd.merge(wave_by_town, vol3yr[["town", "avg_annual_vol"]], on="town", how="outer")
    merged["wave_units"]    = merged["wave_units"].fillna(0)
    merged["avg_annual_vol"] = merged["avg_annual_vol"].fillna(50)
    merged["mop_ratio"] = (merged["wave_units"] / (merged["avg_annual_vol"] * (horizon_months / 12))).clip(0, 5)

    lo, hi = merged["mop_ratio"].min(), merged["mop_ratio"].max()
    merged["mop_score"] = ((merged["mop_ratio"] - lo) / (hi - lo) * 10).clip(0, 10).round(2) if hi > lo else 5.0

    return merged.sort_values("mop_score", ascending=False).reset_index(drop=True)


@st.cache_data
def compute_sers_candidates(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pattern-match HDB blocks against 4 historical SERS criteria.
    Returns a DataFrame of candidate blocks with criteria count and flag.
    """
    SERS_TOWNS = {
        "ANG MO KIO", "BEDOK", "BISHAN", "BUONA VISTA", "BUKIT MERAH",
        "CLEMENTI", "GEYLANG", "JURONG WEST", "KALLANG/WHAMPOA",
        "PASIR RIS", "QUEENSTOWN", "TOA PAYOH", "WOODLANDS", "YISHUN",
        "MARINE PARADE", "TAMPINES",
    }

    max_date = _df["month"].max()
    cutoff3yr = max_date - pd.DateOffset(years=3)
    cutoff2yr = max_date - pd.DateOffset(years=2)

    recent3 = _df[_df["month"] >= cutoff3yr].copy()
    recent2 = _df[_df["month"] >= cutoff2yr].copy()

    # Town medians
    town_psm = recent3.groupby("town")["price_per_sqm"].median().rename("town_median_psm")

    # Block-level metrics (3yr window)
    blk = (
        recent3
        .groupby(["town", "block", "street_name"])
        .agg(
            median_psm=("price_per_sqm", "median"),
            avg_flat_age=("flat_age", "mean"),
            txn_3yr=("resale_price", "size"),
        )
        .reset_index()
    )
    blk["avg_flat_age"] = blk["avg_flat_age"].round(1)

    # 2yr transaction count
    txn2yr = (
        recent2
        .groupby(["town", "block", "street_name"])
        .size()
        .reset_index(name="txn_2yr")
    )
    blk = blk.merge(txn2yr, on=["town", "block", "street_name"], how="left")
    blk["txn_2yr"] = blk["txn_2yr"].fillna(0).astype(int)

    blk = blk.merge(town_psm, on="town", how="left")
    blk["psm_discount_pct"] = (
        (blk["town_median_psm"] - blk["median_psm"]) / blk["town_median_psm"] * 100
    ).round(1)

    # Apply 4 criteria
    blk["c1_age"]   = blk["avg_flat_age"] >= 35
    blk["c2_town"]  = blk["town"].isin(SERS_TOWNS)
    blk["c3_disc"]  = blk["psm_discount_pct"] >= 5
    blk["c4_low_v"] = blk["txn_2yr"] <= 4

    blk["criteria_met"] = (
        blk["c1_age"].astype(int) + blk["c2_town"].astype(int) +
        blk["c3_disc"].astype(int) + blk["c4_low_v"].astype(int)
    )
    blk["flag"] = blk["criteria_met"].map(
        {4: "🔴 High Priority Watch", 3: "🟡 Watch", 2: "ℹ️ Low", 1: "—", 0: "—"}
    )

    return blk[blk["criteria_met"] >= 2].sort_values(
        ["criteria_met", "avg_flat_age"], ascending=[False, False]
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Supply Pressure Score
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pressure:
    st.subheader("📊 Supply Pressure Score (C3)")
    st.markdown(
        "Estimates forward-looking resale supply pressure per town by combining "
        "**MOP unlock waves** (derived from `lease_commence_date + 5`) with URA private pipeline context."
    )

    horizon = st.select_slider(
        "MOP unlock horizon", options=[12, 18, 24, 36], value=24,
        format_func=lambda x: f"{x} months", key="sp_horizon"
    )
    w_mop_pct = st.slider(
        "Weight: MOP pressure vs Private pipeline (%MOP)",
        min_value=50, max_value=100, value=70, step=10, key="sp_wmop"
    ) / 100

    df_resale = load_resale_base()
    mop_df    = compute_mop_pressure(df_resale, horizon_months=horizon)

    # URA private pipeline pressure (CCR/RCR/OCR → proxy for town zones)
    unsold_df = load_ura_unsold()
    private_score = pd.Series(5.0, index=mop_df.index)  # neutral default
    if not unsold_df.empty:
        try:
            pipeline = unsold_df[
                (unsold_df.get("completion_status", pd.Series()) == "Uncompleted") &
                (unsold_df.get("launch_status", pd.Series()) == "Launched") &
                (unsold_df.get("pre_requisites_status", pd.Series()) == "With Pre-Requisites")
            ]
            if pipeline.empty:
                # Try column name variants
                col_names = unsold_df.columns.tolist()
                comp_col = next((c for c in col_names if "complet" in c.lower()), None)
                launch_col = next((c for c in col_names if "launch" in c.lower()), None)
                prereq_col = next((c for c in col_names if "pre" in c.lower()), None)
                if comp_col and launch_col and prereq_col:
                    pipeline = unsold_df[
                        (unsold_df[comp_col] == "Uncompleted") &
                        (unsold_df[launch_col] == "Launched") &
                        (unsold_df[prereq_col] == "With Pre-Requisites")
                    ]
            latest_qtrs = pipeline["quarter"].nlargest(4) if "quarter" in pipeline.columns else pd.Series()
            if not latest_qtrs.empty:
                recent_pipeline = pipeline[pipeline["quarter"].isin(latest_qtrs)]
                seg_units = recent_pipeline.groupby("market_segment")["units"].sum()
                ocr_units = float(seg_units.get("Outside Central Region", 0))
                total_units = float(seg_units.sum()) or 1
                # OCR pressure maps roughly to heartland HDB towns
                ocr_ratio = ocr_units / total_units
                # Score 0–10: higher OCR share → more competition for HDB upgraders
                pipeline_score = min(10, ocr_ratio * 10 * 1.5)
                # Apply evenly as a modifier — private pipeline is national, not town-specific
                private_score = pd.Series(round(pipeline_score, 1), index=mop_df.index)
        except Exception:
            pass

    mop_df["private_score"] = private_score.values
    mop_df["composite"] = (
        w_mop_pct * mop_df["mop_score"] +
        (1 - w_mop_pct) * mop_df["private_score"]
    ).round(1)
    mop_df = mop_df.sort_values("composite", ascending=False)

    # ── Bar chart ─────────────────────────────────────────────────
    fig_bar = go.Figure(go.Bar(
        y=mop_df["town"],
        x=mop_df["composite"],
        orientation="h",
        marker_color=mop_df["composite"].apply(
            lambda v: f"rgb({int(255 * v / 10)}, {int(255 * (1 - v / 10))}, 60)"
        ).tolist(),
        text=mop_df["composite"].apply(lambda v: f"{v:.1f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Composite: %{x:.1f}/10<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis=dict(range=[0, 11], title="Supply Pressure Score (0=low, 10=high)"),
        yaxis=dict(autorange="reversed"),
        height=max(500, len(mop_df) * 22),
        title=f"Town Supply Pressure Score (MOP horizon: {horizon} months)",
        margin=dict(l=160, r=80),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Summary table ──────────────────────────────────────────────
    with st.expander("📋 Full score decomposition table"):
        disp = mop_df[["town", "wave_units", "avg_annual_vol", "mop_score", "private_score", "composite"]].copy()
        disp.columns = ["Town", "MOP Wave Units", "Avg Annual Vol", "MOP Score", "Private Score", "Composite Score"]
        disp["Interpretation"] = disp["Composite Score"].apply(
            lambda v: "🔴 High supply pressure" if v >= 7 else ("🟡 Moderate" if v >= 4 else "🟢 Low")
        )
        st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.warning(
        "🟡 **DATA CONFIDENCE: Medium.** MOP unlock dates use `lease_commence_date + 5` as a proxy — "
        "actual key-collection years may differ by 1–3 years. Private pipeline is aggregate by market segment "
        "(CCR/RCR/OCR) — no project or district-level breakdown without URA REALIS. "
        "Use as a directional signal, not a precise forecast."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SERS Watch List
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sers:
    st.subheader("🏢 SERS Watch List (C5)")

    st.error(
        "⚠️ **IMPORTANT DISCLAIMER:** This list is a **pure statistical pattern-match** "
        "based on public resale transaction data. It does NOT reflect any official HDB SERS "
        "planning decisions or information. **HDB does not pre-announce SERS exercises**, and "
        "blocks on this list have NO special status. Past SERS patterns may not repeat. "
        "Treat this as a research starting point only — **not an investment signal.**"
    )

    st.markdown(
        "Blocks are flagged when they match ≥2 of 4 historical SERS profile criteria:\n"
        "1. **Age ≥ 35 years** (flat age at recent transactions)\n"
        "2. **SERS-prone town** (towns with historical SERS exercises)\n"
        "3. **PSM discount ≥ 5%** vs town median (old blocks often trade at a discount)\n"
        "4. **Low velocity** (≤ 4 transactions in the last 2 years)"
    )

    df_resale = load_resale_base()
    sers_df   = compute_sers_candidates(df_resale)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        min_criteria = st.selectbox("Minimum criteria met", [2, 3, 4], index=0, key="sers_min")
    with col_f2:
        town_filter = st.multiselect("Filter by town", sorted(sers_df["town"].unique()), key="sers_towns")

    filtered = sers_df[sers_df["criteria_met"] >= min_criteria]
    if town_filter:
        filtered = filtered[filtered["town"].isin(town_filter)]

    st.metric("Blocks matching criteria", len(filtered))

    disp_cols = ["flag", "town", "block", "street_name", "avg_flat_age",
                 "median_psm", "town_median_psm", "psm_discount_pct", "txn_2yr", "criteria_met"]
    disp_cols = [c for c in disp_cols if c in filtered.columns]
    st.dataframe(
        filtered[disp_cols].rename(columns={
            "flag": "Flag", "town": "Town", "block": "Block",
            "street_name": "Street", "avg_flat_age": "Avg Age (yrs)",
            "median_psm": "Block PSM ($)", "town_median_psm": "Town PSM ($)",
            "psm_discount_pct": "PSM Discount (%)", "txn_2yr": "Txn (2yr)",
            "criteria_met": "Criteria Met",
        }),
        use_container_width=True, hide_index=True,
    )

    # ── Scatter plot ───────────────────────────────────────────────
    if not filtered.empty:
        fig_sc = px.scatter(
            filtered,
            x="avg_flat_age",
            y="psm_discount_pct",
            color="criteria_met",
            symbol="flag",
            hover_data=["town", "block", "street_name", "txn_2yr"],
            color_continuous_scale="OrRd",
            labels={
                "avg_flat_age": "Average Flat Age (years)",
                "psm_discount_pct": "PSM Discount vs Town Median (%)",
                "criteria_met": "Criteria Met",
            },
            title="SERS Pattern: Age vs PSM Discount",
            size_max=12,
        )
        fig_sc.add_hline(y=5, line_dash="dash", line_color="gray",
                         annotation_text="5% discount threshold")
        fig_sc.add_vline(x=35, line_dash="dash", line_color="gray",
                         annotation_text="35yr age threshold")
        fig_sc.update_layout(height=460)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.caption(
        "🟡 **DATA CONFIDENCE: Low/Experimental.** Pattern-match based on historical SERS "
        "characteristics — no validated predictive model. SERS selection criteria are not "
        "publicly disclosed in full detail. A block on this list is NOT more likely to be "
        "selected for SERS than any other block."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Private Pipeline Dashboard (I2)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.subheader("🏗️ URA Private Residential Pipeline (I2)")
    st.markdown(
        "URA developer data: units launched, sold, and unsold by market segment (CCR/RCR/OCR), "
        "quarterly from 2004. Tracks the private market supply pipeline — "
        "excess private inventory can suppress upgrader demand and affect HDB resale prices."
    )

    launched_df = load_ura_launched()
    sold_df     = load_ura_sold()
    unsold_df   = load_ura_unsold()

    if launched_df.empty and sold_df.empty:
        st.warning("URA pipeline data not found. Run `python src/fetch_data.py` to download.")
    else:
        # ── Prepare quarter timestamps ──────────────────────────────
        def _prep(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            df = df.copy()
            df["units"] = pd.to_numeric(df.get("units", 0), errors="coerce").fillna(0)
            q_raw = df["quarter"].astype(str).str.replace("-Q", "Q", regex=False)
            try:
                df["quarter_dt"] = pd.PeriodIndex(q_raw, freq="Q").to_timestamp()
            except Exception:
                df["quarter_dt"] = pd.to_datetime(df["quarter"], errors="coerce")
            return df

        ldf = _prep(launched_df)
        sdf = _prep(sold_df)

        # ── Policy event lines helper ─────────────────────────────
        cooling_dates = [
            pd.to_datetime(d) for d, _ in POLICY_EVENTS
            if "Cooling" in _
        ]

        # ── Chart 1: Launched vs Sold by segment ─────────────────
        st.markdown("#### Launched vs Sold — by market segment")

        seg_sel = st.multiselect(
            "Market segments",
            ["Core Central Region", "Rest of Central Region", "Outside Central Region"],
            default=["Core Central Region", "Outside Central Region"],
            key="pipe_seg",
        )

        fig_ls = go.Figure()
        colors = {"Core Central Region": "#1f77b4",
                  "Rest of Central Region": "#ff7f0e",
                  "Outside Central Region": "#2ca02c"}

        for seg in seg_sel:
            col = colors.get(seg, "#888")
            l_seg = ldf[ldf["market_segment"] == seg].sort_values("quarter_dt")
            s_seg = sdf[sdf["market_segment"] == seg].sort_values("quarter_dt")
            if not l_seg.empty:
                fig_ls.add_trace(go.Bar(x=l_seg["quarter_dt"], y=l_seg["units"],
                                        name=f"Launched – {seg}", marker_color=col, opacity=0.6))
            if not s_seg.empty:
                fig_ls.add_trace(go.Scatter(x=s_seg["quarter_dt"], y=s_seg["units"],
                                            name=f"Sold – {seg}", line=dict(color=col, width=2)))

        for dt in cooling_dates:
            if not ldf.empty and ldf["quarter_dt"].min() <= dt <= ldf["quarter_dt"].max():
                fig_ls.add_vline(x=dt, line_dash="dot", line_color="crimson", opacity=0.5)

        fig_ls.update_layout(
            barmode="group", height=440,
            xaxis_title="Quarter", yaxis_title="Units",
            hovermode="x unified",
        )
        st.plotly_chart(fig_ls, use_container_width=True)

        # ── Chart 2: Unsold inventory ─────────────────────────────
        if not unsold_df.empty:
            st.markdown("#### Active unsold inventory (Uncompleted + Launched + With Pre-Requisites)")
            udf = _prep(unsold_df).copy()

            # Filter to active pipeline
            comp_col   = next((c for c in udf.columns if "complet" in c.lower()), None)
            launch_col = next((c for c in udf.columns if "launch" in c.lower()), None)
            prereq_col = next((c for c in udf.columns if "pre" in c.lower()), None)

            if comp_col and launch_col and prereq_col:
                active = udf[
                    (udf[comp_col] == "Uncompleted") &
                    (udf[launch_col] == "Launched") &
                    (udf[prereq_col] == "With Pre-Requisites")
                ]
            else:
                active = udf

            if not active.empty and "market_segment" in active.columns:
                active_grp = active.groupby(["quarter_dt", "market_segment"])["units"].sum().reset_index()
                active_seg = active_grp[active_grp["market_segment"].isin(seg_sel)]

                # Compute national total and 8-quarter rolling avg for months-of-inventory
                quarterly_sold_total = (
                    sdf[sdf["market_segment"].isin(seg_sel)]
                    .groupby("quarter_dt")["units"].sum().reset_index(name="sold_q")
                )
                unsold_total = (
                    active_grp[active_grp["market_segment"].isin(seg_sel)]
                    .groupby("quarter_dt")["units"].sum().reset_index(name="unsold")
                )
                moi = unsold_total.merge(quarterly_sold_total, on="quarter_dt", how="left")
                moi["months_of_inventory"] = (moi["unsold"] / moi["sold_q"].replace(0, np.nan) * 3).round(1)

                fig_unsold = px.bar(
                    active_seg,
                    x="quarter_dt", y="units", color="market_segment",
                    barmode="stack",
                    labels={"quarter_dt": "Quarter", "units": "Unsold Units",
                            "market_segment": "Segment"},
                    title="Unsold Active Private Residential Inventory by Quarter",
                )
                fig_unsold.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig_unsold, use_container_width=True)

                # ── Months of Inventory ────────────────────────────
                st.markdown("#### Months of Inventory (unsold / quarterly sold rate × 3)")
                fig_moi = go.Figure()
                fig_moi.add_trace(go.Scatter(
                    x=moi["quarter_dt"], y=moi["months_of_inventory"],
                    fill="tozeroy", line=dict(color="#ff7f0e"),
                    name="Months of Inventory",
                ))
                fig_moi.add_hline(y=12, line_dash="dash", line_color="red",
                                   annotation_text="12mo (elevated risk)")
                fig_moi.add_hline(y=6,  line_dash="dash", line_color="green",
                                   annotation_text="6mo (tight supply)")
                fig_moi.update_layout(
                    xaxis_title="Quarter", yaxis_title="Months of Inventory",
                    height=360, hovermode="x unified",
                )
                st.plotly_chart(fig_moi, use_container_width=True)

                # ── Interpretation ─────────────────────────────────
                if not moi.empty:
                    latest_moi = moi.dropna(subset=["months_of_inventory"]).iloc[-1]
                    hist_avg_moi = moi["months_of_inventory"].median()
                    moi_val = latest_moi["months_of_inventory"]
                    if moi_val > 12:
                        interp = (
                            f"🔴 **Elevated inventory ({moi_val:.1f} months, above 12-month threshold).** "
                            "Private oversupply may slow condo price growth and dampen upgrading demand, "
                            "which could support HDB resale as upgraders stay longer."
                        )
                    elif moi_val < 6:
                        interp = (
                            f"🟢 **Tight supply ({moi_val:.1f} months, below 6-month threshold).** "
                            "Strong take-up relative to launch — private market is absorbing supply quickly. "
                            "This may accelerate HDB-to-condo upgrading, softening HDB resale demand at the margin."
                        )
                    else:
                        interp = (
                            f"🟡 **Balanced market ({moi_val:.1f} months, vs historical median {hist_avg_moi:.1f}).** "
                            "Private pipeline absorption is in line with historical norms."
                        )
                    st.info(interp)

    st.caption(
        "🟡 **DATA CONFIDENCE: Medium.** Aggregate by market segment (CCR/RCR/OCR) — "
        "no project or district-level detail. Months-of-inventory uses same-segment sold units "
        "as denominator; actual project-level take-up rates may differ materially. "
        "Source: URA data.gov.sg datasets d_70824d34defde87d88faccc5d5b1c6ea / "
        "d_e1c5b0df62729e69c82716355ef295ba / d_84d05d45049108f0fd2e99b66bd19cfe."
    )
