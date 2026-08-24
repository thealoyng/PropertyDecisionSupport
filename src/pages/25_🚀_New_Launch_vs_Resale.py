"""
Page 25 — New Launch vs Resale Price Decay (I8)
================================================
Tracks the price premium that new launches command over comparable resale
condos in the same district, and how that gap evolves over time since TOP
(Temporary Occupation Permit).

Requires condo_clean.csv (fetch_data.py + combine_clean_condo.py).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from eda_helpers import load_condo_clean, POLICY_EVENTS

st.set_page_config(page_title="New Launch vs Resale", page_icon="🚀", layout="wide")

st.title("🚀 New Launch vs Resale Premium Decay (I8)")
st.caption(
    "Tracks the price premium new-sale caveats command over resales in the same district "
    "and quarter — and how that gap evolves over time. Identifies overpriced new launches "
    "when the premium exceeds historical norms."
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_gap, tab_decay, tab_screen = st.tabs([
    "📊 New Sale vs Resale Gap",
    "📉 Premium Decay Over Time",
    "🔍 Overpriced Launch Screener",
])


@st.cache_data
def _load():
    df = load_condo_clean()
    if df.empty:
        return df
    df = df[
        df["property_type_broad"].isin(["Condo/Apartment", "Executive Condo (EC)"]) &
        df["price_psm"].notna() &
        (df["price_psm"] > 1000)
    ].copy()
    df["contract_date"] = pd.to_datetime(df["contract_date"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — New Sale vs Resale Gap by Quarter + District
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gap:
    st.subheader("New Sale PSM vs Resale PSM — same district, same quarter")

    df = _load()
    if df.empty:
        st.warning("Condo data not found. Run `python src/fetch_data.py` then `python src/combine_clean_condo.py`.")
    else:
        # Quarterly median PSM by sale type
        new_q = (
            df[df["type_of_sale"] == "New Sale"]
            .groupby(["contract_quarter", "district"])["price_psm"].median()
            .reset_index().rename(columns={"price_psm": "new_psm"})
        )
        resale_q = (
            df[df["type_of_sale"] == "Resale"]
            .groupby(["contract_quarter", "district"])["price_psm"].median()
            .reset_index().rename(columns={"price_psm": "resale_psm"})
        )
        gap_df = new_q.merge(resale_q, on=["contract_quarter", "district"], how="inner")
        gap_df["premium_pct"] = ((gap_df["new_psm"] / gap_df["resale_psm"] - 1) * 100).round(1)
        gap_df["premium_psm"] = (gap_df["new_psm"] - gap_df["resale_psm"]).round(0)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            seg_sel = st.multiselect(
                "Market segment filter (via district)",
                ["All districts", "CCR (D1-11)", "RCR (D12-20)", "OCR (D21-28)"],
                default=["All districts"],
                key="ig_seg",
            )
        with col_g2:
            top_n = st.slider("Districts to show", 5, 28, 10, key="ig_topn")

        # Filter by district groups
        if "CCR (D1-11)" in seg_sel and "RCR (D12-20)" not in seg_sel and "OCR (D21-28)" not in seg_sel and "All districts" not in seg_sel:
            gap_df = gap_df[gap_df["district"] <= 11]
        elif "RCR (D12-20)" in seg_sel and "All districts" not in seg_sel:
            gap_df = gap_df[(gap_df["district"] >= 12) & (gap_df["district"] <= 20)]
        elif "OCR (D21-28)" in seg_sel and "All districts" not in seg_sel:
            gap_df = gap_df[gap_df["district"] >= 21]

        # Latest quarter gap by district
        latest_q = gap_df["contract_quarter"].max()
        latest_gap = gap_df[gap_df["contract_quarter"] == latest_q].copy()
        latest_gap = latest_gap.sort_values("premium_pct", ascending=False).head(top_n)

        if not latest_gap.empty:
            fig_bar = px.bar(
                latest_gap,
                x="district", y="premium_pct",
                color="premium_pct",
                color_continuous_scale="RdYlGn_r",
                text=latest_gap["premium_pct"].apply(lambda x: f"{x:+.1f}%"),
                labels={"district": "District", "premium_pct": "New Launch Premium (%)"},
                title=f"New Sale vs Resale PSM Premium by District — {latest_q}",
            )
            fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_bar.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Time series: aggregate new sale premium over time
        nat_q = (
            gap_df.groupby("contract_quarter")
            .apply(lambda x: pd.Series({
                "median_premium_pct": x["premium_pct"].median(),
                "n_districts": len(x),
            }))
            .reset_index()
        )

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=nat_q["contract_quarter"], y=nat_q["median_premium_pct"],
            fill="tozeroy", name="Median new launch premium",
            line=dict(color="#ff7f0e", width=2),
        ))
        fig_ts.add_hline(y=0, line_dash="dot", line_color="gray")
        hist_med = nat_q["median_premium_pct"].median()
        fig_ts.add_hline(y=hist_med, line_dash="dash", line_color="steelblue",
                         annotation_text=f"Historical median: {hist_med:.1f}%")
        fig_ts.update_layout(
            height=360, title="National median new launch premium over time",
            xaxis_title="Quarter", yaxis_title="New Sale Premium vs Resale (%)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        latest_prem = nat_q.iloc[-1]["median_premium_pct"]
        if latest_prem > hist_med * 1.5:
            st.error(
                f"🔴 **Elevated new launch premium ({latest_prem:.1f}% vs historical median {hist_med:.1f}%).** "
                "New launches are pricing significantly above comparable resales — "
                "this has historically preceded periods of price correction or slower take-up."
            )
        elif latest_prem < 0:
            st.success(
                f"🟢 **New launches are trading at a discount to resale ({latest_prem:.1f}%).** "
                "Rare — indicates either distressed project pricing or data sparsity in new launches this quarter."
            )
        else:
            st.info(
                f"🟡 **Premium in normal range ({latest_prem:.1f}% vs historical median {hist_med:.1f}%).** "
                "New launches carry a typical premium for brand-new units and developer marketing costs."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Premium Decay Over Time Since Transaction Date
# ═══════════════════════════════════════════════════════════════════════════════
with tab_decay:
    st.subheader("📉 New-Sale Premium Decay by Years Since Transaction")
    st.markdown(
        "For each district, compute the average PSM of new-sale transactions vs resale transactions "
        "by 'vintage' (year of transaction). The decay curve shows how the new-launch premium "
        "compresses as projects age and become part of the resale market."
    )

    df = _load()
    if df.empty:
        st.warning("No condo data available.")
    else:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            dist_decay = st.multiselect(
                "Districts (empty = all)",
                list(range(1, 29)),
                default=[9, 10, 15, 19],
                key="decay_dist",
            )
        with col_d2:
            min_txn = st.slider("Min transactions per group", 5, 50, 10, key="decay_min")

        df_filt = df[df["district"].isin(dist_decay)] if dist_decay else df.copy()

        # Group by year of transaction
        df_filt["txn_year"] = df_filt["contract_date"].dt.year

        decay = (
            df_filt[df_filt["type_of_sale"].isin(["New Sale", "Resale"])]
            .groupby(["txn_year", "type_of_sale"])
            .agg(median_psm=("price_psm", "median"), n=("price_psm", "count"))
            .reset_index()
        )
        decay = decay[decay["n"] >= min_txn]

        if not decay.empty:
            new_yr  = decay[decay["type_of_sale"] == "New Sale"][["txn_year", "median_psm"]].rename(columns={"median_psm": "new_psm"})
            res_yr  = decay[decay["type_of_sale"] == "Resale"][["txn_year", "median_psm"]].rename(columns={"median_psm": "resale_psm"})
            decay_m = new_yr.merge(res_yr, on="txn_year", how="inner")
            decay_m["premium_pct"] = ((decay_m["new_psm"] / decay_m["resale_psm"] - 1) * 100).round(1)

            fig_decay = go.Figure()
            fig_decay.add_trace(go.Bar(
                x=decay_m["txn_year"], y=decay_m["new_psm"],
                name="New Sale PSM", marker_color="#ff7f0e", opacity=0.7,
            ))
            fig_decay.add_trace(go.Bar(
                x=decay_m["txn_year"], y=decay_m["resale_psm"],
                name="Resale PSM", marker_color="#1f77b4", opacity=0.7,
            ))
            fig_decay.add_trace(go.Scatter(
                x=decay_m["txn_year"], y=decay_m["premium_pct"],
                name="Premium %", yaxis="y2",
                line=dict(color="crimson", width=2, dash="dot"),
                mode="lines+markers",
            ))
            fig_decay.update_layout(
                barmode="group",
                yaxis=dict(title="Median PSM ($/sqm)"),
                yaxis2=dict(title="New Launch Premium (%)", overlaying="y", side="right",
                            range=[-5, max(50, decay_m["premium_pct"].max() * 1.2)]),
                height=440,
                title="New Sale vs Resale PSM by transaction year (selected districts)",
                hovermode="x unified",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_decay, use_container_width=True)
        else:
            st.info("Not enough transactions for the selected district filter.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Overpriced Launch Screener
# ═══════════════════════════════════════════════════════════════════════════════
with tab_screen:
    st.subheader("🔍 Overpriced New Launch Screener")
    st.markdown(
        "Flag districts where the current new-sale PSM premium is **unusually high** "
        "compared to that district's own historical average — "
        "a potential signal of developer overpricing relative to the local resale market."
    )
    st.error(
        "⚠️ **DISCLAIMER:** This is a statistical pattern screen, not a prediction. "
        "New launches may command higher premiums for legitimate reasons (better specs, "
        "fresh fittings, project branding, future MRT proximity). "
        "Use as a starting point for further due diligence — not as a buy/sell signal."
    )

    df = _load()
    if not df.empty:
        new_q = (
            df[df["type_of_sale"] == "New Sale"]
            .groupby(["contract_quarter", "district"])["price_psm"].median()
            .reset_index().rename(columns={"price_psm": "new_psm"})
        )
        resale_q = (
            df[df["type_of_sale"] == "Resale"]
            .groupby(["contract_quarter", "district"])["price_psm"].median()
            .reset_index().rename(columns={"price_psm": "resale_psm"})
        )
        gap_all = new_q.merge(resale_q, on=["contract_quarter", "district"], how="inner")
        gap_all["premium_pct"] = ((gap_all["new_psm"] / gap_all["resale_psm"] - 1) * 100)

        # Per-district historical stats
        hist_stats = (
            gap_all.groupby("district")["premium_pct"]
            .agg(hist_mean="mean", hist_std="std", hist_median="median", n_qtrs="count")
            .reset_index()
        )

        latest_q = gap_all["contract_quarter"].max()
        latest_gap_all = gap_all[gap_all["contract_quarter"] == latest_q].merge(hist_stats, on="district")
        latest_gap_all["z_score"] = (
            (latest_gap_all["premium_pct"] - latest_gap_all["hist_mean"]) / latest_gap_all["hist_std"]
        ).round(2)
        latest_gap_all["flag"] = latest_gap_all["z_score"].apply(
            lambda z: "🔴 High" if z > 1.5 else ("🟡 Watch" if z > 0.75 else "🟢 Normal")
        )
        latest_gap_all = latest_gap_all.sort_values("z_score", ascending=False)

        threshold_z = st.slider("Flag threshold (z-score)", 0.5, 2.5, 1.0, 0.25, key="screen_z")
        flagged = latest_gap_all[latest_gap_all["z_score"] >= threshold_z]
        st.metric("Districts above threshold", len(flagged))

        fig_screen = px.bar(
            latest_gap_all.head(20),
            x="district", y="z_score",
            color="flag",
            color_discrete_map={"🔴 High": "crimson", "🟡 Watch": "orange", "🟢 Normal": "steelblue"},
            text=latest_gap_all.head(20)["z_score"].apply(lambda x: f"{x:.1f}σ"),
            labels={"district": "District", "z_score": "Premium Z-Score vs District History"},
            title=f"New Launch Premium Anomaly Score by District — {latest_q}",
        )
        fig_screen.add_hline(y=threshold_z, line_dash="dash", line_color="red",
                              annotation_text=f"Flag threshold ({threshold_z}σ)")
        fig_screen.update_layout(height=420, showlegend=True)
        st.plotly_chart(fig_screen, use_container_width=True)

        st.dataframe(
            flagged[["district", "premium_pct", "hist_median", "hist_std", "z_score", "flag", "n_qtrs"]]
            .rename(columns={
                "district": "District", "premium_pct": "Current Premium (%)",
                "hist_median": "Historical Median (%)", "hist_std": "Hist StdDev",
                "z_score": "Z-Score", "flag": "Flag", "n_qtrs": "Qtrs of Data",
            })
            .round(1),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "🟡 **DATA CONFIDENCE: Medium.** Window limited to Aug 2021–present (URA API provides ~3 years). "
        "Z-score meaningful only for districts with ≥4 quarters of both new and resale transactions. "
        "Source: URA caveat-level data via eservice.ura.gov.sg API."
    )
