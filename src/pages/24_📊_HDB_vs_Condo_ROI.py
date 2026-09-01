"""
Page 24 — HDB vs Condo ROI (F3)
=================================
Full return comparison: capital appreciation + rental yield − transaction costs,
matched by town/district and holding period.

Requires condo_clean.csv (run fetch_data.py + combine_clean_condo.py).
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
    load_clean, load_condo_clean, fmt_price, POLICY_EVENTS
)

st.set_page_config(page_title="HDB vs Condo ROI", page_icon="📊", layout="wide")

st.title("📊 HDB vs Condo ROI Comparison (F3)")
st.caption(
    "Full total-return comparison: capital gain + net rental yield − transaction costs, "
    "matched by geography and holding period. Uses individual transaction caveats — "
    "not index-rebased estimates."
)

# ── Constants ─────────────────────────────────────────────────────────────────
# Singapore transaction cost assumptions
STAMP_DUTY_BSD_RATES = [(180_000, 0.01), (180_000, 0.02), (640_000, 0.03), (float("inf"), 0.04)]
ABSD_SC_FIRST   = 0.00   # Singapore citizen, 1st property
ABSD_SC_SECOND  = 0.20   # SC 2nd property (post Dec 2021)
ABSD_PR_FIRST   = 0.05   # PR 1st property
AGENT_COMM      = 0.01   # seller side ~1%
LEGAL_FLAT      = 3_500  # approximate legal fees
RENO_COST_HDB   = 50_000
RENO_COST_CONDO = 80_000

# District → HDB town mapping
DISTRICT_TO_TOWN = {
    1: "CENTRAL AREA", 2: "BUKIT MERAH", 3: "QUEENSTOWN",
    4: "BUKIT MERAH", 5: "CLEMENTI", 6: "CENTRAL AREA",
    7: "CENTRAL AREA", 8: "CENTRAL AREA", 9: "CENTRAL AREA",
    10: "BUKIT TIMAH", 11: "BISHAN", 12: "TOA PAYOH",
    13: "SERANGOON", 14: "GEYLANG", 15: "MARINE PARADE",
    16: "BEDOK", 17: "PASIR RIS", 18: "TAMPINES",
    19: "HOUGANG", 20: "BISHAN", 21: "CLEMENTI",
    22: "JURONG WEST", 23: "BUKIT PANJANG", 24: "CHOA CHU KANG",
    25: "WOODLANDS", 26: "BISHAN", 27: "YISHUN", 28: "SENGKANG",
}
TOWN_TO_DISTRICTS = {}
for d, t in DISTRICT_TO_TOWN.items():
    TOWN_TO_DISTRICTS.setdefault(t, []).append(d)


def calc_bsd(price: float) -> float:
    """Compute Buyer's Stamp Duty for a given purchase price."""
    duty = 0.0
    remaining = price
    for band, rate in STAMP_DUTY_BSD_RATES:
        taxable = min(remaining, band)
        duty += taxable * rate
        remaining -= taxable
        if remaining <= 0:
            break
    return duty


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def _load_hdb():
    df = load_clean()
    df["month"] = pd.to_datetime(df["month"])
    return df


@st.cache_data
def _load_condo():
    return load_condo_clean()


@st.cache_data
def _load_hdb_rental():
    try:
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "rental",
            "hdb_median_rent_by_town.csv"
        )
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_roi, tab_hold, tab_upgrade = st.tabs([
    "📊 ROI Comparison",
    "⏳ Hold Period Analysis",
    "📐 Cost Breakdown",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROI Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_roi:
    st.subheader("Capital Return Comparison by Town (same geography, same window)")
    st.info(
        "Computes median PSM at **buy date** vs **sell date** to estimate capital appreciation, "
        "then adds a gross rental yield estimate and subtracts transaction costs. "
        "Methodology note: uses same-town/district median, not individual unit tracking."
    )

    hdb = _load_hdb()
    condo = _load_condo()

    if condo.empty:
        st.warning("Condo data not found. Run `python src/fetch_data.py` then `python src/combine_clean_condo.py`.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            hold_yrs = st.slider("Holding period (years)", 1, 10, 5, key="roi_hold")
            hdb_types_sel = st.multiselect(
                "HDB flat types", ["3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"],
                default=["4 ROOM", "5 ROOM"], key="roi_hdb_ft"
            )
        with col2:
            absd_profile = st.selectbox(
                "Buyer profile (ABSD)",
                ["SC 1st property (0%)", "SC 2nd property (20%)", "PR 1st property (5%)"],
                key="roi_absd"
            )
            include_reno = st.checkbox("Include renovation cost", value=True, key="roi_reno")

        absd_rate = {"SC 1st property (0%)": 0.0, "SC 2nd property (20%)": 0.20, "PR 1st property (5%)": 0.05}[absd_profile]

        # Compute HDB per-town capital return over hold_yrs
        max_dt   = hdb["month"].max()
        buy_end  = max_dt - pd.DateOffset(years=hold_yrs)
        buy_win  = (buy_end - pd.DateOffset(years=1), buy_end)
        sell_win = (buy_end, max_dt)

        hdb_ft = hdb[hdb["flat_type"].isin(hdb_types_sel)] if hdb_types_sel else hdb

        hdb_buy  = hdb_ft[(hdb_ft["month"] >= buy_win[0]) & (hdb_ft["month"] <= buy_win[1])].groupby("town")["price_per_sqm"].median().rename("buy_psm")
        hdb_sell = hdb_ft[(hdb_ft["month"] >= sell_win[0]) & (hdb_ft["month"] <= sell_win[1])].groupby("town")["price_per_sqm"].median().rename("sell_psm")
        hdb_price_buy  = hdb_ft[(hdb_ft["month"] >= buy_win[0]) & (hdb_ft["month"] <= buy_win[1])].groupby("town")["resale_price"].median().rename("buy_price")

        roi_hdb = pd.concat([hdb_buy, hdb_sell, hdb_price_buy], axis=1).dropna()
        roi_hdb["capital_gain_pct"] = (roi_hdb["sell_psm"] / roi_hdb["buy_psm"] - 1) * 100

        # Approximate HDB rental yield (from median rent data or fixed proxy)
        hdb_rent = _load_hdb_rental()
        if not hdb_rent.empty and "town" in hdb_rent.columns:
            q_str = f"{buy_end.year}-Q{(buy_end.month - 1) // 3 + 1}"
            recent_rent = hdb_rent[hdb_rent["quarter"].astype(str) >= q_str].copy()
            # Convert Arrow dtype to float to avoid median() error
            recent_rent["median_rent"] = pd.to_numeric(recent_rent["median_rent"], errors="coerce")
            town_rent = recent_rent.groupby("town")["median_rent"].median().rename("annual_rent_x12") * 12
            roi_hdb = roi_hdb.join(town_rent, how="left")
            roi_hdb["gross_yield_pct"] = (roi_hdb["annual_rent_x12"] / roi_hdb["buy_price"] * 100).clip(0, 10)
        else:
            roi_hdb["gross_yield_pct"] = 3.5  # proxy: typical HDB gross yield

        # Transaction costs
        roi_hdb["bsd"]        = roi_hdb["buy_price"].apply(calc_bsd)
        roi_hdb["absd"]       = roi_hdb["buy_price"] * absd_rate
        roi_hdb["agent_sell"] = roi_hdb["buy_price"] * (roi_hdb["sell_psm"] / roi_hdb["buy_psm"]) * AGENT_COMM
        roi_hdb["reno"]       = RENO_COST_HDB if include_reno else 0
        roi_hdb["total_cost_pct"] = (
            (roi_hdb["bsd"] + roi_hdb["absd"] + roi_hdb["agent_sell"] + roi_hdb["reno"] + LEGAL_FLAT) /
            roi_hdb["buy_price"] * 100
        )
        roi_hdb["total_return_pct"] = (
            roi_hdb["capital_gain_pct"] +
            roi_hdb["gross_yield_pct"] * hold_yrs -
            roi_hdb["total_cost_pct"]
        ).round(1)
        roi_hdb["annualised_pct"] = ((1 + roi_hdb["total_return_pct"] / 100) ** (1 / hold_yrs) - 1) * 100

        # Compute condo per-district capital return
        condo_strata = condo[
            (condo["property_type_broad"].isin(["Condo/Apartment", "Executive Condo (EC)"])) &
            condo["price_psm"].notna()
        ].copy()

        max_condo_dt = condo_strata["contract_date"].max()
        condo_buy_end  = max_condo_dt - pd.DateOffset(years=hold_yrs)
        c_buy_win  = (condo_buy_end - pd.DateOffset(years=1), condo_buy_end)
        c_sell_win = (condo_buy_end, max_condo_dt)

        c_buy  = condo_strata[(condo_strata["contract_date"] >= c_buy_win[0]) & (condo_strata["contract_date"] <= c_buy_win[1])].groupby("district")["price_psm"].median().rename("buy_psm")
        c_sell = condo_strata[(condo_strata["contract_date"] >= c_sell_win[0]) & (condo_strata["contract_date"] <= c_sell_win[1])].groupby("district")["price_psm"].median().rename("sell_psm")
        c_price_buy = condo_strata[(condo_strata["contract_date"] >= c_buy_win[0]) & (condo_strata["contract_date"] <= c_buy_win[1])].groupby("district")["price"].median().rename("buy_price")

        roi_condo = pd.concat([c_buy, c_sell, c_price_buy], axis=1).dropna()
        roi_condo["capital_gain_pct"] = (roi_condo["sell_psm"] / roi_condo["buy_psm"] - 1) * 100
        roi_condo["gross_yield_pct"]  = 3.2  # typical condo gross yield proxy
        roi_condo["bsd"]        = roi_condo["buy_price"].apply(calc_bsd)
        roi_condo["absd"]       = roi_condo["buy_price"] * absd_rate
        roi_condo["agent_sell"] = roi_condo["buy_price"] * (roi_condo["sell_psm"] / roi_condo["buy_psm"]) * AGENT_COMM
        roi_condo["reno"]       = RENO_COST_CONDO if include_reno else 0
        roi_condo["total_cost_pct"] = (
            (roi_condo["bsd"] + roi_condo["absd"] + roi_condo["agent_sell"] + roi_condo["reno"] + LEGAL_FLAT) /
            roi_condo["buy_price"] * 100
        )
        roi_condo["total_return_pct"] = (
            roi_condo["capital_gain_pct"] +
            roi_condo["gross_yield_pct"] * hold_yrs -
            roi_condo["total_cost_pct"]
        ).round(1)
        roi_condo["annualised_pct"] = ((1 + roi_condo["total_return_pct"] / 100) ** (1 / hold_yrs) - 1) * 100
        roi_condo = roi_condo.reset_index()
        roi_condo["hdb_town"] = roi_condo["district"].map(DISTRICT_TO_TOWN)

        # Merge on geography
        roi_hdb_r   = roi_hdb.reset_index().rename(columns={"town": "hdb_town"})
        roi_condo_t = roi_condo.groupby("hdb_town")["total_return_pct"].mean().reset_index().rename(columns={"total_return_pct": "condo_total_return"})
        roi_hdb_r   = roi_hdb_r.rename(columns={"total_return_pct": "hdb_total_return"})
        merged = roi_hdb_r.merge(roi_condo_t, on="hdb_town", how="inner")
        
        if merged.empty:
            st.error(
                f"⚠️ **No data available for {hold_yrs}-year holding period.**\n\n"
                f"Private condo data only covers **Aug 2021–2026** (~5 years). "
                f"For holding periods >{max_condo_years:.0f} years, there is insufficient data to compute buy/sell pairs.\n\n"
                f"**Try:**\n"
                f"- Reduce holding period to ≤{max_condo_years:.0f} years\n"
                f"- Use **Tab 2** to see HDB-only long-term returns"
            )
            st.stop()
        
        merged["winner"] = merged.apply(
            lambda r: "HDB" if r["hdb_total_return"] > r["condo_total_return"] else "Condo", axis=1
        )

        # ── Comparison bar chart ────────────────────────────────────────────────
        bar_df = merged.melt(
            id_vars="hdb_town",
            value_vars=["hdb_total_return", "condo_total_return"],
            var_name="type", value_name="total_return_pct",
        )
        bar_df["type"] = bar_df["type"].map({"hdb_total_return": "HDB Resale", "condo_total_return": "Private Condo"})
        bar_df = bar_df.sort_values("total_return_pct", ascending=False)

        fig_bar = px.bar(
            bar_df, x="hdb_town", y="total_return_pct", color="type",
            barmode="group",
            labels={"hdb_town": "Town", "total_return_pct": f"Total Return over {hold_yrs}yr (%)", "type": ""},
            title=f"Total Return (capital + yield - costs) over {hold_yrs} years",
            color_discrete_map={"HDB Resale": "#1f77b4", "Private Condo": "#ff7f0e"},
        )
        fig_bar.add_hline(y=0, line_color="gray", line_dash="dash")
        fig_bar.update_layout(height=460, xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Scatter: HDB vs Condo total return ─────────────────────────────────
        fig_sc = px.scatter(
            merged,
            x="hdb_total_return", y="condo_total_return",
            text="hdb_town",
            color="winner",
            color_discrete_map={"HDB": "#1f77b4", "Condo": "#ff7f0e"},
            labels={
                "hdb_total_return": f"HDB Total Return {hold_yrs}yr (%)",
                "condo_total_return": f"Condo Total Return {hold_yrs}yr (%)",
                "winner": "Winner",
            },
            title=f"HDB vs Condo: who wins? ({hold_yrs}yr hold, {absd_profile})",
        )
        returns_flat = merged[["hdb_total_return", "condo_total_return"]].values.flatten()
        max_ret = max(abs(returns_flat)) if len(returns_flat) > 0 else 100
        fig_sc.add_shape(type="line", x0=-max_ret, y0=-max_ret, x1=max_ret, y1=max_ret,
                         line=dict(dash="dot", color="gray"))
        fig_sc.update_traces(textposition="top center", textfont_size=9)
        fig_sc.update_layout(height=460)
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Above the diagonal line = Condo outperforms. Below = HDB outperforms.")

        # ── Summary table ───────────────────────────────────────────────────────
        with st.expander("📋 Full ROI table"):
            tbl = merged[["hdb_town", "hdb_total_return", "condo_total_return", "winner"]].copy()
            tbl.columns = ["Town", f"HDB Return {hold_yrs}yr (%)", f"Condo Return {hold_yrs}yr (%)", "Winner"]
            st.dataframe(tbl.sort_values(f"HDB Return {hold_yrs}yr (%)", ascending=False).reset_index(drop=True),
                         use_container_width=True, hide_index=True)

    st.warning(
        "🟡 **DATA CONFIDENCE: Medium.** Capital gain based on median PSM before/after window, "
        "not individual unit tracking. Rental yield proxied (HDB: from median rent data; "
        "Condo: 3.2% fixed). Transaction costs are standard estimates — actual costs vary. "
        "ABSD, stamp duty, and agent commissions affect net return significantly."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Hold Period Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hold:
    st.subheader("⏳ Capital Appreciation by Hold Period")
    st.markdown(
        "How much has median PSM appreciated for different holding periods? "
        "Select a buy quarter and see how HDB and condo prices evolved."
    )

    hdb = _load_hdb()
    condo = _load_condo()

    if condo.empty:
        st.warning("Condo data not available.")
    else:
        hdb["quarter"] = hdb["month"].dt.to_period("Q").astype(str)
        condo_strata = condo[
            condo["property_type_broad"].isin(["Condo/Apartment", "Executive Condo (EC)"]) &
            condo["price_psm"].notna()
        ].copy()

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            hdb_town_sel = st.selectbox("HDB Town", sorted(hdb["town"].unique()), key="hp_town")
        with col_h2:
            district_sel = st.selectbox(
                "Condo District",
                sorted(TOWN_TO_DISTRICTS.get(hdb_town_sel, list(range(1, 29)))),
                key="hp_dist",
            )

        # Quarterly median PSM
        hdb_q = hdb[hdb["town"] == hdb_town_sel].groupby("quarter")["price_per_sqm"].median().reset_index()
        condo_q = condo_strata[condo_strata["district"] == district_sel].groupby("contract_quarter")["price_psm"].median().reset_index()
        condo_q.columns = ["quarter", "price_per_sqm"]

        # Rebase to first common quarter
        first_q = max(hdb_q["quarter"].min(), condo_q["quarter"].min())
        hdb_base   = hdb_q[hdb_q["quarter"] >= first_q].copy()
        condo_base = condo_q[condo_q["quarter"] >= first_q].copy()

        if not hdb_base.empty and not condo_base.empty:
            hdb_base_val   = hdb_base.iloc[0]["price_per_sqm"]
            condo_base_val = condo_base.iloc[0]["price_per_sqm"]
            hdb_base["indexed"]   = hdb_base["price_per_sqm"]   / hdb_base_val * 100
            condo_base["indexed"] = condo_base["price_per_sqm"] / condo_base_val * 100

            fig_hold = go.Figure()
            fig_hold.add_trace(go.Scatter(
                x=hdb_base["quarter"], y=hdb_base["indexed"],
                name=f"HDB {hdb_town_sel}", line=dict(color="#1f77b4", width=2),
                hovertemplate="Q: %{x}<br>Index: %{y:.1f}<extra></extra>",
            ))
            fig_hold.add_trace(go.Scatter(
                x=condo_base["quarter"], y=condo_base["indexed"],
                name=f"Condo D{district_sel}", line=dict(color="#ff7f0e", width=2),
                hovertemplate="Q: %{x}<br>Index: %{y:.1f}<extra></extra>",
            ))
            fig_hold.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="Base")
            fig_hold.update_layout(
                height=420, title=f"PSM index (base={first_q}=100): {hdb_town_sel} HDB vs District {district_sel} Condo",
                yaxis_title="PSM Index (base=100)", xaxis_title="Quarter",
                hovermode="x unified",
            )
            st.plotly_chart(fig_hold, use_container_width=True)

            # Latest absolute PSMs
            c1, c2 = st.columns(2)
            c1.metric(f"HDB {hdb_town_sel} latest PSM", f"${hdb_base.iloc[-1]['price_per_sqm']:,.0f}/sqm")
            c2.metric(f"Condo D{district_sel} latest PSM", f"${condo_base.iloc[-1]['price_per_sqm']:,.0f}/sqm")
        else:
            st.info("Insufficient data for the selected combination.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Cost Breakdown
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upgrade:
    st.subheader("📐 Transaction Cost Breakdown")
    st.markdown("Understand the full cost structure before comparing raw returns.")

    col_cb1, col_cb2 = st.columns(2)
    with col_cb1:
        purchase_price = st.number_input("Purchase Price ($)", 200_000, 5_000_000, 1_200_000, 50_000, key="cb_price")
        absd_p = st.selectbox(
            "ABSD profile",
            ["SC 1st property (0%)", "SC 2nd property (20%)", "PR 1st property (5%)", "Foreigner (60%)"],
            key="cb_absd",
        )
        prop_class = st.radio("Property class", ["HDB Resale", "Private Condo"], horizontal=True, key="cb_class")
    with col_cb2:
        hold_p = st.slider("Holding period (years)", 1, 15, 5, key="cb_hold")
        gross_yield_p = st.slider("Gross rental yield (%/yr)", 1.0, 8.0, 3.5, 0.1, key="cb_yield")

    absd_rates = {"SC 1st property (0%)": 0.0, "SC 2nd property (20%)": 0.20,
                  "PR 1st property (5%)": 0.05, "Foreigner (60%)": 0.60}
    absd_v = absd_rates[absd_p]

    bsd_v   = calc_bsd(purchase_price)
    absd_v2 = purchase_price * absd_v
    reno_v  = RENO_COST_HDB if prop_class == "HDB Resale" else RENO_COST_CONDO
    agent_v = purchase_price * AGENT_COMM
    total_in  = bsd_v + absd_v2 + reno_v + LEGAL_FLAT
    total_out = agent_v + LEGAL_FLAT
    total_costs = total_in + total_out
    cost_pct = total_costs / purchase_price * 100
    gross_rent_total = purchase_price * (gross_yield_p / 100) * hold_p
    # Net yield after costs (simple)
    net_return_costs = gross_rent_total - total_costs
    breakeven_gain_pct = total_costs / purchase_price * 100

    st.markdown("#### Entry + exit cost breakdown")
    cost_rows = [
        ("BSD (Buyer's Stamp Duty)", bsd_v),
        (f"ABSD ({absd_p})", absd_v2),
        ("Renovation (est.)", reno_v),
        ("Legal fees (entry)", LEGAL_FLAT),
        ("Agent commission (exit ~1%)", agent_v),
        ("Legal fees (exit)", LEGAL_FLAT),
    ]
    cost_df = pd.DataFrame(cost_rows, columns=["Item", "Amount ($)"])
    cost_df["% of Purchase Price"] = (cost_df["Amount ($)"] / purchase_price * 100).round(2)

    fig_cost = px.bar(
        cost_df, x="Item", y="Amount ($)", color="Amount ($)",
        color_continuous_scale="Reds",
        title=f"Transaction costs on ${purchase_price:,.0f} purchase ({prop_class})",
    )
    fig_cost.update_layout(height=360, coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_cost, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transaction costs", f"${total_costs:,.0f}")
    c2.metric("% of purchase price", f"{cost_pct:.1f}%")
    c3.metric("Break-even capital gain needed", f"{breakeven_gain_pct:.1f}%")
    c4.metric(f"Gross rental over {hold_p}yr", f"${gross_rent_total:,.0f}")

    st.caption(
        "BSD rates: 1% on first $180k, 2% on next $180k, 3% on next $640k, 4% on remainder. "
        "ABSD rates as of Dec 2021 revision for SC/PR; 60% for foreigners from 2023. "
        "Renovation and agent fee are estimates — actual figures vary."
    )
