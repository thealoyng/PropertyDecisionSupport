"""
Page 13 — MOP Unlock Calendar
==============================
Smart Money MOP Unlock Calendar — a forward-looking supply signal using
Minimum Occupation Period (MOP) data derived from lease_commence_date.

MOP = 5 years from key collection (approximated by lease_commence_date).
A wave of flats unlocking from MOP creates a predictable supply shock
that historically softens nearby prices 6-12 months later.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from eda_helpers import load_clean, fmt_price, load_condo_clean

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOP Calendar",
    page_icon="📅",
    layout="wide",
)

TODAY = pd.Timestamp.now()
CURRENT_YEAR = TODAY.year

st.title("📅 Smart Money MOP Unlock Calendar")
st.caption(
    "Forward-looking resale supply signal — track Minimum Occupation Period (MOP) unlock waves "
    "to anticipate new supply before it appears in transaction data."
)

st.info(
    "⚠️ **Data caveat:** `lease_commence_date` in the resale dataset is the year the HDB lease "
    "started — this approximates BTO key collection year but may differ by 1–3 years. "
    "MOP dates shown are **estimates**. Always verify against HDB's official BTO completion records."
)

# ── load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def get_df():
    """Load cleaned resale data and add MOP-related columns."""
    df = load_clean()
    df = df.dropna(subset=["lease_commence_date", "town"])
    df["lease_commence_date"] = df["lease_commence_date"].astype(int)
    df["mop_unlock_year"] = df["lease_commence_date"] + 5
    # year column already exists but ensure it's int
    df["year"] = df["year"].astype(int)
    return df


df = get_df()
ALL_TOWNS = sorted(df["town"].unique())

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 MOP Unlock Timeline",
    "🌊 Supply Wave Impact",
    "🏘️ Town-Level Supply Radar",
    "📊 Lease Vintage Analysis",
    "🔗 HDB→Private Bridge"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — MOP Unlock Timeline
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📅 Estimated MOP Unlock Timeline")
    st.markdown(
        "Using **`lease_commence_date + 5`** as the estimated MOP unlock year, this chart "
        "shows the predicted wave of new resale supply entering the market by town. "
        "Data is sourced from Jan 2017 onwards where coverage is most complete."
    )

    @st.cache_data
    def compute_mop_timeline(_df):
        """Group transactions by (town, mop_unlock_year) to estimate supply waves."""
        df17 = _df[_df["year"] >= 2017].copy()
        grp = (
            df17.groupby(["town", "lease_commence_date", "mop_unlock_year"])
            .size()
            .reset_index(name="estimated_units")
        )
        return grp

    mop_timeline = compute_mop_timeline(df)

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        sel_towns_t1 = st.multiselect(
            "Filter by town (leave blank for all)",
            options=ALL_TOWNS,
            default=[],
            key="t1_towns",
            placeholder="All towns",
        )
    with col_ctrl2:
        yr_range = st.slider(
            "MOP unlock year range",
            min_value=2018,
            max_value=2032,
            value=(2022, 2030),
            key="t1_yr",
        )

    tl_filtered = mop_timeline[
        (mop_timeline["mop_unlock_year"] >= yr_range[0]) &
        (mop_timeline["mop_unlock_year"] <= yr_range[1])
    ].copy()

    if sel_towns_t1:
        tl_filtered = tl_filtered[tl_filtered["town"].isin(sel_towns_t1)]

    tl_agg = (
        tl_filtered
        .groupby(["mop_unlock_year", "town"])["estimated_units"]
        .sum()
        .reset_index()
    )

    if tl_agg.empty:
        st.warning("No data for the selected filters.")
    else:
        fig1 = px.bar(
            tl_agg,
            x="mop_unlock_year",
            y="estimated_units",
            color="town",
            title="Estimated MOP Unlock Units by Year and Town",
            labels={
                "mop_unlock_year": "Estimated MOP Unlock Year",
                "estimated_units": "Estimated Units (transaction-count proxy)",
                "town": "Town",
            },
            barmode="stack",
        )
        fig1.add_vline(
            x=CURRENT_YEAR,
            line_dash="dash",
            line_color="crimson",
            annotation_text=f"Now ({CURRENT_YEAR})",
            annotation_position="top right",
            annotation_font_color="crimson",
        )
        fig1.update_layout(
            xaxis=dict(tickmode="linear", dtick=1, title="MOP Unlock Year"),
            yaxis_title="Estimated Units",
            height=540,
            legend=dict(orientation="v", x=1.01, y=1, title="Town"),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Key insight callout
        totals_by_yr = tl_agg.groupby("mop_unlock_year")["estimated_units"].sum()
        peak_yr = int(totals_by_yr.idxmax())
        peak_units = int(totals_by_yr.max())
        near_term = int(totals_by_yr.get(CURRENT_YEAR, 0) + totals_by_yr.get(CURRENT_YEAR + 1, 0))

        st.success(
            f"💡 **Key Insight:** The largest estimated MOP wave in the selected view is "
            f"**{peak_yr}** (~{peak_units:,} units). "
            f"Towns with large MOP waves in **2025–2027** may see increased resale supply, "
            f"which could moderate price growth in those areas. "
            f"Estimated units unlocking in {CURRENT_YEAR}–{CURRENT_YEAR+1}: **{near_term:,}**."
        )

    with st.expander("📋 Show MOP timeline data table"):
        tl_table = (
            tl_agg
            .sort_values(["mop_unlock_year", "estimated_units"], ascending=[True, False])
            .rename(columns={
                "mop_unlock_year": "MOP Unlock Year",
                "town": "Town",
                "estimated_units": "Estimated Units",
            })
        )
        st.dataframe(tl_table, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Supply Wave Impact Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌊 Historical Supply Wave Impact Analysis")
    st.markdown(
        "Did past MOP waves actually change resale transaction volumes and prices? "
        "This section identifies major historical MOP wave years for each town and "
        "compares key metrics in the year **before**, **during**, and **after** each wave."
    )
    st.warning(
        "📝 **Note:** This is descriptive, not causal — MOP wave correlation with price changes "
        "may reflect other concurrent factors (cooling measures, interest rates, economic cycles). "
        "Use this as one signal among many."
    )

    @st.cache_data
    def compute_year_town_stats(_df):
        """Pre-aggregate annual stats per town for fast lookup."""
        return (
            _df.groupby(["year", "town"])
            .agg(
                txn_volume=("resale_price", "count"),
                median_psm=("price_per_sqm", "median"),
                median_price=("resale_price", "median"),
            )
            .reset_index()
        )

    @st.cache_data
    def compute_wave_impact(_df, min_year=1995, max_year=2022):
        """
        For each (town, mop_unlock_year) that represents a major wave,
        compute stats 1yr before, wave year, and 1yr after.
        """
        year_town_stats = compute_year_town_stats(_df)

        # Build wave counts: only historical waves (MOP unlock <= current year - 2)
        wave_df = (
            _df[
                (_df["mop_unlock_year"] >= min_year) &
                (_df["mop_unlock_year"] <= max_year)
            ]
            .groupby(["town", "mop_unlock_year"])
            .size()
            .reset_index(name="wave_units")
        )
        wave_df = wave_df.rename(columns={"mop_unlock_year": "wave_year"})

        # Major waves = above the 75th percentile threshold
        threshold = wave_df["wave_units"].quantile(0.70)
        major_waves = wave_df[wave_df["wave_units"] >= threshold].copy()

        records = []
        for _, row in major_waves.iterrows():
            town = row["town"]
            wy = int(row["wave_year"])
            for offset, label in [(-1, "1yr Before"), (0, "Wave Year"), (1, "1yr After")]:
                yr = wy + offset
                stats_row = year_town_stats[
                    (year_town_stats["town"] == town) & (year_town_stats["year"] == yr)
                ]
                if len(stats_row) > 0:
                    records.append({
                        "town": town,
                        "wave_year": wy,
                        "wave_units": int(row["wave_units"]),
                        "period": label,
                        "period_year": yr,
                        "txn_volume": stats_row["txn_volume"].values[0],
                        "median_psm": stats_row["median_psm"].values[0],
                        "median_price": stats_row["median_price"].values[0],
                    })

        return pd.DataFrame(records)

    wave_impact = compute_wave_impact(df)

    if wave_impact.empty:
        st.warning("Not enough historical data to compute wave impact analysis.")
    else:
        wave_towns = sorted(wave_impact["town"].unique())
        col_t2a, col_t2b = st.columns([2, 1])
        with col_t2a:
            sel_town_t2 = st.selectbox(
                "Select a town to examine (or aggregate all)",
                options=["— All towns (average across waves) —"] + wave_towns,
                key="t2_town",
            )
        with col_t2b:
            metric_t2 = st.radio(
                "Price metric",
                options=["Median PSM (S$/sqm)", "Median Resale Price"],
                key="t2_metric",
                horizontal=True,
            )

        period_order = ["1yr Before", "Wave Year", "1yr After"]

        if sel_town_t2 == "— All towns (average across waves) —":
            plot_data = (
                wave_impact
                .groupby("period")
                .agg(
                    txn_volume=("txn_volume", "mean"),
                    median_psm=("median_psm", "median"),
                    median_price=("median_price", "median"),
                )
                .reset_index()
            )
            title_suffix = "All Towns (Avg Across Major Waves)"
        else:
            plot_data = (
                wave_impact[wave_impact["town"] == sel_town_t2]
                .groupby("period")
                .agg(
                    txn_volume=("txn_volume", "mean"),
                    median_psm=("median_psm", "median"),
                    median_price=("median_price", "median"),
                )
                .reset_index()
            )
            title_suffix = sel_town_t2

        plot_data["period"] = pd.Categorical(
            plot_data["period"], categories=period_order, ordered=True
        )
        plot_data = plot_data.sort_values("period")

        PERIOD_COLORS = {
            "1yr Before": "#636EFA",
            "Wave Year": "#EF553B",
            "1yr After": "#00CC96",
        }

        col_v, col_p = st.columns(2)

        with col_v:
            fig2a = px.bar(
                plot_data,
                x="period",
                y="txn_volume",
                title=f"Avg Transaction Volume Around MOP Wave — {title_suffix}",
                labels={"period": "Period", "txn_volume": "Avg Annual Transactions"},
                color="period",
                color_discrete_map=PERIOD_COLORS,
            )
            fig2a.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig2a, use_container_width=True)

        with col_p:
            price_col = "median_psm" if metric_t2 == "Median PSM (S$/sqm)" else "median_price"
            price_label = "Median PSM (S$/sqm)" if price_col == "median_psm" else "Median Resale Price (S$)"
            fig2b = px.bar(
                plot_data,
                x="period",
                y=price_col,
                title=f"{price_label} Around MOP Wave — {title_suffix}",
                labels={"period": "Period", price_col: price_label},
                color="period",
                color_discrete_map=PERIOD_COLORS,
            )
            fig2b.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig2b, use_container_width=True)

        # Volume change indicator
        if len(plot_data) == 3:
            vol_before = plot_data[plot_data["period"] == "1yr Before"]["txn_volume"].values[0]
            vol_wave = plot_data[plot_data["period"] == "Wave Year"]["txn_volume"].values[0]
            vol_after = plot_data[plot_data["period"] == "1yr After"]["txn_volume"].values[0]
            psm_before = plot_data[plot_data["period"] == "1yr Before"]["median_psm"].values[0]
            psm_after = plot_data[plot_data["period"] == "1yr After"]["median_psm"].values[0]

            vol_chg = (vol_after - vol_before) / vol_before * 100 if vol_before > 0 else 0
            psm_chg = (psm_after - psm_before) / psm_before * 100 if psm_before > 0 else 0

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Vol: 1yr Before", f"{vol_before:,.0f}")
            mc2.metric("Vol: Wave Year", f"{vol_wave:,.0f}", delta=f"{(vol_wave-vol_before)/vol_before*100:+.1f}%")
            mc3.metric("Vol: 1yr After", f"{vol_after:,.0f}", delta=f"{vol_chg:+.1f}% vs before")
            mc4.metric("Median PSM Change (Before→After)", f"{psm_chg:+.1f}%")

        with st.expander("📋 Show major MOP waves identified"):
            disp_cols = ["town", "wave_year", "wave_units", "period", "period_year",
                         "txn_volume", "median_psm", "median_price"]
            if sel_town_t2 == "— All towns (average across waves) —":
                detail = wave_impact[disp_cols].copy()
            else:
                detail = wave_impact[wave_impact["town"] == sel_town_t2][disp_cols].copy()

            detail = detail.rename(columns={
                "town": "Town",
                "wave_year": "MOP Wave Year",
                "wave_units": "Wave Size (units)",
                "period": "Period",
                "period_year": "Calendar Year",
                "txn_volume": "Transactions",
                "median_psm": "Median PSM (S$)",
                "median_price": "Median Price (S$)",
            })
            detail["Median PSM (S$)"] = detail["Median PSM (S$)"].apply(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
            )
            detail["Median Price (S$)"] = detail["Median Price (S$)"].apply(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
            )
            st.dataframe(detail, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Town-Level Supply Radar
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🏘️ Town-Level Supply Radar")
    st.markdown(
        "For each town, we estimate incoming MOP supply relative to current market absorption. "
        "The **Supply Pressure Ratio** = estimated MOP unlocks (24 months) ÷ current annual volume."
    )

    @st.cache_data
    def compute_supply_radar(_df, current_year):
        """Compute MOP supply pressure metrics for every town."""
        # MOP units unlocking in next 12 months (current year only)
        mop_12_df = (
            _df[_df["mop_unlock_year"] == current_year]
            .groupby("town")
            .size()
            .reset_index(name="mop_12m")
        )

        # MOP units unlocking in next 24 months (current year + next year)
        mop_24_df = (
            _df[_df["mop_unlock_year"].isin([current_year, current_year + 1])]
            .groupby("town")
            .size()
            .reset_index(name="mop_24m")
        )

        # Current transaction volume — last 12 months of data
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
        vol_df = (
            _df[_df["month"] >= cutoff]
            .groupby("town")
            .size()
            .reset_index(name="current_vol_12m")
        )

        # Start from full town list so no town is dropped
        all_towns_df = pd.DataFrame({"town": sorted(_df["town"].unique())})
        radar = (
            all_towns_df
            .merge(mop_12_df, on="town", how="left")
            .merge(mop_24_df, on="town", how="left")
            .merge(vol_df, on="town", how="left")
            .fillna(0)
        )

        # Supply pressure ratio
        radar["supply_pressure"] = radar.apply(
            lambda r: r["mop_24m"] / r["current_vol_12m"]
            if r["current_vol_12m"] > 0 else np.nan,
            axis=1,
        )
        return radar.sort_values("supply_pressure", ascending=False, na_position="last")

    supply_radar = compute_supply_radar(df, CURRENT_YEAR)

    if supply_radar.empty:
        st.warning("No data available for supply radar.")
    else:
        high_pressure = supply_radar[supply_radar["supply_pressure"] > 1.0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Towns with High Pressure (>1.0)",
            f"{len(high_pressure)}",
            help="Towns where MOP unlocks (next 24m) exceed current annual volume",
        )
        m2.metric(
            f"MOP Units Unlocking ({CURRENT_YEAR})",
            f"{supply_radar['mop_12m'].sum():,.0f}",
        )
        m3.metric(
            f"MOP Units Unlocking ({CURRENT_YEAR}–{CURRENT_YEAR+1})",
            f"{supply_radar['mop_24m'].sum():,.0f}",
        )
        m4.metric(
            "Current Annual Volume (Last 12m)",
            f"{supply_radar['current_vol_12m'].sum():,.0f}",
        )

        # Bubble chart
        bubble_data = supply_radar[
            (supply_radar["current_vol_12m"] > 0) &
            supply_radar["supply_pressure"].notna()
        ].copy()
        bubble_data["bubble_size"] = (bubble_data["mop_24m"] + 1).clip(lower=1)

        fig3 = px.scatter(
            bubble_data,
            x="current_vol_12m",
            y="supply_pressure",
            size="bubble_size",
            color="town",
            hover_name="town",
            hover_data={
                "mop_12m": True,
                "mop_24m": True,
                "current_vol_12m": True,
                "supply_pressure": ":.2f",
                "bubble_size": False,
                "town": False,
            },
            title=f"Town Supply Pressure Bubble Chart — MOP Horizon: {CURRENT_YEAR}–{CURRENT_YEAR+1}",
            labels={
                "current_vol_12m": "Current Annual Volume (Last 12m Transactions)",
                "supply_pressure": "Supply Pressure Ratio (MOP 24m ÷ Current Vol)",
                "mop_12m": f"MOP Unlocks {CURRENT_YEAR}",
                "mop_24m": f"MOP Unlocks {CURRENT_YEAR}–{CURRENT_YEAR+1}",
            },
            size_max=70,
        )
        fig3.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="crimson",
            annotation_text="Pressure = 1.0 (significant supply risk)",
            annotation_position="top right",
            annotation_font_color="crimson",
        )
        fig3.update_layout(height=580, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.info(
            "💡 **Interpretation:** **Supply Pressure > 1.0** means MOP unlocks could add "
            "significant new resale supply relative to current market absorption — a risk factor "
            "for price appreciation. Bubble size = estimated MOP units (24-month horizon)."
        )

        # Sortable table
        st.subheader("📋 Town Supply Pressure Table")
        display_radar = supply_radar.copy()
        display_radar["supply_pressure_fmt"] = display_radar["supply_pressure"].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
        )
        display_radar["Pressure Signal"] = display_radar["supply_pressure"].apply(
            lambda v: "🔴 High" if (pd.notna(v) and v > 1.0)
            else ("🟡 Moderate" if (pd.notna(v) and v > 0.5) else "🟢 Low")
        )
        st.dataframe(
            display_radar.rename(columns={
                "town": "Town",
                "mop_12m": f"MOP Unlocks ({CURRENT_YEAR})",
                "mop_24m": f"MOP Unlocks ({CURRENT_YEAR}–{CURRENT_YEAR+1})",
                "current_vol_12m": "Current Annual Vol (Last 12m)",
                "supply_pressure_fmt": "Supply Pressure Ratio",
            })[
                ["Town",
                 f"MOP Unlocks ({CURRENT_YEAR})",
                 f"MOP Unlocks ({CURRENT_YEAR}–{CURRENT_YEAR+1})",
                 "Current Annual Vol (Last 12m)",
                 "Supply Pressure Ratio",
                 "Pressure Signal"]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Lease Vintage Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Lease Vintage Analysis")
    st.markdown(
        "Distribution of `lease_commence_date` by decade and town — reveals which towns have "
        "'young' stock (recent BTO completions, high MOP pressure ahead) vs 'mature' stock "
        "(established resale market, mostly older leases). "
        "**Towns with lots of 2018–2022 stock have a supply wave coming.**"
    )

    DECADE_ORDER = ["Pre-1980", "1980s", "1990s", "2000s", "2010s", "2020s"]

    @st.cache_data
    def compute_vintage(_df):
        """Compute decade-level lease vintage distribution by town."""
        def decade_label(yr):
            if yr < 1980:
                return "Pre-1980"
            elif yr < 1990:
                return "1980s"
            elif yr < 2000:
                return "1990s"
            elif yr < 2010:
                return "2000s"
            elif yr < 2020:
                return "2010s"
            else:
                return "2020s"

        df2 = _df.copy()
        df2["vintage_decade"] = df2["lease_commence_date"].apply(decade_label)
        grp = (
            df2.groupby(["town", "vintage_decade"])
            .size()
            .reset_index(name="unit_count")
        )
        return grp

    @st.cache_data
    def compute_eligibility(_df, current_year):
        """Compute MOP-eligible share by town."""
        df2 = _df.copy()
        df2["mop_eligible"] = (df2["lease_commence_date"] + 5) <= current_year
        elig = (
            df2.groupby("town")["mop_eligible"]
            .agg(
                eligible_count="sum",
                total_units="count",
            )
            .reset_index()
        )
        elig["eligible_pct"] = (elig["eligible_count"] / elig["total_units"] * 100).round(1)
        return elig

    @st.cache_data
    def compute_recent_bto(_df):
        """Find towns with 2018–2022 lease_commence_date (imminent MOP waves)."""
        recent = _df[_df["lease_commence_date"].between(2018, 2022)].copy()
        recent["mop_unlock_year"] = recent["lease_commence_date"] + 5
        grp = (
            recent.groupby(["town", "lease_commence_date", "mop_unlock_year"])
            .size()
            .reset_index(name="units")
        )
        return grp

    vintage_grp = compute_vintage(df)
    eligibility = compute_eligibility(df, CURRENT_YEAR)
    recent_bto = compute_recent_bto(df)

    # Town filter
    sel_towns_t4 = st.multiselect(
        "Filter towns (leave blank for all)",
        options=ALL_TOWNS,
        default=[],
        key="t4_towns",
    )

    def apply_town_filter(frame, col="town"):
        if sel_towns_t4:
            return frame[frame[col].isin(sel_towns_t4)].copy()
        return frame.copy()

    vg_filtered = apply_town_filter(vintage_grp)
    vg_filtered["vintage_decade"] = pd.Categorical(
        vg_filtered["vintage_decade"], categories=DECADE_ORDER, ordered=True
    )
    vg_filtered = vg_filtered.sort_values(["town", "vintage_decade"])

    # ── Chart 1: Stacked bar — lease vintage by decade ───────────────────────
    fig4a = px.bar(
        vg_filtered,
        x="town",
        y="unit_count",
        color="vintage_decade",
        title="Lease Vintage Distribution by Town (Stacked by Decade)",
        labels={
            "town": "Town",
            "unit_count": "Transaction Count (proxy for unit stock)",
            "vintage_decade": "Lease Decade",
        },
        barmode="stack",
        category_orders={"vintage_decade": DECADE_ORDER},
        color_discrete_sequence=px.colors.sequential.Plasma_r,
    )
    fig4a.update_layout(
        xaxis_tickangle=-45,
        height=540,
        legend=dict(orientation="h", y=-0.28, title="Lease Decade"),
    )
    st.plotly_chart(fig4a, use_container_width=True)

    # ── Chart 2: MOP-eligible share by town ──────────────────────────────────
    st.subheader("MOP-Eligible Stock Estimate by Town")
    st.markdown(
        "Share of each town's estimated stock already past MOP "
        f"(`lease_commence_date + 5 ≤ {CURRENT_YEAR}`). "
        "Towns near 100% are mature markets; lower % indicates more flats still locked in MOP."
    )

    elig_filtered = apply_town_filter(eligibility)
    elig_sorted = elig_filtered.sort_values("eligible_pct", ascending=True)

    fig4b = px.bar(
        elig_sorted,
        x="eligible_pct",
        y="town",
        orientation="h",
        title=f"% of Stock Already MOP-Eligible (lease_commence_date + 5 ≤ {CURRENT_YEAR})",
        labels={
            "eligible_pct": "MOP-Eligible Share (%)",
            "town": "Town",
        },
        color="eligible_pct",
        color_continuous_scale="RdYlGn",
        range_color=[50, 100],
        text="eligible_pct",
    )
    fig4b.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig4b.update_layout(
        height=620,
        coloraxis_showscale=True,
        coloraxis_colorbar_title="% Eligible",
        xaxis=dict(range=[0, 110]),
    )
    st.plotly_chart(fig4b, use_container_width=True)

    # ── Chart 3: 2018-2022 BTO cohort breakdown ───────────────────────────────
    st.subheader("🔍 Near-Term MOP Wave — 2018–2022 BTO Cohorts (Unlock: 2023–2027)")
    st.markdown(
        "Towns with significant volumes of flats carrying `lease_commence_date` 2018–2022 "
        "are facing MOP expiry between **2023 and 2027** — the highest near-term supply risk window. "
        "Grouped by MOP unlock year."
    )

    bto_filtered = apply_town_filter(recent_bto)

    if bto_filtered.empty:
        st.info("No 2018–2022 lease_commence_date data for selected towns.")
    else:
        bto_filtered["mop_unlock_year"] = bto_filtered["mop_unlock_year"].astype(str)
        town_totals = bto_filtered.groupby("town")["units"].sum().sort_values(ascending=False)
        town_order = town_totals.index.tolist()

        fig4c = px.bar(
            bto_filtered,
            x="town",
            y="units",
            color="mop_unlock_year",
            barmode="group",
            title="2018–2022 BTO Cohorts by Town — MOP Unlock Years 2023–2027",
            labels={
                "town": "Town",
                "units": "Estimated Units",
                "mop_unlock_year": "MOP Unlock Year",
            },
            category_orders={
                "town": town_order,
                "mop_unlock_year": ["2023", "2024", "2025", "2026", "2027"],
            },
            color_discrete_sequence=px.colors.sequential.Oranges[2:],
        )
        fig4c.update_layout(
            xaxis_tickangle=-45,
            height=480,
            legend_title="MOP Unlock Year",
        )
        st.plotly_chart(fig4c, use_container_width=True)

        # Summary table for recent BTO
        with st.expander("📋 Show 2018–2022 BTO cohort data"):
            bto_tbl = (
                bto_filtered.groupby(["town", "mop_unlock_year"])["units"]
                .sum()
                .reset_index()
                .sort_values(["mop_unlock_year", "units"], ascending=[True, False])
                .rename(columns={
                    "town": "Town",
                    "mop_unlock_year": "MOP Unlock Year",
                    "units": "Estimated Units",
                })
            )
            st.dataframe(bto_tbl, use_container_width=True, hide_index=True)

    # ── footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Data current as of last CSV update. Page generated {TODAY.strftime('%d %b %Y')}. "
        "All MOP estimates use `lease_commence_date + 5` as the unlock year proxy. "
        "Verify against HDB's official BTO completion records before making property decisions."
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — HDB→Private Demand Bridge
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔗 HDB→Private Demand Bridge (I6)")
    st.caption(
        "MOP-cleared HDB supply in a town acts as a leading indicator for private condo "
        "demand in adjacent areas. Upgraders selling their MOP'd HDB flat typically reinvest "
        "into nearby private mass-market condos (OCR) 6–18 months later."
    )
    st.info(
        "**The thesis:** A large MOP wave in town X → upgrader pool with new equity → "
        "increased demand for private condos in the same or adjacent district → price support "
        "in that OCR/RCR private pocket 6–18 months later. This is most visible in towns "
        "historically associated with EC/condo upgrading (Tampines, Sengkang, Punggol, Woodlands, Jurong)."
    )

    @st.cache_data
    def compute_bridge(_df):
        # MOP wave estimates by town and year (same logic as Tab 1)
        df_2017 = _df[_df["year"] >= 2017].copy()
        df_2017["mop_yr"] = df_2017["lease_commence_date"] + 5
        mop_by_town_yr = (df_2017.groupby(["town", "mop_yr"])
                          .agg(mop_units=("resale_price", "count"))
                          .reset_index())

        # Resale volume in the year after MOP wave
        vol_by_town_yr = (_df.groupby(["town", "year"])
                          .agg(resale_vol=("resale_price", "count"))
                          .reset_index())

        # Merge: MOP wave year t → resale volume year t+1 (upgrader activation lag)
        mop_by_town_yr["lag_yr"] = mop_by_town_yr["mop_yr"] + 1
        bridge = mop_by_town_yr.merge(
            vol_by_town_yr.rename(columns={"year": "lag_yr", "resale_vol": "vol_1yr_after"}),
            on=["town", "lag_yr"], how="left"
        )
        return bridge, mop_by_town_yr, vol_by_town_yr

    bridge_df, mop_df, vol_df = compute_bridge(df)

    TODAY_YR = pd.Timestamp.now().year
    b_col1, b_col2 = st.columns([1, 3])
    with b_col1:
        b_towns = st.multiselect(
            "Towns to highlight",
            sorted(df["town"].unique()),
            default=["TAMPINES", "SENGKANG", "PUNGGOL", "WOODLANDS", "JURONG WEST"]
            if all(t in df["town"].unique() for t in ["TAMPINES", "SENGKANG", "PUNGGOL"]) else [],
            key="bridge_towns"
        )

    with b_col2:
        # Chart: MOP units by town over time
        mop_plot = mop_df[mop_df["town"].isin(b_towns)] if b_towns else mop_df.copy()
        mop_plot = mop_plot[(mop_plot["mop_yr"] >= 2015) & (mop_plot["mop_yr"] <= TODAY_YR + 3)]

        if len(mop_plot) > 0:
            fig_bridge = px.bar(
                mop_plot, x="mop_yr", y="mop_units", color="town", barmode="stack",
                labels={"mop_yr": "MOP Wave Year", "mop_units": "Estimated MOP Units", "town": "Town"},
                title="Estimated MOP Wave by Town (upgrader equity pool timing)",
            )
            fig_bridge.add_vline(x=TODAY_YR, line_dash="dash", line_color="red",
                                  annotation_text=f"Now ({TODAY_YR})")
            st.plotly_chart(fig_bridge, use_container_width=True)

    # Historical correlation: did past MOP waves predict resale volume surges?
    st.markdown("#### Historical: Did MOP waves precede resale volume increases?")
    bridge_hist = bridge_df[(bridge_df["mop_yr"] >= 2015) & (bridge_df["mop_yr"] <= TODAY_YR - 2)].dropna(subset=["vol_1yr_after"])
    if len(bridge_hist) > 5:
        fig_corr = px.scatter(
            bridge_hist, x="mop_units", y="vol_1yr_after", color="town",
            trendline="ols",
            labels={"mop_units": "MOP units in year T", "vol_1yr_after": "Resale volume in year T+1"},
            title="MOP wave size vs resale volume the following year (per town)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Forward-looking table
    future_mop = mop_df[(mop_df["mop_yr"] >= TODAY_YR) & (mop_df["mop_yr"] <= TODAY_YR + 3)]
    if len(future_mop) > 0:
        st.markdown("#### Upcoming MOP waves — potential upgrader pools")
        pivot = future_mop.pivot_table(index="town", columns="mop_yr", values="mop_units", aggfunc="sum").fillna(0)
        st.dataframe(pivot.astype(int), use_container_width=True)

    # ── I6 FULL: Real private condo volume in adjacent districts ──────────────
    st.markdown("---")
    st.markdown("#### Real private condo transaction volume in MOP-adjacent districts (I6)")

    TOWN_TO_OCR_DISTRICTS = {
        "TAMPINES":    [18], "PASIR RIS": [18],
        "SENGKANG":    [19], "PUNGGOL":   [19],
        "HOUGANG":     [19], "WOODLANDS": [25, 27],
        "SEMBAWANG":   [27], "YISHUN":    [27],
        "JURONG WEST": [22], "JURONG EAST": [22],
        "CHOA CHU KANG": [23], "BUKIT PANJANG": [23],
        "ANG MO KIO":  [20], "BISHAN":    [20],
        "BEDOK":       [16], "GEYLANG":   [14],
        "CLEMENTI":    [5],  "QUEENSTOWN": [3],
        "SERANGOON":   [13], "BUKIT MERAH": [3, 4],
        "TOA PAYOH":   [12], "BUKIT TIMAH": [10, 21],
        "MARINE PARADE": [15], "KALLANG/WHAMPOA": [12, 13],
    }

    @st.cache_data
    def _load_condo_vol():
        cdf = load_condo_clean()
        if cdf.empty:
            return pd.DataFrame()
        cdf = cdf[
            cdf["property_type_broad"].isin(["Condo/Apartment", "Executive Condo (EC)"])
        ].copy()
        cdf["contract_date"] = pd.to_datetime(cdf["contract_date"])
        cdf["year"] = cdf["contract_date"].dt.year
        vol = cdf.groupby(["district", "year"]).size().reset_index(name="condo_vol")
        return vol

    condo_vol_df = _load_condo_vol()

    if condo_vol_df.empty:
        st.info("Condo volume data not yet available — run `python src/fetch_data.py` to fetch URA caveat data.")
    else:
        towns_with_map = [t for t in (b_towns if b_towns else list(TOWN_TO_OCR_DISTRICTS.keys())) if t in TOWN_TO_OCR_DISTRICTS]
        if towns_with_map:
            rows = []
            for t in towns_with_map:
                dists = TOWN_TO_OCR_DISTRICTS.get(t, [])
                for d in dists:
                    sub = condo_vol_df[condo_vol_df["district"] == d]
                    for _, r in sub.iterrows():
                        rows.append({"town": t, "district": d, "year": r["year"], "condo_vol": r["condo_vol"]})
            if rows:
                bridge_condo = pd.DataFrame(rows)
                # Add MOP wave by town from bridge_df
                mop_yr_vol = mop_df[mop_df["town"].isin(towns_with_map)].copy()
                # Merge condo volume with MOP wave on lag (MOP year → condo volume next year)
                mop_yr_vol["lag_yr"] = mop_yr_vol["mop_yr"] + 1
                bridge_full = mop_yr_vol.merge(
                    bridge_condo.rename(columns={"year": "lag_yr"})
                    .groupby(["town", "lag_yr"])["condo_vol"].sum().reset_index(),
                    on=["town", "lag_yr"], how="left",
                )

                fig_bridge2 = go.Figure()
                colors = px.colors.qualitative.Set2
                for i, town in enumerate(towns_with_map[:6]):
                    t_data = bridge_full[bridge_full["town"] == town].sort_values("mop_yr")
                    if t_data.empty:
                        continue
                    col = colors[i % len(colors)]
                    fig_bridge2.add_trace(go.Bar(
                        x=t_data["mop_yr"], y=t_data["mop_units"],
                        name=f"{town} MOP", opacity=0.5,
                        marker_color=col, yaxis="y",
                    ))
                    t_condo = t_data.dropna(subset=["condo_vol"])
                    if not t_condo.empty:
                        fig_bridge2.add_trace(go.Scatter(
                            x=t_condo["lag_yr"], y=t_condo["condo_vol"],
                            name=f"{town} Condo Vol (T+1)",
                            line=dict(color=col, width=2),
                            yaxis="y2", mode="lines+markers",
                        ))
                fig_bridge2.update_layout(
                    barmode="overlay",
                    yaxis=dict(title="Estimated MOP Units"),
                    yaxis2=dict(title="Private Condo Transactions (T+1)", overlaying="y", side="right"),
                    height=450,
                    title="MOP wave (bars) vs private condo volume in adjacent districts 1yr later (lines)",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=-0.3),
                )
                st.plotly_chart(fig_bridge2, use_container_width=True)

                # Correlation summary
                bridge_corr = bridge_full.dropna(subset=["mop_units", "condo_vol"])
                if len(bridge_corr) >= 5:
                    corr_val = bridge_corr["mop_units"].corr(bridge_corr["condo_vol"])
                    if abs(corr_val) >= 0.5:
                        st.success(
                            f"🟢 **Correlation confirmed: r = {corr_val:.2f}** — MOP wave size positively "
                            "correlated with private condo volume in adjacent districts the following year."
                        )
                    else:
                        st.info(
                            f"🟡 **Weak correlation: r = {corr_val:.2f}** — MOP waves and adjacent private "
                            "volume don't show a clear link in this dataset window. "
                            "The effect may be masked by broader market forces."
                        )

    st.warning(
        "DATA CONFIDENCE: Medium. "
        "The upgrader bridge thesis is directionally supported by academic and industry research "
        "but is hard to isolate precisely in public data. MOP unit counts are estimated from "
        "lease_commence_date (see Tab 1 caveats). Private condo volume from URA caveat data "
        "(Aug 2021+) — pre-2021 private volume not available."
    )
