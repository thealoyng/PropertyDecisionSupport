"""
Page 16 🏢 HDB vs Private Residential
======================================
Compares HDB resale prices against URA private residential property prices
using the URA Property Price Index (PPI) and aggregate transaction volumes.

Key methodology note:
  - URA PPI is an index (base 2009-Q1 = 100), NOT raw prices.
  - HDB resale is re-indexed on the same base so both series are comparable.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from eda_helpers import load_clean, fmt_price, fmt_pct, POLICY_EVENTS, load_condo_clean

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HDB vs Private",
    page_icon="🏢",
    layout="wide",
)

st.title("🏢 HDB vs Private Residential")
st.caption(
    "Compare HDB resale price trends against URA private residential property "
    "price indices — covering price growth, rolling returns, correlation, and market volumes."
)

# ── data paths ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PPI_CSV = os.path.join(DATA_DIR, "private_property", "ura_ppi.csv")
AGG_CSV = os.path.join(DATA_DIR, "private_property", "ura_private_transactions_agg.csv")


# ── cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_ura_ppi():
    """Load URA Property Price Index data."""
    df = pd.read_csv(PPI_CSV)
    df["quarter"] = df["quarter"].str.strip()
    return df


@st.cache_data
def load_ura_agg():
    """Load URA aggregate quarterly transaction data."""
    df = pd.read_csv(AGG_CSV)
    df["quarter"] = df["quarter"].str.strip()
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data
def build_hdb_index():
    """
    Compute quarterly median price-per-sqm from HDB resale data and
    rebase to 2009-Q1 = 100, matching the URA PPI base period.
    Returns a DataFrame with columns: quarter (str "YYYY-Qn"), hdb_index.
    """
    df = load_clean()
    # Build quarter string in the same "YYYY-Qn" format as URA PPI
    df["quarter_str"] = (
        df["month"].dt.to_period("Q").astype(str).str.replace("Q", "-Q", regex=False)
    )
    quarterly = (
        df.groupby("quarter_str")["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"quarter_str": "quarter", "price_per_sqm": "median_psm"})
    )
    quarterly = quarterly.sort_values("quarter").reset_index(drop=True)
    # Base: 2009-Q1
    base_rows = quarterly[quarterly["quarter"] == "2009-Q1"]
    if base_rows.empty:
        raise ValueError("2009-Q1 not found in HDB resale data; cannot build index.")
    base_psm = base_rows["median_psm"].iloc[0]
    quarterly["hdb_index"] = (quarterly["median_psm"] / base_psm) * 100
    return quarterly


@st.cache_data
def build_merged_index():
    """
    Merge HDB index with URA PPI (Non-Landed and All Residential) on quarter.
    Returns a DataFrame filtered to quarters from 2000-Q1 onwards.
    """
    hdb = build_hdb_index()
    ppi = load_ura_ppi()

    non_landed = (
        ppi[ppi["property_type"] == "Non-Landed"]
        .rename(columns={"index": "ura_nonlanded"})
        [["quarter", "ura_nonlanded"]]
    )
    all_res = (
        ppi[ppi["property_type"] == "All Residential"]
        .rename(columns={"index": "ura_allres"})
        [["quarter", "ura_allres"]]
    )

    merged = (
        hdb
        .merge(non_landed, on="quarter", how="outer")
        .merge(all_res, on="quarter", how="outer")
        .sort_values("quarter")
        .reset_index(drop=True)
    )
    # Filter from 2000-Q1 onwards
    merged = merged[merged["quarter"] >= "2000-Q1"].copy()
    return merged


@st.cache_data
def build_hdb_annual_volume():
    """Annual HDB resale transaction counts."""
    df = load_clean()
    annual = (
        df.groupby("year")
        .size()
        .reset_index(name="hdb_units")
    )
    return annual


# ── helper: convert quarter str to Timestamp (mid-quarter) ────────────────────

def quarter_to_ts(q: str) -> pd.Timestamp:
    """Convert '2009-Q1' -> Timestamp('2009-01-01')."""
    try:
        return pd.Period(q.replace("-Q", "Q"), freq="Q").to_timestamp()
    except Exception:
        return pd.NaT


def add_policy_vlines(fig, quarters_present, from_year=2000):
    """Add dashed vertical lines for key policy events."""
    for idx, (date_str, label) in enumerate(POLICY_EVENTS):
        ts = pd.Timestamp(date_str)
        if ts.year < from_year:
            continue
        fig.add_vline(
            x=ts,
            line_width=1,
            line_dash="dot",
            line_color="#94a3b8",
        )
        y_anchor = 1.0 if idx % 2 == 0 else 0.90
        fig.add_annotation(
            x=ts,
            y=y_anchor,
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=8, color="#64748b"),
            textangle=-90,
            xanchor="left",
        )
    return fig


# ── load data ─────────────────────────────────────────────────────────────────
try:
    merged = build_merged_index()
    hdb_index_df = build_hdb_index()
    ppi_df = load_ura_ppi()
    agg_df = load_ura_agg()
    data_ok = True
except Exception as e:
    st.error(f"Failed to load data: {e}")
    data_ok = False
    st.stop()

# ── tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Index Comparison",
    "📊 Growth Rate Comparison",
    "🔄 Correlation & Divergence",
    "📦 Private Market Volume",
    "🔬 Unit-Level PSM (F2)",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price Index Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Price Index Comparison (base 2009-Q1 = 100)")
    st.caption(
        "Both indices are rebased to 100 at 2009-Q1. HDB index uses quarterly "
        "median price-per-sqm from resale transactions. URA PPI uses caveats lodged."
    )

    # ── KPI cards ──────────────────────────────────────────────────────────────
    latest_q = merged.dropna(subset=["hdb_index", "ura_nonlanded"]).iloc[-1]
    latest_quarter = latest_q["quarter"]

    hdb_now = latest_q["hdb_index"]
    ura_nl_now = latest_q["ura_nonlanded"]
    ura_ar_now = latest_q["ura_allres"] if not pd.isna(latest_q["ura_allres"]) else ura_nl_now

    hdb_chg = hdb_now - 100
    ura_chg = ura_nl_now - 100
    faster = "HDB" if hdb_chg > ura_chg else "Private (Non-Landed)"
    gap = ura_nl_now - hdb_now

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"HDB Index ({latest_quarter})",
        f"{hdb_now:.1f}",
        f"{hdb_chg:+.1f} pts since 2009-Q1 ({hdb_chg:.0f}%)",
        help="HDB resale median PSM rebased to 100 at 2009-Q1.",
    )
    k2.metric(
        f"URA Non-Landed Index ({latest_quarter})",
        f"{ura_nl_now:.1f}",
        f"{ura_chg:+.1f} pts since 2009-Q1 ({ura_chg:.0f}%)",
        help="URA Private Non-Landed PPI rebased to 100 at 2009-Q1.",
    )
    k3.metric(
        "Faster grower since 2009",
        faster,
        help="Which market showed higher index growth since 2009-Q1.",
    )
    k4.metric(
        "Private premium over HDB",
        f"{gap:+.1f} pts",
        help="URA Non-Landed index minus HDB index as of latest quarter. "
             "Positive = private has outgrown HDB since 2009.",
    )

    st.divider()

    # ── line chart ─────────────────────────────────────────────────────────────
    plot_df = merged.copy()
    plot_df["ts"] = plot_df["quarter"].apply(quarter_to_ts)
    plot_df = plot_df.dropna(subset=["ts"])

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=plot_df["ts"], y=plot_df["ura_allres"],
        name="URA All Residential",
        mode="lines",
        line=dict(color="#d1d5db", width=1.5),
        opacity=0.7,
    ))
    fig1.add_trace(go.Scatter(
        x=plot_df["ts"], y=plot_df["ura_nonlanded"],
        name="URA Non-Landed",
        mode="lines",
        line=dict(color="#f97316", width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=plot_df["ts"], y=plot_df["hdb_index"],
        name="HDB Resale",
        mode="lines",
        line=dict(color="#2563eb", width=2.5),
    ))
    # Base reference line
    fig1.add_hline(
        y=100, line_dash="dash", line_color="#6b7280", line_width=1,
        annotation_text="Base: 2009-Q1 = 100",
        annotation_position="bottom left",
        annotation_font=dict(size=9, color="#6b7280"),
    )
    fig1 = add_policy_vlines(fig1, plot_df["quarter"].tolist())
    fig1.update_layout(
        yaxis_title="Price Index (2009-Q1 = 100)",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, b=20),
        height=500,
    )
    st.plotly_chart(fig1, width="stretch")

    # ── methodology note ───────────────────────────────────────────────────────
    with st.expander("ℹ️ Data confidence & methodology"):
        st.markdown("""
**Data Confidence: Medium**

- Indices are based on aggregate medians — not matched pairs or hedonic adjustment.
- HDB index uses resale transactions only; URA PPI uses caveats lodged (includes new sales).
- Direct comparison should be taken as **broad directional**, not precise.
- HDB index is derived from `price_per_sqm` quarterly medians; the base quarter (2009-Q1)
  must have sufficient observations for the index to be reliable.

> **Note:** For individual condo transaction comparisons at the unit level, this page will be
> enhanced when URA API individual transaction data is integrated. The current view shows
> aggregate price index trends only.
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Growth Rate Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Annualised Growth Rate Comparison")
    st.caption(
        "Rolling annualised returns for HDB and Private Non-Landed indices, "
        "computed from the most recent available quarter."
    )

    @st.cache_data
    def compute_rolling_returns():
        m = build_merged_index().copy()
        m["ts"] = m["quarter"].apply(quarter_to_ts)
        m = m.dropna(subset=["hdb_index", "ura_nonlanded", "ts"]).sort_values("ts").reset_index(drop=True)

        results = []
        latest = m.iloc[-1]
        hdb_now_val = latest["hdb_index"]
        ura_now_val = latest["ura_nonlanded"]
        ts_now = latest["ts"]

        for n_years, label in [(1, "1-Year"), (3, "3-Year"), (5, "5-Year"), (10, "10-Year")]:
            ts_ago = ts_now - pd.DateOffset(years=n_years)
            # find closest row
            idx_ago = (m["ts"] - ts_ago).abs().idxmin()
            row_ago = m.iloc[idx_ago]
            actual_years = (ts_now - row_ago["ts"]).days / 365.25
            if actual_years < 0.5:
                continue
            hdb_ago = row_ago["hdb_index"]
            ura_ago = row_ago["ura_nonlanded"]
            hdb_ann = ((hdb_now_val / hdb_ago) ** (1 / actual_years) - 1) * 100
            ura_ann = ((ura_now_val / ura_ago) ** (1 / actual_years) - 1) * 100
            results.append({
                "Horizon": label,
                "HDB Resale (% p.a.)": round(hdb_ann, 2),
                "URA Non-Landed (% p.a.)": round(ura_ann, 2),
            })
        return pd.DataFrame(results)

    @st.cache_data
    def compute_decade_returns():
        m = build_merged_index().copy()
        m["ts"] = m["quarter"].apply(quarter_to_ts)
        m = m.dropna(subset=["hdb_index", "ura_nonlanded", "ts"]).sort_values("ts").reset_index(drop=True)

        decades = [
            ("2000s", "2000-Q1", "2009-Q4"),
            ("2010s", "2010-Q1", "2019-Q4"),
            ("2020s", "2020-Q1", None),
        ]
        rows = []
        for label, start_q, end_q in decades:
            sub = m[m["quarter"] >= start_q]
            if end_q:
                sub = sub[sub["quarter"] <= end_q]
            if len(sub) < 4:
                continue
            # compute YoY-equiv from start to end of decade
            ts_start = sub.iloc[0]["ts"]
            ts_end = sub.iloc[-1]["ts"]
            yrs = (ts_end - ts_start).days / 365.25
            if yrs < 0.5:
                continue
            hdb_ann = ((sub.iloc[-1]["hdb_index"] / sub.iloc[0]["hdb_index"]) ** (1 / yrs) - 1) * 100
            ura_ann = ((sub.iloc[-1]["ura_nonlanded"] / sub.iloc[0]["ura_nonlanded"]) ** (1 / yrs) - 1) * 100
            rows.append({
                "Decade": label,
                "HDB Resale (% p.a.)": round(hdb_ann, 2),
                "URA Non-Landed (% p.a.)": round(ura_ann, 2),
            })
        return pd.DataFrame(rows)

    rolling_df = compute_rolling_returns()
    decade_df = compute_decade_returns()

    # ── bar chart: rolling returns ──────────────────────────────────────────────
    if not rolling_df.empty:
        fig2a = go.Figure()
        fig2a.add_trace(go.Bar(
            x=rolling_df["Horizon"],
            y=rolling_df["HDB Resale (% p.a.)"],
            name="HDB Resale",
            marker_color="#2563eb",
        ))
        fig2a.add_trace(go.Bar(
            x=rolling_df["Horizon"],
            y=rolling_df["URA Non-Landed (% p.a.)"],
            name="URA Non-Landed",
            marker_color="#f97316",
        ))
        fig2a.update_layout(
            barmode="group",
            yaxis_title="Annualised return (% p.a.)",
            xaxis_title="Time horizon",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=40, b=20),
            height=380,
        )
        st.plotly_chart(fig2a, width="stretch")
        st.dataframe(rolling_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Insufficient data to compute rolling returns.")

    st.divider()

    # ── decade comparison ──────────────────────────────────────────────────────
    st.subheader("Decade-by-decade comparison")
    if not decade_df.empty:
        fig2b = go.Figure()
        fig2b.add_trace(go.Bar(
            x=decade_df["Decade"],
            y=decade_df["HDB Resale (% p.a.)"],
            name="HDB Resale",
            marker_color="#2563eb",
        ))
        fig2b.add_trace(go.Bar(
            x=decade_df["Decade"],
            y=decade_df["URA Non-Landed (% p.a.)"],
            name="URA Non-Landed",
            marker_color="#f97316",
        ))
        fig2b.update_layout(
            barmode="group",
            yaxis_title="Avg annualised return (% p.a.)",
            xaxis_title="",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=40, b=20),
            height=360,
        )
        st.plotly_chart(fig2b, width="stretch")
        st.dataframe(decade_df, use_container_width=True, hide_index=True)

    st.info(
        "💡 **Key insight:** HDB resale price growth has historically been more stable "
        "than private, with lower peaks and shallower troughs. Private prices are more "
        "sensitive to cooling measures, interest rate cycles, and foreign demand."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Correlation & Divergence
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Correlation & Divergence between HDB and Private")
    st.caption(
        "Rolling correlation between year-on-year growth rates, and the index spread "
        "showing when HDB and private prices diverge."
    )

    @st.cache_data
    def compute_correlation_series():
        m = build_merged_index().copy()
        m["ts"] = m["quarter"].apply(quarter_to_ts)
        m = m.dropna(subset=["hdb_index", "ura_nonlanded", "ts"]).sort_values("ts").reset_index(drop=True)

        # YoY growth (4-quarter lag)
        m["hdb_yoy"] = m["hdb_index"].pct_change(4) * 100
        m["ura_yoy"] = m["ura_nonlanded"].pct_change(4) * 100

        # Rolling 4-quarter correlation (pandas 2+ requires plain Series, not Rolling)
        m["rolling_corr"] = (
            m["hdb_yoy"]
            .rolling(8, min_periods=4)
            .corr(m["ura_yoy"])
        )

        # Rebased spread: start both at 100 and compute divergence
        first_valid = m.dropna(subset=["hdb_index", "ura_nonlanded"]).index[0]
        hdb_base = m.loc[first_valid, "hdb_index"]
        ura_base = m.loc[first_valid, "ura_nonlanded"]
        m["hdb_rebased"] = m["hdb_index"] / hdb_base * 100
        m["ura_rebased"] = m["ura_nonlanded"] / ura_base * 100
        m["spread"] = m["hdb_rebased"] - m["ura_rebased"]

        return m

    corr_df = compute_correlation_series()

    # ── rolling correlation chart ───────────────────────────────────────────────
    st.markdown("#### Rolling 8-quarter correlation of YoY growth rates")
    fig3a = go.Figure()
    fig3a.add_trace(go.Scatter(
        x=corr_df["ts"],
        y=corr_df["rolling_corr"],
        mode="lines",
        name="Rolling correlation",
        line=dict(color="#7c3aed", width=2),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.10)",
    ))
    fig3a.add_hline(y=0, line_dash="dash", line_color="#9ca3af", line_width=1)
    fig3a.add_hline(
        y=0.7, line_dash="dot", line_color="#16a34a", line_width=1,
        annotation_text="High correlation (0.7)",
        annotation_position="top left",
        annotation_font=dict(size=9),
    )
    fig3a = add_policy_vlines(fig3a, corr_df["quarter"].tolist())
    fig3a.update_layout(
        yaxis_title="Correlation coefficient",
        yaxis=dict(range=[-1.1, 1.1]),
        xaxis_title="",
        hovermode="x unified",
        margin=dict(t=40, b=20),
        height=380,
    )
    st.plotly_chart(fig3a, width="stretch")
    st.caption(
        "When correlation drops below 0.7, the two markets are decoupling. "
        "Low or negative correlation often coincides with policy events that "
        "affect private and HDB markets differently."
    )

    st.divider()

    # ── spread / divergence chart ───────────────────────────────────────────────
    st.markdown("#### HDB vs Private: Relative performance (rebased to equal start)")
    fig3b = go.Figure()
    fig3b.add_trace(go.Scatter(
        x=corr_df["ts"], y=corr_df["ura_rebased"],
        name="Private Non-Landed",
        mode="lines",
        line=dict(color="#f97316", width=2),
    ))
    fig3b.add_trace(go.Scatter(
        x=corr_df["ts"], y=corr_df["hdb_rebased"],
        name="HDB Resale",
        mode="lines",
        line=dict(color="#2563eb", width=2),
    ))
    # Filled area showing spread direction
    fig3b.add_trace(go.Scatter(
        x=pd.concat([corr_df["ts"], corr_df["ts"][::-1]]),
        y=pd.concat([corr_df["ura_rebased"], corr_df["hdb_rebased"][::-1]]),
        fill="toself",
        fillcolor="rgba(249,115,22,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
        name="Gap",
    ))
    fig3b = add_policy_vlines(fig3b, corr_df["quarter"].tolist())
    fig3b.update_layout(
        yaxis_title="Rebased index (equal start = 100)",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, b=20),
        height=420,
    )
    st.plotly_chart(fig3b, width="stretch")

    # ── spread line ────────────────────────────────────────────────────────────
    st.markdown("#### Spread: HDB minus Private (positive = HDB outperforming)")
    fig3c = go.Figure()
    spread_clean = corr_df.dropna(subset=["spread"])
    colors = np.where(spread_clean["spread"] >= 0, "#2563eb", "#f97316")
    fig3c.add_trace(go.Bar(
        x=spread_clean["ts"],
        y=spread_clean["spread"],
        marker_color=colors,
        name="HDB - Private spread",
        hovertemplate="%{x|%Y-Q%q}<br>Spread: %{y:+.1f} pts<extra></extra>",
    ))
    fig3c.add_hline(y=0, line_dash="dash", line_color="#6b7280", line_width=1)
    fig3c = add_policy_vlines(fig3c, corr_df["quarter"].tolist())
    fig3c.update_layout(
        yaxis_title="Index spread (HDB − Private)",
        xaxis_title="",
        margin=dict(t=40, b=20),
        height=320,
    )
    st.plotly_chart(fig3c, width="stretch")
    st.caption(
        "🔵 Blue bars = HDB outperforming private on a rebased basis. "
        "🟠 Orange bars = Private outperforming HDB. "
        "Notable: ABSD hike in Dec 2021 disproportionately dampened private demand."
    )

    with st.expander("ℹ️ Data confidence"):
        st.markdown("""
**Data Confidence: Medium**

- YoY growth computed from quarterly index values — small sample sizes in early periods
  may produce noisy correlation estimates.
- Rebased spread is sensitive to the chosen start date (earliest common quarter).
- Treat divergence signals as directional rather than precise.
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Private Market Volume
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Private Market Transaction Volume")
    st.caption(
        "URA private residential quarterly transaction volumes by sale type, "
        "compared with HDB resale volumes."
    )

    # ── stacked bar: private volume by type_of_sale ────────────────────────────
    st.markdown("#### Private transactions by sale type (quarterly)")

    agg = agg_df.copy()
    agg["ts"] = agg["quarter"].apply(quarter_to_ts)
    agg = agg.dropna(subset=["ts"]).sort_values("ts")

    # Aggregate across sale_status to get total per quarter per type_of_sale
    agg_by_type = (
        agg.groupby(["ts", "quarter", "type_of_sale"])["units"]
        .sum()
        .reset_index()
    )

    sale_types = agg_by_type["type_of_sale"].unique().tolist()
    color_map = {
        "New Sale": "#f97316",
        "Resale": "#2563eb",
        "Sub Sale": "#a855f7",
    }

    fig4a = go.Figure()
    for stype in sale_types:
        sub = agg_by_type[agg_by_type["type_of_sale"] == stype]
        fig4a.add_trace(go.Bar(
            x=sub["ts"],
            y=sub["units"],
            name=stype,
            marker_color=color_map.get(stype, "#6b7280"),
            hovertemplate=f"{stype}<br>%{{x|%Y-Q%q}}: %{{y:,}} units<extra></extra>",
        ))
    fig4a.update_layout(
        barmode="stack",
        yaxis_title="Units sold",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, b=20),
        height=420,
    )
    st.plotly_chart(fig4a, width="stretch")

    st.divider()

    # ── annual HDB vs Private comparison ───────────────────────────────────────
    st.markdown("#### Annual volume: Private vs HDB Resale")

    # Private annual
    agg["year"] = agg["ts"].dt.year
    private_annual = (
        agg.groupby("year")["units"]
        .sum()
        .reset_index()
        .rename(columns={"units": "private_units"})
    )
    # HDB annual
    hdb_annual = build_hdb_annual_volume()

    vol_merged = private_annual.merge(hdb_annual, on="year", how="outer").sort_values("year")
    vol_merged = vol_merged[vol_merged["year"] >= 2000]

    fig4b = go.Figure()
    fig4b.add_trace(go.Bar(
        x=vol_merged["year"],
        y=vol_merged["hdb_units"],
        name="HDB Resale",
        marker_color="#2563eb",
        opacity=0.85,
    ))
    fig4b.add_trace(go.Bar(
        x=vol_merged["year"],
        y=vol_merged["private_units"],
        name="Private (all types)",
        marker_color="#f97316",
        opacity=0.85,
    ))
    fig4b.update_layout(
        barmode="group",
        yaxis_title="Annual units",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, b=20),
        height=380,
    )
    st.plotly_chart(fig4b, width="stretch")

    # ── summary table ──────────────────────────────────────────────────────────
    with st.expander("📋 Annual volume data table"):
        display_vol = vol_merged.copy()
        display_vol.columns = ["Year", "Private Units", "HDB Resale Units"]
        display_vol["Private Units"] = display_vol["Private Units"].fillna(0).astype(int)
        display_vol["HDB Resale Units"] = display_vol["HDB Resale Units"].fillna(0).astype(int)
        st.dataframe(display_vol, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Data confidence"):
        st.markdown("""
**Data Confidence: High** for unit counts.

- Private transaction data from URA starts at **1999-Q4** only.
- Includes New Sales, Sub-Sales, and Resales across all private residential types.
- HDB resale counts from resale caveat records (same source as price data).
- Volume figures exclude private rentals and commercial transactions.
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Unit-Level PSM Comparison (F2 — requires URA API key)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔬 Unit-Level PSM: HDB vs Condo (F2)")
    st.markdown(
        "Direct price-per-sqm comparison between HDB resale transactions and "
        "URA private caveat-level data, over the same time window. "
        "**No index rebasing** — raw $/sqm on comparable floor areas."
    )

    @st.cache_data
    def load_psm_comparison():
        """Merge recent HDB resale and condo clean PSM data."""
        hdb = load_clean()
        hdb["month"] = pd.to_datetime(hdb["month"])
        hdb["quarter"] = hdb["month"].dt.to_period("Q").astype(str)
        hdb["psm"] = hdb["price_per_sqm"]
        hdb["source"] = "HDB Resale"
        hdb["type"] = hdb["flat_type"]
        hdb_out = hdb[["quarter", "month", "source", "type", "psm", "town"]].copy()

        condo = load_condo_clean()
        if condo.empty:
            return hdb_out, pd.DataFrame()
        condo_strata = condo[
            (condo["property_type_broad"].isin(["Condo/Apartment", "Executive Condo (EC)"])) &
            condo["price_psm"].notna() &
            (condo["price_psm"] > 1000)
        ].copy()
        condo_strata["month"] = condo_strata["contract_date"]
        condo_strata["quarter"] = condo_strata["contract_quarter"]
        condo_strata["source"] = "Private Condo"
        condo_strata["type"] = condo_strata["property_type_broad"]
        condo_strata["psm"] = condo_strata["price_psm"]
        condo_strata["town"] = condo_strata["market_segment"]
        condo_out = condo_strata[["quarter", "month", "source", "type", "psm", "town"]].copy()
        return hdb_out, condo_out

    hdb_df, condo_df = load_psm_comparison()

    if condo_df.empty:
        st.warning(
            "Condo data not found. Run `python src/fetch_data.py` then "
            "`python src/combine_clean_condo.py` to generate the data."
        )
    else:
        # ── Controls ────────────────────────────────────────────────────────────
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            hdb_types = st.multiselect(
                "HDB flat types",
                ["3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"],
                default=["4 ROOM", "5 ROOM"],
                key="f2_hdb_types",
            )
        with col_c2:
            condo_types = st.multiselect(
                "Condo types",
                ["Condo/Apartment", "Executive Condo (EC)"],
                default=["Condo/Apartment"],
                key="f2_condo_types",
            )
        with col_c3:
            sale_filter = st.multiselect(
                "Condo sale type",
                ["New Sale", "Resale", "Sub Sale"],
                default=["New Sale", "Resale"],
                key="f2_sale_type",
            )

        # Overlap window: condo data starts Aug 2021
        condo_start = pd.to_datetime("2021-08-01")
        hdb_recent = hdb_df[
            (hdb_df["month"] >= condo_start) &
            (hdb_df["type"].isin(hdb_types))
        ] if hdb_types else hdb_df[hdb_df["month"] >= condo_start]

        condo_recent_raw = load_condo_clean()
        if not condo_recent_raw.empty and sale_filter:
            condo_recent_raw = condo_recent_raw[condo_recent_raw["type_of_sale"].isin(sale_filter)]

        condo_recent = condo_df[condo_df["type"].isin(condo_types)] if condo_types else condo_df

        # ── Quarterly median PSM ────────────────────────────────────────────────
        hdb_q = hdb_recent.groupby("quarter")["psm"].median().reset_index()
        hdb_q["source"] = "HDB Resale (selected types)"
        condo_q = condo_recent.groupby("quarter")["psm"].median().reset_index()
        condo_q["source"] = "Private Condo"

        combined_q = pd.concat([hdb_q, condo_q], ignore_index=True)
        combined_q = combined_q.sort_values("quarter")

        fig_line = px.line(
            combined_q,
            x="quarter", y="psm", color="source",
            markers=True,
            labels={"quarter": "Quarter", "psm": "Median PSM ($/sqm)", "source": ""},
            title="Quarterly Median PSM: HDB Resale vs Private Condo (overlap window)",
            color_discrete_map={
                "HDB Resale (selected types)": "#1f77b4",
                "Private Condo": "#ff7f0e",
            },
        )
        fig_line.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)

        # ── PSM gap metric ─────────────────────────────────────────────────────
        latest_q = combined_q["quarter"].max()
        latest = combined_q[combined_q["quarter"] == latest_q]
        hdb_psm_latest   = latest[latest["source"].str.startswith("HDB")]["psm"].values
        condo_psm_latest = latest[latest["source"] == "Private Condo"]["psm"].values
        if len(hdb_psm_latest) and len(condo_psm_latest):
            gap = condo_psm_latest[0] - hdb_psm_latest[0]
            ratio = condo_psm_latest[0] / hdb_psm_latest[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("HDB Median PSM", f"${hdb_psm_latest[0]:,.0f}")
            c2.metric("Condo Median PSM", f"${condo_psm_latest[0]:,.0f}")
            c3.metric("Condo Premium", f"${gap:,.0f}  ({ratio:.1f}x HDB)",
                      delta=f"as of {latest_q}")

        # ── Distribution violin ────────────────────────────────────────────────
        st.markdown("#### PSM distribution — all transactions since Aug 2021")

        violin_data = []
        if not hdb_recent.empty:
            hdb_sample = hdb_recent[["psm"]].copy()
            hdb_sample["source"] = "HDB Resale"
            violin_data.append(hdb_sample)
        if not condo_recent.empty:
            condo_sample = condo_recent[["psm"]].copy()
            condo_sample["source"] = "Private Condo"
            violin_data.append(condo_sample)

        if violin_data:
            vdf = pd.concat(violin_data)
            # Cap at 99th percentile to avoid skew
            cap = vdf["psm"].quantile(0.99)
            vdf = vdf[vdf["psm"] <= cap]
            fig_vio = px.violin(
                vdf, y="psm", x="source", color="source", box=True,
                labels={"psm": "PSM ($/sqm)", "source": ""},
                title=f"PSM distribution (capped at 99th pct = ${cap:,.0f}/sqm)",
                color_discrete_map={
                    "HDB Resale": "#1f77b4",
                    "Private Condo": "#ff7f0e",
                },
            )
            fig_vio.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig_vio, use_container_width=True)

        # ── District vs HDB town scatter ───────────────────────────────────────
        st.markdown("#### Condo PSM by District vs HDB PSM by Town (same geography)")

        # District → HDB town approximate mapping
        DISTRICT_TO_HDB_TOWN = {
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

        hdb_town_psm = (
            hdb_recent.groupby("town")["psm"].median()
            .reset_index().rename(columns={"psm": "hdb_psm", "town": "hdb_town"})
        )

        condo_all = load_condo_clean()
        if not condo_all.empty:
            condo_all = condo_all[
                condo_all["property_type_broad"].isin(["Condo/Apartment", "Executive Condo (EC)"]) &
                condo_all["price_psm"].notna()
            ]
            if sale_filter:
                condo_all = condo_all[condo_all["type_of_sale"].isin(sale_filter)]
            condo_dist_psm = (
                condo_all.groupby("district")["price_psm"].median()
                .reset_index().rename(columns={"price_psm": "condo_psm"})
            )
            condo_dist_psm["hdb_town"] = condo_dist_psm["district"].map(DISTRICT_TO_HDB_TOWN)
            scatter_df = condo_dist_psm.merge(hdb_town_psm, on="hdb_town", how="inner")
            scatter_df["premium_pct"] = ((scatter_df["condo_psm"] - scatter_df["hdb_psm"]) / scatter_df["hdb_psm"] * 100).round(1)

            fig_sc = px.scatter(
                scatter_df,
                x="hdb_psm", y="condo_psm",
                text="hdb_town",
                color="premium_pct",
                color_continuous_scale="RdYlGn_r",
                labels={
                    "hdb_psm": "HDB Median PSM ($/sqm)",
                    "condo_psm": "Condo Median PSM ($/sqm)",
                    "premium_pct": "Condo Premium (%)",
                },
                title="HDB vs Condo PSM by neighbourhood (district → HDB town mapping)",
                size_max=14,
            )
            # Add 1:1 and 2:1 reference lines
            max_val = max(scatter_df["hdb_psm"].max(), scatter_df["condo_psm"].max())
            fig_sc.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                             line=dict(dash="dot", color="gray"), name="1:1")
            fig_sc.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val*2,
                             line=dict(dash="dash", color="lightcoral"), name="2x")
            fig_sc.update_traces(textposition="top center", textfont_size=9)
            fig_sc.update_layout(height=480)
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption("Grey dotted = 1:1 parity. Red dashed = condo costs 2x HDB.")

        st.success(
            "**DATA CONFIDENCE: High.** PSM derived directly from individual transaction caveats "
            "(HDB: resale_price / floor_area_sqm; Condo: URA caveat price / strata area). "
            "District→HDB-town mapping is approximate (many-to-one). "
            "Landed property and land-area transactions are excluded from PSM calculations."
        )
