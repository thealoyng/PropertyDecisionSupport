"""
Page 23 — Strategy Backtesting Framework
==========================================
Section 15: Rolling historical backtests of buy/sell signals derived from
the Opportunity Score and Fair Value Model — validating whether the signals
predicted actual price outperformance.

Non-negotiable rule: NO look-ahead bias. When evaluating a 2018 opportunity,
only information available in 2018 is used (no future transactions, no
post-2018 prices in training data).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from eda_helpers import load_clean, fmt_price, fmt_pct

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strategy Backtesting",
    page_icon="🔬",
    layout="wide",
)

# ── constants ─────────────────────────────────────────────────────────────────
BACKTEST_CONFIDENCE_NOTE = (
    "📊 **Data Confidence: Low / Experimental** — "
    "Backtest results use the same historical dataset that was used to design the "
    "Opportunity Score signals. This creates **training-set contamination**: the signals "
    "may appear to 'work' simply because they were calibrated on the same data. "
    "A true walk-forward validation would require monthly model retraining on "
    "out-of-sample data, which is computationally expensive and not yet implemented."
)

SURVIVORSHIP_WARNING = (
    "⚠️ **Survivorship bias**: only blocks with sufficient post-signal transactions "
    "are measurable. Blocks that were never re-sold after the signal date are excluded, "
    "which systematically overstates apparent returns."
)

NO_LOOKAHEAD_NOTE = (
    "🔒 **Look-ahead bias prevention**: for each evaluation quarter Q, signals are "
    "computed using ONLY transactions up to and including Q. No future data (after Q) "
    "enters the signal calculation."
)

PAST_PERF_DISCLAIMER = (
    "⚠️ **Past outperformance does not guarantee future outperformance.** "
    "Historical backtest results are indicative only and should not be relied upon "
    "as a basis for investment decisions."
)

HOLD_PERIODS = {"1 year": 1, "3 years": 3, "5 years": 5}
SIGNAL_TYPES = ["Value Score (PSM discount)", "Velocity (Liquidity)", "Combined"]

# ── data loading ──────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading transaction data…")
def get_base_df():
    df = load_clean()
    df["month"] = pd.to_datetime(df["month"])
    df = df.dropna(subset=["price_per_sqm", "resale_price", "town"])
    # quarter column used throughout backtesting
    df["quarter"] = df["month"].dt.to_period("Q")
    return df


# ── backtest engine ───────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Running backtest…")
def run_signal_backtest(
    _df: pd.DataFrame,
    signal_type: str,
    town_filter: str,
    hold_years: int,
    min_txn: int,
    discount_threshold: float,
    start_year: int,
    end_year: int,
) -> tuple:
    """
    Rolling backtest of buy signals derived from the Opportunity Score.

    Parameters
    ----------
    _df               : full cleaned transactions DataFrame (no future data leaks
                        because slicing is done per-quarter inside the loop)
    signal_type       : one of SIGNAL_TYPES
    town_filter       : town name or "All"
    hold_years        : 1, 3, or 5 — hold period after signal quarter
    min_txn           : minimum transactions in last 4 quarters to qualify
    discount_threshold: fractional discount vs town median required for signal
                        (e.g. 0.08 means block PSM must be ≥ 8 % below town median)
    start_year        : first year of backtest range (inclusive)
    end_year          : last signal year (inclusive); must be ≤ data_max - hold_years

    Returns
    -------
    (results_df, quartile_df, metrics_dict)
      results_df  : one row per signal quarter with portfolio_return, benchmark_return
      quartile_df : signal-quartile → avg hold-period return table
      metrics_dict: hit_rate, sharpe, max_drawdown, n_quarters, n_signals_total
    """
    df = _df.copy()
    if town_filter != "All":
        df = df[df["town"] == town_filter]

    if len(df) < 50:
        return pd.DataFrame(), pd.DataFrame(), {}

    hold_offset = pd.DateOffset(years=hold_years)

    # Build quarterly index for backtest
    min_q = pd.Period(f"{start_year}Q1", freq="Q")
    max_q = pd.Period(f"{end_year}Q4", freq="Q")

    quarters = pd.period_range(min_q, max_q, freq="Q")

    rows = []

    for q in quarters:
        q_end = q.to_timestamp(how="end")
        q_end_future = (q.to_timestamp(how="start") + hold_offset)

        # ── NO look-ahead: only data up to q_end ──────────────────────────
        hist = df[df["month"] <= q_end].copy()

        # Last 4 quarters window for velocity / transaction count
        q4_start = (q_end - pd.DateOffset(months=12))
        recent = hist[hist["month"] > q4_start]

        if len(recent) < 5:
            continue

        # ── town median PSM (computed on history only) ─────────────────────
        town_psm = (
            hist.groupby("town")["price_per_sqm"]
            .median()
            .rename("town_psm")
        )

        # ── block aggregates from recent 4 quarters ────────────────────────
        blk = (
            recent.groupby(["block", "street_name", "town"])
            .agg(
                txn_count=("resale_price", "size"),
                median_psm=("price_per_sqm", "median"),
            )
            .reset_index()
        )
        blk = blk[blk["txn_count"] >= min_txn].copy()
        blk = blk.join(town_psm, on="town")
        blk = blk.dropna(subset=["town_psm"])
        blk["disc_pct"] = (
            (blk["median_psm"] - blk["town_psm"]) / blk["town_psm"]
        )

        if len(blk) < 3:
            continue

        # ── signal generation ──────────────────────────────────────────────
        if signal_type == "Value Score (PSM discount)":
            signal_mask = blk["disc_pct"] <= -discount_threshold
        elif signal_type == "Velocity (Liquidity)":
            med_vel = blk["txn_count"].median()
            signal_mask = blk["txn_count"] >= med_vel * 1.5
        else:  # Combined
            med_vel = blk["txn_count"].median()
            signal_mask = (
                (blk["disc_pct"] <= -discount_threshold)
                & (blk["txn_count"] >= med_vel)
            )

        signal_blocks = blk[signal_mask].copy()
        n_signals = len(signal_blocks)

        if n_signals == 0:
            continue

        # ── measure actual appreciation over hold period ───────────────────
        future = df[
            (df["month"] > q_end)
            & (df["month"] <= q_end_future)
        ].copy()

        if len(future) < 5:
            continue

        # Portfolio: signal blocks only
        sig_keys = signal_blocks.set_index(["block", "street_name"])
        future_sig = future[
            future.apply(
                lambda r: (r["block"], r["street_name"]) in sig_keys.index,
                axis=1,
            )
        ]

        # Benchmark: all blocks in scope
        if len(future_sig) < 3:
            continue

        sig_entry_psm = signal_blocks["median_psm"].median()
        sig_exit_psm = future_sig["price_per_sqm"].median()

        bm_entry_psm = recent["price_per_sqm"].median()
        bm_exit_psm = future["price_per_sqm"].median()

        if sig_entry_psm <= 0 or bm_entry_psm <= 0:
            continue

        portfolio_return = (sig_exit_psm - sig_entry_psm) / sig_entry_psm
        benchmark_return = (bm_exit_psm - bm_entry_psm) / bm_entry_psm
        excess_return = portfolio_return - benchmark_return

        # ── score quartile → return mapping (for Tab 1 quartile table) ────
        blk["signal_score"] = -blk["disc_pct"]  # higher discount = higher score
        blk["quartile"] = pd.qcut(
            blk["signal_score"], q=4,
            labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
            duplicates="drop",
        )

        for _q_label, _q_grp in blk.groupby("quartile", observed=True):
            _q_keys = _q_grp.set_index(["block", "street_name"])
            _q_future = future[
                future.apply(
                    lambda r: (r["block"], r["street_name"]) in _q_keys.index,
                    axis=1,
                )
            ]
            if len(_q_future) >= 3:
                _q_entry = _q_grp["median_psm"].median()
                _q_exit = _q_future["price_per_sqm"].median()
                if _q_entry > 0:
                    rows.append({
                        "quarter": str(q),
                        "quartile": str(_q_label),
                        "portfolio_return": portfolio_return,
                        "benchmark_return": benchmark_return,
                        "excess_return": excess_return,
                        "q_return": (_q_exit - _q_entry) / _q_entry,
                        "n_signals": n_signals,
                        "n_future_txns": len(future_sig),
                    })

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), {}

    results_df = pd.DataFrame(rows).drop_duplicates(
        subset=["quarter", "quartile"]
    )

    # Collapse to per-quarter summary (deduplicate quartile rows)
    summary_df = (
        results_df.drop_duplicates(subset=["quarter"])
        .sort_values("quarter")
        .reset_index(drop=True)
    )

    # Quartile return table
    quartile_df = (
        results_df.groupby("quartile", observed=True)["q_return"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "Avg Return", "median": "Median Return", "count": "Obs"})
    )
    quartile_df["Avg Return"] = quartile_df["Avg Return"] * 100
    quartile_df["Median Return"] = quartile_df["Median Return"] * 100

    # Risk metrics
    n_q = len(summary_df)
    if n_q == 0:
        return summary_df, quartile_df, {}

    excess = summary_df["excess_return"]
    hit_rate = (excess > 0).mean() * 100

    # Sharpe-like: mean excess return / std dev (annualised crude approximation)
    sharpe = float(excess.mean() / excess.std()) if excess.std() > 0 else np.nan

    # Max drawdown on cumulative portfolio returns
    cum_ret = (1 + summary_df["portfolio_return"]).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100

    metrics = {
        "hit_rate": hit_rate,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_quarters": n_q,
        "n_signals_total": int(summary_df["n_signals"].sum()),
        "mean_excess_return": float(excess.mean() * 100),
        "mean_portfolio_return": float(summary_df["portfolio_return"].mean() * 100),
        "mean_benchmark_return": float(summary_df["benchmark_return"].mean() * 100),
    }

    return summary_df, quartile_df, metrics


@st.cache_data(show_spinner="Computing town-level signal accuracy…")
def compute_town_accuracy(
    _df: pd.DataFrame,
    hold_years: int,
    discount_threshold: float,
    min_txn: int,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    For each town, compute the hit rate (% of quarters where undervalued blocks
    outperformed the town benchmark) for a given hold period.

    Returns a DataFrame: town × hold_years → hit_rate (%)
    """
    df = _df.copy()
    towns = sorted(df["town"].dropna().unique().tolist())
    hold_offset = pd.DateOffset(years=hold_years)

    min_q = pd.Period(f"{start_year}Q1", freq="Q")
    max_q = pd.Period(f"{end_year}Q4", freq="Q")
    quarters = pd.period_range(min_q, max_q, freq="Q")

    town_rows = []

    for town in towns:
        tdf = df[df["town"] == town]
        hits, total = 0, 0

        for q in quarters:
            q_end = q.to_timestamp(how="end")
            q_end_future = (q.to_timestamp(how="start") + hold_offset)

            hist = tdf[tdf["month"] <= q_end]
            q4_start = q_end - pd.DateOffset(months=12)
            recent = hist[hist["month"] > q4_start]

            if len(recent) < min_txn:
                continue

            town_psm_val = hist["price_per_sqm"].median()
            if town_psm_val <= 0:
                continue

            blk = (
                recent.groupby(["block", "street_name"])
                .agg(
                    txn_count=("resale_price", "size"),
                    median_psm=("price_per_sqm", "median"),
                )
                .reset_index()
            )
            blk = blk[blk["txn_count"] >= min_txn].copy()
            blk["disc_pct"] = (blk["median_psm"] - town_psm_val) / town_psm_val
            signal_blocks = blk[blk["disc_pct"] <= -discount_threshold]

            if len(signal_blocks) == 0:
                continue

            future = tdf[
                (tdf["month"] > q_end) & (tdf["month"] <= q_end_future)
            ]
            if len(future) < 3:
                continue

            sig_keys = signal_blocks.set_index(["block", "street_name"])
            future_sig = future[
                future.apply(
                    lambda r: (r["block"], r["street_name"]) in sig_keys.index,
                    axis=1,
                )
            ]

            if len(future_sig) < 2:
                continue

            sig_entry = signal_blocks["median_psm"].median()
            sig_exit = future_sig["price_per_sqm"].median()
            bm_entry = recent["price_per_sqm"].median()
            bm_exit = future["price_per_sqm"].median()

            if sig_entry <= 0 or bm_entry <= 0:
                continue

            port_ret = (sig_exit - sig_entry) / sig_entry
            bm_ret = (bm_exit - bm_entry) / bm_entry

            if port_ret > bm_ret:
                hits += 1
            total += 1

        if total > 0:
            town_rows.append({
                "Town": town,
                "Hit Rate (%)": round(hits / total * 100, 1),
                "Quarters Evaluated": total,
                f"Hold {hold_years}yr": round(hits / total * 100, 1),
            })

    return pd.DataFrame(town_rows).sort_values("Hit Rate (%)", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Computing multi-period town heatmap…")
def compute_town_heatmap(
    _df: pd.DataFrame,
    hold_years_list: list,
    discount_threshold: float,
    min_txn: int,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Returns a pivot table: rows = towns, columns = hold periods (1yr, 3yr, 5yr),
    values = hit rate (%).
    """
    frames = []
    for hy in hold_years_list:
        acc_df = compute_town_accuracy(
            _df,
            hold_years=hy,
            discount_threshold=discount_threshold,
            min_txn=min_txn,
            start_year=start_year,
            end_year=end_year,
        )
        if len(acc_df) > 0:
            acc_df = acc_df[["Town", "Hit Rate (%)"]].rename(
                columns={"Hit Rate (%)": f"{hy}yr"}
            )
            frames.append(acc_df.set_index("Town"))

    if not frames:
        return pd.DataFrame()

    pivot = pd.concat(frames, axis=1).fillna(np.nan)
    return pivot.reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔬 Strategy Backtesting Framework")
st.caption(
    "Section 15 — Rolling historical backtests of Opportunity Score signals. "
    "Validates whether discounted blocks actually outperformed their town benchmarks."
)

# Load data
df_base = get_base_df()
ALL_TOWNS = sorted(df_base["town"].dropna().unique().tolist())

# Determine data date range
data_min_yr = int(df_base["month"].dt.year.min())
data_max_yr = int(df_base["month"].dt.year.max())

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔬 Backtest Parameters")

hold_label = st.sidebar.selectbox(
    "Hold period", list(HOLD_PERIODS.keys()), index=0, key="bt_hold",
)
hold_years = HOLD_PERIODS[hold_label]

signal_type = st.sidebar.selectbox(
    "Signal type", SIGNAL_TYPES, index=0, key="bt_sig",
)

sel_town = st.sidebar.selectbox(
    "Town (Tab 1)", ["All"] + ALL_TOWNS, key="bt_town",
)

discount_pct = st.sidebar.slider(
    "Discount threshold (%)",
    min_value=2, max_value=25, value=8, step=1,
    key="bt_disc",
    help="Block PSM must be this % below town median to trigger a buy signal.",
)
discount_threshold = discount_pct / 100.0

min_txn = st.sidebar.slider(
    "Min transactions (last 4 qtrs)", 3, 20, 5, key="bt_mintxn",
)

# Backtest date range — end year must leave room for hold period
default_end = min(data_max_yr - hold_years, data_max_yr - 1)
bt_start = st.sidebar.slider(
    "Backtest start year",
    min_value=max(data_min_yr, 2010), max_value=default_end - 1,
    value=max(data_min_yr, 2010), key="bt_sy",
)
bt_end = st.sidebar.slider(
    "Backtest end year",
    min_value=bt_start + 1, max_value=default_end,
    value=min(default_end, 2021), key="bt_ey",
)

with st.sidebar.expander("ℹ️ Disclaimer", expanded=False):
    st.markdown(PAST_PERF_DISCLAIMER)
    st.markdown(SURVIVORSHIP_WARNING)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 Signal Validation",
    "🏘️ Town-Level Signal Accuracy",
    "🎯 Methodology & Limitations",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📈 Signal Validation (Core Backtest)")

    # Prominent warnings
    st.error(BACKTEST_CONFIDENCE_NOTE)

    warn_col1, warn_col2 = st.columns(2)
    with warn_col1:
        st.warning(SURVIVORSHIP_WARNING)
    with warn_col2:
        st.info(NO_LOOKAHEAD_NOTE)

    st.caption(
        f"**Configuration:** Signal = *{signal_type}* · "
        f"Town = *{sel_town}* · Hold = *{hold_label}* · "
        f"Discount ≥ *{discount_pct}%* · Min txns = *{min_txn}* · "
        f"Range = *{bt_start}–{bt_end}*"
    )

    run_bt = st.button("▶ Run Backtest", key="run_bt", type="primary")

    if run_bt:
        prog = st.progress(0, text="Initialising backtest…")

        prog.progress(10, text="Computing rolling signals (no look-ahead)…")
        summary_df, quartile_df, metrics = run_signal_backtest(
            df_base,
            signal_type=signal_type,
            town_filter=sel_town,
            hold_years=hold_years,
            min_txn=min_txn,
            discount_threshold=discount_threshold,
            start_year=bt_start,
            end_year=bt_end,
        )
        prog.progress(100, text="Done.")

        if len(summary_df) == 0:
            st.warning(
                "No backtest results generated. Try:\n"
                "- Lowering the **Discount threshold** (fewer signals will be filtered out)\n"
                "- Lowering **Min transactions**\n"
                "- Widening the **Backtest date range**\n"
                "- Selecting **All** towns instead of a specific town"
            )
        else:
            # ── key metrics ───────────────────────────────────────────────
            st.subheader("Key Metrics")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric(
                "Quarters Evaluated",
                f"{metrics.get('n_quarters', 0):,}",
            )
            mc2.metric(
                "Signal Hit Rate",
                f"{metrics.get('hit_rate', 0):.1f}%",
                help="% of quarters where signal portfolio beat benchmark",
            )
            mc3.metric(
                "Mean Excess Return",
                f"{metrics.get('mean_excess_return', 0):.2f}%",
                help="Portfolio return minus benchmark return (per hold period)",
            )
            mc4.metric(
                "Sharpe-like Ratio",
                f"{metrics.get('sharpe', float('nan')):.2f}"
                if not np.isnan(metrics.get("sharpe", float("nan")))
                else "N/A",
                help="Mean excess return / std dev of excess returns",
            )
            mc5.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.1f}%",
                help="Maximum cumulative drawdown on portfolio returns",
            )

            st.caption(
                f"Total signal triggers across all quarters: "
                f"**{metrics.get('n_signals_total', 0):,} blocks**"
            )

            # ── time series chart ─────────────────────────────────────────
            st.subheader("Portfolio vs Benchmark Returns Over Time")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=summary_df["quarter"],
                y=summary_df["portfolio_return"] * 100,
                mode="lines+markers",
                name="Signal Portfolio",
                line=dict(color="#2ca02c", width=2),
            ))
            fig_ts.add_trace(go.Scatter(
                x=summary_df["quarter"],
                y=summary_df["benchmark_return"] * 100,
                mode="lines+markers",
                name="Benchmark (All Blocks)",
                line=dict(color="#1f77b4", width=2, dash="dash"),
            ))
            fig_ts.add_hline(y=0, line_dash="dot", line_color="grey")
            fig_ts.update_layout(
                xaxis_title="Signal Quarter",
                yaxis_title=f"Return over {hold_label} (%)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=60, r=20, t=40, b=60),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Excess return chart
            fig_exc = go.Figure()
            fig_exc.add_trace(go.Bar(
                x=summary_df["quarter"],
                y=summary_df["excess_return"] * 100,
                marker_color=[
                    "#2ca02c" if v >= 0 else "#d62728"
                    for v in summary_df["excess_return"]
                ],
                name="Excess Return",
            ))
            fig_exc.add_hline(y=0, line_dash="solid", line_color="black")
            fig_exc.update_layout(
                xaxis_title="Signal Quarter",
                yaxis_title="Excess Return vs Benchmark (%)",
                height=300,
                showlegend=False,
                margin=dict(l=60, r=20, t=30, b=60),
            )
            st.plotly_chart(fig_exc, use_container_width=True)
            st.caption(
                "🟢 Green bars = signal portfolio outperformed benchmark  |  "
                "🔴 Red bars = underperformed"
            )

            # ── quartile table ────────────────────────────────────────────
            st.subheader("Signal Quartile → Average Return")
            st.caption(
                "Do blocks with the deepest discounts (Q4) actually outperform "
                "shallow-discount blocks (Q1)?"
            )
            if len(quartile_df) > 0:
                st.dataframe(
                    quartile_df,
                    use_container_width=True,
                    column_config={
                        "Avg Return": st.column_config.NumberColumn(format="%.2f%%"),
                        "Median Return": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                )

                fig_q = px.bar(
                    quartile_df,
                    x="quartile",
                    y="Avg Return",
                    color="Avg Return",
                    color_continuous_scale=["#d62728", "#ffbb78", "#98df8a", "#2ca02c"],
                    labels={"quartile": "Signal Quartile", "Avg Return": f"Avg Return over {hold_label} (%)"},
                    title="Avg Return by Signal Quartile",
                )
                fig_q.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_q, use_container_width=True)
            else:
                st.info("Insufficient data to compute quartile returns.")

            # ── raw data expander ─────────────────────────────────────────
            with st.expander("📋 Raw backtest data"):
                st.dataframe(summary_df, use_container_width=True)
    else:
        st.info(
            "Configure backtest parameters in the sidebar, then click **▶ Run Backtest**.\n\n"
            "⏱ Estimated run time: 10–60 seconds depending on date range and town selection."
        )

    # Always show the disclaimer at the bottom
    st.divider()
    st.warning(PAST_PERF_DISCLAIMER)
    st.caption(
        "**Model contamination note:** The signals in this backtest were designed by inspecting "
        "the same historical data being backtested. A truly unbiased test would require a signal "
        "designed on pre-2010 data and validated on post-2010 data only. Real walk-forward "
        "validation would require monthly model retraining, which is computationally expensive "
        "and not yet implemented."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TOWN-LEVEL SIGNAL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🏘️ Town-Level Signal Accuracy")
    st.info(
        "📊 **Data Confidence: Low / Experimental** — "
        "Towns with fewer transactions have noisier signals. "
        "Results should be treated as indicative, not definitive."
    )

    st.markdown(
        f"For each town: percentage of signal quarters (using a **{discount_pct}% discount** "
        f"threshold and **≥{min_txn} transactions**) where undervalued blocks actually "
        f"outperformed the town benchmark over the selected hold period."
    )

    run_town = st.button("▶ Run Town Accuracy Analysis", key="run_town", type="primary")

    if run_town:
        prog2 = st.progress(0, text="Computing town accuracy…")

        # Single hold period first
        prog2.progress(20, text=f"Computing {hold_label} accuracy…")
        town_df_single = compute_town_accuracy(
            df_base,
            hold_years=hold_years,
            discount_threshold=discount_threshold,
            min_txn=min_txn,
            start_year=bt_start,
            end_year=bt_end,
        )

        # Multi-period heatmap
        prog2.progress(60, text="Computing multi-period heatmap (1yr, 3yr, 5yr)…")
        heatmap_df = compute_town_heatmap(
            df_base,
            hold_years_list=[1, 3, 5],
            discount_threshold=discount_threshold,
            min_txn=min_txn,
            start_year=bt_start,
            end_year=min(bt_end, data_max_yr - 5),  # 5yr hold needs 5yr runway
        )
        prog2.progress(100, text="Done.")

        if len(town_df_single) == 0:
            st.warning(
                "No town accuracy results. Try lowering the discount threshold "
                "or minimum transactions, or widening the date range."
            )
        else:
            # Summary bar chart
            st.subheader(f"Signal Hit Rate by Town ({hold_label} hold)")
            fig_town = px.bar(
                town_df_single.sort_values("Hit Rate (%)", ascending=True),
                x="Hit Rate (%)",
                y="Town",
                orientation="h",
                color="Hit Rate (%)",
                color_continuous_scale=["#d62728", "#ffbb78", "#2ca02c"],
                range_color=[0, 100],
                title=f"% of Quarters where Undervalued Blocks Outperformed ({hold_label} hold)",
                labels={"Hit Rate (%)": "Hit Rate (%)"},
            )
            fig_town.add_vline(x=50, line_dash="dash", line_color="grey",
                               annotation_text="50% baseline")
            fig_town.update_layout(height=max(400, len(town_df_single) * 20))
            st.plotly_chart(fig_town, use_container_width=True)

            # Best and worst
            bcol, wcol = st.columns(2)
            top5 = town_df_single.head(5)
            bot5 = town_df_single.tail(5).sort_values("Hit Rate (%)")
            with bcol:
                st.markdown("**🏆 Top 5 Towns (highest signal accuracy)**")
                st.dataframe(top5[["Town", "Hit Rate (%)", "Quarters Evaluated"]],
                             use_container_width=True, hide_index=True)
            with wcol:
                st.markdown("**⚠️ Bottom 5 Towns (lowest signal accuracy)**")
                st.dataframe(bot5[["Town", "Hit Rate (%)", "Quarters Evaluated"]],
                             use_container_width=True, hide_index=True)

            st.caption(
                "Towns with fewer than 5 quarters evaluated may show unreliable hit rates. "
                "Focus on towns with 10+ quarters of evaluation data for meaningful conclusions."
            )

            # Multi-period heatmap
            if len(heatmap_df) > 0:
                st.subheader("Multi-Period Heatmap: Town × Hold Period → Hit Rate (%)")
                st.caption(
                    "Cells show the % of signal quarters where the strategy beat the "
                    "benchmark. Only periods with sufficient data are shown."
                )
                hp_cols = [c for c in ["1yr", "3yr", "5yr"] if c in heatmap_df.columns]
                if hp_cols:
                    heat_pivot = heatmap_df.set_index("Town")[hp_cols]
                    fig_heat = px.imshow(
                        heat_pivot.values,
                        x=hp_cols,
                        y=heat_pivot.index.tolist(),
                        color_continuous_scale="RdYlGn",
                        zmin=0, zmax=100,
                        labels=dict(x="Hold Period", y="Town", color="Hit Rate (%)"),
                        title="Town × Hold Period — Signal Hit Rate (%)",
                    )
                    fig_heat.update_layout(
                        height=max(400, len(heat_pivot) * 18),
                        margin=dict(l=120, r=20, t=60, b=40),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Multi-period heatmap unavailable — insufficient data for all three hold periods.")

        # Always show caveat
        st.divider()
        st.warning(SURVIVORSHIP_WARNING)
    else:
        st.info(
            "Configure parameters in the sidebar, then click **▶ Run Town Accuracy Analysis**.\n\n"
            "⏱ This may take 30–90 seconds for the multi-period heatmap."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY & LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎯 Methodology & Limitations")

    st.markdown("""
## 1. How the Backtest is Constructed

For each evaluation quarter **Q** in the selected date range:

1. **Signal generation (no look-ahead):**
   - Only transactions up to and including Q are used.
   - Block-level median PSM and transaction velocity are computed from the
     **last 4 quarters** of history (≤ Q).
   - A "buy" signal fires when a block's median PSM is ≥ *discount_threshold*%
     below the town median PSM, and the block has ≥ *min_txn* transactions
     in the trailing 4 quarters.

2. **Hold period:**
   - After the signal quarter, the strategy "holds" for 1, 3, or 5 years.
   - Actual return = `(median_PSM_at_exit − median_PSM_at_entry) / median_PSM_at_entry`.

3. **Benchmark:**
   - The benchmark is the return on the **same town/scope** over the identical hold period
     (no signal filtering — all qualifying blocks).

4. **Signal quartiles:**
   - Blocks are ranked by discount depth into Q1–Q4 quartiles.
   - Q4 (deepest discounts) should outperform Q1 if the signal has alpha.
""")

    st.markdown("""
## 2. Look-Ahead Bias Prevention

> **Non-negotiable rule**: when evaluating a 2018-Q1 opportunity, only
> information available before 2018-Q1 is used.

- The signal computation uses a **strict cutoff** at the end of each evaluation quarter.
- Town median PSM and velocity benchmarks are re-computed fresh for each quarter.
- No price normalisation, model coefficients, or derived features use post-signal data.

**Remaining risk:** the signal thresholds themselves (e.g. "8% discount") were chosen
by the analyst after inspecting the full dataset. This is a form of **in-sample
parameter selection**, not a true walk-forward design.
""")

    st.markdown("""
## 3. Known Limitations

### 3a. Survivorship Bias (Major)
Only blocks that **have subsequent transactions** after the signal date can be measured.
Blocks that received signals but were never re-sold in the hold period are excluded.
This systematically overstates apparent returns because:
- "Good" blocks (priced fairly) tend to trade again.
- "Problem" blocks (structural defects, unpopular layout) may never re-sell,
  but their negative outcome is invisible.

### 3b. No Renovation / Condition Data (Signal Noise)
Two identical blocks can differ by $100k due to renovation quality.
Without listing condition data, the PSM discount signal conflates:
- Genuine undervaluation → real alpha
- Deferred maintenance / unappealing units → justified discount (no alpha)

### 3c. Block-Level Granularity Only
The analysis aggregates all units in a block. Within a block:
- Floor level matters (upper floors command 5–15% premium)
- Facing direction affects value
- Renovation vintage varies

A signal that fires on block-level data may not fire on the specific unit available.

### 3d. Training Set Contamination
The Opportunity Score's dimension weightings (Valuation 30%, Liquidity 20%, etc.)
were calibrated by the analyst on the **same historical dataset** being backtested.
This means the backtest is implicitly in-sample — the model "knew" which dimensions
were historically predictive when it was designed.

A truly uncontaminated test would require:
1. Fixing all signal parameters using data up to year X.
2. Running the backtest on data from year X onwards only.
3. Never revisiting parameters after seeing the out-of-sample results.

### 3e. Transaction Cost Assumptions
""")

    txn_cost = st.slider(
        "Assumed round-trip transaction cost (%)",
        min_value=0.5, max_value=5.0, value=2.5, step=0.5, key="txn_cost",
    )
    st.markdown(
        f"At **{txn_cost:.1f}%** round-trip cost, a signal generating "
        f"**+3% excess return** over 1 year becomes **+{3.0 - txn_cost:.1f}%** net — "
        f"{'positive' if (3.0 - txn_cost) > 0 else 'negative or marginal'} alpha. "
        "HDB resale costs typically include BSD (1–4%), agent fees (~1%), legal fees, "
        "and any ABSD applicable to the buyer. For a typical 4-room resale "
        "the total friction is **2–4%** of the purchase price."
    )

    st.markdown("""
### 3f. Market Efficiency Friction
Even when the signal detects a genuine discount, execution friction may prevent capture:
- **Listing availability**: the discounted block may have no units currently for sale.
- **Seller knowledge**: sellers increasingly price-match online data (HDB resale portal,
  99.co, PropertyGuru), reducing the persistence of discounts.
- **Negotiation dynamics**: a block trading at a "discount" may reflect genuine buyer
  reluctance (school catchment, neighbours, construction noise nearby) that persists.

## 4. What Future Data Would Improve the Backtest

| Data Source | Benefit |
|---|---|
| Listing data (ask price + days on market) | Filter signals by listing availability; measure time-to-liquidation |
| Unit condition / renovation ratings | Separate structural discounts from genuine alpha |
| Floor-level transaction data | Sub-block precision for signal generation |
| Monthly HDB block supply releases | Better supply-side signal |
| Walk-forward model retraining | Remove training-set contamination |

## 5. Why a "Smart" Signal Might Not Outperform in Practice

Even a theoretically sound signal may fail to generate alpha due to:
1. **Market efficiency**: professional investors and developers already exploit block-level
   PSM discounts; the edge erodes quickly.
2. **Execution friction**: 2–4% round-trip costs require sustained alpha to break even.
3. **Illiquidity**: HDB units take months to transact; by the time an offer is accepted,
   the discount may have closed.
4. **Policy risk**: cooling measures can be announced overnight, instantly altering
   expected returns across the board.
5. **Concentration risk**: a "top-scored" block may represent a single street or micro-market
   with idiosyncratic risks not captured in the aggregate signal.

---
""")

    st.warning(PAST_PERF_DISCLAIMER)
    st.warning(BACKTEST_CONFIDENCE_NOTE)
