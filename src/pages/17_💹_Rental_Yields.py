"""
Page 17 - Smart Money Rental Yields
=====================================
Combines 206k individual rental transactions (Jan 2021+) with historical
median rent data (2005-Q2+) and resale prices to compute and visualise
rental yields for Singapore HDB flats.

Tabs:
  1. Gross Yield by Town & Flat Type
  2. Yield Compression Over Time
  3. Net Yield / IRR Calculator
  4. Rental Market Explorer
  5. Buy vs Rent Decision
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

from eda_helpers import load_clean, fmt_price, fmt_pct, DATA_DIR

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rental Yields",
    page_icon="\U0001f4b9",
    layout="wide",
)

# ── constants ──────────────────────────────────────────────────────────────────
RENTAL_TX_CSV = os.path.join(DATA_DIR, "rental", "hdb_rental_transactions.csv")
MEDIAN_RENT_CSV = os.path.join(DATA_DIR, "rental", "hdb_median_rent_by_town.csv")

FLAT_TYPE_MAP = {
    "1-RM": "1 ROOM",
    "2-RM": "2 ROOM",
    "3-RM": "3 ROOM",
    "4-RM": "4 ROOM",
    "5-RM": "5 ROOM",
    "EXEC": "EXECUTIVE",
    "MULTI-GEN": "MULTI GENERATION",
}

FLAT_TYPE_ORDER = [
    "1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM",
    "5 ROOM", "EXECUTIVE", "MULTI GENERATION",
]


# ── data loaders (cached) ──────────────────────────────────────────────────────

@st.cache_data
def load_rental_tx():
    """Load HDB rental transactions (Jan 2021+)."""
    df = pd.read_csv(RENTAL_TX_CSV)
    # Normalise flat_type: "4-ROOM" -> "4 ROOM"; "EXECUTIVE" stays "EXECUTIVE"
    df["flat_type"] = df["flat_type"].str.replace("-", " ", n=1)
    df["monthly_rent"] = pd.to_numeric(df["monthly_rent"], errors="coerce")
    df["rent_approval_date"] = pd.to_datetime(df["rent_approval_date"], format="%Y-%m")
    df["year"] = df["rent_approval_date"].dt.year
    df["month_str"] = df["rent_approval_date"].dt.strftime("%Y-%m")
    # Quarter in "2021Q1" format (to match resale quarter column)
    df["quarter"] = df["rent_approval_date"].dt.to_period("Q").astype(str)
    df["town"] = df["town"].str.upper().str.strip()
    return df


@st.cache_data
def load_median_rent():
    """Load HDB median rent by town (2005-Q2+). Maps flat_type codes and drops na."""
    df = pd.read_csv(MEDIAN_RENT_CSV)
    # Normalise quarter from "2005-Q2" -> "2005Q2" to align with resale quarters
    df["quarter"] = df["quarter"].str.replace("-Q", "Q", regex=False)
    df["flat_type"] = df["flat_type"].map(FLAT_TYPE_MAP).fillna(df["flat_type"])
    df["median_rent"] = pd.to_numeric(df["median_rent"], errors="coerce")
    df = df.dropna(subset=["median_rent"])
    df["town"] = df["town"].str.upper().str.strip()
    return df


@st.cache_data
def load_resale_enriched():
    """Load cleaned resale dataset with parsed month and quarter columns."""
    df = load_clean()
    df["month_dt"] = pd.to_datetime(df["month"])
    df["year"] = df["month_dt"].dt.year
    df["quarter"] = df["month_dt"].dt.to_period("Q").astype(str)
    df["town"] = df["town"].str.upper().str.strip()
    return df


# ── helper functions ───────────────────────────────────────────────────────────

def monthly_payment_calc(principal: float, annual_rate_pct: float, tenure_years: int) -> float:
    """Standard annuity monthly payment."""
    if principal <= 0 or tenure_years <= 0:
        return 0.0
    r = annual_rate_pct / 100 / 12
    n = tenure_years * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def compute_irr(cash_flows: list) -> float:
    """Compute IRR (as %) using scipy.optimize.brentq; fallback to numpy_financial."""
    try:
        from scipy.optimize import brentq

        def _npv(rate, cfs):
            return sum(cf / (1 + rate) ** t for t, cf in enumerate(cfs))

        try:
            irr = brentq(_npv, -0.9999, 50.0, args=(cash_flows,), maxiter=1000)
            return irr * 100
        except ValueError:
            return float("nan")
    except ImportError:
        pass

    try:
        import numpy_financial as npf
        val = npf.irr(cash_flows)
        return float(val) * 100 if not math.isnan(float(val)) else float("nan")
    except (ImportError, Exception):
        pass

    return float("nan")


def compute_bsd(price: float) -> float:
    """Buyer Stamp Duty - Singapore 2024 progressive rates."""
    brackets = [
        (180_000, 0.01),
        (180_000, 0.02),
        (640_000, 0.03),
        (500_000, 0.04),
        (1_500_000, 0.05),
        (float("inf"), 0.06),
    ]
    bsd = 0.0
    remaining = price
    for band, rate in brackets:
        if remaining <= 0:
            break
        taxable = min(remaining, band)
        bsd += taxable * rate
        remaining -= taxable
    return bsd


def remaining_loan_balance(principal: float, annual_rate_pct: float,
                           monthly_pmt: float, months_paid: int) -> float:
    """Remaining loan balance after months_paid payments."""
    r = annual_rate_pct / 100 / 12
    if r > 0 and principal > 0:
        bal = principal * (1 + r) ** months_paid - \
              monthly_pmt * ((1 + r) ** months_paid - 1) / r
    elif principal > 0:
        bal = max(0.0, principal - monthly_pmt * months_paid)
    else:
        bal = 0.0
    return max(0.0, bal)


def confidence_badge(level: str, text: str) -> None:
    icons = {"High": "\U0001f7e2", "Medium": "\U0001f7e1", "Low": "\U0001f534"}
    icon = icons.get(level, "\u26aa")
    st.info(f"{icon} **Data Confidence: {level}** — {text}")


# ── page header ────────────────────────────────────────────────────────────────
st.title("\U0001f4b9 Smart Money Rental Yields")
st.caption(
    "Singapore HDB rental yield analysis — 206k individual rental transactions (Jan 2021+), "
    "historical median rents (2005-Q2+), and resale prices combined."
)

# ── load all data upfront ──────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    rental = load_rental_tx()
    med_rent_hist = load_median_rent()
    resale = load_resale_enriched()

# Pre-compute rolling 12-month cutoffs (used across multiple tabs)
max_rent_date = rental["rent_approval_date"].max()
cutoff_rent = max_rent_date - pd.DateOffset(months=11)

max_resale_date = resale["month_dt"].max()
cutoff_resale = max_resale_date - pd.DateOffset(months=11)

# 12-month slices
rental_12m = rental[rental["rent_approval_date"] >= cutoff_rent]
resale_12m = resale[resale["month_dt"] >= cutoff_resale]

# Median monthly rent by (town, flat_type) — last 12 months of rental tx
med_rent_12m = (
    rental_12m.groupby(["town", "flat_type"])["monthly_rent"]
    .median()
    .reset_index()
    .rename(columns={"monthly_rent": "median_rent_12m"})
)

# Median resale price by (town, flat_type) — last 12 months
med_price_12m = (
    resale_12m.groupby(["town", "flat_type"])["resale_price"]
    .median()
    .reset_index()
    .rename(columns={"resale_price": "median_price_12m"})
)

# Merged yield table
yield_df = med_rent_12m.merge(med_price_12m, on=["town", "flat_type"], how="inner")
yield_df = yield_df.dropna(subset=["median_rent_12m", "median_price_12m"])
yield_df["gross_yield"] = (
    yield_df["median_rent_12m"] * 12 / yield_df["median_price_12m"] * 100
)

# ── tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "\U0001f4b9 Gross Yield by Town & Type",
    "\U0001f4c9 Yield Compression Over Time",
    "\U0001f4b0 Net Yield / IRR Calculator",
    "\U0001f5fa\ufe0f Rental Market Explorer",
    "\u2696\ufe0f Buy vs Rent Decision",
    "\U0001f4ca Real Returns (CPI-adj)",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Gross Yield by Town & Flat Type
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("\U0001f4b9 Gross Rental Yield by Town & Flat Type")
    st.write(
        "**Gross yield = (Median Annual Rent / Median Resale Price) \xd7 100.** "
        "Based on the most recent 12 months of rental transactions and resale data."
    )

    if yield_df.empty:
        st.warning("Insufficient data to compute yields. Check data files.")
    else:
        # ── KPI cards ──────────────────────────────────────────────────────
        best_row = yield_df.loc[yield_df["gross_yield"].idxmax()]
        avg_yield = yield_df["gross_yield"].mean()

        # Highest-price town (by median resale in last 12m, across flat types)
        hp_town = (
            med_price_12m.groupby("town")["median_price_12m"].max().idxmax()
        )
        hp_yield_vals = yield_df[yield_df["town"] == hp_town]["gross_yield"]
        hp_yield = hp_yield_vals.mean() if not hp_yield_vals.empty else float("nan")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "\U0001f3c6 Best Yield Combo",
            f"{best_row['gross_yield']:.2f}%",
            f"{best_row['town']} \u00b7 {best_row['flat_type']}",
        )
        c2.metric("\U0001f4ca Avg HDB Gross Yield", f"{avg_yield:.2f}%")
        c3.metric(
            "\U0001f3d9\ufe0f Highest-Price Town Yield",
            f"{hp_yield:.2f}%" if not math.isnan(hp_yield) else "N/A",
            hp_town,
        )

        # ── heatmap ────────────────────────────────────────────────────────
        st.subheader("Gross Yield Heatmap — Town \xd7 Flat Type")
        pivot = yield_df.pivot_table(
            index="town", columns="flat_type", values="gross_yield", aggfunc="mean"
        )
        col_order = [c for c in FLAT_TYPE_ORDER if c in pivot.columns]
        pivot = pivot[col_order].sort_index()

        if not pivot.empty:
            fig_heat = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                title="Gross Rental Yield % (Last 12 Months)",
                labels={"color": "Gross Yield %"},
                text_auto=".1f",
            )
            fig_heat.update_layout(
                height=600,
                coloraxis_colorbar=dict(title="Yield %"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── top-15 bar chart ───────────────────────────────────────────────
        st.subheader("\U0001f3c6 Top 15 Town + Flat Type by Gross Yield")
        top15 = yield_df.nlargest(15, "gross_yield").copy()
        top15["label"] = top15["town"] + " \u00b7 " + top15["flat_type"]
        top15_sorted = top15.sort_values("gross_yield")

        fig_bar = px.bar(
            top15_sorted,
            x="gross_yield",
            y="label",
            orientation="h",
            color="gross_yield",
            color_continuous_scale="RdYlGn",
            text=top15_sorted["gross_yield"].map(lambda x: f"{x:.2f}%"),
            title="Top 15 Gross Yield Combinations",
            labels={"gross_yield": "Gross Yield %", "label": ""},
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    confidence_badge(
        "Medium",
        "Gross yield assumes the flat is rented continuously at the current median market rent "
        "and purchased at the current median resale price. It excludes: vacancy, maintenance costs "
        "($200\u2013800/month), property tax (~4\u201316% of Annual Value), agent fees "
        "(0.5\u20131 month\u2019s rent), and financing costs. True net yield is typically 1\u20132% lower.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Yield Compression Over Time
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("\U0001f4c9 Yield Compression Over Time")
    st.write(
        "How gross yields have changed as resale prices rose faster than rents (2005-Q2 to present)."
    )

    # Quarterly median resale price by (quarter, town, flat_type)
    q_resale = (
        resale.groupby(["quarter", "town", "flat_type"])["resale_price"]
        .median()
        .reset_index()
        .rename(columns={"resale_price": "median_resale_price"})
    )

    # Merge with historical median rent (quarters already aligned to "2005Q2" format)
    hist = med_rent_hist.merge(
        q_resale, on=["quarter", "town", "flat_type"], how="inner"
    )
    hist = hist.dropna(subset=["median_rent", "median_resale_price"])
    hist["gross_yield"] = hist["median_rent"] * 12 / hist["median_resale_price"] * 100

    # Convert quarter string back to timestamp for plotting
    hist["quarter_dt"] = pd.PeriodIndex(hist["quarter"], freq="Q").to_timestamp()
    hist = hist.sort_values("quarter_dt")

    # ── national average yield over time ──────────────────────────────────
    st.subheader("National Average Gross Yield Over Time")
    nat_yield = (
        hist.groupby("quarter_dt")["gross_yield"]
        .mean()
        .reset_index()
        .rename(columns={"gross_yield": "avg_gross_yield"})
    )

    if not nat_yield.empty:
        recent_avg = nat_yield["avg_gross_yield"].iloc[-8:].mean()
        fig_nat = px.line(
            nat_yield,
            x="quarter_dt",
            y="avg_gross_yield",
            title="National Average HDB Gross Rental Yield (2005-Q2 to Present)",
            labels={"quarter_dt": "Quarter", "avg_gross_yield": "Avg Gross Yield (%)"},
        )
        fig_nat.update_traces(line_color="#2E86AB", line_width=2)
        fig_nat.add_hline(
            y=recent_avg,
            line_dash="dash",
            line_color="crimson",
            annotation_text=f"Recent avg: {recent_avg:.1f}%",
            annotation_position="bottom right",
        )
        fig_nat.update_layout(height=420)
        st.plotly_chart(fig_nat, use_container_width=True)

        # Key insight
        early_mask = nat_yield["quarter_dt"].dt.year.between(2010, 2013)
        recent_mask = nat_yield["quarter_dt"].dt.year >= 2023
        early_avg = nat_yield.loc[early_mask, "avg_gross_yield"].mean()
        recent_avg2 = nat_yield.loc[recent_mask, "avg_gross_yield"].mean()
        if not math.isnan(early_avg) and not math.isnan(recent_avg2):
            st.info(
                f"\U0001f4ca **Key Insight:** Gross yields compressed from "
                f"**{early_avg:.1f}%** (2010\u20132013) to **{recent_avg2:.1f}%** "
                f"(2023\u2013present) as resale prices outpaced rental growth."
            )

    # ── yield by flat type — last 10 years ────────────────────────────────
    st.subheader("Gross Yield by Flat Type (Last 10 Years)")
    cutoff_10y = hist["quarter_dt"].max() - pd.DateOffset(years=10)
    hist_10y = hist[hist["quarter_dt"] >= cutoff_10y]
    yield_by_type = (
        hist_10y.groupby(["quarter_dt", "flat_type"])["gross_yield"]
        .mean()
        .reset_index()
    )
    type_order_present = [t for t in FLAT_TYPE_ORDER if t in yield_by_type["flat_type"].unique()]

    if not yield_by_type.empty:
        fig_types = px.line(
            yield_by_type,
            x="quarter_dt",
            y="gross_yield",
            color="flat_type",
            category_orders={"flat_type": type_order_present},
            title="Gross Yield by Flat Type (Last 10 Years)",
            labels={
                "quarter_dt": "Quarter",
                "gross_yield": "Gross Yield (%)",
                "flat_type": "Flat Type",
            },
        )
        fig_types.update_layout(height=420)
        st.plotly_chart(fig_types, use_container_width=True)

    # ── scatter: rent growth vs price growth 2021-now ─────────────────────
    st.subheader("Rent Growth vs Price Growth by Town (2021\u2013Now)")
    st.caption("Towns above the diagonal have rents rising faster than prices (yield-supportive).")

    yr_start_scatter = 2021
    yr_end_scatter = resale["year"].max()

    rent_grp = (
        rental[rental["year"].isin([yr_start_scatter, yr_end_scatter])]
        .groupby(["town", "year"])["monthly_rent"]
        .median()
        .reset_index()
    )
    rent_piv = rent_grp.pivot(index="town", columns="year", values="monthly_rent").dropna()

    price_grp = (
        resale[resale["year"].isin([yr_start_scatter, yr_end_scatter])]
        .groupby(["town", "year"])["resale_price"]
        .median()
        .reset_index()
    )
    price_piv = price_grp.pivot(index="town", columns="year", values="resale_price").dropna()

    if (
        not rent_piv.empty
        and not price_piv.empty
        and yr_start_scatter in rent_piv.columns
        and yr_end_scatter in rent_piv.columns
        and yr_start_scatter in price_piv.columns
        and yr_end_scatter in price_piv.columns
        and yr_end_scatter != yr_start_scatter
    ):
        n_yrs = yr_end_scatter - yr_start_scatter
        rent_piv["rent_growth_pct"] = (
            (rent_piv[yr_end_scatter] / rent_piv[yr_start_scatter]) ** (1 / n_yrs) - 1
        ) * 100
        price_piv["price_growth_pct"] = (
            (price_piv[yr_end_scatter] / price_piv[yr_start_scatter]) ** (1 / n_yrs) - 1
        ) * 100

        scatter_df = (
            rent_piv[["rent_growth_pct"]]
            .reset_index()
            .merge(price_piv[["price_growth_pct"]].reset_index(), on="town")
            .dropna()
        )

        if not scatter_df.empty:
            all_vals = pd.concat(
                [scatter_df["price_growth_pct"], scatter_df["rent_growth_pct"]]
            )
            v_min, v_max = all_vals.min() - 1, all_vals.max() + 1

            fig_sc = px.scatter(
                scatter_df,
                x="price_growth_pct",
                y="rent_growth_pct",
                text="town",
                title=(
                    f"Avg Annual Rent Growth vs Price Growth by Town "
                    f"({yr_start_scatter}\u2013{yr_end_scatter})"
                ),
                labels={
                    "price_growth_pct": "Annual Price Growth (%)",
                    "rent_growth_pct": "Annual Rent Growth (%)",
                },
                color="rent_growth_pct",
                color_continuous_scale="RdYlGn",
            )
            fig_sc.add_shape(
                type="line",
                x0=v_min, y0=v_min, x1=v_max, y1=v_max,
                line=dict(dash="dash", color="gray", width=1.5),
            )
            fig_sc.add_annotation(
                x=v_max, y=v_max,
                text="Equal growth line",
                showarrow=False,
                font=dict(color="gray", size=11),
                xshift=-80,
                yshift=10,
            )
            fig_sc.update_traces(textposition="top center", marker=dict(size=10))
            fig_sc.update_layout(height=520, coloraxis_showscale=False)
            st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Insufficient data to compute town-level rent vs price growth comparison.")

    confidence_badge(
        "Medium",
        "Historical yields use HDB published quarterly median rents merged with median resale "
        "prices per quarter. Coverage varies by town/type; sparse cells are excluded.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Net Yield / IRR Calculator
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("\U0001f4b0 Net Yield / IRR Calculator")
    st.write(
        "Full net yield and IRR calculation for a user-specified flat. "
        "Pre-filled with current market medians for the selected town and flat type."
    )

    all_towns_sorted = sorted(resale["town"].dropna().unique().tolist())
    all_ft_sorted = [ft for ft in FLAT_TYPE_ORDER if ft in resale["flat_type"].unique()]

    col_s1, col_s2 = st.columns(2)
    sel_town_irr = col_s1.selectbox("Town", all_towns_sorted, key="irr_town_sel")
    sel_type_irr = col_s2.selectbox("Flat Type", all_ft_sorted, key="irr_type_sel")

    # Pre-fill defaults from market medians
    mask_r12 = (
        (resale["town"] == sel_town_irr)
        & (resale["flat_type"] == sel_type_irr)
        & (resale["month_dt"] >= cutoff_resale)
    )
    def_price_raw = resale.loc[mask_r12, "resale_price"].median()
    def_price = int(def_price_raw) if not math.isnan(float(def_price_raw)) else 500_000

    mask_rnt12 = (
        (rental["town"] == sel_town_irr)
        & (rental["flat_type"] == sel_type_irr)
        & (rental["rent_approval_date"] >= cutoff_rent)
    )
    def_rent_raw = rental.loc[mask_rnt12, "monthly_rent"].median()
    def_rent = int(def_rent_raw) if not math.isnan(float(def_rent_raw)) else 2_500

    with st.form("irr_calc_form"):
        st.subheader("Property & Rental Inputs")
        fi1, fi2 = st.columns(2)
        purchase_price = fi1.number_input(
            "Purchase Price ($)", min_value=100_000, max_value=5_000_000,
            value=def_price, step=10_000,
        )
        monthly_rent_inp = fi2.number_input(
            "Monthly Rent ($)", min_value=500, max_value=20_000,
            value=def_rent, step=50,
        )

        st.subheader("Annual Cost Assumptions")
        fc1, fc2, fc3 = st.columns(3)
        vacancy_pct = fc1.slider("Vacancy (%)", 0.0, 20.0, 8.0, step=0.5)
        annual_maint = fc2.number_input(
            "Annual Maintenance ($)", min_value=0, max_value=20_000,
            value=2_400, step=100,
        )
        agent_months = fc3.slider(
            "Agent Fee (months/yr)", 0.0, 2.0, 1.0, step=0.5,
            help="Typical: 1 month per year (split across tenancy renewals)",
        )

        ft1, ft2 = st.columns(2)
        prop_tax_pct = ft1.slider(
            "Property Tax (% of Annual Value)", 2.0, 16.0, 10.0, step=0.5,
            help="Annual Value \u2248 70% of gross annual rent. Owner-occupied rates are lower.",
        )
        use_loan = ft2.checkbox("Use financing (mortgage loan)?", value=False)

        if use_loan:
            fl1, fl2, fl3 = st.columns(3)
            down_pct = fl1.slider("Down Payment (%)", 10, 50, 25, step=5)
            loan_rate = fl2.slider("Interest Rate (% p.a.)", 1.0, 6.0, 2.6, step=0.1)
            loan_tenure = fl3.slider("Loan Tenure (years)", 5, 30, 25, step=1)
        else:
            down_pct, loan_rate, loan_tenure = 100, 0.0, 1

        st.subheader("Investment Horizon & Growth")
        fh1, fh2 = st.columns(2)
        holding_years = fh1.slider("Holding Period (years)", 3, 30, 5)
        price_growth = fh2.slider(
            "Assumed Annual Price Growth (%)", -3.0, 10.0, 3.0, step=0.5,
        )

        irr_submitted = st.form_submit_button("\U0001f4ca Calculate", type="primary")

    # Always compute (shows defaults on first load; updates on submit)
    # ── core computations ────────────────────────────────────────────────────
    gross_annual_rent = monthly_rent_inp * 12 * (1 - vacancy_pct / 100)
    annual_value = monthly_rent_inp * 12 * 0.7          # AV \u2248 70% of gross annual rent
    property_tax = prop_tax_pct / 100 * annual_value
    agent_fee_annual = agent_months * monthly_rent_inp
    annual_costs = annual_maint + property_tax + agent_fee_annual
    noi = gross_annual_rent - annual_costs

    gross_yield_calc = monthly_rent_inp * 12 / purchase_price * 100
    net_yield_calc = noi / purchase_price * 100

    down_payment = purchase_price * down_pct / 100
    loan_amount = purchase_price - down_payment if use_loan else 0.0
    monthly_mort = monthly_payment_calc(loan_amount, loan_rate, loan_tenure) if use_loan else 0.0
    annual_mort = monthly_mort * 12

    cash_on_cash = (
        (noi - annual_mort) / down_payment * 100
        if use_loan and down_payment > 0
        else None
    )

    bsd_calc = compute_bsd(purchase_price)
    exit_value = purchase_price * (1 + price_growth / 100) ** holding_years
    sale_agent_fee = exit_value * 0.02
    net_sale_proceeds = exit_value - sale_agent_fee

    # Build annual cash flows for IRR
    initial_outflow = -(down_payment + bsd_calc) if use_loan else -(purchase_price + bsd_calc)
    annual_net_cf = noi - annual_mort

    cf_list = [initial_outflow]
    for yr in range(1, holding_years + 1):
        if yr < holding_years:
            cf_list.append(annual_net_cf)
        else:
            # Final year: operating income + net sale proceeds - remaining loan
            if use_loan and loan_amount > 0:
                rem_bal = remaining_loan_balance(
                    loan_amount, loan_rate, monthly_mort,
                    min(yr, loan_tenure) * 12,
                )
                final_cf = annual_net_cf + net_sale_proceeds - purchase_price - rem_bal
            else:
                final_cf = annual_net_cf + net_sale_proceeds - purchase_price
            cf_list.append(final_cf)

    irr_result = compute_irr(cf_list)

    # Breakeven rent: monthly_rent such that NOI = 0
    # NOI = r*12*(1-vac) - tax_pct/100 * r*12*0.7 - agent_m*r - maint = 0
    rent_coeff = (
        12 * (1 - vacancy_pct / 100)
        - prop_tax_pct / 100 * 12 * 0.7
        - agent_months
    )
    breakeven_rent = annual_maint / rent_coeff if rent_coeff > 0 else float("nan")

    # ── KPI metrics ──────────────────────────────────────────────────────────
    st.subheader("Results")
    km1, km2, km3 = st.columns(3)
    km1.metric("\U0001f4c8 Gross Yield", f"{gross_yield_calc:.2f}%")
    km2.metric("\U0001f4b0 Net Yield", f"{net_yield_calc:.2f}%")
    if cash_on_cash is not None:
        km3.metric("\U0001f3e6 Cash-on-Cash Return", f"{cash_on_cash:.2f}%")
    else:
        km3.metric(
            "\U0001f3e6 IRR",
            f"{irr_result:.2f}%" if not math.isnan(irr_result) else "N/A",
        )

    km4, km5, km6 = st.columns(3)
    km4.metric("\U0001f3e0 Est. Exit Value", fmt_price(exit_value))
    km5.metric("\U0001f4ca Annual NOI", fmt_price(noi))
    km6.metric(
        "\u2696\ufe0f Breakeven Monthly Rent",
        fmt_price(breakeven_rent) if not math.isnan(breakeven_rent) else "N/A",
        help="Monthly rent at which net operating income turns positive",
    )

    if use_loan:
        st.metric(
            f"IRR over {holding_years}-year holding period",
            f"{irr_result:.2f}%" if not math.isnan(irr_result) else "N/A",
        )

    # ── summary boxes ─────────────────────────────────────────────────────────
    with st.expander("Annual Cost Breakdown"):
        cb1, cb2, cb3, cb4 = st.columns(4)
        cb1.metric("Gross Rent", fmt_price(gross_annual_rent))
        cb2.metric("Maintenance", fmt_price(annual_maint))
        cb3.metric("Property Tax", fmt_price(property_tax))
        cb4.metric("Agent Fees", fmt_price(agent_fee_annual))
        if use_loan:
            st.metric("Annual Mortgage", fmt_price(annual_mort))

    # ── year-by-year cash flow table ─────────────────────────────────────────
    st.subheader("Year-by-Year Cash Flow")
    cf_rows = []
    cumulative_net = initial_outflow
    for yr_i in range(1, holding_years + 1):
        prop_val_yr = purchase_price * (1 + price_growth / 100) ** yr_i
        yr_cf = annual_net_cf
        cumulative_net += yr_cf
        cf_rows.append({
            "Year": yr_i,
            "Rent Income ($)": f"{gross_annual_rent:,.0f}",
            "All Costs ($)": f"{annual_costs + annual_mort:,.0f}",
            "Net Cash Flow ($)": f"{yr_cf:,.0f}",
            "Cumulative CF ($)": f"{cumulative_net:,.0f}",
            "Property Value ($)": f"{prop_val_yr:,.0f}",
        })
    st.dataframe(pd.DataFrame(cf_rows), use_container_width=True, hide_index=True)

    # Cash flow chart
    cf_chart_df = pd.DataFrame({
        "Year": list(range(1, holding_years + 1)),
        "Annual Net CF": [annual_net_cf] * holding_years,
    })
    cf_chart_df["Cumulative"] = cf_chart_df["Annual Net CF"].cumsum() + initial_outflow

    fig_cf = go.Figure()
    fig_cf.add_bar(
        x=cf_chart_df["Year"], y=cf_chart_df["Annual Net CF"],
        name="Annual Net CF", marker_color="#2E86AB",
    )
    fig_cf.add_scatter(
        x=cf_chart_df["Year"], y=cf_chart_df["Cumulative"],
        name="Cumulative CF", line=dict(color="darkorange", width=2.5),
        mode="lines+markers",
    )
    fig_cf.add_hline(y=0, line_dash="dash", line_color="crimson", line_width=1)
    fig_cf.update_layout(
        title="Cash Flow Over Holding Period",
        xaxis_title="Year",
        yaxis_title="$ (SGD)",
        height=380,
        barmode="relative",
    )
    st.plotly_chart(fig_cf, use_container_width=True)

    confidence_badge(
        "Medium",
        "Calculator uses simplified assumptions. Property tax, vacancy, and maintenance estimates "
        "are illustrative. Consult a licensed financial advisor and IRAS for actual tax obligations.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Rental Market Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("\U0001f5fa\ufe0f Rental Market Explorer")
    st.write("Explore individual HDB rental approval records (Jan 2021\u2013present).")

    exp_c1, exp_c2 = st.columns(2)
    exp_town = exp_c1.selectbox(
        "Town",
        ["All Towns"] + sorted(rental["town"].dropna().unique().tolist()),
        key="exp_town_sel",
    )
    exp_type = exp_c2.selectbox(
        "Flat Type",
        ["All Types"] + [ft for ft in FLAT_TYPE_ORDER if ft in rental["flat_type"].unique()],
        key="exp_type_sel",
    )

    rental_date_min = rental["rent_approval_date"].min().to_pydatetime()
    rental_date_max = rental["rent_approval_date"].max().to_pydatetime()
    exp_d1, exp_d2 = st.columns(2)
    date_from = exp_d1.date_input(
        "From (month)", value=rental_date_min,
        min_value=rental_date_min, max_value=rental_date_max, key="exp_from",
    )
    date_to = exp_d2.date_input(
        "To (month)", value=rental_date_max,
        min_value=rental_date_min, max_value=rental_date_max, key="exp_to",
    )

    exp_df = rental.copy()
    if exp_town != "All Towns":
        exp_df = exp_df[exp_df["town"] == exp_town]
    if exp_type != "All Types":
        exp_df = exp_df[exp_df["flat_type"] == exp_type]
    exp_df = exp_df[
        (exp_df["rent_approval_date"] >= pd.Timestamp(date_from))
        & (exp_df["rent_approval_date"] <= pd.Timestamp(date_to))
    ]

    st.caption(f"**{len(exp_df):,}** rental approval records match your filters.")

    if exp_df.empty:
        st.warning("No records match the selected filters.")
    else:
        # ── distribution + trend ───────────────────────────────────────────
        dist_col, trend_col = st.columns(2)

        with dist_col:
            st.subheader("Rent Distribution")
            q1_r = exp_df["monthly_rent"].quantile(0.25)
            q3_r = exp_df["monthly_rent"].quantile(0.75)
            fig_hist = px.histogram(
                exp_df, x="monthly_rent", nbins=60,
                title="Distribution of Monthly Rents",
                labels={"monthly_rent": "Monthly Rent (SGD)"},
                color_discrete_sequence=["#2E86AB"],
            )
            fig_hist.add_vline(
                x=exp_df["monthly_rent"].median(),
                line_dash="dash", line_color="crimson",
                annotation_text=f"Median: ${exp_df['monthly_rent'].median():,.0f}",
            )
            fig_hist.update_layout(height=360)
            st.plotly_chart(fig_hist, use_container_width=True)

        with trend_col:
            st.subheader("Median Rent Trend Over Time")
            rent_trend = (
                exp_df.groupby("month_str")["monthly_rent"]
                .median()
                .reset_index()
                .rename(columns={"monthly_rent": "median_rent"})
                .sort_values("month_str")
            )
            fig_trend = px.line(
                rent_trend, x="month_str", y="median_rent",
                title="Median Monthly Rent Over Time",
                labels={"month_str": "Month", "median_rent": "Median Rent (SGD)"},
            )
            fig_trend.update_traces(line_color="#E84855", line_width=2)
            fig_trend.update_layout(height=360)
            st.plotly_chart(fig_trend, use_container_width=True)

        # ── top 10 most active blocks ──────────────────────────────────────
        st.subheader("\U0001f3d8\ufe0f Top 10 Most Active Rental Blocks")
        top_blocks = (
            exp_df.groupby(["block", "street_name"])
            .size()
            .reset_index(name="rental_count")
            .nlargest(10, "rental_count")
        )
        top_blocks["address"] = (
            top_blocks["block"].astype(str) + " " + top_blocks["street_name"]
        )
        top_blocks_sorted = top_blocks.sort_values("rental_count")
        fig_blocks = px.bar(
            top_blocks_sorted,
            x="rental_count", y="address", orientation="h",
            title="Top 10 Blocks by Rental Volume (Selected Filters)",
            labels={"rental_count": "No. of Rental Approvals", "address": ""},
            color_discrete_sequence=["#F7B731"],
        )
        fig_blocks.update_layout(height=380)
        st.plotly_chart(fig_blocks, use_container_width=True)

        # ── market depth ───────────────────────────────────────────────────
        st.subheader("\U0001f4ca Market Depth — Rental Approvals per Month")
        depth = (
            exp_df.groupby("month_str")
            .size()
            .reset_index(name="approvals")
            .sort_values("month_str")
        )
        fig_depth = px.bar(
            depth, x="month_str", y="approvals",
            title="Number of Rental Approvals per Month (Selected Filters)",
            labels={"month_str": "Month", "approvals": "No. of Approvals"},
            color_discrete_sequence=["#20BF6B"],
        )
        fig_depth.update_layout(height=360)
        st.plotly_chart(fig_depth, use_container_width=True)

    confidence_badge(
        "High",
        "Individual rental approval records from HDB (owner-declared; HDB does not verify "
        "rent amounts). Records reflect registered tenancies, not private sub-lettings.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Buy vs Rent Decision
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("\u2696\ufe0f Buy vs Rent Decision Calculator")
    st.write(
        "At what holding period does buying beat renting the same flat? "
        "Accounts for equity build-up, opportunity cost of down payment, and rent inflation."
    )

    # Overall market medians for default pre-fill
    overall_price_med_raw = resale_12m["resale_price"].median()
    overall_price_med = int(overall_price_med_raw) if not math.isnan(float(overall_price_med_raw)) else 500_000

    overall_rent_med_raw = rental_12m["monthly_rent"].median()
    overall_rent_med = int(overall_rent_med_raw) if not math.isnan(float(overall_rent_med_raw)) else 2_500

    with st.form("bvr_calc_form"):
        st.subheader("Property & Financing")
        bv1, bv2 = st.columns(2)
        bvr_price = bv1.number_input(
            "Purchase Price ($)", min_value=100_000, max_value=5_000_000,
            value=overall_price_med, step=10_000,
        )
        bvr_eq_rent = bv2.number_input(
            "Equivalent Monthly Rent (renting same flat, $)",
            min_value=500, max_value=20_000,
            value=overall_rent_med, step=50,
        )

        bv3, bv4 = st.columns(2)
        bvr_down_pct = bv3.slider("Down Payment (%)", 10, 50, 25, step=5)
        bvr_rate = bv4.slider("Mortgage Rate (% p.a.)", 1.0, 6.0, 2.6, step=0.1)

        bv5, bv6 = st.columns(2)
        bvr_tenure_yr = bv5.slider("Loan Tenure (years)", 5, 30, 25)
        bvr_horizon = bv6.slider("Analysis Horizon (years)", 3, 30, 15)

        st.subheader("Growth & Cost Assumptions")
        bg1, bg2 = st.columns(2)
        bvr_price_gr = bg1.slider("Annual Price Appreciation (%)", -3.0, 10.0, 3.0, step=0.5)
        bvr_rent_inf = bg2.slider("Annual Rent Inflation (%)", 0.0, 8.0, 3.0, step=0.5)

        bc1, bc2 = st.columns(2)
        bvr_maint_yr = bc1.number_input(
            "Annual Maintenance ($)", min_value=0, max_value=10_000,
            value=2_400, step=100,
        )
        bvr_opp_rate = bc2.slider(
            "Opportunity Cost of Down Payment (% p.a.)", 0.0, 8.0, 3.5, step=0.5,
            help="Expected annual return if down payment was invested in a portfolio instead",
        )

        bvr_submit = st.form_submit_button("\U0001f4ca Compute Breakeven", type="primary")

    # ── computations ─────────────────────────────────────────────────────────
    bvr_down_amt = bvr_price * bvr_down_pct / 100
    bvr_loan_amt = bvr_price - bvr_down_amt
    bvr_bsd = compute_bsd(bvr_price)
    bvr_monthly_mort = monthly_payment_calc(bvr_loan_amt, bvr_rate, bvr_tenure_yr)
    bvr_annual_mort = bvr_monthly_mort * 12

    buy_cumulative_costs = []
    rent_cumulative_costs = []
    equity_list = []
    breakeven_yr = None

    cum_buy = bvr_down_amt + bvr_bsd   # initial cash outlay
    cum_rent = 0.0
    curr_rent = float(bvr_eq_rent)
    curr_prop_val = float(bvr_price)

    for yr_bvr in range(1, bvr_horizon + 1):
        # Property value grows
        curr_prop_val *= 1 + bvr_price_gr / 100

        # Annual buying running costs
        ann_prop_tax_bvr = curr_rent * 12 * 0.7 * 0.10   # 10% of AV
        ann_buy_running = bvr_annual_mort + bvr_maint_yr + ann_prop_tax_bvr
        cum_buy += ann_buy_running

        # Remaining loan balance
        paid_months_bvr = min(yr_bvr, bvr_tenure_yr) * 12
        rem_bal_bvr = remaining_loan_balance(
            bvr_loan_amt, bvr_rate, bvr_monthly_mort, paid_months_bvr
        )
        equity_bvr = curr_prop_val - rem_bal_bvr

        # Opportunity cost of the initial down payment
        opp_cost_bvr = bvr_down_amt * ((1 + bvr_opp_rate / 100) ** yr_bvr - 1)

        # Net cost of buying = cumulative cash spent - equity gained above down pmt + opp cost
        equity_above_down = max(0.0, equity_bvr - bvr_down_amt)
        net_buy_cost = cum_buy - equity_above_down + opp_cost_bvr

        # Annual renting cost (rent inflates each year)
        cum_rent += curr_rent * 12
        curr_rent *= 1 + bvr_rent_inf / 100

        buy_cumulative_costs.append(net_buy_cost)
        rent_cumulative_costs.append(cum_rent)
        equity_list.append(equity_bvr)

        if breakeven_yr is None and net_buy_cost <= cum_rent:
            breakeven_yr = yr_bvr

    bvr_chart_df = pd.DataFrame({
        "Year": list(range(1, bvr_horizon + 1)),
        "Net Cost of Buying": buy_cumulative_costs,
        "Cost of Renting": rent_cumulative_costs,
        "Property Equity": equity_list,
    })

    fig_bvr = go.Figure()
    fig_bvr.add_scatter(
        x=bvr_chart_df["Year"], y=bvr_chart_df["Net Cost of Buying"],
        name="Net Cost of Buying (after equity)",
        line=dict(color="#2E86AB", width=2.5), mode="lines+markers",
    )
    fig_bvr.add_scatter(
        x=bvr_chart_df["Year"], y=bvr_chart_df["Cost of Renting"],
        name="Cumulative Cost of Renting",
        line=dict(color="#E84855", width=2.5), mode="lines+markers",
    )
    fig_bvr.add_scatter(
        x=bvr_chart_df["Year"], y=bvr_chart_df["Property Equity"],
        name="Property Equity",
        line=dict(color="#20BF6B", width=1.5, dash="dot"), mode="lines",
    )

    if breakeven_yr is not None:
        fig_bvr.add_vline(
            x=breakeven_yr, line_dash="dash", line_color="gold", line_width=2.5,
            annotation_text=f"Breakeven: Year {breakeven_yr}",
            annotation_position="top right",
            annotation_font_color="gold",
        )

    fig_bvr.update_layout(
        title="Buy vs Rent: Cumulative Cost Comparison",
        xaxis_title="Years Held",
        yaxis_title="Cumulative Cost (SGD)",
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bvr, use_container_width=True)

    # ── summary metrics ───────────────────────────────────────────────────────
    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Monthly Mortgage", fmt_price(bvr_monthly_mort))
    bm2.metric("BSD (Stamp Duty)", fmt_price(bvr_bsd))
    bm3.metric("Down Payment", fmt_price(bvr_down_amt))
    if breakeven_yr is not None:
        bm4.metric("\u2696\ufe0f Breakeven", f"Year {breakeven_yr}")
    else:
        bm4.metric(
            "\u2696\ufe0f Breakeven",
            f">{bvr_horizon} yrs",
            delta="Renting cheaper in this horizon",
            delta_color="inverse",
        )

    if breakeven_yr is not None:
        st.success(
            f"\U0001f3e0 **Buying beats renting after Year {breakeven_yr}** under these assumptions. "
            f"After the breakeven, the cumulative cost of buying (net of equity) falls below "
            f"cumulative rent paid."
        )
    else:
        st.warning(
            f"\U0001f4cb **Renting remains cheaper throughout the {bvr_horizon}-year horizon** "
            f"under these assumptions. Try increasing price appreciation or decreasing rent inflation."
        )

    with st.expander("Methodology Notes"):
        st.markdown(
            """
**Net Cost of Buying** = cumulative mortgage payments + maintenance + property tax
\u2212 equity gained above the down payment + opportunity cost of down payment invested elsewhere.

**Cost of Renting** = cumulative rent paid (rent grows at the inflation rate you specify).

**Breakeven year** = first year where Net Cost of Buying \u2264 Cost of Renting.

**Not included:** CPF Ordinary Account usage, HDB grants (which reduce effective purchase price),
ABSD (if applicable), Seller Stamp Duty on early exit, sub-letting income, renovation costs.
            """
        )

    confidence_badge(
        "Medium",
        "Simplified model for illustration. Actual outcomes depend on financing structure, "
        "CPF usage, tax obligations, market conditions, and individual circumstances. "
        "Always consult a licensed financial advisor before transacting.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Real Returns (CPI-Adjusted) (E4)
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("\U0001f4ca CPI-Adjusted Real Returns (E4)")
    st.caption(
        "Are HDB resale buyers actually beating inflation, or just keeping up in nominal terms? "
        "This tab adjusts historical HDB PSM growth against CPI to show real purchasing power changes."
    )

    from eda_helpers import load_cpi  # import here to avoid circular dependency issues

    cpi_df = load_cpi()

    if cpi_df.empty:
        st.warning(
            "CPI data not yet fetched. Run `python src/fetch_data.py` from the repo root to download it."
        )
    else:
        @st.cache_data
        def compute_real_returns(_resale_df, _cpi_df):
            # Annual HDB median PSM by flat type
            annual_psm = (_resale_df.groupby(["year", "flat_type"])["price_per_sqm"]
                          .median().reset_index())
            annual_psm = annual_psm[annual_psm["year"] >= 2000]

            # CPI All Items \u2014 annual average
            all_items = _cpi_df[_cpi_df["series"] == "All Items"].copy()
            all_items["year"] = pd.to_datetime(all_items["date"]).dt.year
            cpi_annual = all_items.groupby("year")["cpi_index"].mean().reset_index()

            # Merge + compute real PSM (nominal / cpi_index * 100)
            merged = annual_psm.merge(cpi_annual, on="year", how="left")
            merged["real_psm"] = merged["price_per_sqm"] / merged["cpi_index"] * 100
            return merged

        resale_for_cpi = resale
        real_df = compute_real_returns(resale_for_cpi, cpi_df)

        e4_col1, e4_col2 = st.columns([1, 3])
        with e4_col1:
            e4_flat = st.selectbox("Flat type", ["All types"] + sorted(real_df["flat_type"].unique()),
                                   key="e4_flat")
            e4_base_yr = st.slider("Base year (index = 100)", 2000, 2020, 2010, key="e4_base")

        plot_df = real_df.copy()
        if e4_flat != "All types":
            plot_df = plot_df[plot_df["flat_type"] == e4_flat]

        # Rebase to base year
        base_vals = plot_df[plot_df["year"] == e4_base_yr].set_index("flat_type")[["price_per_sqm", "real_psm"]]

        def rebase(row):
            ft = row["flat_type"]
            if ft in base_vals.index:
                row["nominal_idx"] = row["price_per_sqm"] / base_vals.loc[ft, "price_per_sqm"] * 100
                row["real_idx"] = row["real_psm"] / base_vals.loc[ft, "real_psm"] * 100
            else:
                row["nominal_idx"] = float("nan")
                row["real_idx"] = float("nan")
            return row

        plot_df = plot_df.apply(rebase, axis=1)

        with e4_col2:
            fig_real = go.Figure()
            for ft in plot_df["flat_type"].unique():
                sub = plot_df[plot_df["flat_type"] == ft].sort_values("year")
                fig_real.add_trace(go.Scatter(x=sub["year"], y=sub["nominal_idx"],
                                               name=f"{ft} (nominal)", line=dict(dash="dot"), opacity=0.6))
                fig_real.add_trace(go.Scatter(x=sub["year"], y=sub["real_idx"],
                                               name=f"{ft} (real)", line=dict(width=2)))
            fig_real.add_hline(y=100, line_dash="dash", line_color="black",
                                annotation_text=f"Base: {e4_base_yr}=100")
            fig_real.update_layout(
                title=f"Nominal vs Real HDB PSM Index (base {e4_base_yr}=100)",
                xaxis_title="Year", yaxis_title="Index (base=100)",
                height=450,
            )
            st.plotly_chart(fig_real, use_container_width=True)

        # Summary: total nominal vs real return since base year
        latest_yr = plot_df["year"].max()
        latest = plot_df[plot_df["year"] == latest_yr][["flat_type", "nominal_idx", "real_idx"]].set_index("flat_type")
        if not latest.empty:
            st.markdown(f"#### Returns from {e4_base_yr} to {latest_yr}")
            summary_rows = []
            for ft, row in latest.iterrows():
                summary_rows.append({
                    "Flat Type": ft,
                    f"Nominal return (since {e4_base_yr})": f"{row['nominal_idx']-100:+.0f}%",
                    "Real return (CPI-adj)": f"{row['real_idx']-100:+.0f}%",
                    "Inflation erosion": f"{(row['nominal_idx'] - row['real_idx']):+.0f}pp",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.info(
            "**CPI base year:** SINGSTAT 2024=100. Real returns = nominal PSM growth deflated by "
            "All Items CPI. Data: HDB resale (data.gov.sg) + SINGSTAT CPI, 2000-present. "
            "DATA CONFIDENCE: Medium. CPI measures general household inflation, not property-specific inflation. "
            "True housing returns also depend on rental income, transaction costs, and leverage."
        )


# ── footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "\U0001f4b9 **Rental Yields** | Singapore HDB Property Decision Support | "
    "Data sources: HDB Rental Transactions, HDB Median Rent by Town, HDB Resale Prices"
)
