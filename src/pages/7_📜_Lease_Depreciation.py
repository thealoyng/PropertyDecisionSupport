"""
Page 7 — Lease Depreciation
============================
Analysis of lease decay and flat age effects on Singapore HDB resale
prices.  Explores how remaining lease length drives pricing, identifies
depreciation curves by flat type and town, and examines the critical
60-year lease threshold for HDB loan eligibility.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess

from eda_helpers import load_clean, fmt_price, fmt_pct

# ── page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Lease Depreciation",
    page_icon="📜",
    layout="wide",
)

st.title("📜 Lease Depreciation")
st.caption(
    "How the 99-year leasehold structure shapes HDB resale values — "
    "depreciation curves, lease threshold effects, and flat age analysis."
)

with st.expander("ℹ️ About the 99-year lease system", expanded=False):
    st.markdown(
        """
        All HDB flats in Singapore are sold on a **99-year leasehold** basis.
        The remaining lease directly affects:

        - **CPF usage** — buyers can only use CPF savings if the remaining
          lease covers the youngest buyer until age 95.
        - **HDB loan eligibility** — flats with very short leases may not
          qualify for an HDB loan; banks also impose stricter limits.
        - **Perceived value** — the market prices in the "time value" of a
          depreciating asset, leading to observable depreciation curves.
        - **60-year threshold** — flats with fewer than ~60 years of
          remaining lease face significantly tighter financing, reducing
          the pool of eligible buyers and putting downward pressure on
          prices.

        This page explores how lease decay manifests in transaction data.
        """
    )

# ── load data ────────────────────────────────────────────────────
df = load_clean()

# Drop rows with missing lease / price info needed for this page
df = df.dropna(subset=["remaining_lease_yrs", "price_per_sqm", "flat_age"])

# ── helper: 5-year lease bins ────────────────────────────────────
def lease_bin(yrs):
    """Bin remaining_lease_yrs into 5-year bands."""
    if pd.isna(yrs):
        return None
    lo = int(yrs // 5) * 5
    return f"{lo}–{lo + 5}"


def lease_bin_sort_key(label):
    """Extract lower bound for sorting lease bin labels."""
    try:
        return int(label.split("–")[0])
    except Exception:
        return 0


df["lease_band"] = df["remaining_lease_yrs"].apply(lease_bin)

# ── KPI metric cards ────────────────────────────────────────────
avg_lease_all = df["remaining_lease_yrs"].mean()

recent_cutoff = df["year"].max() - 2  # most recent 3 years
df_recent = df[df["year"] >= recent_cutoff]
avg_lease_recent = df_recent["remaining_lease_yrs"].mean()

pct_below_60 = (df["remaining_lease_yrs"] < 60).mean() * 100

median_above = df.loc[df["remaining_lease_yrs"] >= 60, "price_per_sqm"].median()
median_below = df.loc[df["remaining_lease_yrs"] < 60, "price_per_sqm"].median()
median_diff = median_above - median_below

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Avg remaining lease (all time)",
    f"{avg_lease_all:.1f} yrs",
    help="Mean remaining lease across every transaction in the dataset.",
)
k2.metric(
    "Avg remaining lease (recent 3 yr)",
    f"{avg_lease_recent:.1f} yrs",
    delta=f"{avg_lease_recent - avg_lease_all:+.1f} yrs vs all-time",
    help="Mean remaining lease for transactions in the most recent 3 years.",
)
k3.metric(
    "Transactions with lease < 60 yrs",
    f"{pct_below_60:.1f}%",
    help="Share of all transactions where remaining lease is below 60 years.",
)
k4.metric(
    "Median $/sqm gap (≥60 vs <60)",
    fmt_price(median_diff),
    help="Difference in median price per sqm between flats with ≥60 and <60 years remaining.",
)

st.divider()

# ── tabs ─────────────────────────────────────────────────────────
(
    tab_scatter,
    tab_curves,
    tab_town,
    tab_threshold,
    tab_age,
    tab_depr_rate,
    tab_volume,
) = st.tabs(
    [
        "🔵 Price vs Lease (scatter)",
        "📉 Cohort depreciation",
        "🏘️ By town",
        "⚠️ 60-yr threshold",
        "🏗️ Flat age cohorts",
        "📊 Depreciation rate",
        "📦 Transaction volume",
    ]
)

# ================================================================
# 1. Price per sqm vs remaining lease — scatter + LOWESS
# ================================================================
with tab_scatter:
    st.subheader("Price per sqm vs remaining lease")
    st.caption(
        "Each dot is one transaction (sampled to 20 000 for performance). "
        "The LOWESS curve shows the smoothed central tendency."
    )

    sample = df.sample(n=min(20_000, len(df)), random_state=42)

    fig1 = px.scatter(
        sample.sort_values("remaining_lease_yrs"),
        x="remaining_lease_yrs",
        y="price_per_sqm",
        color="flat_type",
        opacity=0.25,
        labels={
            "remaining_lease_yrs": "Remaining lease (years)",
            "price_per_sqm": "Price per sqm ($)",
            "flat_type": "Flat type",
        },
    )

    # LOWESS smoothing on the full sample (sorted)
    sorted_sample = sample.dropna(subset=["remaining_lease_yrs", "price_per_sqm"]).sort_values(
        "remaining_lease_yrs"
    )
    smooth = lowess(
        sorted_sample["price_per_sqm"].values,
        sorted_sample["remaining_lease_yrs"].values,
        frac=0.3,
    )
    fig1.add_trace(
        go.Scatter(
            x=smooth[:, 0],
            y=smooth[:, 1],
            mode="lines",
            name="LOWESS trend",
            line=dict(color="#ef4444", width=3),
        )
    )

    fig1.update_layout(
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30),
    )
    st.plotly_chart(fig1, width='stretch')

# ================================================================
# 2. Cohort depreciation curves (by lease-commencement decade)
# ================================================================
with tab_curves:
    st.subheader("Cohort depreciation curves")
    st.caption(
        "Median price per sqm within 5-year remaining-lease bands, "
        "one line per lease-commencement decade (cohort). "
        "Read left → right: as remaining lease increases, prices rise — "
        "confirming true lease depreciation once era/inflation is controlled."
    )

    with st.expander("Why cohort analysis?", expanded=False):
        st.markdown(
            """
            **The era-confounding problem with naïve depreciation curves**

            A simple chart of *median price per sqm vs remaining lease* — grouping
            all transactions regardless of when they happened — produces a
            misleading result: it often shows that flats with **more lease
            remaining** are *cheaper* than short-lease flats. That is backwards.

            Why does it happen? Because flats with 90+ years of lease remaining
            were sold in the **1990s**, when HDB prices were far lower. Flats with
            40–50 years remaining are sold **today** at much higher prices — but
            that reflects 30 years of Singapore price inflation, **not** the value
            of a shorter lease.

            **Cohort analysis is the fix.** By grouping flats built in the same
            decade (same lease-commencement cohort), we compare prices within a
            set of buildings whose physical age and neighbourhood characteristics
            are similar. As the cohort's remaining lease ticks down across
            transaction years, we observe *true* lease depreciation — isolating
            it from era/inflation effects.

            Each line below tracks flats built in the same decade as their
            remaining lease ticks down over multiple transaction years — this
            isolates true depreciation from era/inflation effects.
            """
        )

    # -- Cohort column: lease_commence_date decade --------------------------------
    def _cohort_decade(yr):
        if pd.isna(yr):
            return None
        yr = int(yr)
        if yr < 1960:
            return None  # negligible sample
        decade = (yr // 10) * 10
        return f"{decade}s"

    df["_cohort_tab2"] = df["lease_commence_date"].apply(_cohort_decade)
    df_c2 = df.dropna(subset=["_cohort_tab2", "lease_band"]).copy()

    cohort_curves = (
        df_c2.groupby(["_cohort_tab2", "lease_band"])["price_per_sqm"]
        .median()
        .reset_index()
        .rename(columns={"_cohort_tab2": "Cohort"})
    )
    cohort_curves["_sort"] = cohort_curves["lease_band"].apply(lease_bin_sort_key)
    cohort_curves = cohort_curves.sort_values(["Cohort", "_sort"])

    # X-axis: LOW → HIGH left-to-right (lease running out on the left)
    all_bands_sorted = sorted(
        cohort_curves["lease_band"].unique(), key=lease_bin_sort_key
    )

    fig2 = px.line(
        cohort_curves,
        x="lease_band",
        y="price_per_sqm",
        color="Cohort",
        markers=True,
        labels={
            "lease_band": "Remaining lease band (years)",
            "price_per_sqm": "Median price per sqm ($)",
            "Cohort": "Lease decade",
        },
        category_orders={"lease_band": all_bands_sorted},
    )
    fig2.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=all_bands_sorted,
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30),
    )
    st.plotly_chart(fig2, width='stretch')

    st.info(
        "📌 In recent years only the older cohorts (1970s–1980s) have flats "
        "with < 60 years of remaining lease, so those lines extend furthest "
        "to the left. Newer cohorts (2000s–2010s) still carry high remaining "
        "lease and appear only on the right side of the chart — their "
        "depreciation story will become visible over future decades."
    )

# ================================================================
# 3. Depreciation by town (user-selectable)
# ================================================================
with tab_town:
    st.subheader("Depreciation by town")
    st.caption(
        "Compare how remaining lease affects pricing across different towns."
    )

    all_towns = sorted(df["town"].unique())
    default_towns = all_towns[:4] if len(all_towns) >= 4 else all_towns
    selected_towns = st.multiselect(
        "Select 3–5 towns to compare",
        all_towns,
        default=default_towns,
        max_selections=5,
    )

    if len(selected_towns) < 1:
        st.info("Please select at least one town.")
    else:
        town_curves = (
            df[df["town"].isin(selected_towns)]
            .groupby(["town", "lease_band"])["price_per_sqm"]
            .median()
            .reset_index()
        )
        town_curves["_sort"] = town_curves["lease_band"].apply(lease_bin_sort_key)
        town_curves = town_curves.sort_values("_sort")

        fig3 = px.line(
            town_curves,
            x="lease_band",
            y="price_per_sqm",
            color="town",
            markers=True,
            labels={
                "lease_band": "Remaining lease band (years)",
                "price_per_sqm": "Median price per sqm ($)",
                "town": "Town",
            },
        )
        fig3.update_layout(
            xaxis=dict(
                categoryorder="array",
                categoryarray=sorted(
                    town_curves["lease_band"].unique(), key=lease_bin_sort_key
                ),
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
            ),
            margin=dict(t=30),
        )
        st.plotly_chart(fig3, width='stretch')

# ================================================================
# 4. Lease threshold analysis — ≥60 vs <60 years
# ================================================================
with tab_threshold:
    st.subheader("Lease threshold analysis: ≥ 60 vs < 60 years")
    st.caption(
        "Comparing price distributions above and below the critical 60-year "
        "remaining lease threshold."
    )

    df["lease_group"] = np.where(
        df["remaining_lease_yrs"] >= 60,
        "≥ 60 years",
        "< 60 years",
    )

    fig4 = go.Figure()
    for grp, colour in [("≥ 60 years", "#2563eb"), ("< 60 years", "#dc2626")]:
        subset = df[df["lease_group"] == grp]["price_per_sqm"]
        fig4.add_trace(
            go.Violin(
                y=subset,
                name=grp,
                box_visible=True,
                meanline_visible=True,
                line_color=colour,
                fillcolor=colour,
                opacity=0.55,
            )
        )

    fig4.update_layout(
        yaxis_title="Price per sqm ($)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30),
    )

    # annotation box
    fig4.add_annotation(
        text=(
            "Flats with < 60 years remaining lease face<br>"
            "restricted CPF usage and tighter bank loan<br>"
            "limits, reducing the buyer pool significantly."
        ),
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.95,
        showarrow=False,
        font=dict(size=11, color="#475569"),
        align="left",
        bordercolor="#94a3b8",
        borderwidth=1,
        borderpad=6,
        bgcolor="#f8fafc",
    )

    st.plotly_chart(fig4, width='stretch')

    # supporting stats table
    stats = (
        df.groupby("lease_group")["price_per_sqm"]
        .agg(["count", "median", "mean", "std"])
        .rename(
            columns={
                "count": "Transactions",
                "median": "Median $/sqm",
                "mean": "Mean $/sqm",
                "std": "Std dev",
            }
        )
    )
    st.dataframe(stats.style.format("{:,.0f}"), width='stretch')

# ================================================================
# 5. Flat age vs price per sqm — cohorts by decade
# ================================================================
with tab_age:
    st.subheader("Flat age vs price per sqm by lease-commencement cohort")
    st.caption(
        "Each line represents flats whose lease commenced in the same decade. "
        "Shows how differently-aged cohorts depreciate over time."
    )

    def decade_cohort(yr):
        if pd.isna(yr):
            return None
        yr = int(yr)
        if yr < 1970:
            return None  # too few records
        decade = (yr // 10) * 10
        return f"{decade}s"

    df["cohort"] = df["lease_commence_date"].apply(decade_cohort)
    df_cohort = df.dropna(subset=["cohort"]).copy()

    # bin flat_age into 2-year buckets for smoother lines
    df_cohort["age_bin"] = (df_cohort["flat_age"] // 2) * 2

    cohort_agg = (
        df_cohort.groupby(["cohort", "age_bin"])["price_per_sqm"]
        .median()
        .reset_index()
        .sort_values("age_bin")
    )

    fig5 = px.line(
        cohort_agg,
        x="age_bin",
        y="price_per_sqm",
        color="cohort",
        markers=True,
        labels={
            "age_bin": "Flat age (years)",
            "price_per_sqm": "Median price per sqm ($)",
            "cohort": "Lease decade",
        },
    )
    fig5.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30),
    )
    st.plotly_chart(fig5, width='stretch')

# ================================================================
# 6. Annual depreciation rate by town (slope, recent 5 years)
# ================================================================
with tab_depr_rate:
    st.subheader("Annual depreciation rate by town")
    st.caption(
        "Estimated slope of price_per_sqm vs remaining_lease_yrs using only "
        "the most recent 5 years of transactions.  More negative = faster "
        "depreciation per year of lease lost."
    )

    recent_5y = df[df["year"] >= (df["year"].max() - 4)].copy()

    slopes = []
    for town, grp in recent_5y.groupby("town"):
        x = grp["remaining_lease_yrs"].values
        y = grp["price_per_sqm"].values
        if len(x) < 30:
            continue
        # simple OLS slope
        x_mean = x.mean()
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        slopes.append({"town": town, "slope": round(slope, 1)})

    slopes_df = pd.DataFrame(slopes).sort_values("slope")
    slopes_df["colour"] = np.where(slopes_df["slope"] >= 0, "#16a34a", "#dc2626")

    fig6 = go.Figure(
        go.Bar(
            x=slopes_df["slope"],
            y=slopes_df["town"],
            orientation="h",
            marker_color=slopes_df["colour"],
            hovertemplate="Town: %{y}<br>Slope: %{x:+.1f} $/sqm per lease yr<extra></extra>",
        )
    )
    fig6.update_layout(
        xaxis_title="$/sqm per remaining-lease year (slope)",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=max(450, len(slopes_df) * 22),
        margin=dict(t=20, l=140),
    )
    st.plotly_chart(fig6, width='stretch')

    st.info(
        "A slope of **+50** means each additional year of remaining lease "
        "is associated with ~$50 higher price per sqm in that town (recent "
        "5 years). Negative slopes indicate prices falling with lease decay."
    )

# ================================================================
# 7. Remaining lease vs transaction volume
# ================================================================
with tab_volume:
    st.subheader("Transaction volume by remaining lease band")
    st.caption(
        "Number of transactions in each 5-year remaining-lease bucket.  "
        "Reveals whether old-lease flats still trade in significant volume."
    )

    vol = (
        df.groupby("lease_band")
        .size()
        .reset_index(name="count")
    )
    vol["_sort"] = vol["lease_band"].apply(lease_bin_sort_key)
    vol = vol.sort_values("_sort")

    fig7 = px.bar(
        vol,
        x="lease_band",
        y="count",
        text="count",
        labels={
            "lease_band": "Remaining lease band (years)",
            "count": "Transaction count",
        },
        color_discrete_sequence=["#6366f1"],
    )
    fig7.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=sorted(vol["lease_band"].unique(), key=lease_bin_sort_key),
        ),
        margin=dict(t=20),
    )
    fig7.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig7, width='stretch')
