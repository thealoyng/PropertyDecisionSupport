"""
Page 2 — Univariate Distributions
==================================
Explore the shape, spread, and skew of every key numeric and categorical
variable in the HDB resale dataset.  Each chart is designed to surface a
specific insight (e.g. price inflation across decades, lease-decay risk,
floor-level premiums).
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eda_helpers import load_clean, decade_label, storey_band

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# ── page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Distributions", page_icon="📉", layout="wide")

# ── load data ────────────────────────────────────────────────────
df = load_clean()
df["decade"] = df["year"].apply(decade_label)
df["storey_band"] = df["storey_mid"].apply(storey_band)

# ── title ────────────────────────────────────────────────────────
st.title("📉 Univariate Distributions")
st.markdown(
    "A deep dive into how each variable is distributed across **{:,}** "
    "HDB resale transactions (1990–present).  Use these charts to spot "
    "skewness, outliers, and structural shifts before building any model.".format(len(df))
)

# ── summary metrics ──────────────────────────────────────────────
st.subheader("Key statistics at a glance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean resale price", f"${df['resale_price'].mean():,.0f}")
m2.metric("Median resale price", f"${df['resale_price'].median():,.0f}")
m3.metric("Std deviation", f"${df['resale_price'].std():,.0f}")
m4.metric("Total transactions", f"{len(df):,}")

st.divider()

# =====================================================================
# 1. Resale price histogram + KDE overlay
# =====================================================================
st.subheader("1 · Resale price distribution")
st.markdown(
    "The histogram shows the overall shape of resale prices.  Toggle between "
    "**All time** and a specific decade to see how the distribution has shifted "
    "rightward with inflation and policy changes."
)

decade_options = ["All time"] + sorted(df["decade"].unique())
selected_decade = st.selectbox("Time period", decade_options, key="price_decade")

if selected_decade == "All time":
    subset = df["resale_price"]
    chart_title = "Resale price distribution — All time"
else:
    subset = df.loc[df["decade"] == selected_decade, "resale_price"]
    chart_title = f"Resale price distribution — {selected_decade}"

# Build histogram with KDE overlay
fig1 = px.histogram(
    subset,
    nbins=120,
    labels={"value": "Resale price (SGD)", "count": "Transactions"},
    title=chart_title,
    opacity=0.7,
    color_discrete_sequence=["#636EFA"],
)
fig1.update_layout(bargap=0.02, showlegend=False)

# KDE curve
try:
    kde_x = np.linspace(subset.min(), subset.max(), 500)
    kde_y = gaussian_kde(subset.dropna())(kde_x)
    # Scale KDE to histogram height
    bin_width = (subset.max() - subset.min()) / 120
    kde_y_scaled = kde_y * len(subset) * bin_width
    fig1.add_trace(
        go.Scatter(x=kde_x, y=kde_y_scaled, mode="lines",
                   line=dict(color="#EF553B", width=2.5),
                   name="KDE")
    )
    fig1.update_layout(showlegend=True)
except Exception:
    pass  # skip KDE if too few data points

st.plotly_chart(fig1, width='stretch')

st.divider()

# =====================================================================
# 2. Price box plots by decade
# =====================================================================
st.subheader("2 · Price spread by decade")
st.markdown(
    "Box plots reveal the median, IQR, and outlier range for each decade.  "
    "Notice how **variance** grows alongside median prices in recent decades."
)

decade_order = ["1990s", "2000s", "2010s", "2020s"]
fig2 = px.box(
    df, x="decade", y="resale_price",
    category_orders={"decade": decade_order},
    color="decade",
    labels={"decade": "Decade", "resale_price": "Resale price (SGD)"},
    title="Resale price by decade",
    color_discrete_sequence=px.colors.qualitative.Safe,
)
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, width='stretch')

st.divider()

# =====================================================================
# 3. Floor area histogram faceted by flat type
# =====================================================================
st.subheader("3 · Floor area by flat type")
st.markdown(
    "Each sub-plot shows the size distribution for one flat type.  "
    "Older flat models often have slightly different footprints, creating "
    "multi-modal peaks within a single flat type."
)

flat_type_order = sorted(df["flat_type"].unique())
fig3 = px.histogram(
    df, x="floor_area_sqm",
    facet_col="flat_type", facet_col_wrap=3,
    category_orders={"flat_type": flat_type_order},
    labels={"floor_area_sqm": "Floor area (sqm)", "count": "Count"},
    title="Floor area distribution by flat type",
    color_discrete_sequence=["#00CC96"],
    nbins=50,
)
fig3.update_layout(height=700, bargap=0.03, showlegend=False)
fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig3, width='stretch')

st.divider()

# =====================================================================
# 4. Storey midpoint distribution
# =====================================================================
st.subheader("4 · Storey midpoint distribution")
st.markdown(
    "Which floors trade most?  The bulk of HDB stock sits between storeys "
    "4–12.  High-floor units (16+) are scarcer and command a premium."
)

fig4 = px.histogram(
    df, x="storey_mid",
    nbins=60,
    labels={"storey_mid": "Storey midpoint", "count": "Transactions"},
    title="Distribution of storey midpoint",
    color_discrete_sequence=["#AB63FA"],
)
fig4.update_layout(bargap=0.03, showlegend=False)
st.plotly_chart(fig4, width='stretch')

st.divider()

# =====================================================================
# 5. Flat age distribution
# =====================================================================
st.subheader("5 · Flat age at point of sale")
st.markdown(
    "Flat age is the gap between the transaction year and lease commencement.  "
    "Peaks around 10–15 years indicate the typical MOP-driven resale cycle.  "
    "The right tail shows ageing estates approaching lease-decay concerns."
)

fig5 = px.histogram(
    df, x="flat_age",
    nbins=60,
    labels={"flat_age": "Flat age (years)", "count": "Transactions"},
    title="Flat age distribution",
    opacity=0.7,
    color_discrete_sequence=["#FFA15A"],
)
fig5.update_layout(bargap=0.03, showlegend=False)

try:
    age_vals = df["flat_age"].dropna()
    kde_x5 = np.linspace(age_vals.min(), age_vals.max(), 400)
    kde_y5 = gaussian_kde(age_vals)(kde_x5)
    bw5 = (age_vals.max() - age_vals.min()) / 60
    kde_y5_scaled = kde_y5 * len(age_vals) * bw5
    fig5.add_trace(
        go.Scatter(x=kde_x5, y=kde_y5_scaled, mode="lines",
                   line=dict(color="#EF553B", width=2.5), name="KDE")
    )
    fig5.update_layout(showlegend=True)
except Exception:
    pass

st.plotly_chart(fig5, width='stretch')

st.divider()

# =====================================================================
# 6. Remaining lease distribution
# =====================================================================
st.subheader("6 · Remaining lease years")
st.markdown(
    "Banks typically require **≥ 60 years** of remaining lease for a full "
    "HDB loan.  The red line marks that threshold — flats to the left face "
    "financing constraints and potential value erosion."
)

fig6 = px.histogram(
    df, x="remaining_lease_yrs",
    nbins=60,
    labels={"remaining_lease_yrs": "Remaining lease (years)", "count": "Transactions"},
    title="Remaining lease distribution",
    color_discrete_sequence=["#19D3F3"],
)
fig6.add_vline(
    x=60, line_dash="dash", line_color="red", line_width=2,
    annotation_text="60-yr loan threshold",
    annotation_position="top left",
    annotation_font_color="red",
)
fig6.update_layout(bargap=0.03, showlegend=False)
st.plotly_chart(fig6, width='stretch')

st.divider()

# =====================================================================
# 7. Price per sqm histogram
# =====================================================================
st.subheader("7 · Price per sqm (normalised price)")
st.markdown(
    "Dividing resale price by floor area removes the size effect and lets you "
    "compare value across flat types.  The long right tail captures premium "
    "locations like Central Area and Bishan."
)

fig7 = px.histogram(
    df, x="price_per_sqm",
    nbins=120,
    labels={"price_per_sqm": "Price per sqm (SGD)", "count": "Transactions"},
    title="Price per sqm distribution",
    color_discrete_sequence=["#FF6692"],
)
fig7.update_layout(bargap=0.02, showlegend=False)
st.plotly_chart(fig7, width='stretch')

st.divider()

# =====================================================================
# 8. Flat type proportion — pie + bar
# =====================================================================
st.subheader("8 · Flat type proportions")
st.markdown(
    "4-ROOM flats dominate the resale market, followed by 3-ROOM.  "
    "The pie chart shows share; the bar chart adds exact counts for "
    "easier comparison of smaller categories."
)

type_counts = (
    df["flat_type"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "flat_type", "flat_type": "flat_type", "count": "count"})
)
# Ensure column names are consistent regardless of pandas version
if "count" not in type_counts.columns:
    type_counts.columns = ["flat_type", "count"]

type_counts["pct"] = (type_counts["count"] / type_counts["count"].sum() * 100).round(1)

col_pie, col_bar = st.columns(2)

with col_pie:
    fig8a = px.pie(
        type_counts, names="flat_type", values="count",
        title="Transaction share by flat type",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.35,
    )
    fig8a.update_traces(textinfo="label+percent", textposition="outside")
    st.plotly_chart(fig8a, width='stretch')

with col_bar:
    fig8b = px.bar(
        type_counts.sort_values("count"),
        x="count", y="flat_type",
        orientation="h",
        text="pct",
        labels={"count": "Transactions", "flat_type": "Flat type"},
        title="Transaction count by flat type",
        color_discrete_sequence=["#636EFA"],
    )
    fig8b.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig8b.update_layout(showlegend=False, yaxis_title="")
    st.plotly_chart(fig8b, width='stretch')

st.divider()

# =====================================================================
# 9. Town transaction count
# =====================================================================
st.subheader("9 · Transactions by town")
st.markdown(
    "Mature estates like **Tampines**, **Jurong West**, and **Bedok** lead "
    "in transaction volume, reflecting their larger housing stock.  Newer "
    "towns like **Punggol** and **Sengkang** are catching up fast."
)

town_counts = (
    df["town"]
    .value_counts()
    .reset_index()
)
if "count" not in town_counts.columns:
    town_counts.columns = ["town", "count"]

fig9 = px.bar(
    town_counts.sort_values("count"),
    x="count", y="town",
    orientation="h",
    labels={"count": "Transactions", "town": "Town"},
    title="Total resale transactions by town",
    color_discrete_sequence=["#00CC96"],
)
fig9.update_layout(height=700, showlegend=False, yaxis_title="")
st.plotly_chart(fig9, width='stretch')

st.divider()

# =====================================================================
# 10. Flat model frequency — top 15
# =====================================================================
st.subheader("10 · Top 15 flat models")
st.markdown(
    "HDB has built dozens of flat models over the decades.  The chart below "
    "shows the **15 most common** models in the resale market.  'Model A' and "
    "'Improved' dominate, while premium models like 'DBSS' and 'Type S2' "
    "are comparatively rare."
)

model_counts = (
    df["flat_model"]
    .value_counts()
    .head(15)
    .reset_index()
)
if "count" not in model_counts.columns:
    model_counts.columns = ["flat_model", "count"]

fig10 = px.bar(
    model_counts.sort_values("count"),
    x="count", y="flat_model",
    orientation="h",
    labels={"count": "Transactions", "flat_model": "Flat model"},
    title="Top 15 flat models by transaction count",
    color_discrete_sequence=["#EF553B"],
)
fig10.update_layout(height=550, showlegend=False, yaxis_title="")
st.plotly_chart(fig10, width='stretch')

# ── footer ───────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data: HDB Resale Flat Prices (data.gov.sg) · "
    "Page 2 of EDA series · Built with Streamlit + Plotly"
)
