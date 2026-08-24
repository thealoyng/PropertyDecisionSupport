"""
Home — Singapore Property Intelligence
=======================================
Welcome hub: quick-stats bar, decision-tree journey guide,
and one-click navigation cards to every section.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

st.set_page_config(
    page_title="Singapore Property Intelligence",
    page_icon="🏠",
    layout="wide",
)

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🏠 Singapore Property Intelligence")
st.markdown(
    "**Data-driven tools for every step of the property decision** — "
    "from reading the market to closing the deal."
)
st.divider()

# ── Quick-stats bar ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("HDB transactions", "978k+", "1990 – 2026")
c2.metric("Years of history", "35+", "HDB resale")
c3.metric("HDB towns covered", "27", "all planning areas")
c4.metric("Private caveats", "134k+", "Aug 2021 – 2026")

st.divider()

# ── Journey guide ──────────────────────────────────────────────────────────────
st.subheader("Where are you in your journey?")
st.caption(
    "This app follows the natural Singapore property decision flow. "
    "Start by reading the market, then decide on property type, then research and negotiate. "
    "Use the sidebar to jump to any section at any time."
)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("#### 📊 Step 1 — Read the Market")
    st.markdown("Is now a good time to buy? What cycle are we in? What supply is coming?")
    st.page_link("pages/14_📈_Market_Regime.py",    label="→ Market Regime",       icon="📈")
    st.page_link("pages/4_📅_Temporal_Trends.py",   label="→ Temporal Trends",     icon="📅")
    st.page_link("pages/22_🏗️_Supply_Intelligence.py", label="→ Supply Intelligence", icon="🏗️")

    st.markdown("---")
    st.markdown("#### 🏠 Step 3 — Find Your Property")
    st.markdown(
        "Search by lifestyle fit, validate the price, find undervalued units. "
        "Toggle between **HDB** and **Private** modes on each tool."
    )
    st.page_link("pages/18_🧭_Property_Scout.py",   label="→ Property Scout",       icon="🧭")
    st.page_link("pages/12_📐_Fair_Value.py",        label="→ Fair Value Model",     icon="📐")
    st.page_link("pages/19_📋_Comps_Report.py",      label="→ Comps Report",         icon="📋")

    st.markdown("---")
    st.markdown("#### 🔬 Step 5 — Deep Research")
    st.markdown("For analysts: statistical tests, backtesting, and raw data exploration.")
    st.page_link("pages/10_🔬_Statistical.py",       label="→ Statistical Analysis", icon="🔬")
    st.page_link("pages/23_🔬_Backtesting.py",       label="→ Strategy Backtesting", icon="🔬")

with col_b:
    st.markdown("#### ⚖️ Step 2 — HDB or Private?")
    st.markdown(
        "Compare returns, market dynamics, and decide which type fits your goals. "
        "If leaning private: is new launch or resale better value?"
    )
    st.page_link("pages/16_🏢_HDB_vs_Private.py",   label="→ HDB vs Private",       icon="🏢")
    st.page_link("pages/24_📊_HDB_vs_Condo_ROI.py", label="→ HDB vs Condo ROI",     icon="📊")
    st.page_link("pages/25_🚀_New_Launch_vs_Resale.py", label="→ New Launch vs Resale", icon="🚀")

    st.markdown("---")
    st.markdown("#### 💰 Step 4 — Run the Numbers")
    st.markdown(
        "Mortgage affordability, CPF eligibility, grants, ABSD, rental yield, "
        "lease decay, and BTO payment stages."
    )
    st.page_link("pages/15_🧮_Smart_Calculator.py",  label="→ Smart Calculator",     icon="🧮")
    st.page_link("pages/17_💹_Rental_Yields.py",     label="→ Rental Yields",        icon="💹")
    st.page_link("pages/7_📜_Lease_Depreciation.py", label="→ Lease Depreciation",   icon="📜")

    st.markdown("---")
    st.markdown("#### 🛠️ Data & Methodology")
    st.markdown("Understand the data sources, coverage, and confidence levels.")
    st.page_link("pages/1_📊_Data_Quality.py",       label="→ Data Quality",         icon="📊")

st.divider()

# ── About ──────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this app"):
    st.markdown("""
**Singapore Property Intelligence** is an open-source decision-support platform
built on public datasets from HDB, URA, and data.gov.sg.

**Data sources:**
- HDB resale transaction records (data.gov.sg, Open Data Licence) — 978k+ rows, 1990 to present
- URA private residential transaction caveats (URA API) — 134k+ rows, Aug 2021 to present
- URA developer pipeline (units launched / sold / unsold, by quarter)
- HDB median rental data (data.gov.sg)
- MRT station coordinates, amenity data (NEA, PA, NParks, HPB, MOE)
- CPI data (SingStat)

**Key tools:** Streamlit · Pandas · Plotly · Pydeck · Folium · Scikit-learn

**Disclaimer:** All analysis is for informational purposes only and does not constitute
financial or property advice. Always verify with official HDB / URA / CPF sources
and consult a licensed professional before making any property decision.
""")
