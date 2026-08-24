"""
Singapore Property Intelligence — Navigation Router
=====================================================
Entry point for the Streamlit multi-page app.
Uses st.navigation() to define 7 use-case-driven sidebar sections
ordered by the Singapore property decision journey.

Run locally:
    streamlit run src/app.py
"""
import streamlit as st

# ── Section: Home ──────────────────────────────────────────────────────────────
home = st.Page("pages/00_🏠_Home.py", title="Home", icon="🏠", default=True)

# ── Section 1: Read the Market ─────────────────────────────────────────────────
market_regime   = st.Page("pages/14_📈_Market_Regime.py",        title="Market Regime",        icon="📈")
temporal_trends = st.Page("pages/4_📅_Temporal_Trends.py",       title="Temporal Trends",      icon="📅")
spatial_analysis= st.Page("pages/5_🗺️_Spatial_Analysis.py",     title="Spatial Analysis",     icon="🗺️")
market_dynamics = st.Page("pages/8_📊_Market_Dynamics.py",       title="Market Dynamics",      icon="📊")
supply_intel    = st.Page("pages/22_🏗️_Supply_Intelligence.py",  title="Supply Intelligence",  icon="🏗️")
mop_calendar    = st.Page("pages/13_📅_MOP_Calendar.py",         title="MOP Calendar",         icon="📅")
bto_dev         = st.Page("pages/9_🔗_Cross_Dataset.py",         title="BTO & Developments",   icon="🔗")

# ── Section 2: HDB or Private? ─────────────────────────────────────────────────
hdb_vs_private  = st.Page("pages/16_🏢_HDB_vs_Private.py",      title="HDB vs Private",       icon="🏢")
roi_compare     = st.Page("pages/24_📊_HDB_vs_Condo_ROI.py",    title="HDB vs Condo ROI",     icon="📊")
nl_vs_resale    = st.Page("pages/25_🚀_New_Launch_vs_Resale.py", title="New Launch vs Resale", icon="🚀")

# ── Section 3: Find Your Property ─────────────────────────────────────────────
prop_scout      = st.Page("pages/18_🧭_Property_Scout.py",       title="Property Scout",       icon="🧭")
location_intel  = st.Page("pages/21_🌳_Location_Intel.py",       title="Location Intelligence",icon="🌳")
fair_value      = st.Page("pages/12_📐_Fair_Value.py",           title="Fair Value Model",     icon="📐")
value_finder    = st.Page("pages/11_🔍_Value_Finder.py",         title="Value Finder",         icon="🔍")
comps_report    = st.Page("pages/19_📋_Comps_Report.py",         title="Comps Report",         icon="📋")
opp_score       = st.Page("pages/20_⭐_Opportunity_Score.py",    title="Opportunity Score",    icon="⭐")

# ── Section 4: Run the Numbers ─────────────────────────────────────────────────
smart_calc      = st.Page("pages/15_🧮_Smart_Calculator.py",     title="Smart Calculator",     icon="🧮")
rental_yields   = st.Page("pages/17_💹_Rental_Yields.py",        title="Rental Yields",        icon="💹")
lease_dep       = st.Page("pages/7_📜_Lease_Depreciation.py",    title="Lease Depreciation",   icon="📜")

# ── Section 5: Deep Research ───────────────────────────────────────────────────
stat_analysis   = st.Page("pages/10_🔬_Statistical.py",          title="Statistical Analysis", icon="🔬")
flat_chars      = st.Page("pages/6_🏗️_Flat_Characteristics.py",  title="Flat Characteristics", icon="🏗️")
price_deep      = st.Page("pages/3_💰_Price_Deep_Dive.py",       title="Price Deep Dive",      icon="💰")
distributions   = st.Page("pages/2_📉_Distributions.py",         title="Distributions",        icon="📉")
backtesting     = st.Page("pages/23_🔬_Backtesting.py",          title="Strategy Backtesting", icon="🔬")

# ── Section 6: Data & Methodology ─────────────────────────────────────────────
data_quality    = st.Page("pages/1_📊_Data_Quality.py",          title="Data Quality",         icon="📊")

# ── Build navigation ───────────────────────────────────────────────────────────
pg = st.navigation({
    "":                       [home],
    "📊 Read the Market":     [market_regime, temporal_trends, spatial_analysis,
                               market_dynamics, supply_intel, mop_calendar, bto_dev],
    "⚖️ HDB or Private?":    [hdb_vs_private, roi_compare, nl_vs_resale],
    "🏠 Find Your Property":  [prop_scout, location_intel, fair_value,
                               value_finder, comps_report, opp_score],
    "💰 Run the Numbers":     [smart_calc, rental_yields, lease_dep],
    "🔬 Deep Research":       [stat_analysis, flat_chars, price_deep,
                               distributions, backtesting],
    "🛠️ Data & Methodology":  [data_quality],
})

pg.run()
