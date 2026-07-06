"""
Shared data-loading and utility functions for EDA Streamlit pages.

All pages import from here so data is loaded once and cached.
Handles: raw CSVs, cleaned resale data, supplementary datasets,
         and coordinate joins.
"""
import os
import re
import pandas as pd
import streamlit as st

# ── paths (relative to project root; Streamlit runs from src/) ──
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

RAW_FILES = [
    "Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv",
    "Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv",
    "Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv",
    "Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv",
    "Resale flat prices based on registration date from Jan-2017 onwards.csv",
]

RAW_LABELS = [
    "1990-1999 (Approval)",
    "2000-Feb 2012 (Approval)",
    "Mar 2012-Dec 2014 (Registration)",
    "Jan 2015-Dec 2016 (Registration)",
    "Jan 2017-present (Registration)",
]

CLEAN_CSV = os.path.join(DATA_DIR, "resale_clean_1990_present.csv")
BTO_CSV = os.path.join(DATA_DIR, "bto_projects.csv")
FUTURE_CSV = os.path.join(DATA_DIR, "future_developments.csv")
MRT_CSV = os.path.join(DATA_DIR, "mrt_stations.csv")
COORDS_CSV = os.path.join(DATA_DIR, "address_coords.csv")

# Singapore policy events for timeline annotation
POLICY_EVENTS = [
    ("1996-05", "Anti-speculation measures"),
    ("1997-07", "Asian Financial Crisis begins"),
    ("2003-04", "SARS outbreak"),
    ("2009-09", "Cooling: 1st seller stamp duty"),
    ("2010-02", "Cooling: SSD extended, LTV tightened"),
    ("2010-08", "Cooling: 3rd round, lower LTV"),
    ("2011-01", "Cooling: Additional Buyer Stamp Duty"),
    ("2013-01", "Cooling: ABSD raised, TDSR introduced"),
    ("2013-08", "Mortgage Servicing Ratio 30% for HDB"),
    ("2018-07", "Cooling: ABSD raised 5% across board"),
    ("2020-02", "COVID-19: Circuit Breaker starts"),
    ("2021-12", "Cooling: ABSD +5%, TDSR tightened"),
    ("2022-09", "Cooling: 15-month wait for private"),
    ("2023-04", "Cooling: ABSD foreigners -> 60%"),
    ("2024-08", "Prime/Plus/Standard classification"),
]

LINE_COLORS = {
    "NSL": "#D42E12", "EWL": "#009645", "CCL": "#FA9E0D",
    "NEL": "#9B26AF", "DTL": "#005EC4", "TEL": "#9D5B25",
}
LINE_NAMES = {
    "NSL": "North-South", "EWL": "East-West", "CCL": "Circle",
    "NEL": "North East", "DTL": "Downtown", "TEL": "Thomson-East Coast",
}

# ── data loaders (cached) ──

@st.cache_data
def load_raw_files():
    """Load all 5 raw CSV files, returning a dict of {label: DataFrame}."""
    frames = {}
    for fname, label in zip(RAW_FILES, RAW_LABELS):
        path = os.path.join(RAW_DIR, fname)
        frames[label] = pd.read_csv(path)
    return frames


@st.cache_data
def load_clean():
    """Load the combined & cleaned resale dataset."""
    df = pd.read_csv(CLEAN_CSV, parse_dates=["month"])
    return df


@st.cache_data
def load_clean_with_coords():
    """Load cleaned resale data with block-level coordinates joined in."""
    df = load_clean()
    if os.path.exists(COORDS_CSV):
        coords = pd.read_csv(COORDS_CSV)
        coords["block"] = coords["block"].astype(str)
        df["block"] = df["block"].astype(str)
        df = df.merge(coords, on=["block", "street_name"], how="left")
    return df


@st.cache_data
def load_bto():
    try:
        return pd.read_csv(BTO_CSV)
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_future():
    try:
        return pd.read_csv(FUTURE_CSV)
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_mrt():
    try:
        return pd.read_csv(MRT_CSV)
    except FileNotFoundError:
        return pd.DataFrame()


# ── utility functions ──

def fmt_price(val):
    """Format a price as $X,XXX."""
    if pd.isna(val):
        return "N/A"
    return f"${val:,.0f}"


def fmt_pct(val):
    """Format a percentage."""
    if pd.isna(val):
        return "N/A"
    return f"{val:+.1f}%"


def trend_arrow(val):
    """Return an arrow character for a growth value."""
    if pd.isna(val):
        return "-"
    if val > 2:
        return "▲"
    elif val < -2:
        return "▼"
    return "▶"


def decade_label(year):
    """Map a year to its decade label."""
    if year < 2000:
        return "1990s"
    elif year < 2010:
        return "2000s"
    elif year < 2020:
        return "2010s"
    else:
        return "2020s"


def parse_remaining_lease(val):
    """Parse remaining lease to decimal years (shared with combine_clean)."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.replace(".", "").isdigit():
        return float(s)
    yrs = re.search(r"(\d+)\s*year", s)
    mos = re.search(r"(\d+)\s*month", s)
    y = int(yrs.group(1)) if yrs else 0
    m = int(mos.group(1)) if mos else 0
    return round(y + m / 12, 2)


def storey_band(mid):
    """Map storey midpoint to a human-readable band."""
    if pd.isna(mid):
        return "Unknown"
    mid = float(mid)
    if mid <= 3:
        return "01-03"
    elif mid <= 6:
        return "04-06"
    elif mid <= 9:
        return "07-09"
    elif mid <= 12:
        return "10-12"
    elif mid <= 15:
        return "13-15"
    elif mid <= 21:
        return "16-21"
    elif mid <= 30:
        return "22-30"
    else:
        return "31+"


# Town approximate centroids for fallback when block coords unavailable
TOWN_CENTROIDS = {
    "ANG MO KIO": (1.3691, 103.8454),
    "BEDOK": (1.3236, 103.9273),
    "BISHAN": (1.3526, 103.8491),
    "BUKIT BATOK": (1.3590, 103.7637),
    "BUKIT MERAH": (1.2819, 103.8239),
    "BUKIT PANJANG": (1.3774, 103.7719),
    "BUKIT TIMAH": (1.3294, 103.8021),
    "CENTRAL AREA": (1.2789, 103.8536),
    "CHOA CHU KANG": (1.3840, 103.7470),
    "CLEMENTI": (1.3162, 103.7649),
    "GEYLANG": (1.3201, 103.8918),
    "HOUGANG": (1.3612, 103.8863),
    "JURONG EAST": (1.3329, 103.7436),
    "JURONG WEST": (1.3404, 103.7090),
    "KALLANG/WHAMPOA": (1.3100, 103.8651),
    "LIM CHU KANG": (1.4305, 103.7172),
    "MARINE PARADE": (1.3020, 103.9072),
    "PASIR RIS": (1.3721, 103.9474),
    "PUNGGOL": (1.4053, 103.9024),
    "QUEENSTOWN": (1.2942, 103.7861),
    "SEMBAWANG": (1.4491, 103.8185),
    "SENGKANG": (1.3868, 103.8914),
    "SERANGOON": (1.3554, 103.8679),
    "TAMPINES": (1.3496, 103.9568),
    "TOA PAYOH": (1.3343, 103.8563),
    "WOODLANDS": (1.4382, 103.7891),
    "YISHUN": (1.4304, 103.8354),
}
