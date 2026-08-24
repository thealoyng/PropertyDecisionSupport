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

# Prefer the gzip-compressed version (12 MB vs 104 MB uncompressed).
# pandas reads .gz natively; fall back to uncompressed for local dev convenience.
_CLEAN_CSV_GZ  = os.path.join(DATA_DIR, "resale", "resale_clean_1990_present.csv.gz")
_CLEAN_CSV_RAW = os.path.join(DATA_DIR, "resale", "resale_clean_1990_present.csv")
CLEAN_CSV = _CLEAN_CSV_GZ if os.path.exists(_CLEAN_CSV_GZ) else _CLEAN_CSV_RAW
BTO_CSV = os.path.join(DATA_DIR, "bto_projects.csv")
FUTURE_CSV = os.path.join(DATA_DIR, "future_developments.csv")
MRT_CSV = os.path.join(DATA_DIR, "mrt_stations.csv")
COORDS_CSV = os.path.join(DATA_DIR, "address_coords.csv")

# ── new data paths (Smart Money expansion) ──
RENTAL_TXN_CSV      = os.path.join(DATA_DIR, "rental", "hdb_rental_transactions.csv")
RENTAL_MEDIAN_CSV   = os.path.join(DATA_DIR, "rental", "hdb_median_rent_by_town.csv")
URA_PPI_CSV         = os.path.join(DATA_DIR, "private_property", "ura_ppi.csv")
URA_TXN_AGG_CSV     = os.path.join(DATA_DIR, "private_property", "ura_private_transactions_agg.csv")

# ── Phase 2: amenities + finance ──
AMENITIES_DIR       = os.path.join(DATA_DIR, "amenities")
HAWKER_CSV          = os.path.join(AMENITIES_DIR, "hawker_centres.csv")
CC_CSV              = os.path.join(AMENITIES_DIR, "community_clubs.csv")
PARKS_CSV           = os.path.join(AMENITIES_DIR, "parks.csv")
POLYCLINICS_CSV     = os.path.join(AMENITIES_DIR, "polyclinics.csv")
SCHOOLS_CSV         = os.path.join(AMENITIES_DIR, "schools.csv")
CPI_CSV             = os.path.join(DATA_DIR, "finance", "cpi_monthly.csv")

# ── Phase 3 (I2): URA developer pipeline data ──
URA_LAUNCHED_CSV    = os.path.join(DATA_DIR, "private_property", "ura_units_launched.csv")
URA_SOLD_CSV        = os.path.join(DATA_DIR, "private_property", "ura_units_sold.csv")
URA_UNSOLD_CSV      = os.path.join(DATA_DIR, "private_property", "ura_units_unsold.csv")
CONDO_CLEAN_CSV     = os.path.join(DATA_DIR, "private_property", "condo_clean.csv")

# Flat-type normalisation maps (rental datasets use different conventions)
RENTAL_FLAT_TYPE_MAP = {
    "1-RM": "1 ROOM", "2-RM": "2 ROOM", "3-RM": "3 ROOM",
    "4-RM": "4 ROOM", "5-RM": "5 ROOM",
    "EXEC": "EXECUTIVE", "MULTI-GEN": "MULTI GENERATION",
    # hdb_rental_transactions uses hyphen format
    "1-ROOM": "1 ROOM", "2-ROOM": "2 ROOM", "3-ROOM": "3 ROOM",
    "4-ROOM": "4 ROOM", "5-ROOM": "5 ROOM",
    "EXECUTIVE": "EXECUTIVE", "MULTI GENERATION": "MULTI GENERATION",
}

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


@st.cache_data
def load_rental_transactions():
    """Load individual HDB rental approval records (Jan 2021+).
    Normalises flat_type to match resale dataset convention (e.g. '4 ROOM').
    Returns empty DataFrame if file not yet fetched (run src/fetch_data.py).
    """
    try:
        df = pd.read_csv(RENTAL_TXN_CSV)
        df["monthly_rent"] = pd.to_numeric(df["monthly_rent"], errors="coerce")
        df["rent_approval_date"] = pd.to_datetime(df["rent_approval_date"], format="%Y-%m")
        df["year"] = df["rent_approval_date"].dt.year
        df["month_num"] = df["rent_approval_date"].dt.month
        # Normalise flat_type
        df["flat_type"] = df["flat_type"].str.strip().map(
            lambda x: RENTAL_FLAT_TYPE_MAP.get(x, x.replace("-", " "))
        )
        return df
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_rental_medians():
    """Load HDB median rent by town & flat type (quarterly, 2005-Q2+).
    Normalises flat_type and converts median_rent to numeric (NaN for 'na').
    """
    try:
        df = pd.read_csv(RENTAL_MEDIAN_CSV)
        df["median_rent"] = pd.to_numeric(df["median_rent"], errors="coerce")
        df["flat_type"] = df["flat_type"].str.strip().map(
            lambda x: RENTAL_FLAT_TYPE_MAP.get(x, x)
        )
        # Parse quarter string "2005-Q2" -> period
        df["quarter_dt"] = pd.PeriodIndex(df["quarter"].str.replace("-Q", "Q"), freq="Q").to_timestamp()
        return df
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_ura_ppi():
    """Load URA Private Residential Property Price Index (quarterly, base 2009-Q1=100).
    Columns: quarter (str), property_type (str), index (float).
    """
    try:
        df = pd.read_csv(URA_PPI_CSV)
        df["index"] = pd.to_numeric(df["index"], errors="coerce")
        df["quarter_dt"] = pd.PeriodIndex(df["quarter"].str.replace("-Q", "Q"), freq="Q").to_timestamp()
        return df
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_ura_txn_agg():
    """Load URA private residential quarterly transaction aggregate.
    Columns: quarter, type_of_sale, sale_status, units.
    """
    try:
        df = pd.read_csv(URA_TXN_AGG_CSV)
        df["units"] = pd.to_numeric(df["units"], errors="coerce")
        df["quarter_dt"] = pd.PeriodIndex(df["quarter"].str.replace("-Q", "Q"), freq="Q").to_timestamp()
        return df
    except FileNotFoundError:
        return pd.DataFrame()


# ── Phase 2 loaders ──────────────────────────────────────────────

def _load_amenity(path: str, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    """Generic amenity CSV loader — returns empty DataFrame if not yet fetched."""
    try:
        df = pd.read_csv(path)
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        return df.dropna(subset=[lat_col, lon_col])
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_hawker_centres() -> pd.DataFrame:
    """Hawker centres with lat/lon (NEA). Columns: name, lat, lon, postal_code, street, status."""
    return _load_amenity(HAWKER_CSV)


@st.cache_data
def load_community_clubs() -> pd.DataFrame:
    """Community clubs with lat/lon (PA). Columns: name, lat, lon, postal_code, street."""
    return _load_amenity(CC_CSV)


@st.cache_data
def load_parks() -> pd.DataFrame:
    """Parks with lat/lon (NParks). Columns: name, lat, lon."""
    return _load_amenity(PARKS_CSV)


@st.cache_data
def load_polyclinics() -> pd.DataFrame:
    """Polyclinics with lat/lon (HPB). Columns: name, lat, lon, postal_code, street."""
    return _load_amenity(POLYCLINICS_CSV)


@st.cache_data
def load_schools() -> pd.DataFrame:
    """Schools geocoded via OneMap (MOE).
    Columns: school_name, address, postal_code, lat, lon, mainlevel_code, type_code, zone_code.
    mainlevel_code: PRIMARY | SECONDARY | JUNIOR COLLEGE | MIXED LEVELS
    """
    return _load_amenity(SCHOOLS_CSV)


@st.cache_data
def load_cpi() -> pd.DataFrame:
    """CPI monthly data (SINGSTAT, base 2024=100).
    Columns: series (str), date (datetime), cpi_index (float).
    Key series: 'All Items', 'Accommodation', 'Food', 'Transport', 'Healthcare'.
    """
    try:
        df = pd.read_csv(CPI_CSV, parse_dates=["date"])
        df["cpi_index"] = pd.to_numeric(df["cpi_index"], errors="coerce")
        return df.dropna(subset=["cpi_index"])
    except FileNotFoundError:
        return pd.DataFrame()


# ── Phase 3 loaders (I2 — URA developer pipeline) ────────────────

def _load_ura_pipeline(path: str) -> pd.DataFrame:
    """Generic loader for URA quarterly pipeline CSVs. Returns empty DF if missing."""
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        if "units" in df.columns:
            df["units"] = pd.to_numeric(df["units"], errors="coerce")
        # Normalise quarter column: "2024-Q1" or "2024Q1" → Period
        if "quarter" in df.columns:
            q_str = df["quarter"].astype(str).str.replace("-Q", "Q", regex=False)
            df["quarter_dt"] = pd.PeriodIndex(q_str, freq="Q").to_timestamp()
        return df
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_ura_launched() -> pd.DataFrame:
    """URA units launched by market segment, quarterly (2004+, 270 rows).
    Columns: quarter, market_segment, units, quarter_dt.
    market_segment: 'Core Central Region' | 'Rest of Central Region' | 'Outside Central Region'
    """
    return _load_ura_pipeline(URA_LAUNCHED_CSV)


@st.cache_data
def load_ura_sold() -> pd.DataFrame:
    """URA units sold (uncompleted) by market segment, quarterly (2004+, 270 rows).
    Same schema as load_ura_launched().
    """
    return _load_ura_pipeline(URA_SOLD_CSV)


@st.cache_data
def load_condo_clean() -> pd.DataFrame:
    """
    URA individual private residential transaction caveats, cleaned.
    Source: combine_clean_condo.py (run after fetch_data.py Phase 3).
    Coverage: ~3 years (latest 4 postal-district batches from URA API).
    134k+ rows; Aug 2021 – present.

    Key columns:
        project, street, district, district_name, market_segment,
        property_type, property_type_broad (Condo/Apartment | EC | Landed | Strata Landed),
        type_of_sale (New Sale | Sub Sale | Resale),
        tenure_clean (Freehold | 99 years | 999 years | ...),
        floor_range, no_of_units, type_of_area,
        contract_date (datetime), contract_year, contract_month, contract_quarter,
        price, nett_price, area_sqm, price_psm, nett_price_psm,
        lat, lon (WGS84, 79% coverage — landed has no coords)
    """
    if not os.path.exists(CONDO_CLEAN_CSV):
        return pd.DataFrame()
    df = pd.read_csv(CONDO_CLEAN_CSV, parse_dates=["contract_date"])
    df["price_psm"]      = pd.to_numeric(df["price_psm"],      errors="coerce")
    df["nett_price_psm"] = pd.to_numeric(df["nett_price_psm"], errors="coerce")
    df["area_sqm"]       = pd.to_numeric(df["area_sqm"],       errors="coerce")
    df["price"]          = pd.to_numeric(df["price"],          errors="coerce")
    df["district"]       = pd.to_numeric(df["district"],       errors="coerce").astype("Int64")
    return df


@st.cache_data
def load_ura_unsold() -> pd.DataFrame:
    """URA unsold units with planning approvals, quarterly (2006+, 960 rows).
    Columns: quarter, market_segment, type_of_development, status, type_of_sale, units, quarter_dt.
    Filter for: type_of_development='Uncompleted', status='Launched', type_of_sale='With Pre-Requisites'
    to get the active inventory pipeline.
    """
    return _load_ura_pipeline(URA_UNSOLD_CSV)


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
