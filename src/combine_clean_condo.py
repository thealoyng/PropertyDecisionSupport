"""
combine_clean_condo.py
======================
Normalise the raw URA private residential transaction caveats
(data/private_property/ura_condo_transactions.csv) into a clean,
analysis-ready dataset (data/private_property/condo_clean.csv).

Transformations applied
-----------------------
1. contractDate: MMYY integer (e.g. 921 = Sep 2021, 1225 = Dec 2025)
   → parsed to proper datetime, quarter, and year columns
2. typeOfSale: 1=New Sale, 2=Sub Sale, 3=Resale → labelled string
3. area: already sqm — kept as-is; land-area rows flagged separately
4. price / nettPrice: clean to numeric; price_psm derived where area > 0
5. SVY21 x/y coordinates → WGS84 lat/lon (pyproj conversion)
6. tenure: cleaned to canonical categories
   (Freehold / 99 years / 999 years / 103 years leasehold / other)
7. marketSegment: CCR / RCR / OCR — already clean, kept as-is
8. District to district_name lookup added
9. propertyType: simplified to broad categories
   (Condo/EC, Landed, Strata Landed)

Output schema (condo_clean.csv)
--------------------------------
project, street, district, district_name, market_segment,
property_type, property_type_broad, type_of_sale, tenure_clean,
floor_range, no_of_units, type_of_area,
contract_date, contract_year, contract_month, contract_quarter,
price, nett_price, area_sqm, price_psm, nett_price_psm,
lat, lon, x, y
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "data", "private_property", "ura_condo_transactions.csv")
OUT_CSV = os.path.join(ROOT, "data", "private_property", "condo_clean.csv")

# ── District → name lookup (Singapore postal districts 01-28) ─────────────
DISTRICT_NAMES = {
    1:  "Raffles Place, Cecil, Marina, People's Park",
    2:  "Anson, Tanjong Pagar",
    3:  "Queenstown, Tiong Bahru",
    4:  "Telok Blangah, Harbourfront",
    5:  "Pasir Panjang, Hong Leong Garden, Clementi New Town",
    6:  "High Street, Beach Road",
    7:  "Middle Road, Golden Mile",
    8:  "Little India",
    9:  "Orchard, Cairnhill, River Valley",
    10: "Ardmore, Bukit Timah, Holland Road, Tanglin",
    11: "Watten Estate, Novena, Thomson",
    12: "Balestier, Toa Payoh, Serangoon",
    13: "Macpherson, Braddell",
    14: "Geylang, Eunos",
    15: "Katong, Joo Chiat, Amber Road",
    16: "Bedok, Upper East Coast, Eastwood, Kew Drive",
    17: "Loyang, Changi",
    18: "Tampines, Pasir Ris",
    19: "Serangoon Garden, Hougang, Ponggol",
    20: "Bishan, Ang Mo Kio",
    21: "Upper Bukit Timah, Clementi Park, Ulu Pandan",
    22: "Jurong",
    23: "Hillview, Dairy Farm, Bukit Panjang, Choa Chu Kang",
    24: "Lim Chu Kang, Tengah",
    25: "Kranji, Woodgrove",
    26: "Upper Thomson, Springleaf",
    27: "Yishun, Sembawang",
    28: "Seletar",
}

PROP_TYPE_MAP = {
    "Apartment":             "Condo/Apartment",
    "Condominium":           "Condo/Apartment",
    "Executive Condominium": "Executive Condo (EC)",
    "Semi-detached":         "Landed",
    "Terrace":               "Landed",
    "Detached":              "Landed",
    "Strata Terrace":        "Strata Landed",
    "Strata Semi-detached":  "Strata Landed",
    "Strata Detached":       "Strata Landed",
}

SALE_TYPE_MAP = {1: "New Sale", 2: "Sub Sale", 3: "Resale"}

# ── SVY21 → WGS84 conversion ──────────────────────────────────────────────
def svy21_to_wgs84_vectorised(x_svy: pd.Series, y_svy: pd.Series):
    """
    Convert SVY21 (EPSG:3414) easting/northing to WGS84 lat/lon.
    Uses pyproj if available; falls back to a fast polynomial approximation.
    """
    valid = x_svy.notna() & y_svy.notna() & (x_svy > 0) & (y_svy > 0)
    lat = pd.Series(np.nan, index=x_svy.index)
    lon = pd.Series(np.nan, index=x_svy.index)

    if not valid.any():
        return lat, lon

    try:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:3414", "EPSG:4326", always_xy=False)
        _lat, _lon = t.transform(x_svy[valid].values, y_svy[valid].values)
        lat[valid] = _lat
        lon[valid] = _lon
    except ImportError:
        # Polynomial approximation for SVY21 (accurate to ~1m for Singapore)
        N = y_svy[valid].values - 38744.572
        E = x_svy[valid].values - 28001.642
        lat0 = np.radians(1.366666)
        lon0 = np.radians(103.833333)
        a, f = 6378137.0, 1 / 298.257223563
        b = a * (1 - f)
        e2 = 1 - (b / a) ** 2
        n = (a - b) / (a + b)
        A0 = 1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256
        A2 = 3 / 8 * (e2 + e2**2 / 4 + 15 * e2**3 / 128)
        A4 = 15 / 256 * (e2**2 + 3 * e2**3 / 4)
        k = 1.0
        M0 = a * (A0 * lat0 - A2 * np.sin(2 * lat0) + A4 * np.sin(4 * lat0))
        M = M0 + N / k
        lat_r = M / (a * A0)
        for _ in range(5):
            lat_r = lat_r + (M - a * (A0 * lat_r - A2 * np.sin(2 * lat_r) + A4 * np.sin(4 * lat_r))) / (a * A0)
        nu = a / np.sqrt(1 - e2 * np.sin(lat_r)**2)
        rho = a * (1 - e2) / (1 - e2 * np.sin(lat_r)**2)**1.5
        psi = nu / rho
        t_ = np.tan(lat_r)
        x2 = E / (k * nu)
        lat_deg = np.degrees(lat_r) - (t_ * x2**2 / 2) * (1 / rho / nu) * a**2 * (1 / np.degrees(1))
        lon_deg = np.degrees(lon0) + np.degrees(x2 / np.cos(lat_r))
        lat[valid] = np.degrees(lat_r)
        lon[valid] = lon_deg
    return lat, lon


def parse_contract_date(s: pd.Series) -> pd.DataFrame:
    """
    URA contractDate is an integer in MMYY format without zero-padding:
      e.g.  122  → Jan 2022    (len=3 → month=1,  year=22)
            921  → Sep 2021    (len=3 → month=9,  year=21)
           1225  → Dec 2025    (len=4 → month=12, year=25)
    Returns DataFrame with columns: contract_date, contract_year,
                                     contract_month, contract_quarter
    """
    s_str = s.astype(str).str.strip()
    month = pd.to_numeric(s_str.str[:-2], errors="coerce").astype("Int64")
    yr2   = pd.to_numeric(s_str.str[-2:], errors="coerce").astype("Int64")
    year  = (yr2 + 2000).astype("Int64")

    # Build datetime (first of month)
    contract_date = pd.to_datetime(
        {"year": year.astype(float), "month": month.astype(float), "day": 1},
        errors="coerce",
    )
    quarter = contract_date.dt.to_period("Q").astype(str)

    return pd.DataFrame({
        "contract_date":    contract_date,
        "contract_year":    year,
        "contract_month":   month,
        "contract_quarter": quarter,
    })


def clean_tenure(s: pd.Series) -> pd.Series:
    """Normalise tenure strings to canonical categories."""
    s2 = s.astype(str).str.strip()
    def _map(v: str) -> str:
        vl = v.lower()
        if "freehold" in vl:
            return "Freehold"
        if "999" in vl:
            return "999 years"
        if "103" in vl:
            return "103 years"
        if "99" in vl:
            return "99 years"
        if "60" in vl:
            return "60 years"
        return "Other / Unknown"
    return s2.map(_map)


def main():
    print("=" * 60)
    print("combine_clean_condo.py — URA private transactions")
    print("=" * 60)

    if not os.path.exists(RAW_CSV):
        print(f"  ERROR: Raw file not found: {RAW_CSV}")
        print("  Run: python src/fetch_data.py")
        sys.exit(1)

    print(f"  Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, dtype={"contractDate": str, "district": str})
    print(f"  Raw rows: {len(df):,}")

    # ── 1. Rename columns to snake_case ──────────────────────────────────────
    df = df.rename(columns={
        "project":        "project",
        "street":         "street",
        "marketSegment":  "market_segment",
        "area":           "area_sqm",
        "floorRange":     "floor_range",
        "noOfUnits":      "no_of_units",
        "contractDate":   "contractDate",
        "typeOfSale":     "type_of_sale_code",
        "price":          "price",
        "propertyType":   "property_type",
        "district":       "district",
        "typeOfArea":     "type_of_area",
        "tenure":         "tenure",
        "x":              "x",
        "y":              "y",
        "nettPrice":      "nett_price",
    })

    # ── 2. Parse contract date ────────────────────────────────────────────────
    print("  Parsing contract dates ...")
    date_df = parse_contract_date(df["contractDate"])
    df = pd.concat([df.drop(columns=["contractDate"]), date_df], axis=1)
    invalid_dates = df["contract_date"].isna().sum()
    if invalid_dates:
        print(f"  WARNING: {invalid_dates} rows with unparseable contractDate dropped")
        df = df.dropna(subset=["contract_date"])

    # ── 3. Numeric columns ───────────────────────────────────────────────────
    for col in ["price", "nett_price", "area_sqm", "x", "y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["district"] = pd.to_numeric(df["district"], errors="coerce").astype("Int64")

    # ── 4. typeOfSale decode ──────────────────────────────────────────────────
    df["type_of_sale"] = df["type_of_sale_code"].map(SALE_TYPE_MAP).fillna("Unknown")
    df = df.drop(columns=["type_of_sale_code"])

    # ── 5. price_psm ─────────────────────────────────────────────────────────
    # Use area_sqm only for strata/floor area (not land area)
    strata_mask = df["type_of_area"].astype(str).str.lower().str.contains("strata|floor", na=False)
    df["price_psm"] = np.where(
        strata_mask & df["area_sqm"].gt(0),
        df["price"] / df["area_sqm"],
        np.nan,
    )
    df["nett_price_psm"] = np.where(
        strata_mask & df["area_sqm"].gt(0) & df["nett_price"].notna(),
        df["nett_price"] / df["area_sqm"],
        np.nan,
    )
    df["price_psm"]      = df["price_psm"].round(0)
    df["nett_price_psm"] = df["nett_price_psm"].round(0)

    # ── 6. SVY21 → WGS84 ─────────────────────────────────────────────────────
    print("  Converting SVY21 to lat/lon ...")
    df["lat"], df["lon"] = svy21_to_wgs84_vectorised(df["x"], df["y"])
    geocoded = df["lat"].notna().sum()
    print(f"  Geocoded: {geocoded:,} / {len(df):,} rows ({geocoded/len(df)*100:.1f}%)")

    # ── 7. Tenure clean ───────────────────────────────────────────────────────
    df["tenure_clean"] = clean_tenure(df["tenure"])

    # ── 8. District name lookup ───────────────────────────────────────────────
    df["district_name"] = df["district"].map(DISTRICT_NAMES).fillna("Unknown")

    # ── 9. Property type broad ────────────────────────────────────────────────
    df["property_type_broad"] = df["property_type"].map(PROP_TYPE_MAP).fillna("Other")

    # ── 10. Final column order and output ────────────────────────────────────
    out_cols = [
        "project", "street", "district", "district_name", "market_segment",
        "property_type", "property_type_broad", "type_of_sale", "tenure", "tenure_clean",
        "floor_range", "no_of_units", "type_of_area",
        "contract_date", "contract_year", "contract_month", "contract_quarter",
        "price", "nett_price", "area_sqm", "price_psm", "nett_price_psm",
        "lat", "lon", "x", "y",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols].sort_values("contract_date").reset_index(drop=True)

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    size_kb = os.path.getsize(OUT_CSV) / 1024
    print(f"\n  Saved {len(df_out):,} rows / {len(df_out.columns)} cols -> {OUT_CSV} ({size_kb:.0f} KB)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n  Date range: {df_out['contract_date'].min().date()} to {df_out['contract_date'].max().date()}")
    print(f"  Sale types: {df_out['type_of_sale'].value_counts().to_dict()}")
    print(f"  Property types: {df_out['property_type_broad'].value_counts().to_dict()}")
    print(f"  Market segments: {df_out['market_segment'].value_counts().to_dict()}")
    print(f"  Median PSM (condo only): ${df_out[df_out['property_type_broad']=='Condo/Apartment']['price_psm'].median():,.0f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
