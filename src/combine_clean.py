"""
Combine & clean all HDB resale data (1990 -> present)

"""
import re
import pandas as pd

UP = "data/raw/"
FILES = [
    "Resale_Flat_Prices__Based_on_Approval_Date___1990_-_1999.csv",
    "Resale_Flat_Prices__Based_on_Approval_Date___2000_-_Feb_2012.csv",
    "Resale_Flat_Prices__Based_on_Registration_Date___From_Mar_2012_to_Dec_2014.csv",
    "Resale_Flat_Prices__Based_on_Registration_Date___From_Jan_2015_to_Dec_2016.csv",
    "Resale_flat_prices_based_on_registration_date_from_Jan-2017_onwards.csv",
]

def parse_remaining_lease(val):
    """2017+ format: '61 years 04 months' -> 61.33; 2015-16: '70' -> 70.0"""
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

def storey_mid(s):
    try:
        lo, hi = str(s).upper().split(" TO ")
        return (int(lo) + int(hi)) / 2
    except Exception:
        return None

frames = []
for f in FILES:
    df = pd.read_csv(UP + f)
    df["source_file"] = f
    frames.append(df)
    print(f"loaded {f}: {len(df):,} rows, has_remaining_lease={'remaining_lease' in df.columns}")

raw = pd.concat(frames, ignore_index=True)
print(f"\nCombined raw: {len(raw):,} rows")

# --- harmonize ---
raw["month"] = pd.to_datetime(raw["month"], errors="coerce")
raw["year"] = raw["month"].dt.year
raw["resale_price"] = pd.to_numeric(raw["resale_price"], errors="coerce")
raw["floor_area_sqm"] = pd.to_numeric(raw["floor_area_sqm"], errors="coerce")
raw["lease_commence_date"] = pd.to_numeric(raw["lease_commence_date"], errors="coerce")

# standardize text fields
for c in ["town", "flat_type", "street_name"]:
    raw[c] = raw[c].astype(str).str.strip().str.upper()
raw["flat_model"] = raw["flat_model"].astype(str).str.strip().str.title()
# unify a known flat_type variant
raw["flat_type"] = raw["flat_type"].str.replace("MULTI-GENERATION", "MULTI GENERATION")

# remaining lease: use given value, else compute from lease_commence_date
raw["remaining_lease_yrs"] = raw.get("remaining_lease").apply(parse_remaining_lease) \
    if "remaining_lease" in raw.columns else None
computed = 99 - (raw["year"] - raw["lease_commence_date"])
raw["remaining_lease_yrs"] = raw["remaining_lease_yrs"].fillna(computed)

# engineered features
raw["price_per_sqm"] = (raw["resale_price"] / raw["floor_area_sqm"]).round(0)
raw["flat_age"] = raw["year"] - raw["lease_commence_date"]
raw["storey_mid"] = raw["storey_range"].apply(storey_mid)

before = len(raw)
clean = raw.dropna(subset=["resale_price", "floor_area_sqm", "town", "flat_type", "month"])
print(f"Dropped {before - len(clean):,} rows missing essentials -> {len(clean):,} clean rows")

cols = ["month", "year", "town", "flat_type", "block", "street_name",
        "storey_range", "storey_mid", "floor_area_sqm", "flat_model",
        "lease_commence_date", "flat_age", "remaining_lease_yrs",
        "resale_price", "price_per_sqm"]
clean = clean[cols]
out = "data/resale_clean_1990_present.csv"
clean.to_csv(out, index=False)
print(f"Saved -> {out}")

# --- EDA ---
print("\n" + "=" * 60)
print("EXPLORATORY SUMMARY (1990 -> present)")
print("=" * 60)
print(f"Rows: {len(clean):,}")
print(f"Date range: {clean['month'].min():%b %Y} to {clean['month'].max():%b %Y}")
print(f"Towns: {clean['town'].nunique()} | Flat types: {clean['flat_type'].nunique()}")
print("\nResale price ($) distribution:")
print(clean["resale_price"].describe().round(0).to_string())
print("\nMedian price by flat type:")
print(clean.groupby("flat_type")["resale_price"].median().sort_values().round(0).to_string())
print("\nTop 6 towns by median price/sqm (recent 3 yrs):")
recent = clean[clean["year"] >= clean["year"].max() - 2]
print(recent.groupby("town")["price_per_sqm"].median().sort_values(ascending=False).head(6).round(0).to_string())
print("\nMissing values:")
miss = clean.isna().sum()
print(miss[miss > 0].to_string() or "  none")
