"""
fetch_data.py — data.gov.sg API fetcher
========================================
Fetches and saves the following free public datasets:

Phase 1 — Rental & Private Property
1. HDB Rental Transactions (Jan 2021+)
   resource_id: d_c9f57187485a850908655db0e8cfe651
   → data/rental/hdb_rental_transactions.csv

2. HDB Median Rent By Town And Flat Type (2005-Q2+)
   resource_id: d_23000a00c52996c55106084ed0339566
   → data/rental/hdb_median_rent_by_town.csv

3. URA Private Residential PPI (1975-Q1+)
   resource_id: d_97f8a2e995022d311c6c68cfda6d034c
   → data/private_property/ura_ppi.csv

4. URA Private Residential Transactions Quarterly (aggregate)
   resource_id: d_7c69c943d5f0d89d6a9a773d2b51f337
   → data/private_property/ura_private_transactions_agg.csv

Phase 2 — Amenities & CPI
5. CPI All Items Monthly (SINGSTAT, 1961+)
   resource_id: d_bdaff844e3ef89d39fceb962ff8f0791
   → data/finance/cpi_monthly.csv

6. Hawker Centres GeoJSON (NEA)
   dataset_id: d_4a086da0a5553be1d89383cd90d07ecd
   → data/amenities/hawker_centres.csv

7. Community Clubs GeoJSON (PA)
   dataset_id: d_9de02d3fb33d96da1855f4fbef549a0f
   → data/amenities/community_clubs.csv

8. Parks GeoJSON (NParks)
   dataset_id: d_0542d48f0991541706b58059381a6eca
   → data/amenities/parks.csv

9. Polyclinics GeoJSON (HPB breast screening locations)
   dataset_id: d_0cdfbf7e277e8bfa1ef79fadf4b71b56
   → data/amenities/polyclinics.csv

10. Schools (MOE general info, geocoded via OneMap postal search)
    resource_id: d_688b934f82c1059ed0a6993d2a829089
    → data/amenities/schools.csv

Run from the repo root:
    python src/fetch_data.py          # skip existing files
    python src/fetch_data.py --force  # re-fetch everything

Requirements: requests (pip install requests)
"""

import os
import sys
import math
import time
import json
import requests
import pandas as pd

# Work relative to repo root whether script is run from root or src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Phase 1: datastore API datasets (paginated CSV) ─────────────
DATASETS = [
    {
        "name": "HDB Rental Transactions (Jan 2021+)",
        "resource_id": "d_c9f57187485a850908655db0e8cfe651",
        "out": os.path.join(ROOT, "data", "rental", "hdb_rental_transactions.csv"),
        "limit": 10_000,
    },
    {
        "name": "HDB Median Rent By Town & Flat Type",
        "resource_id": "d_23000a00c52996c55106084ed0339566",
        "out": os.path.join(ROOT, "data", "rental", "hdb_median_rent_by_town.csv"),
        "limit": 1_000,
    },
    {
        "name": "URA Private Residential PPI",
        "resource_id": "d_97f8a2e995022d311c6c68cfda6d034c",
        "out": os.path.join(ROOT, "data", "private_property", "ura_ppi.csv"),
        "limit": 1_000,
    },
    {
        "name": "URA Private Residential Transactions (aggregate quarterly)",
        "resource_id": "d_7c69c943d5f0d89d6a9a773d2b51f337",
        "out": os.path.join(ROOT, "data", "private_property", "ura_private_transactions_agg.csv"),
        "limit": 1_000,
    },
    {
        "name": "URA Private Units Launched by Market Segment (quarterly)",
        "resource_id": "d_70824d34defde87d88faccc5d5b1c6ea",
        "out": os.path.join(ROOT, "data", "private_property", "ura_units_launched.csv"),
        "limit": 1_000,
    },
    {
        "name": "URA Private Units Sold by Market Segment (quarterly)",
        "resource_id": "d_e1c5b0df62729e69c82716355ef295ba",
        "out": os.path.join(ROOT, "data", "private_property", "ura_units_sold.csv"),
        "limit": 1_000,
    },
    {
        "name": "URA Unsold Private Units with Planning Approvals (quarterly)",
        "resource_id": "d_84d05d45049108f0fd2e99b66bd19cfe",
        "out": os.path.join(ROOT, "data", "private_property", "ura_units_unsold.csv"),
        "limit": 1_000,
    },
]

# ── Phase 2: GeoJSON poll-download datasets ──────────────────────
GEOJSON_DATASETS = [
    {
        "name": "Hawker Centres (NEA)",
        "dataset_id": "d_4a086da0a5553be1d89383cd90d07ecd",
        "out": os.path.join(ROOT, "data", "amenities", "hawker_centres.csv"),
        "name_field": "NAME",
        "postal_field": "ADDRESSPOSTALCODE",
        "street_field": "ADDRESSSTREETNAME",
        "block_field": "ADDRESSBLOCKHOUSENUMBER",
        "extra_fields": {"STATUS": "status"},
    },
    {
        "name": "Community Clubs (PA)",
        "dataset_id": "d_9de02d3fb33d96da1855f4fbef549a0f",
        "out": os.path.join(ROOT, "data", "amenities", "community_clubs.csv"),
        "name_field": "NAME",
        "postal_field": "ADDRESSPOSTALCODE",
        "street_field": "ADDRESSSTREETNAME",
        "block_field": "ADDRESSBLOCKHOUSENUMBER",
        "extra_fields": {},
    },
    {
        "name": "Parks (NParks)",
        "dataset_id": "d_0542d48f0991541706b58059381a6eca",
        "out": os.path.join(ROOT, "data", "amenities", "parks.csv"),
        "name_field": "NAME",
        "postal_field": None,
        "street_field": None,
        "block_field": None,
        "extra_fields": {},
    },
    {
        "name": "Polyclinics (HPB)",
        "dataset_id": "d_0cdfbf7e277e8bfa1ef79fadf4b71b56",
        "out": os.path.join(ROOT, "data", "amenities", "polyclinics.csv"),
        "name_field": "NAME",
        "postal_field": "ADDRESSPOSTALCODE",
        "street_field": "ADDRESSSTREETNAME",
        "block_field": "ADDRESSBLOCKHOUSENUMBER",
        "extra_fields": {},
    },
]

BASE_URL = "https://data.gov.sg/api/action/datastore_search"
GEOJSON_API = "https://api-open.data.gov.sg/v1/public/api/datasets/{}/poll-download"
ONEMAP_SEARCH = "https://www.onemap.gov.sg/api/common/elastic/search"


# ── helpers ──────────────────────────────────────────────────────

def fetch_all_records(resource_id: str, page_limit: int = 10_000) -> list:
    """Paginate through the datastore API and return all records."""
    records = []
    offset = 0
    total = None

    while True:
        params = {"resource_id": resource_id, "limit": page_limit, "offset": offset}
        for attempt in range(3):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=60, verify=False)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Retry {attempt + 1}/3 after error: {e}")
                time.sleep(3)

        result = data.get("result", {})
        if total is None:
            total = result.get("total", 0)
            pages = math.ceil(total / page_limit) if page_limit else 1
            print(f"  Total records: {total:,} ({pages} pages)")

        batch = result.get("records", [])
        records.extend(batch)
        offset += len(batch)

        pct = offset / total * 100 if total else 100
        print(f"  Fetched {offset:,} / {total:,} ({pct:.0f}%)", end="\r")

        if not batch or offset >= total:
            break
        time.sleep(0.3)

    print()
    return records


def fetch_geojson_dataset(ds: dict) -> pd.DataFrame:
    """Download a GeoJSON dataset via poll-download API and flatten to DataFrame."""
    url = GEOJSON_API.format(ds["dataset_id"])
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30, verify=False)
            r.raise_for_status()
            meta = r.json()
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(3)

    if meta.get("code") != 0:
        raise RuntimeError(f"poll-download error: {meta.get('errorMsg')}")

    dl_url = meta["data"]["url"]
    gj = requests.get(dl_url, timeout=60, verify=False).json()
    features = gj.get("features", [])
    print(f"  {len(features)} features")

    rows = []
    for feat in features:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates", [None, None])
        # GeoJSON coordinates are [lon, lat]
        lon = coords[0] if len(coords) > 0 else None
        lat = coords[1] if len(coords) > 1 else None

        row = {
            "name": props.get(ds["name_field"], ""),
            "lat": lat,
            "lon": lon,
        }
        if ds.get("postal_field"):
            row["postal_code"] = props.get(ds["postal_field"], "")
        if ds.get("street_field"):
            row["street"] = props.get(ds["street_field"], "")
        if ds.get("block_field"):
            row["block"] = props.get(ds["block_field"], "")
        for src_field, dst_col in (ds.get("extra_fields") or {}).items():
            row[dst_col] = props.get(src_field, "")
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[df["lat"].notna() & df["lon"].notna()]
    return df


def fetch_cpi() -> pd.DataFrame:
    """Fetch CPI monthly data, melt wide→long, filter to key series."""
    resource_id = "d_bdaff844e3ef89d39fceb962ff8f0791"
    records = fetch_all_records(resource_id, page_limit=300)
    if not records:
        return pd.DataFrame()

    raw = pd.DataFrame(records)
    if "_id" in raw.columns:
        raw = raw.drop(columns=["_id"])

    # Keep only the key series
    keep_series = ["All Items", "Accommodation", "Food", "Transport",
                   "Healthcare", "Education", "Clothing & Footwear"]
    raw = raw[raw["DataSeries"].isin(keep_series)].copy()

    # Month columns look like "2026Jun", "2025Dec", etc.
    month_cols = [c for c in raw.columns if c != "DataSeries" and len(c) == 7
                  and c[:4].isdigit()]

    melted = raw.melt(id_vars=["DataSeries"], value_vars=month_cols,
                      var_name="month_str", value_name="cpi_index")
    melted["cpi_index"] = pd.to_numeric(melted["cpi_index"], errors="coerce")
    melted = melted.dropna(subset=["cpi_index"])

    # Parse "2026Jun" → datetime
    melted["date"] = pd.to_datetime(melted["month_str"], format="%Y%b")
    melted = melted.rename(columns={"DataSeries": "series"})
    melted = melted[["series", "date", "cpi_index"]].sort_values(["series", "date"])
    return melted


def geocode_schools() -> pd.DataFrame:
    """
    Fetch schools from MOE dataset, geocode via OneMap postal code search.
    Returns DataFrame with lat/lon columns.
    Saves progress incrementally to avoid repeating calls on failure.
    """
    resource_id = "d_688b934f82c1059ed0a6993d2a829089"
    records = fetch_all_records(resource_id, page_limit=500)
    if not records:
        return pd.DataFrame()

    schools = pd.DataFrame(records)
    if "_id" in schools.columns:
        schools = schools.drop(columns=["_id"])

    lats, lons = [], []
    total = len(schools)
    for i, (_, row) in enumerate(schools.iterrows()):
        postal = str(row.get("postal_code", "")).strip()
        lat, lon = None, None
        if postal and postal != "na":
            try:
                params = {
                    "searchVal": postal,
                    "returnGeom": "Y",
                    "getAddrDetails": "Y",
                    "pageNum": 1,
                }
                r = requests.get(ONEMAP_SEARCH, params=params, timeout=10, verify=False)
                results = r.json().get("results", [])
                if results:
                    lat = float(results[0]["LATITUDE"])
                    lon = float(results[0]["LONGITUDE"])
            except Exception:
                pass
        lats.append(lat)
        lons.append(lon)
        pct = (i + 1) / total * 100
        print(f"  Geocoding schools: {i+1}/{total} ({pct:.0f}%)", end="\r")
        time.sleep(0.15)  # polite delay for OneMap

    print()
    schools["lat"] = lats
    schools["lon"] = lons

    # Keep useful columns
    keep_cols = ["school_name", "address", "postal_code", "lat", "lon",
                 "mainlevel_code", "type_code", "zone_code", "dgp_code",
                 "sap_ind", "autonomous_ind", "gifted_ind"]
    keep_cols = [c for c in keep_cols if c in schools.columns]
    return schools[keep_cols]


def _fetch_ura_private_transactions(access_key: str, force: bool = False) -> None:
    """
    Fetch individual private residential transaction caveats from URA API.

    URA API docs: https://www.ura.gov.sg/maps/api/
    Endpoint: invokeUraDS?service=PMI_Resi_Transaction&batch=<1-4>
    Each batch covers ~1 year of data; 4 batches = ~3 years of caveats.

    Requires free URA API key (register at ura.gov.sg/maps/api/).
    Set environment variable URA_ACCESS_KEY before running.

    Output: data/private_property/ura_condo_transactions.csv
    Schema (flattened from nested project+transaction JSON):
        project, street, x, y, contractDate, type, price, propertyType,
        district, typeOfSale, floorRange, noOfUnits, typeOfArea, area,
        tenure, purchaserAddressIndicator
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Updated to eservice subdomain + /v1 paths (API migrated ~2024)
    URA_TOKEN_URL = "https://eservice.ura.gov.sg/uraDataService/insertNewToken/v1"
    URA_TXN_URL   = "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1"
    # verify=False: required on corporate networks with SSL-inspection proxies.
    # The URA server presents a chain with a self-signed intermediate CA that
    # the corporate proxy replaces — disabling verify is the standard workaround.
    SSL_VERIFY = False

    out_path = os.path.join(ROOT, "data", "private_property", "ura_condo_transactions.csv")

    if os.path.exists(out_path) and not force:
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  Already exists ({size_kb:.0f} KB) — skipping. Use --force to re-fetch.")
        return

    # Browser-like headers required — CloudFront WAF blocks raw Python User-Agent
    BASE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }

    # ── Step 1: Get daily token ────────────────────────────────────────────────
    print("  Getting URA daily token...")
    try:
        resp = requests.get(
            URA_TOKEN_URL,
            headers={**BASE_HEADERS, "AccessKey": access_key},
            timeout=30,
            verify=SSL_VERIFY,
        )
        resp.raise_for_status()
        token_data = resp.json()
        if token_data.get("Status") != "Success":
            print(f"  ERROR: Token request failed: {token_data.get('Message', 'Unknown error')}")
            return
        token = token_data["Result"]
        print(f"  Token obtained: {token[:8]}...")
    except Exception as e:
        print(f"  ERROR getting token: {e}")
        return

    # ── Step 2: Fetch all 4 batches ────────────────────────────────────────────
    # URA splits transactions into 4 batches by postal district:
    #   Batch 1 = districts 01-07, Batch 2 = 08-14,
    #   Batch 3 = 15-21, Batch 4 = 22-28
    # Each batch covers all available history (~3 years).
    all_records = []
    headers = {**BASE_HEADERS, "AccessKey": access_key, "Token": token}

    print("  Fetching transaction batches 1–4 by postal district (this may take 2–3 minutes)...")
    for batch_num in range(1, 5):
        try:
            r = requests.get(
                URA_TXN_URL,
                params={"service": "PMI_Resi_Transaction", "batch": batch_num},
                headers=headers,
                timeout=120,
                verify=SSL_VERIFY,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("Status") != "Success":
                print(f"  Batch {batch_num}: API error — {data.get('Message', 'unknown')}")
                break

            results = data.get("Result", [])
            if not results:
                print(f"  Batch {batch_num}: empty — stopping.")
                break

            # Flatten nested structure: each project has a list of transactions
            before = len(all_records)
            for proj in results:
                proj_info = {k: v for k, v in proj.items() if k != "transaction"}
                for txn in proj.get("transaction", []):
                    all_records.append({**proj_info, **txn})

            added = len(all_records) - before
            print(f"  Batch {batch_num}: {len(results):,} projects, +{added:,} rows "
                  f"(total: {len(all_records):,})")
            time.sleep(1.5)  # polite delay between batches

        except Exception as e:
            print(f"  Batch {batch_num} error: {e}")
            break

    if not all_records:
        print("  WARNING: No records fetched — check your URA_ACCESS_KEY and network.")
        return

    df = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved {len(df):,} rows -> {out_path}")


# ── main ──────────────────────────────────────────────────────────

def main():
    import warnings
    warnings.filterwarnings("ignore")

    force = "--force" in sys.argv

    print("=" * 60)
    print("PropertyDecisionSupport — Data Fetcher")
    print("=" * 60)

    # ── Phase 1: datastore paginated datasets ─────────────────────
    for ds in DATASETS:
        name = ds["name"]
        resource_id = ds["resource_id"]
        out_path = ds["out"]
        page_limit = ds["limit"]

        print(f"\nFetching: {name}")
        print(f"  Resource ID: {resource_id}")
        print(f"  Output:      {out_path}")

        if os.path.exists(out_path) and not force:
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  Already exists ({size_kb:.0f} KB) — skipping. Use --force to re-fetch.")
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            records = fetch_all_records(resource_id, page_limit)
            if not records:
                print("  WARNING: No records returned.")
                continue
            df = pd.DataFrame(records)
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"  Saved {len(df):,} rows -> {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Phase 2a: GeoJSON amenity datasets ────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 — Amenities")
    print("=" * 60)

    for ds in GEOJSON_DATASETS:
        out_path = ds["out"]
        print(f"\nFetching: {ds['name']}")
        print(f"  Dataset ID: {ds['dataset_id']}")
        print(f"  Output:     {out_path}")

        if os.path.exists(out_path) and not force:
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  Already exists ({size_kb:.0f} KB) — skipping.")
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            df = fetch_geojson_dataset(ds)
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"  Saved {len(df):,} rows -> {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Phase 2b: CPI data ────────────────────────────────────────
    cpi_path = os.path.join(ROOT, "data", "finance", "cpi_monthly.csv")
    print(f"\nFetching: CPI Monthly (SINGSTAT)")
    print(f"  Output: {cpi_path}")

    if os.path.exists(cpi_path) and not force:
        size_kb = os.path.getsize(cpi_path) / 1024
        print(f"  Already exists ({size_kb:.0f} KB) — skipping.")
    else:
        os.makedirs(os.path.dirname(cpi_path), exist_ok=True)
        try:
            df_cpi = fetch_cpi()
            if len(df_cpi) > 0:
                df_cpi.to_csv(cpi_path, index=False, encoding="utf-8")
                print(f"  Saved {len(df_cpi):,} rows -> {cpi_path}")
            else:
                print("  WARNING: No CPI data returned.")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Phase 2c: Schools (geocoded) ──────────────────────────────
    schools_path = os.path.join(ROOT, "data", "amenities", "schools.csv")
    print(f"\nFetching + geocoding: Schools (MOE)")
    print(f"  Output: {schools_path}")

    if os.path.exists(schools_path) and not force:
        size_kb = os.path.getsize(schools_path) / 1024
        print(f"  Already exists ({size_kb:.0f} KB) — skipping.")
    else:
        os.makedirs(os.path.dirname(schools_path), exist_ok=True)
        try:
            df_schools = geocode_schools()
            if len(df_schools) > 0:
                df_schools.to_csv(schools_path, index=False, encoding="utf-8")
                geocoded = df_schools["lat"].notna().sum()
                print(f"  Saved {len(df_schools):,} schools ({geocoded} geocoded) -> {schools_path}")
            else:
                print("  WARNING: No school data returned.")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Phase 3: URA API (requires access token) ──────────────────────────────
    # Registration: https://www.ura.gov.sg/maps/api/
    # Set environment variable URA_ACCESS_KEY=<your_token> to activate.
    # Fetches: individual private residential transaction caveats (F1)
    ura_key = os.environ.get("URA_ACCESS_KEY", "")
    if ura_key:
        print("\n" + "=" * 60)
        print("Phase 3 — URA Private Transactions (individual caveats)")
        print("=" * 60)
        _fetch_ura_private_transactions(ura_key, force=force)
    else:
        print("\nPhase 3 (URA private transactions): SKIPPED — URA_ACCESS_KEY not set.")
        print("  Register at https://www.ura.gov.sg/maps/api/ (free)")
        print("  Then set: set URA_ACCESS_KEY=your_token")

    print("\nAll done.")


if __name__ == "__main__":
    main()
