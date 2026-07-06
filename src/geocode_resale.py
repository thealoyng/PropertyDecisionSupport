"""
Geocode unique HDB resale addresses (block + street_name) via OneMap API.

Extracts unique (block, street_name) pairs from the cleaned resale data,
queries OneMap for coordinates, and saves a lookup CSV that can be joined
back to the full dataset.

OneMap is Singapore's official map service — no API key needed for search.

Run from the project root:
    python src/geocode_resale.py

Output:
    data/address_coords.csv  (block, street_name, lat, lon)
"""
import os
import time
import urllib3
import requests
import pandas as pd

# Suppress InsecureRequestWarning when SSL verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CLEAN_CSV = "data/resale_clean_1990_present.csv"
OUTPUT_CSV = "data/address_coords.csv"
ONEMAP = "https://www.onemap.gov.sg/api/common/elastic/search"
DELAY = 0.3  # seconds between API calls


def geocode(query, verify_ssl=True):
    """Return (lat, lon) for a search term, or (None, None) if not found."""
    params = {"searchVal": query, "returnGeom": "Y",
              "getAddrDetails": "Y", "pageNum": 1}
    try:
        r = requests.get(ONEMAP, params=params, timeout=15, verify=verify_ssl)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            top = results[0]
            return round(float(top["LATITUDE"]), 5), round(float(top["LONGITUDE"]), 5)
    except Exception as e:
        print(f"  ! failed for '{query}': {e}")
    return None, None


def main():
    # Load cleaned data and extract unique addresses
    print(f"Loading {CLEAN_CSV}...")
    df = pd.read_csv(CLEAN_CSV, usecols=["block", "street_name"])
    unique = df.drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique):,} unique (block, street_name) pairs")

    # Resume from existing progress if available
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        done_keys = set(zip(existing["block"].astype(str), existing["street_name"]))
        remaining = unique[~unique.apply(
            lambda r: (str(r["block"]), r["street_name"]) in done_keys, axis=1
        )].reset_index(drop=True)
        print(f"Already geocoded: {len(existing):,} | Remaining: {len(remaining):,}")
    else:
        existing = pd.DataFrame(columns=["block", "street_name", "lat", "lon"])
        remaining = unique

    if len(remaining) == 0:
        print("All addresses already geocoded!")
        return

    # Test SSL connectivity; fall back to verify=False if needed
    verify_ssl = True
    try:
        requests.get(ONEMAP, params={"searchVal": "test"}, timeout=10)
    except requests.exceptions.SSLError:
        print("  SSL verification failed (corporate proxy?), retrying without verification...")
        verify_ssl = False

    # Geocode remaining addresses in batches, saving progress periodically
    results = []
    batch_size = 100
    total = len(remaining)

    for i, (_, row) in enumerate(remaining.iterrows()):
        query = f"{row['block']} {row['street_name']}"
        lat, lon = geocode(query, verify_ssl=verify_ssl)
        results.append({
            "block": row["block"],
            "street_name": row["street_name"],
            "lat": lat,
            "lon": lon,
        })

        status = "OK" if lat is not None else "MISS"
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {query:<40} -> {status}")

        # Save progress every batch_size records
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            batch_df = pd.DataFrame(results)
            combined = pd.concat([existing, batch_df], ignore_index=True)
            combined.to_csv(OUTPUT_CSV, index=False)
            existing = combined
            results = []
            hit = combined["lat"].notna().sum()
            print(f"  -- Saved progress: {len(combined):,} total "
                  f"({hit:,} geocoded, {len(combined)-hit:,} missing)")

        time.sleep(DELAY)

    # Final summary
    final = pd.read_csv(OUTPUT_CSV)
    hit = final["lat"].notna().sum()
    miss = final["lat"].isna().sum()
    print(f"\nDone! {len(final):,} addresses total: "
          f"{hit:,} geocoded ({hit/len(final)*100:.1f}%), "
          f"{miss:,} missing ({miss/len(final)*100:.1f}%)")
    print(f"Saved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
