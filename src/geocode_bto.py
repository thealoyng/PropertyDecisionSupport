"""
Geocode BTO projects using the free OneMap Search API.
Reads data/bto_projects.csv, looks up accurate coordinates for each
project's search_term, and overwrites the lat/lon columns.

OneMap is Singapore's official map service — no API key needed for search.

Run from the project root:
    python src/geocode_bto.py
"""
import time
import requests
import pandas as pd

CSV = "data/bto_projects.csv"
ONEMAP = "https://www.onemap.gov.sg/api/common/elastic/search"


def geocode(query):
    """Return (lat, lon) for a search term, or (None, None) if not found."""
    params = {"searchVal": query, "returnGeom": "Y",
              "getAddrDetails": "Y", "pageNum": 1}
    try:
        r = requests.get(ONEMAP, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            top = results[0]
            return float(top["LATITUDE"]), float(top["LONGITUDE"])
    except Exception as e:
        print(f"  ! failed for '{query}': {e}")
    return None, None


if __name__ == "__main__":
    for path in ["data/bto_projects.csv",
                 "data/future_developments.csv",
                 "data/mrt_stations.csv"]:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"(skipping {path} — not found)")
            continue
        print(f"\nGeocoding {len(df)} rows in {path} via OneMap...\n")

        for i, row in df.iterrows():
            lat, lon = geocode(row["search_term"])
            if lat is not None:
                df.at[i, "lat"] = round(lat, 5)
                df.at[i, "lon"] = round(lon, 5)
                print(f"  {str(row['name']):<30} -> {lat:.5f}, {lon:.5f}")
            else:
                print(f"  {str(row['name']):<30} -> kept approximate "
                      f"({row['lat']}, {row['lon']})")
            time.sleep(0.3)  # be polite to the API

        df.to_csv(path, index=False)
        print(f"Saved updated coordinates -> {path}")
