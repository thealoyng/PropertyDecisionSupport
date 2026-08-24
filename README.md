# PropertyDecisionSupport

A comprehensive **property decision-support system** for Singapore HDB buyers, investors, and real estate agents — built on 978k+ HDB resale transactions, rental data, amenity databases, and price index comparisons.

> **Data scope:** 978,391 HDB resale transactions · Jan 1990 – May 2026 · 27 towns · 7 flat types

---

## Features

### Home app (`app.py`) — 5 tabs

| Tab | What it does |
|---|---|
| **Trends** | Median price over time by town and flat type, annual transaction volume |
| **Location map** | BTO projects (Standard / Plus / Prime), MRT stations, URA Master Plan future growth areas |
| **Explorer map** | Interactive block-level pydeck map — colour by PSM, price, volume, avg lease, or value anomalies; heatmap / hex / scatter modes; animated year slider |
| **Payment breakdown** | Staged BTO payment timeline with CPF grants |
| **Forecast** | Prophet time-series price projection |

### EDA pages (`src/pages/`) — 21 analytical pages

#### Descriptive analytics (Pages 1–10)
| Page | Focus |
|---|---|
| 1 · Data Quality | Schema, missing values, coverage gaps, raw vs clean row counts |
| 2 · Distributions | Price and PSM distributions by flat type, storey, floor area |
| 3 · Price Deep Dive | Violin plots, million-dollar analysis, percentile bands, town×type matrix, PSM vs floor area |
| 4 · Temporal Trends | Price and volume over time, policy event overlays, YoY growth |
| 5 · Spatial Analysis | Town bubble map, MRT proximity premium, price along rail lines |
| 6 · Flat Characteristics | Flat model breakdown, storey premium, floor area vs price, flat model PSM differential |
| 7 · Lease Depreciation | Remaining lease vs price, 60-yr threshold effect, cohort-based depreciation (era-corrected) |
| 8 · Market Dynamics | Price-volume cycles, market momentum, concentration heatmaps |
| 9 · Cross-Dataset | HDB resale × BTO projects × MRT × URA future developments |
| 10 · Statistical | Correlation matrix, regression, outlier detection, LOWESS smoothing |

#### Smart Money — Valuation & Intelligence (Pages 11–14)
| Page | Tabs | What it does |
|---|---|---|
| 11 · Value Finder | 5 | Undervalued areas (block vs 1km neighbourhood), comps finder, percentile pricer, floor premium validator, MRT proximity premium |
| 12 · Fair Value | 5 | Hedonic fair value model (Ridge + GB), fair value lookup with 90% intervals, mispricing detector map, comparable engine, distressed sale proxy |
| 13 · MOP Calendar | 5 | MOP unlock timeline to 2030, supply wave impact, town supply radar, lease vintage analysis, HDB→private demand bridge |
| 14 · Market Regime | 6 | Regime classifier, entry/exit seasonality, cooling era event studies, stigma persistence detector, recovery rates, price-vs-volume divergence |

#### Smart Money — Calculators & Cross-Market (Pages 15–17)
| Page | Tabs | What it does |
|---|---|---|
| 15 · Smart Calculator | 5 | Mortgage/MSR/TDSR, CPF eligibility timeline, grant calculator (EHG/PHG/Family), ABSD by buyer profile, HDB→condo upgrade waterfall |
| 16 · HDB vs Private | 4 | HDB PSM index vs URA PPI (2009 base), growth rate comparison, correlation & divergence, private market volume |
| 17 · Rental Yields | 6 | Gross yield heatmap (6.33% avg, best: Bukit Merah 2-Room 10.14%), yield compression over time, net yield/IRR calculator, rental market explorer, buy-vs-rent breakeven, CPI-adjusted real returns |

#### Smart Money — Agent Tools (Pages 18–20)
| Page | Tabs | What it does |
|---|---|---|
| 18 · Property Scout | 3 | Client fit shortlister (budget + lifestyle → ranked towns), town side-by-side comparison, lifestyle-weighted town matching (amenity density + MRT) |
| 19 · Comps Report | — | One-click agent report: enter address → comparable transactions, implied fair value, floor premium, market trend, shareable text summary + CSV export |
| 20 · Opportunity Score | 3 | Opportunity screener (6-dimension decomposed score per block), block desirability fingerprint, negotiation leverage report |

#### Smart Money — Location Intelligence (Page 21)
| Page | Tabs | What it does |
|---|---|---|
| 21 · Location Intel | 4 | Amenity map (129 hawker centres, 128 CCs, 461 parks, 23 polyclinics, 337 schools), town walkability/amenity scores, block-level accessibility calculator, school proximity premium analysis |

#### Smart Money — Supply Intelligence & Backtesting (Pages 22–23)
| Page | Tabs | What it does |
|---|---|---|
| 22 · Supply Intelligence | 3 | Supply pressure score per town (MOP waves + private pipeline), SERS pattern watch list (4-criterion match with explicit caveats), URA developer pipeline dashboard (launched/sold/unsold + months-of-inventory) |
| 23 · Backtesting | 3 | Rolling walk-forward signal validation (value score vs benchmark return), town-level hit-rate heatmap (1yr/3yr/5yr hold), methodology & limitations (look-ahead bias, survivorship bias) |

#### Private Property Track — Tier 3 (Pages 24–25, requires URA API key)
| Page | Tabs | What it does |
|---|---|---|
| 24 · HDB vs Condo ROI | 3 | Full total-return comparison (capital gain + yield − transaction costs) matched by town/district and holding period; break-even cost breakdown with BSD/ABSD calculator |
| 25 · New Launch vs Resale | 3 | New-sale premium vs resale by district and quarter; premium decay curve by transaction year; overpriced-launch screener using z-score anomaly detection |

**Also upgraded (Tier 3 data):**
- Page 16 Tab 5: Unit-level PSM comparison (HDB vs condo, same neighbourhood), quarterly trend, violin distribution, scatter by district
- Page 13 Tab 5: Real private condo volume in MOP-adjacent districts, dual-axis MOP-wave vs condo-volume chart, correlation analysis

---

## Data Architecture

```
PropertyDecisionSupport/
├── raw/                                    # Original HDB CSVs (source of truth)
│   ├── Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv
│   ├── Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv
│   ├── Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv
│   ├── Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv
│   └── Resale flat prices based on registration date from Jan-2017 onwards.csv
│
├── data/                                   # Processed / supplementary datasets
│   ├── resale_clean_1990_present.csv       # Main cleaned dataset (978k rows)
│   ├── address_coords.csv                  # Geocoded block coordinates (9,972 unique addresses)
│   ├── bto_projects.csv                    # BTO launch projects with classification
│   ├── future_developments.csv             # URA Master Plan 2025 growth areas
│   ├── mrt_stations.csv                    # MRT station locations by line
│   │
│   ├── rental/                             # HDB rental data (fetched via fetch_data.py)
│   │   ├── hdb_rental_transactions.csv     # 207k individual rental approvals, Jan 2021+
│   │   └── hdb_median_rent_by_town.csv     # Quarterly medians by town/type, 2005-Q2+
│   │
│   ├── private_property/                   # URA private residential data
│   │   ├── ura_ppi.csv                     # Private property price index, 1975-Q1+
│   │   └── ura_private_transactions_agg.csv # Aggregate quarterly transactions
│   │
│   ├── amenities/                          # Government amenity databases (fetched via fetch_data.py)
│   │   ├── hawker_centres.csv              # 129 hawker centres with coordinates (NEA)
│   │   ├── community_clubs.csv             # 128 community clubs with coordinates (PA)
│   │   ├── parks.csv                       # 461 parks with coordinates (NParks)
│   │   ├── polyclinics.csv                 # 23 polyclinics with coordinates (HPB)
│   │   └── schools.csv                     # 337 schools geocoded via OneMap (MOE)
│   │
│   ├── finance/                            # Macroeconomic data
│   │   └── cpi_monthly.csv                 # CPI All Items + sub-indices, 2000+ (SINGSTAT)
│   │
│   └── metadata/                           # Data governance
│       ├── source_registry.yml             # Every data source: provider, coverage, confidence
│       ├── analysis_registry.yml           # Every analysis: assumptions, gaps, status
│       └── data_dictionary.yml             # Field-level documentation
│
├── src/                                    # Application source code
│   ├── app.py                              # Main Streamlit app (home page)
│   ├── combine_clean.py                    # Data pipeline: raw CSVs → cleaned CSV
│   ├── fetch_data.py                       # API fetcher: rental, amenities, CPI (re-runnable)
│   ├── geocode_resale.py                   # One-time geocoding of block addresses
│   ├── eda_helpers.py                      # Shared data loaders and utility functions
│   └── pages/                             # 21 analytical pages
│       ├── 1_📊_Data_Quality.py
│       ├── 2_📉_Distributions.py
│       ├── 3_💰_Price_Deep_Dive.py
│       ├── 4_📅_Temporal_Trends.py
│       ├── 5_🗺️_Spatial_Analysis.py
│       ├── 6_🏗️_Flat_Characteristics.py
│       ├── 7_📜_Lease_Depreciation.py
│       ├── 8_📊_Market_Dynamics.py
│       ├── 9_🔗_Cross_Dataset.py
│       ├── 10_🔬_Statistical.py
│       ├── 11_🔍_Value_Finder.py
│       ├── 12_📐_Fair_Value.py
│       ├── 13_📅_MOP_Calendar.py
│       ├── 14_📈_Market_Regime.py
│       ├── 15_🧮_Smart_Calculator.py
│       ├── 16_🏢_HDB_vs_Private.py
│       ├── 17_💹_Rental_Yields.py
│       ├── 18_🧭_Property_Scout.py
│       ├── 19_📋_Comps_Report.py
│       ├── 20_⭐_Opportunity_Score.py
│       └── 21_🌳_Location_Intel.py
│
└── docs/                                   # Reference charts (PNG)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- The 5 raw HDB CSV files in `raw/` (download from [data.gov.sg](https://data.gov.sg/collections/189/view))

### Install dependencies

```bash
pip install streamlit pandas plotly pydeck folium streamlit-folium \
            statsmodels scikit-learn scipy numpy-financial
```

### 1. Build the cleaned dataset

Run once, or again whenever you add new raw CSV files:

```bash
python src/combine_clean.py
```

### 2. Fetch supplementary data

Fetches rental, amenity, and CPI data from data.gov.sg (free, no API key required):

```bash
python src/fetch_data.py          # skip files that already exist
python src/fetch_data.py --force  # re-fetch everything (monthly refresh)
```

This downloads: HDB rental transactions, median rents, URA private PPI, hawker centres, community clubs, parks, polyclinics, CPI monthly, and geocodes 337 schools via OneMap (~60s total).

### 3. Run the app

```bash
cd src
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Data Pipeline Detail

### Cleaned dataset columns

| Column | Type | Description |
|---|---|---|
| `month` | date | Transaction month (parsed from raw) |
| `year` | int | Transaction year |
| `town` | str | HDB town (27 towns, uppercase) |
| `flat_type` | str | 1 ROOM / 2 ROOM / 3 ROOM / 4 ROOM / 5 ROOM / EXECUTIVE / MULTI GENERATION |
| `block` | str | Block number |
| `street_name` | str | Street name (uppercase) |
| `storey_range` | str | Raw storey band (e.g. "10 TO 12") |
| `storey_mid` | float | Midpoint of storey band |
| `floor_area_sqm` | float | Floor area in square metres |
| `flat_model` | str | Flat model (title-cased) |
| `lease_commence_date` | int | Year lease started (= build year) |
| `flat_age` | int | Age at time of transaction (years) |
| `remaining_lease_yrs` | float | Remaining lease at time of transaction |
| `resale_price` | float | Transaction price (SGD) |
| `price_per_sqm` | float | Resale price ÷ floor area |

### Schema notes

The 5 raw files span 36 years and have schema differences:
- `remaining_lease` only appears from 2015 onwards; earlier values are back-computed as `99 - flat_age`
- Pre-2012 data uses **approval date**; post-Mar 2012 uses **registration date**
- `remaining_lease` format changed from a plain integer (2015–2016) to a string `"61 years 04 months"` (2017+)
- `flat_model` casing varies across files (all normalised to title case)

### Geocoding (already done — do not re-run unless data changes)

Block-level coordinates for all 9,972 unique `(block, street_name)` pairs were fetched once via the [OneMap API](https://www.onemap.gov.sg/apidocs/) and stored in `data/address_coords.csv`. The Explorer Map uses this for precise block-level dot placement.

```bash
# Only run if address data changes:
python src/geocode_resale.py   # ~40–75 min, fetches ~10k addresses
```

---

## Data Sources

| Dataset | Source | Provider | Update frequency |
|---|---|---|---|
| HDB resale prices | [data.gov.sg](https://data.gov.sg/collections/189/view) | HDB | Monthly |
| HDB rental transactions | [data.gov.sg](https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view) | HDB | Monthly |
| HDB median rent | [data.gov.sg](https://data.gov.sg/datasets/d_23000a00c52996c55106084ed0339566/view) | HDB | Quarterly |
| URA Private PPI | [data.gov.sg](https://data.gov.sg/datasets/d_97f8a2e995022d311c6c68cfda6d034c/view) | URA | Quarterly |
| Hawker centres | [data.gov.sg](https://data.gov.sg/datasets/d_4a086da0a5553be1d89383cd90d07ecd/view) | NEA | Periodic |
| Community clubs | [data.gov.sg](https://data.gov.sg/datasets/d_9de02d3fb33d96da1855f4fbef549a0f/view) | PA | Periodic |
| Parks | [data.gov.sg](https://data.gov.sg/datasets/d_0542d48f0991541706b58059381a6eca/view) | NParks | Periodic |
| Polyclinics | [data.gov.sg](https://data.gov.sg/datasets/d_0cdfbf7e277e8bfa1ef79fadf4b71b56/view) | HPB | Periodic |
| Schools | [data.gov.sg](https://data.gov.sg/datasets/d_688b934f82c1059ed0a6993d2a829089/view) | MOE | Annual |
| CPI monthly | [data.gov.sg](https://data.gov.sg/datasets/d_bdaff844e3ef89d39fceb962ff8f0791/view) | SINGSTAT | Monthly |
| BTO projects | HDB sales launch announcements (compiled manually) | HDB | Per launch exercise |
| MRT stations | LTA / OneMap (compiled manually) | LTA | As network expands |
| Future developments | [URA Draft Master Plan 2025](https://www.ura.gov.sg/Corporate/Planning/Master-Plan) | URA | Indicative |
| Block coordinates | [OneMap API](https://www.onemap.gov.sg/apidocs/) | SLA | One-time geocoding |

---

## Refreshing Data

When HDB publishes new resale data:

1. Download the latest CSV from data.gov.sg
2. Place it in `raw/` (or replace the existing "Jan-2017 onwards" file)
3. Re-run `python src/combine_clean.py`
4. Restart the Streamlit app (or press **R** in the browser to reload)

To refresh rental, amenity, and CPI data (monthly):
```bash
python src/fetch_data.py --force
```

---

## Data Confidence Framework

Every analysis in this app carries an explicit confidence label:

| Level | Meaning |
|---|---|
| **High** | Direct from transaction records or official government datasets; minimal assumptions |
| **Medium** | Derived/modelled; proxy variables used; assumptions stated in the UI |
| **Low** | Experimental; sparse data; high uncertainty; use as directional indicator only |

Key confidence notes:
- **Fair Value Model (Page 12)**: Medium — R² ~70-80%; no unit-level data (renovation, facing, view)
- **Opportunity Score (Page 20)**: Medium — historical patterns, not forward guarantees
- **School Proximity Premium (Page 21)**: Medium — correlation only; location endogeneity
- **MOP Calendar (Page 13)**: High for unlock dates (deterministic); Medium for price impact

---

## Roadmap — What's Next

### Immediate (no new data needed)
- Backtesting framework: did the Opportunity Score actually predict outperformance historically?
- G16 One-Click Comps PDF export (Page 19 already has CSV + text; PDF needs additional library)
- Portfolio tracker: ongoing valuation monitoring for past clients

### URA API key — COMPLETE ✅
All Tier 3 features are now live. 134,599 private caveat transactions fetched (Aug 2021–Aug 2026).

**Monthly refresh:** `python src/fetch_data.py --force` (requires `URA_ACCESS_KEY` env var set)

**Data pipeline:** `python src/combine_clean_condo.py` normalises raw caveats → `condo_clean.csv`

### Institutional layer (free public data, higher effort)
- I2: Developer unsold inventory tracker (URA monthly data)
- I3: REIT cap-rate benchmarking (SGX filings)
- I6: MOP-cleared HDB supply as leading indicator for private condo demand

---

## Known Limitations

- **HDB resale only (individual transactions)** — private residential requires URA API key
- **No listing data** — asking prices, days-on-market, price cuts require SRX/PropertyGuru licensing
- **Fair value model** — no renovation, facing, view, or unit condition data; R² ~70-80%
- **School proximity** — 1km zones are approximate; MOE priority admission zones are the legal boundary
- **Block coordinates** — ~97% precisely geocoded; remainder fall back to town centroid
- **CPI base** — SINGSTAT 2024=100; comparisons before 2000 may be less reliable
