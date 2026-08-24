"""
Page 21 - Location Intelligence
================================
Amenity mapping, walkability scoring, block-level accessibility, and
school proximity premium analysis for Singapore HDB towns and blocks.

Tabs:
  B1. Amenity Map          — interactive pydeck map of all 5 amenity types
  B2. Walkability Score    — composite score for all 27 HDB towns
  B3. Block Accessibility  — personalised accessibility for a specific block
  B5. School Proximity     — PSM premium for being near a primary school
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from eda_helpers import (
    load_clean,
    load_hawker_centres,
    load_community_clubs,
    load_parks,
    load_polyclinics,
    load_schools,
    load_mrt,
    fmt_price,
    TOWN_CENTROIDS,
    DATA_DIR,
)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Location Intel",
    page_icon="\U0001f333",
    layout="wide",
)

# ── constants ──────────────────────────────────────────────────────────────────
COORDS_CSV = os.path.join(DATA_DIR, "address_coords.csv")

AMENITY_COLORS = {
    "\U0001f35c Hawker Centres": [255, 100, 0],
    "\U0001f3db\ufe0f Community Clubs": [0, 150, 255],
    "\U0001f333 Parks": [0, 200, 100],
    "\U0001f3e5 Polyclinics": [200, 0, 200],
    "\U0001f3eb Schools": [255, 200, 0],
}

AMENITY_LABELS = list(AMENITY_COLORS.keys())


# ── helpers ────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorised haversine distance (km).
    lat1 / lon1 are scalars; lat2_arr / lon2_arr are 1-D array-like.
    """
    R = 6371.0
    lat1_r = np.radians(float(lat1))
    lon1_r = np.radians(float(lon1))
    lat2_r = np.radians(np.asarray(lat2_arr, dtype=float))
    lon2_r = np.radians(np.asarray(lon2_arr, dtype=float))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@st.cache_data
def load_address_coords():
    """Load address_coords.csv (block, street_name, lat, lon).
    Returns empty DataFrame if not yet generated (run src/geocode_resale.py).
    """
    try:
        df = pd.read_csv(COORDS_CSV)
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["block"] = df["block"].astype(str).str.strip()
        return df.dropna(subset=["lat", "lon"])
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def get_town_streets():
    """Return {town -> sorted list of unique streets} from the clean resale data."""
    resale = load_clean()
    result = {}
    for town, grp in resale.groupby("town"):
        result[town] = sorted(grp["street_name"].dropna().unique())
    return result


@st.cache_data
def compute_walkability(_hawker, _cc, _parks, _poly, _mrt):
    """Compute composite walkability / amenity-access score for every TOWN_CENTROID."""
    rows = []
    for town, (t_lat, t_lon) in TOWN_CENTROIDS.items():

        def _cnt(df, radius_km):
            if df.empty:
                return 0
            return int((haversine_km(t_lat, t_lon, df["lat"].values, df["lon"].values) <= radius_km).sum())

        n_hawker = _cnt(_hawker, 1.0)
        n_cc = _cnt(_cc, 2.0)
        n_parks = _cnt(_parks, 0.5)
        n_poly = _cnt(_poly, 3.0)

        if not _mrt.empty:
            mrt_min_km = float(
                haversine_km(t_lat, t_lon, _mrt["lat"].values, _mrt["lon"].values).min()
            )
        else:
            mrt_min_km = 2.0

        hawker_score = min(n_hawker * 2, 10)
        cc_score = min(n_cc * 3, 10)
        park_score = min(n_parks * 2, 10)
        poly_score = min(n_poly * 5, 10)
        mrt_score = max(0.0, 10.0 - mrt_min_km * 5.0)

        composite = (
            hawker_score * 0.30
            + cc_score * 0.15
            + park_score * 0.20
            + poly_score * 0.15
            + mrt_score * 0.20
        )

        rows.append({
            "Town": town,
            "Hawker (1km)": n_hawker,
            "CC (2km)": n_cc,
            "Parks (500m)": n_parks,
            "Poly (3km)": n_poly,
            "MRT dist (km)": round(mrt_min_km, 2),
            "Hawker Score": round(hawker_score, 1),
            "CC Score": round(cc_score, 1),
            "Park Score": round(park_score, 1),
            "Poly Score": round(poly_score, 1),
            "MRT Score": round(mrt_score, 1),
            "Composite Score": round(composite, 2),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Composite Score", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data
def compute_school_premium(_resale_df, _schools_df, _coords_df):
    """
    Compute PSM premium for HDB flats within 1 km of a primary school.

    Returns
    -------
    town_prem : DataFrame  — columns: town, premium_pct (avg across flat types)
    detail    : DataFrame  — columns: town, flat_type, outside_1km_psm,
                             within_1km_psm, premium_pct
    Both may be empty DataFrames on failure.
    """
    # Filter to primary schools only
    s = _schools_df.copy()
    if "mainlevel_code" in s.columns:
        primary = s[s["mainlevel_code"] == "PRIMARY"].copy()
    else:
        primary = s.copy()

    if primary.empty or _coords_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    s_lats = primary["lat"].values.astype(float)
    s_lons = primary["lon"].values.astype(float)

    # Merge resale with block-level coordinates
    work = _resale_df.copy()
    work["block"] = work["block"].astype(str).str.strip()
    coords_sub = _coords_df[["block", "street_name", "lat", "lon"]].copy()
    work = work.merge(coords_sub, on=["block", "street_name"], how="left")
    work = work.dropna(subset=["lat", "lon", "price_per_sqm"])

    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Sample for performance
    if len(work) > 50_000:
        work = work.sample(50_000, random_state=42)

    w_lats = work["lat"].values.astype(float)
    w_lons = work["lon"].values.astype(float)

    # Chunked vectorised haversine to find distance to nearest primary school
    chunk_size = 5_000
    min_dists = np.zeros(len(work), dtype=float)
    for i in range(0, len(work), chunk_size):
        cl = w_lats[i: i + chunk_size]
        co = w_lons[i: i + chunk_size]
        # Shape: (chunk, n_schools)
        dlat = np.radians(s_lats) - np.radians(cl[:, None])
        dlon = np.radians(s_lons) - np.radians(co[:, None])
        cos1 = np.cos(np.radians(cl[:, None]))
        cos2 = np.cos(np.radians(s_lats))
        a = np.sin(dlat / 2) ** 2 + cos1 * cos2 * np.sin(dlon / 2) ** 2
        chunk_dists = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        min_dists[i: i + chunk_size] = chunk_dists.min(axis=1)

    work = work.copy()
    work["dist_nearest_pri_km"] = min_dists
    work["within_1km"] = work["dist_nearest_pri_km"] <= 1.0

    # Median PSM grouped by town + flat_type × within_1km flag
    grp = (
        work.groupby(["town", "flat_type", "within_1km"])["price_per_sqm"]
        .median()
        .reset_index()
    )
    piv = grp.pivot_table(
        index=["town", "flat_type"],
        columns="within_1km",
        values="price_per_sqm",
    )
    col_map = {}
    if False in piv.columns:
        col_map[False] = "outside_1km_psm"
    if True in piv.columns:
        col_map[True] = "within_1km_psm"
    piv = piv.rename(columns=col_map).reset_index()

    if "within_1km_psm" not in piv.columns or "outside_1km_psm" not in piv.columns:
        return pd.DataFrame(), pd.DataFrame()

    piv = piv.dropna(subset=["within_1km_psm", "outside_1km_psm"])
    piv["premium_pct"] = (piv["within_1km_psm"] / piv["outside_1km_psm"] - 1) * 100

    town_prem = (
        piv.groupby("town")["premium_pct"]
        .mean()
        .reset_index()
        .sort_values("premium_pct", ascending=False)
        .reset_index(drop=True)
    )
    return town_prem, piv


# ── load amenity data (module level — all are @st.cache_data) ──────────────────
hawker_df = load_hawker_centres()
cc_df = load_community_clubs()
parks_df = load_parks()
poly_df = load_polyclinics()
schools_df = load_schools()
mrt_df = load_mrt()
coords_df = load_address_coords()

AMENITY_DATA = {
    "\U0001f35c Hawker Centres": hawker_df,
    "\U0001f3db\ufe0f Community Clubs": cc_df,
    "\U0001f333 Parks": parks_df,
    "\U0001f3e5 Polyclinics": poly_df,
    "\U0001f3eb Schools": schools_df,
}

# ── page header ────────────────────────────────────────────────────────────────
st.title("\U0001f333 Location Intelligence")
st.caption(
    "Amenity coverage, walkability scores, block-level accessibility, and school "
    "proximity premium analysis for Singapore HDB towns and blocks."
)

# ── tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f5fe\ufe0f Amenity Map",
    "\U0001f6b6 Walkability Score",
    "\U0001f4cd Block Accessibility",
    "\U0001f3eb School Proximity Premium",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Amenity Map (B1)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("\U0001f5fe\ufe0f Amenity Map (B1)")

    all_empty = all(df.empty for df in AMENITY_DATA.values())
    if all_empty:
        st.warning("Run src/fetch_data.py to fetch amenity data")
    else:
        # ── filters ───────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns([1, 2, 2])
        town_opts = ["All towns"] + sorted(TOWN_CENTROIDS.keys())
        sel_town = fc1.selectbox("Town filter", town_opts, key="b1_town")
        sel_amenities = fc2.multiselect(
            "Amenity types to show",
            AMENITY_LABELS,
            default=AMENITY_LABELS,
            key="b1_amenities",
        )
        school_levels = []
        if "\U0001f3eb Schools" in sel_amenities:
            school_levels = fc3.multiselect(
                "School levels",
                ["PRIMARY", "SECONDARY", "JUNIOR COLLEGE"],
                default=["PRIMARY", "SECONDARY", "JUNIOR COLLEGE"],
                key="b1_school_levels",
            )

        # ── pydeck view state ─────────────────────────────────────────────────
        if sel_town != "All towns" and sel_town in TOWN_CENTROIDS:
            ctr_lat, ctr_lon = TOWN_CENTROIDS[sel_town]
            zoom = 13
        else:
            ctr_lat, ctr_lon = 1.3521, 103.8198  # Singapore geographic centroid
            zoom = 11

        view_state = pdk.ViewState(
            latitude=ctr_lat, longitude=ctr_lon, zoom=zoom, pitch=0
        )

        # ── build ScatterplotLayers ───────────────────────────────────────────
        layers = []
        for label in sel_amenities:
            df = AMENITY_DATA[label].copy()
            if df.empty:
                continue

            # School level sub-filter
            if label == "\U0001f3eb Schools" and school_levels and "mainlevel_code" in df.columns:
                df = df[df["mainlevel_code"].isin(school_levels)]

            if df.empty:
                continue

            # Normalise name column (schools use school_name)
            if "school_name" in df.columns and "name" not in df.columns:
                df = df.rename(columns={"school_name": "name"})

            color = AMENITY_COLORS[label]
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df[["name", "lat", "lon"]].dropna(),
                get_position="[lon, lat]",
                get_color=color + [200],
                get_radius=150,
                pickable=True,
                radius_min_pixels=4,
                radius_max_pixels=15,
            ))

        if layers:
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip={"text": "{name}"},
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            )
            st.pydeck_chart(deck)
        else:
            st.info(
                "No amenities selected or all selected types have no data yet. "
                "Select at least one amenity type."
            )

        # ── summary table ─────────────────────────────────────────────────────
        st.subheader("Amenity Summary")
        if sel_town != "All towns" and sel_town in TOWN_CENTROIDS:
            c_lat, c_lon = TOWN_CENTROIDS[sel_town]
            count_col = f"Count (within 1\u2009km of {sel_town})"
        else:
            c_lat, c_lon = 1.3521, 103.8198
            count_col = "Count (within 1\u2009km of SG centroid)"

        tbl_rows = []
        for label, df in AMENITY_DATA.items():
            count_island = len(df) if not df.empty else 0
            if not df.empty:
                dists = haversine_km(c_lat, c_lon, df["lat"].values, df["lon"].values)
                count_1km = int((dists <= 1.0).sum())
            else:
                count_1km = "-"
            tbl_rows.append({
                "Amenity Type": label,
                "Count (island-wide)": count_island,
                count_col: count_1km,
            })
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

        st.caption(
            "Hawker: 129 centres (NEA, Nov 2025). Community Clubs: 128 (PA, Sep 2025). "
            "Parks: 461 (NParks, Apr 2026). Polyclinics: 23 (HPB). "
            "Schools: 337 (MOE, geocoded via OneMap). "
            "All coordinates from official government sources."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Walkability Score (B2)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("\U0001f6b6 Walkability Score (B2)")

    missing_b2 = [
        name for name, df in [
            ("Hawker Centres", hawker_df),
            ("Community Clubs", cc_df),
            ("Parks", parks_df),
            ("Polyclinics", poly_df),
            ("MRT stations", mrt_df),
        ]
        if df.empty
    ]

    if missing_b2:
        st.warning("Run src/fetch_data.py to fetch amenity data")
    else:
        walk_df = compute_walkability(hawker_df, cc_df, parks_df, poly_df, mrt_df)

        # ── ranked bar chart ──────────────────────────────────────────────────
        st.subheader("Towns Ranked by Composite Walkability Score")
        bar_df = walk_df.sort_values("Composite Score")
        fig_bar = px.bar(
            bar_df,
            x="Composite Score",
            y="Town",
            orientation="h",
            color="Composite Score",
            color_continuous_scale="RdYlGn",
            text=bar_df["Composite Score"].map(lambda x: f"{x:.1f}"),
            title="Composite Walkability / Amenity Access Score by Town (0\u201310)",
            labels={"Composite Score": "Score (0\u201310)"},
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=720, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── radar chart: top 5 ────────────────────────────────────────────────
        st.subheader("Top 5 Towns \u2014 Component Breakdown (Radar)")
        top5 = walk_df.head(5)
        categories = [
            "Hawker Score", "CC Score", "Park Score", "Poly Score", "MRT Score",
        ]
        fig_radar = go.Figure()
        for _, row in top5.iterrows():
            vals = [row[c] for c in categories] + [row[categories[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=row["Town"],
                opacity=0.75,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="Top 5 Towns: Walkability Component Scores",
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── full data table ───────────────────────────────────────────────────
        st.subheader("All 27 Towns \u2014 Detailed Scores")
        display_cols = [
            "Town", "Hawker (1km)", "CC (2km)", "Parks (500m)", "Poly (3km)",
            "MRT dist (km)", "Composite Score",
        ]
        st.dataframe(walk_df[display_cols], use_container_width=True, hide_index=True)

        st.info(
            "Scores are based on proximity from town centroids. Individual block-level scores "
            "will vary. Data from official government sources (2024\u20132026)."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Block Accessibility (B3)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("\U0001f4cd Block Accessibility (B3)")

    missing_b3 = any(
        df.empty for df in [hawker_df, cc_df, parks_df, poly_df, mrt_df, schools_df]
    )

    if missing_b3:
        st.warning("Run src/fetch_data.py to fetch amenity data")
    else:
        # ── inputs ────────────────────────────────────────────────────────────
        town_street_map = get_town_streets()
        towns_sorted = sorted(TOWN_CENTROIDS.keys())

        b3c1, b3c2, b3c3 = st.columns([1, 2, 1])
        b3_town = b3c1.selectbox("Town", towns_sorted, key="b3_town")

        streets_in_town = town_street_map.get(b3_town, [])
        b3_street = b3c2.selectbox(
            "Street",
            streets_in_town if streets_in_town else ["(No street data)"],
            key="b3_street",
        )
        b3_block = b3c3.text_input("Block (optional)", value="", key="b3_block")

        st.subheader("Amenity Weights (for reference)")
        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        _w_hawker = wc1.slider("Hawker", 0, 10, 5, key="b3_w_hawker")
        _w_mrt = wc2.slider("MRT", 0, 10, 5, key="b3_w_mrt")
        _w_school = wc3.slider("School", 0, 10, 3, key="b3_w_school")
        _w_park = wc4.slider("Park", 0, 10, 3, key="b3_w_park")
        _w_poly = wc5.slider("Polyclinic", 0, 10, 2, key="b3_w_poly")

        # ── determine block lat / lon ─────────────────────────────────────────
        block_lat, block_lon = TOWN_CENTROIDS[b3_town]
        source_label = f"town centroid ({b3_town})"

        if not coords_df.empty and b3_street not in ("(No street data)", ""):
            mask = coords_df["street_name"] == b3_street
            if b3_block.strip():
                mask = mask & (
                    coords_df["block"].str.upper() == b3_block.strip().upper()
                )
            match = coords_df[mask]
            if not match.empty:
                block_lat = float(match.iloc[0]["lat"])
                block_lon = float(match.iloc[0]["lon"])
                addr = (
                    f"{b3_block.strip()} {b3_street}"
                    if b3_block.strip()
                    else b3_street
                )
                source_label = addr

        st.caption(
            f"\U0001f4cd Coordinates from: **{source_label}** "
            f"({block_lat:.4f}\u00b0N, {block_lon:.4f}\u00b0E)"
        )

        # ── helpers (inline, using block_lat/lon closure) ─────────────────────
        def _nearest(df, name_col):
            """Return (name, distance_km) for the amenity nearest to the block."""
            if df.empty:
                return "N/A", float("nan")
            dists = haversine_km(
                block_lat, block_lon, df["lat"].values, df["lon"].values
            )
            idx = int(np.argmin(dists))
            nm = str(df[name_col].iloc[idx])[:40]
            return nm, round(float(dists[idx]), 2)

        def _count_r(df, radius_km):
            """Count amenities within radius_km of the block."""
            if df.empty:
                return 0
            dists = haversine_km(
                block_lat, block_lon, df["lat"].values, df["lon"].values
            )
            return int((dists <= radius_km).sum())

        # Prepare schools DataFrame with unified "name" column
        sch = schools_df.copy()
        if "school_name" in sch.columns:
            sch = sch.rename(columns={"school_name": "name"})
        if "mainlevel_code" in sch.columns:
            pri_sch = sch[sch["mainlevel_code"] == "PRIMARY"]
            sec_sch = sch[sch["mainlevel_code"] == "SECONDARY"]
        else:
            pri_sch = sch
            sec_sch = sch

        near_hawker, near_hawker_km = _nearest(hawker_df, "name")
        near_mrt, near_mrt_km = _nearest(mrt_df, "name")
        near_poly, near_poly_km = _nearest(poly_df, "name")
        near_pri, near_pri_km = _nearest(pri_sch, "name")
        near_sec, near_sec_km = _nearest(sec_sch, "name")
        near_park, near_park_km = _nearest(parks_df, "name")

        # ── KPI cards (2 rows × 3 cols) ───────────────────────────────────────
        st.subheader("Nearest Amenities")
        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("\U0001f35c Nearest Hawker", near_hawker, f"{near_hawker_km:.2f} km")
        kc2.metric("\U0001f687 Nearest MRT", near_mrt, f"{near_mrt_km:.2f} km")
        kc3.metric("\U0001f3e5 Nearest Polyclinic", near_poly, f"{near_poly_km:.2f} km")

        kc4, kc5, kc6 = st.columns(3)
        kc4.metric("\U0001f3eb Nearest Pri School", near_pri, f"{near_pri_km:.2f} km")
        kc5.metric("\U0001f3eb Nearest Sec School", near_sec, f"{near_sec_km:.2f} km")
        kc6.metric("\U0001f333 Nearest Park", near_park, f"{near_park_km:.2f} km")

        # ── radius count table ────────────────────────────────────────────────
        st.subheader("Amenity Counts by Radius")
        radius_rows = []
        for r in [0.5, 1.0, 2.0]:
            radius_rows.append({
                "Radius (km)": r,
                "Hawker Centres": _count_r(hawker_df, r),
                "MRT Stations": _count_r(mrt_df, r),
                "Parks": _count_r(parks_df, r),
                "Schools (all levels)": _count_r(sch, r),
            })
        st.dataframe(
            pd.DataFrame(radius_rows), use_container_width=True, hide_index=True
        )

        # ── mini pydeck map (amenities within 2 km) ───────────────────────────
        st.subheader("Nearby Amenities Map (within 2 km)")

        map_layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame([{
                    "name": f"\U0001f4cd {source_label}",
                    "lat": block_lat,
                    "lon": block_lon,
                }]),
                get_position="[lon, lat]",
                get_color=[255, 0, 0, 255],
                get_radius=200,
                radius_min_pixels=8,
                pickable=True,
            )
        ]

        amenity_map_spec = [
            ("\U0001f35c Hawker Centres", hawker_df, "name"),
            ("\U0001f3db\ufe0f Community Clubs", cc_df, "name"),
            ("\U0001f333 Parks", parks_df, "name"),
            ("\U0001f3e5 Polyclinics", poly_df, "name"),
            ("\U0001f3eb Schools", sch, "name"),
            ("\U0001f687 MRT", mrt_df, "name"),
        ]
        for label, df, nc in amenity_map_spec:
            if df.empty:
                continue
            dists = haversine_km(
                block_lat, block_lon, df["lat"].values, df["lon"].values
            )
            nearby = df[dists <= 2.0].copy()
            if nearby.empty:
                continue
            # Ensure name column is present
            if nc not in nearby.columns:
                first_str_col = nearby.select_dtypes("object").columns
                if len(first_str_col):
                    nearby = nearby.rename(columns={first_str_col[0]: nc})
            color = AMENITY_COLORS.get(label, [128, 128, 128])
            map_layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=nearby[[nc, "lat", "lon"]].rename(columns={nc: "name"}),
                get_position="[lon, lat]",
                get_color=color + [200],
                get_radius=100,
                radius_min_pixels=5,
                pickable=True,
            ))

        st.pydeck_chart(pdk.Deck(
            layers=map_layers,
            initial_view_state=pdk.ViewState(
                latitude=block_lat, longitude=block_lon, zoom=14, pitch=0
            ),
            tooltip={"text": "{name}"},
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        ))

        st.caption(
            "\U0001f7e2 **Data Confidence: High** \u2014 Coordinates from government sources "
            "(OneMap). Distances are straight-line (Haversine), not walking distance."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — School Proximity Premium (B5)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("\U0001f3eb School Proximity Premium (B5)")
    st.caption(
        "Quantify the PSM premium for HDB flats within 1\u2009km of a primary school."
    )

    if schools_df.empty:
        st.warning("Run src/fetch_data.py to compute this analysis.")
    else:
        with st.spinner(
            "Computing school proximity premium\u2026 "
            "(samples up to 50\u202fk transactions)"
        ):
            resale_b5 = load_clean()
            if "price_per_sqm" not in resale_b5.columns:
                resale_b5 = resale_b5.copy()
                resale_b5["price_per_sqm"] = (
                    resale_b5["resale_price"] / resale_b5["floor_area_sqm"]
                )
            town_prem_df, detail_df = compute_school_premium(
                resale_b5, schools_df, coords_df
            )

        if town_prem_df.empty:
            st.warning(
                "Could not compute school proximity premium. "
                "Ensure address_coords.csv is available (run src/geocode_resale.py, "
                "then src/fetch_data.py)."
            )
        else:
            # ── bar chart: premium by town ─────────────────────────────────────
            st.subheader("School Proximity Premium by Town")
            bar_prem = town_prem_df.sort_values("premium_pct")
            fig_prem = px.bar(
                bar_prem,
                x="premium_pct",
                y="town",
                orientation="h",
                color="premium_pct",
                color_continuous_scale="RdYlGn",
                text=bar_prem["premium_pct"].map(lambda x: f"{x:+.1f}%"),
                title="Average PSM Premium for Flats within 1\u2009km of a Primary School",
                labels={"premium_pct": "Premium (%)", "town": "Town"},
            )
            fig_prem.update_traces(textposition="outside")
            fig_prem.update_layout(height=700, coloraxis_showscale=False)
            st.plotly_chart(fig_prem, use_container_width=True)

            # ── scatter: within vs outside PSM ────────────────────────────────
            if not detail_df.empty and "within_1km_psm" in detail_df.columns:
                st.subheader(
                    "Within 1\u2009km vs Outside 1\u2009km Median PSM "
                    "(by Town \u00d7 Flat Type)"
                )
                max_val = max(
                    detail_df["outside_1km_psm"].max(),
                    detail_df["within_1km_psm"].max(),
                )
                fig_sc = px.scatter(
                    detail_df,
                    x="outside_1km_psm",
                    y="within_1km_psm",
                    color="town",
                    hover_data=["flat_type", "premium_pct"],
                    title="Median PSM: Within 1\u2009km vs Outside 1\u2009km of Primary School",
                    labels={
                        "outside_1km_psm": "Median PSM \u2014 Outside 1\u2009km ($)",
                        "within_1km_psm": "Median PSM \u2014 Within 1\u2009km ($)",
                    },
                )
                fig_sc.add_shape(
                    type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                    line=dict(dash="dash", color="gray", width=1),
                )
                fig_sc.add_annotation(
                    x=max_val, y=max_val,
                    text="Equal PSM line",
                    showarrow=False,
                    font=dict(color="gray", size=10),
                    xshift=-65,
                )
                fig_sc.update_layout(height=480)
                st.plotly_chart(fig_sc, use_container_width=True)

            # ── key finding callout ────────────────────────────────────────────
            avg_prem = town_prem_df["premium_pct"].mean()
            top_row = town_prem_df.iloc[0]
            st.success(
                f"\U0001f4ca **Key Finding:** On average, HDB flats within 1\u2009km of a "
                f"primary school command a **{avg_prem:+.1f}% PSM premium**. "
                f"The premium is strongest in **{top_row['town']}** "
                f"({top_row['premium_pct']:+.1f}%)."
            )

            st.warning(
                "\u26a0\ufe0f **DATA CONFIDENCE: Medium.** This is a correlation, not causation. "
                "Flats near schools also tend to be near MRT stations, hawker centres, and "
                "other amenities. The true school-specific premium is smaller than the raw "
                "comparison suggests."
            )


# ── footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "\U0001f333 **Location Intelligence** | Singapore HDB Property Decision Support | "
    "Data: NEA, PA, NParks, HPB, MOE, OneMap (2024\u20132026)"
)
