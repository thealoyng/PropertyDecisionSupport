"""
Page 9 — Cross-Dataset Analysis
================================
Cross-references resale transaction data with BTO project pipeline,
MRT station network, and URA future development plans to surface
spatial and market-structure insights.
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
    load_bto,
    load_mrt,
    load_future,
    TOWN_CENTROIDS,
    LINE_COLORS,
    LINE_NAMES,
    fmt_price,
)

# ── page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Cross-Dataset", page_icon="🔗", layout="wide")

st.title("🔗 Cross-Dataset Analysis")
st.caption(
    "Cross-referencing HDB resale transactions with BTO projects, "
    "MRT stations, and URA future development plans."
)

# ── load data ────────────────────────────────────────────────────
df = load_clean()
bto = load_bto()
mrt = load_mrt()
future = load_future()

# Normalise BTO classification column (may be "classification" or "cls")
if not bto.empty:
    if "classification" in bto.columns and "cls" not in bto.columns:
        bto = bto.rename(columns={"classification": "cls"})
    elif "cls" not in bto.columns and "classification" not in bto.columns:
        bto["cls"] = "Unknown"

# ── helper: nearest town from lat/lon ────────────────────────────
def nearest_town(lat, lon):
    """Return the TOWN_CENTROIDS key closest to (lat, lon) by Euclidean distance."""
    best, best_d = None, float("inf")
    for town, (t_lat, t_lon) in TOWN_CENTROIDS.items():
        d = (lat - t_lat) ** 2 + (lon - t_lon) ** 2
        if d < best_d:
            best, best_d = town, d
    return best


# ── recent resale baseline (last 3 years) ────────────────────────
max_year = int(df["year"].max())
recent = df[df["year"] >= max_year - 2].copy()

town_psm = (
    recent.groupby("town")["price_per_sqm"]
    .median()
    .reset_index()
    .rename(columns={"price_per_sqm": "median_psm"})
)

# ── BTO town list ────────────────────────────────────────────────
bto_towns = set()
if not bto.empty and "town" in bto.columns:
    bto_towns = set(bto["town"].str.upper().str.strip().unique())

# ── KPI metrics ──────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("BTO projects", len(bto) if not bto.empty else 0)
k2.metric("MRT stations", len(mrt) if not mrt.empty else 0)
k3.metric("Future developments", len(future) if not future.empty else 0)
k4.metric(
    "Towns with upcoming BTO",
    len(bto_towns),
    help="Distinct towns that have at least one BTO project in the pipeline.",
)

st.divider()

# ── tabs ─────────────────────────────────────────────────────────
(
    tab_bto_vs,
    tab_cls,
    tab_mrt_line,
    tab_spatial,
    tab_future,
    tab_supply,
    tab_interchange,
) = st.tabs(
    [
        "🏘️ BTO vs non-BTO towns",
        "🏷️ Classification premium",
        "🚇 MRT line analysis",
        "🗺️ Unified spatial view",
        "🔭 Future dev baseline",
        "📦 Supply vs volume",
        "🔀 Interchange premium",
    ]
)

# ================================================================
# 1. Resale prices in BTO towns vs non-BTO towns
# ================================================================
with tab_bto_vs:
    st.subheader("Resale prices: BTO towns vs non-BTO towns")
    st.caption(
        f"Median price per sqm over the last 3 years ({max_year-2}–{max_year}). "
        "Towns with at least one upcoming BTO project are labelled 'BTO town'."
    )

    if not bto_towns:
        st.warning("No BTO project data available.")
    else:
        psm = town_psm.copy()
        psm["has_bto"] = psm["town"].apply(
            lambda t: "BTO town" if t.upper().strip() in bto_towns else "Non-BTO town"
        )
        psm = psm.sort_values("median_psm", ascending=False)

        fig1 = px.bar(
            psm,
            x="town",
            y="median_psm",
            color="has_bto",
            color_discrete_map={
                "BTO town": "#2563eb",
                "Non-BTO town": "#94a3b8",
            },
            labels={
                "median_psm": "Median price per sqm ($)",
                "town": "Town",
                "has_bto": "",
            },
            barmode="group",
        )
        fig1.update_layout(
            xaxis_tickangle=-45,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
            ),
            margin=dict(t=30),
        )
        st.plotly_chart(fig1, width='stretch')

        # summary statistics
        bto_med = psm.loc[psm["has_bto"] == "BTO town", "median_psm"].median()
        non_med = psm.loc[psm["has_bto"] == "Non-BTO town", "median_psm"].median()
        if pd.notna(bto_med) and pd.notna(non_med) and non_med > 0:
            diff_pct = (bto_med / non_med - 1) * 100
            st.info(
                f"Median PSM in BTO towns: **{fmt_price(bto_med)}** vs "
                f"non-BTO towns: **{fmt_price(non_med)}** "
                f"({diff_pct:+.1f}% difference)."
            )

# ================================================================
# 2. BTO classification vs resale premiums
# ================================================================
with tab_cls:
    st.subheader("BTO classification vs resale premiums")
    st.caption(
        "For each BTO project's town, the town-level median resale price per "
        "sqm (last 3 years) is looked up. Box plot groups by classification."
    )

    if bto.empty or "cls" not in bto.columns:
        st.warning("No BTO classification data available.")
    else:
        bto_cls = bto.copy()
        bto_cls["town_upper"] = bto_cls["town"].str.upper().str.strip()
        bto_cls = bto_cls.merge(
            town_psm.rename(columns={"town": "town_upper"}),
            on="town_upper",
            how="left",
        )
        bto_cls = bto_cls.dropna(subset=["median_psm"])

        if bto_cls.empty:
            st.warning("Could not match BTO towns to resale data.")
        else:
            cls_order = ["Standard", "Plus", "Prime"]
            bto_cls["cls"] = pd.Categorical(
                bto_cls["cls"], categories=cls_order, ordered=True
            )
            bto_cls = bto_cls.sort_values("cls")

            fig2 = px.box(
                bto_cls,
                x="cls",
                y="median_psm",
                color="cls",
                color_discrete_map={
                    "Standard": "#16a34a",
                    "Plus": "#f59e0b",
                    "Prime": "#dc2626",
                },
                points="all",
                labels={
                    "median_psm": "Median resale price per sqm ($)",
                    "cls": "BTO classification",
                },
            )
            fig2.update_layout(showlegend=False, margin=dict(t=30))
            st.plotly_chart(fig2, width='stretch')

            # summary table
            cls_summary = (
                bto_cls.groupby("cls", observed=True)["median_psm"]
                .agg(["median", "mean", "count"])
                .reset_index()
            )
            cls_summary.columns = [
                "Classification",
                "Median PSM",
                "Mean PSM",
                "# Projects",
            ]
            cls_summary["Median PSM"] = cls_summary["Median PSM"].apply(fmt_price)
            cls_summary["Mean PSM"] = cls_summary["Mean PSM"].apply(fmt_price)
            st.dataframe(cls_summary, hide_index=True, width='stretch')

# ================================================================
# 3. MRT line analysis — average resale PSM per line
# ================================================================
with tab_mrt_line:
    st.subheader("Average resale price per sqm by MRT line")
    st.caption(
        "Each MRT station is mapped to its nearest town (Euclidean distance "
        "on centroids). The town's median resale PSM (last 3 years) proxies "
        "the station's neighbourhood price."
    )

    if mrt.empty:
        st.warning("No MRT station data available.")
    else:
        mrt_cp = mrt.copy()
        mrt_cp["nearest_town"] = mrt_cp.apply(
            lambda r: nearest_town(r["lat"], r["lon"])
            if pd.notna(r["lat"]) and pd.notna(r["lon"])
            else None,
            axis=1,
        )
        mrt_cp = mrt_cp.merge(
            town_psm, left_on="nearest_town", right_on="town", how="left"
        )
        mrt_cp = mrt_cp.dropna(subset=["median_psm"])

        # Expand multi-line stations so each line gets a row
        rows = []
        for _, r in mrt_cp.iterrows():
            primary = r.get("line", "")
            all_lines = str(r.get("lines", primary)).split(",")
            for ln in all_lines:
                ln = ln.strip()
                if ln:
                    rows.append(
                        {
                            "station": r["name"],
                            "line": ln,
                            "median_psm": r["median_psm"],
                        }
                    )
        mrt_expanded = pd.DataFrame(rows)

        if mrt_expanded.empty:
            st.warning("Could not match MRT stations to resale data.")
        else:
            line_avg = (
                mrt_expanded.groupby("line")["median_psm"]
                .mean()
                .reset_index()
                .sort_values("median_psm", ascending=False)
            )
            line_avg["line_name"] = line_avg["line"].map(LINE_NAMES).fillna(
                line_avg["line"]
            )
            line_avg["color"] = line_avg["line"].map(LINE_COLORS).fillna("#888888")

            fig3 = go.Figure(
                go.Bar(
                    x=line_avg["line_name"],
                    y=line_avg["median_psm"],
                    marker_color=line_avg["color"],
                    hovertemplate="%{x}<br>Avg PSM: $%{y:,.0f}<extra></extra>",
                )
            )
            fig3.update_layout(
                yaxis_title="Average median price per sqm ($)",
                xaxis_title="MRT line",
                margin=dict(t=30),
            )
            st.plotly_chart(fig3, width='stretch')

            # per-station detail
            with st.expander("Station-level detail"):
                detail = mrt_expanded.copy()
                detail["line_name"] = (
                    detail["line"].map(LINE_NAMES).fillna(detail["line"])
                )
                detail["median_psm_fmt"] = detail["median_psm"].apply(fmt_price)
                detail = detail.sort_values(
                    ["line", "median_psm"], ascending=[True, False]
                )
                st.dataframe(
                    detail[["station", "line_name", "median_psm_fmt"]].rename(
                        columns={
                            "station": "Station",
                            "line_name": "Line",
                            "median_psm_fmt": "Median PSM",
                        }
                    ),
                    hide_index=True,
                    width='stretch',
                )

# ================================================================
# 4. Unified spatial view (pydeck)
# ================================================================
with tab_spatial:
    st.subheader("Unified spatial view")
    st.caption(
        "Town centroids coloured by median price per sqm, BTO projects "
        "as blue squares, and future developments as purple diamonds."
    )

    # --- Town centroid layer (large circles, green→red gradient) ---
    centroid_rows = []
    for town, (lat, lon) in TOWN_CENTROIDS.items():
        match = town_psm.loc[town_psm["town"] == town]
        med = float(match["median_psm"].iloc[0]) if len(match) else np.nan
        centroid_rows.append(
            {"town": town, "lat": lat, "lon": lon, "median_psm": med}
        )
    centroid_df = pd.DataFrame(centroid_rows).dropna(subset=["median_psm"])

    # Map median_psm to colour (green=low → red=high)
    psm_min = centroid_df["median_psm"].min()
    psm_max = centroid_df["median_psm"].max()
    psm_range = psm_max - psm_min if psm_max != psm_min else 1

    def psm_to_rgb(val):
        t = (val - psm_min) / psm_range  # 0 → 1
        r = int(255 * t)
        g = int(255 * (1 - t))
        return [r, g, 60, 180]

    centroid_df["color"] = centroid_df["median_psm"].apply(psm_to_rgb)

    town_layer = pdk.Layer(
        "ScatterplotLayer",
        data=centroid_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=600,
        pickable=True,
        auto_highlight=True,
    )

    # --- BTO layer (small blue squares) ---
    layers = [town_layer]
    if not bto.empty and "lat" in bto.columns and "lon" in bto.columns:
        bto_valid = bto.dropna(subset=["lat", "lon"]).copy()
        bto_valid["color"] = [[37, 99, 235, 200]] * len(bto_valid)
        bto_layer = pdk.Layer(
            "ScatterplotLayer",
            data=bto_valid,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=300,
            pickable=True,
            stroked=True,
            get_line_color=[37, 99, 235, 255],
            line_width_min_pixels=1,
        )
        layers.append(bto_layer)

    # --- Future developments layer (purple diamonds) ---
    if not future.empty and "lat" in future.columns and "lon" in future.columns:
        fut_valid = future.dropna(subset=["lat", "lon"]).copy()
        fut_valid["color"] = [[147, 51, 234, 200]] * len(fut_valid)
        fut_layer = pdk.Layer(
            "ScatterplotLayer",
            data=fut_valid,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=350,
            pickable=True,
        )
        layers.append(fut_layer)

    view_state = pdk.ViewState(
        latitude=1.3521,
        longitude=103.8198,
        zoom=10.5,
        pitch=0,
    )

    tooltip = {
        "html": "<b>{town}</b>{name}<br/>Median PSM: ${median_psm}",
        "style": {"backgroundColor": "#1e293b", "color": "white"},
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/light-v10",
        )
    )

    # Legend
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "🟢🔴 **Town centroids** — circle colour = median PSM "
        f"(green = {fmt_price(psm_min)}, red = {fmt_price(psm_max)})"
    )
    c2.markdown("🔵 **BTO projects** — small blue circles")
    c3.markdown("🟣 **Future developments** — purple circles")

# ================================================================
# 5. Future development areas: baseline prices
# ================================================================
with tab_future:
    st.subheader("Future development areas — baseline resale prices")
    st.caption(
        "Each planned development is mapped to its nearest HDB town. "
        "Current median resale PSM and year-over-year growth provide a "
        "baseline for evaluating future potential."
    )

    if future.empty:
        st.warning("No future development data available.")
    else:
        # YoY growth per town
        prev_year = max_year - 1
        psm_this = (
            recent[recent["year"] == max_year]
            .groupby("town")["price_per_sqm"]
            .median()
            .reset_index()
            .rename(columns={"price_per_sqm": "psm_now"})
        )
        psm_prev = (
            recent[recent["year"] == prev_year]
            .groupby("town")["price_per_sqm"]
            .median()
            .reset_index()
            .rename(columns={"price_per_sqm": "psm_prev"})
        )
        yoy_df = psm_this.merge(psm_prev, on="town", how="left")
        yoy_df["yoy_growth"] = (
            (yoy_df["psm_now"] / yoy_df["psm_prev"] - 1) * 100
        ).round(1)

        rows = []
        for _, r in future.iterrows():
            if pd.notna(r.get("lat")) and pd.notna(r.get("lon")):
                town = nearest_town(r["lat"], r["lon"])
            else:
                town = "N/A"
            match = yoy_df.loc[yoy_df["town"] == town]
            med_psm = (
                float(match["psm_now"].iloc[0]) if len(match) else np.nan
            )
            yoy = float(match["yoy_growth"].iloc[0]) if len(match) else np.nan
            rows.append(
                {
                    "Development Name": r.get("name", ""),
                    "Type": r.get("type", ""),
                    "Nearest Town": town,
                    "Median PSM": fmt_price(med_psm),
                    "YoY Growth": f"{yoy:+.1f}%" if pd.notna(yoy) else "N/A",
                    "Timeline": r.get("horizon", ""),
                }
            )

        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, hide_index=True, width='stretch')

# ================================================================
# 6. BTO units planned vs resale volume
# ================================================================
with tab_supply:
    st.subheader("BTO units planned vs annual resale volume")
    st.caption(
        "For towns with upcoming BTO projects, compare the planned supply "
        "(BTO units) against the annual resale transaction volume (last full "
        "year)."
    )

    if bto.empty:
        st.warning("No BTO project data available.")
    else:
        # BTO units per town
        bto_supply = bto.copy()
        bto_supply["town_upper"] = bto_supply["town"].str.upper().str.strip()
        if "units" in bto_supply.columns:
            bto_supply["units_num"] = pd.to_numeric(
                bto_supply["units"], errors="coerce"
            )
        else:
            bto_supply["units_num"] = np.nan
        bto_agg = (
            bto_supply.groupby("town_upper")
            .agg(planned_units=("units_num", "sum"), n_projects=("name", "count"))
            .reset_index()
        )

        # Resale volume last full year
        last_full_year = max_year if df[df["year"] == max_year].shape[0] > 0 else max_year - 1
        resale_vol = (
            df[df["year"] == last_full_year]
            .groupby("town")
            .size()
            .reset_index(name="resale_txns")
        )
        resale_vol["town_upper"] = resale_vol["town"].str.upper().str.strip()

        merged = bto_agg.merge(resale_vol[["town_upper", "resale_txns"]], on="town_upper", how="left")
        merged = merged.dropna(subset=["planned_units"])
        merged = merged.sort_values("planned_units", ascending=False)

        if merged.empty:
            st.info("No BTO towns with known unit counts to display.")
        else:
            fig6 = go.Figure()
            fig6.add_trace(
                go.Bar(
                    x=merged["town_upper"],
                    y=merged["planned_units"],
                    name="Planned BTO units",
                    marker_color="#2563eb",
                )
            )
            fig6.add_trace(
                go.Bar(
                    x=merged["town_upper"],
                    y=merged["resale_txns"],
                    name=f"Resale transactions ({last_full_year})",
                    marker_color="#f59e0b",
                )
            )
            fig6.update_layout(
                barmode="group",
                yaxis_title="Count",
                xaxis_title="Town",
                xaxis_tickangle=-45,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
                margin=dict(t=30),
            )
            st.plotly_chart(fig6, width='stretch')

            # Ratio table
            merged["supply_ratio"] = (
                merged["planned_units"] / merged["resale_txns"]
            ).round(2)
            ratio_tbl = merged[
                ["town_upper", "planned_units", "resale_txns", "supply_ratio"]
            ].rename(
                columns={
                    "town_upper": "Town",
                    "planned_units": "Planned BTO units",
                    "resale_txns": f"Resale txns ({last_full_year})",
                    "supply_ratio": "BTO / Resale ratio",
                }
            )
            with st.expander("Supply-to-volume ratio table"):
                st.dataframe(ratio_tbl, hide_index=True, width='stretch')

# ================================================================
# 7. MRT interchange premium
# ================================================================
with tab_interchange:
    st.subheader("MRT interchange premium")
    st.caption(
        "Interchange stations (serving 2+ lines) are identified by a comma "
        "in the 'lines' column. For each station we find the nearest town "
        "and compare resale PSM between interchange and single-line towns."
    )

    if mrt.empty:
        st.warning("No MRT station data available.")
    else:
        mrt_ic = mrt.copy()
        mrt_ic["is_interchange"] = mrt_ic["lines"].fillna("").str.contains(",")
        mrt_ic["nearest_town"] = mrt_ic.apply(
            lambda r: nearest_town(r["lat"], r["lon"])
            if pd.notna(r["lat"]) and pd.notna(r["lon"])
            else None,
            axis=1,
        )
        mrt_ic = mrt_ic.merge(
            town_psm, left_on="nearest_town", right_on="town", how="left"
        )
        mrt_ic = mrt_ic.dropna(subset=["median_psm"])

        # De-duplicate towns — a town is "interchange" if ANY station near it
        # is an interchange
        town_ic = (
            mrt_ic.groupby("nearest_town")
            .agg(
                has_interchange=("is_interchange", "any"),
                median_psm=("median_psm", "first"),
            )
            .reset_index()
        )
        town_ic["station_type"] = town_ic["has_interchange"].map(
            {True: "Interchange town", False: "Single-line town"}
        )

        fig7 = px.box(
            town_ic,
            x="station_type",
            y="median_psm",
            color="station_type",
            color_discrete_map={
                "Interchange town": "#2563eb",
                "Single-line town": "#94a3b8",
            },
            points="all",
            labels={
                "median_psm": "Median resale price per sqm ($)",
                "station_type": "",
            },
        )
        fig7.update_layout(showlegend=False, margin=dict(t=30))
        st.plotly_chart(fig7, width='stretch')

        # Summary stats
        ic_med = town_ic.loc[
            town_ic["station_type"] == "Interchange town", "median_psm"
        ].median()
        sl_med = town_ic.loc[
            town_ic["station_type"] == "Single-line town", "median_psm"
        ].median()
        if pd.notna(ic_med) and pd.notna(sl_med) and sl_med > 0:
            prem = (ic_med / sl_med - 1) * 100
            st.info(
                f"Interchange town median PSM: **{fmt_price(ic_med)}** vs "
                f"single-line town: **{fmt_price(sl_med)}** "
                f"({prem:+.1f}% premium)."
            )

        with st.expander("Town-level detail"):
            detail_ic = town_ic.sort_values("median_psm", ascending=False).copy()
            detail_ic["median_psm_fmt"] = detail_ic["median_psm"].apply(fmt_price)
            st.dataframe(
                detail_ic[["nearest_town", "station_type", "median_psm_fmt"]].rename(
                    columns={
                        "nearest_town": "Town",
                        "station_type": "Type",
                        "median_psm_fmt": "Median PSM",
                    }
                ),
                hide_index=True,
                width='stretch',
            )
