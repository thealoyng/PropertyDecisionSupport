"""
Page 1 — Data Quality & Integrity Assessment
=============================================
Audits the 5 raw HDB resale CSV files and the cleaned combined dataset
for schema differences, missing values, duplicates, casing inconsistencies,
format variations, and outliers.
"""

import sys
import os

import streamlit as st

st.set_page_config(page_title="Data Quality", page_icon="📊", layout="wide")

# ── shared helpers (one directory up from pages/) ──
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eda_helpers import (
    load_raw_files,
    load_clean,
    RAW_FILES,
    RAW_LABELS,
    parse_remaining_lease,
    POLICY_EVENTS,
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── load data ──
raw_dict = load_raw_files()           # {label: DataFrame}
df_clean = load_clean()               # cleaned combined DataFrame

# =====================================================================
# HEADER + KEY METRICS
# =====================================================================
st.title("📊 Data Quality & Integrity")
st.markdown(
    "A systematic audit of the **5 raw HDB resale CSV files** (1990 → present) "
    "and the **cleaned combined dataset**, checking schema consistency, missing "
    "values, duplicates, casing issues, format variations, and outliers."
)

total_raw_rows = sum(len(v) for v in raw_dict.values())
date_min = df_clean["month"].min()
date_max = df_clean["month"].max()
n_cols_clean = len(df_clean.columns)
overall_missing_pct = (df_clean.isna().sum().sum() / np.prod(df_clean.shape)) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total rows (clean)", f"{len(df_clean):,}")
m2.metric("Date range", f"{date_min:%b %Y} – {date_max:%b %Y}")
m3.metric("Columns (clean)", n_cols_clean)
m4.metric("Overall missing %", f"{overall_missing_pct:.2f}%")

st.divider()

# =====================================================================
# 1. SCHEMA COMPARISON TABLE
# =====================================================================
with st.expander("1️⃣ Schema Comparison across Raw Files", expanded=True):
    st.markdown(
        "Each raw CSV covers a different era and may have **different columns**. "
        "In particular, `remaining_lease` only appears from Jan 2015 onward."
    )

    # Collect all unique columns across all files (preserving a sensible order)
    all_cols_ordered: list[str] = []
    for label in RAW_LABELS:
        for c in raw_dict[label].columns:
            if c not in all_cols_ordered:
                all_cols_ordered.append(c)

    schema_rows = []
    for col in all_cols_ordered:
        row = {"Column": col}
        for label in RAW_LABELS:
            present = col in raw_dict[label].columns
            row[label] = "✅" if present else "❌"
        schema_rows.append(row)

    schema_df = pd.DataFrame(schema_rows)

    def _highlight_missing(val):
        if val == "❌":
            return "background-color: #ffcccc; color: #990000; font-weight: bold"
        return ""

    st.dataframe(
        schema_df.style.map(_highlight_missing, subset=RAW_LABELS),
        width='stretch',
        hide_index=True,
    )

    # Summary note
    missing_cols = schema_df[schema_df[RAW_LABELS].apply(
        lambda row: "❌" in row.values, axis=1
    )]["Column"].tolist()
    if missing_cols:
        st.info(
            f"**Columns not present in all files:** {', '.join(f'`{c}`' for c in missing_cols)}. "
            "The cleaning pipeline computes missing values where possible "
            "(e.g. `remaining_lease` from `lease_commence_date`)."
        )

# =====================================================================
# 2. MISSING VALUES HEATMAP (cleaned dataset)
# =====================================================================
with st.expander("2️⃣ Missing Values Heatmap (Cleaned Dataset)", expanded=True):
    st.markdown(
        "Percentage of missing (NaN) values for each column in the cleaned dataset. "
        "Darker colour = higher proportion missing."
    )

    miss_pct = (df_clean.isna().mean() * 100).round(4)
    miss_df = miss_pct.reset_index()
    miss_df.columns = ["Column", "Missing %"]

    fig_miss = go.Figure(
        go.Heatmap(
            z=[miss_df["Missing %"].values],
            x=miss_df["Column"].values,
            y=["Missing %"],
            colorscale="Reds",
            text=[[f"{v:.2f}%" for v in miss_df["Missing %"].values]],
            texttemplate="%{text}",
            textfont=dict(size=12),
            hovertemplate="Column: %{x}<br>Missing: %{text}<extra></extra>",
            colorbar=dict(title="% Missing"),
        )
    )
    fig_miss.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=80),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_miss, width='stretch')

    # Also show as a small table for screen readers / exact values
    cols_with_missing = miss_df[miss_df["Missing %"] > 0].sort_values(
        "Missing %", ascending=False
    )
    if len(cols_with_missing):
        st.dataframe(cols_with_missing, width='stretch', hide_index=True)
    else:
        st.success("No missing values in the cleaned dataset.")

# =====================================================================
# 3. ROW COUNT BY SOURCE FILE
# =====================================================================
with st.expander("3️⃣ Row Count by Source File", expanded=True):
    st.markdown(
        "Volume of transactions in each raw CSV. Later files tend to be larger "
        "because registration-based reporting captures more transactions."
    )

    counts = pd.DataFrame(
        {"Source File": RAW_LABELS,
         "Rows": [len(raw_dict[l]) for l in RAW_LABELS]}
    )

    fig_rows = px.bar(
        counts,
        y="Source File",
        x="Rows",
        orientation="h",
        text="Rows",
        color="Rows",
        color_continuous_scale="Blues",
    )
    fig_rows.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_rows.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_rows, width='stretch')

    st.caption(f"**Total raw rows across all files:** {total_raw_rows:,}")

# =====================================================================
# 4. DATE COVERAGE TIMELINE (Gantt-style)
# =====================================================================
with st.expander("4️⃣ Date Coverage Timeline", expanded=True):
    st.markdown(
        "Min and max `month` in each raw file shown as a Gantt-style bar. "
        "Overlapping bands indicate periods where two files both contain data; "
        "gaps indicate periods with no coverage."
    )

    timeline_rows = []
    for label in RAW_LABELS:
        rdf = raw_dict[label]
        dates = pd.to_datetime(rdf["month"], errors="coerce")
        timeline_rows.append({
            "Source": label,
            "Start": dates.min(),
            "End": dates.max(),
        })
    timeline_df = pd.DataFrame(timeline_rows)

    fig_gantt = px.timeline(
        timeline_df,
        x_start="Start",
        x_end="End",
        y="Source",
        color="Source",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_gantt.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        xaxis_title="",
    )
    st.plotly_chart(fig_gantt, width='stretch')

    # Overlap / gap detection
    st.markdown("**Coverage details:**")
    for i, row in timeline_df.iterrows():
        st.write(f"- **{row['Source']}**: {row['Start']:%b %Y} → {row['End']:%b %Y}")
    # Check for overlaps
    overlaps = []
    for i in range(len(timeline_df) - 1):
        end_i = timeline_df.iloc[i]["End"]
        start_next = timeline_df.iloc[i + 1]["Start"]
        if end_i >= start_next:
            overlaps.append(
                f"Overlap between *{timeline_df.iloc[i]['Source']}* "
                f"and *{timeline_df.iloc[i+1]['Source']}* "
                f"({start_next:%b %Y} – {end_i:%b %Y})"
            )
        elif (start_next - end_i).days > 31:
            overlaps.append(
                f"Gap between *{timeline_df.iloc[i]['Source']}* "
                f"and *{timeline_df.iloc[i+1]['Source']}* "
                f"({end_i:%b %Y} → {start_next:%b %Y})"
            )
    if overlaps:
        for o in overlaps:
            st.warning(o)
    else:
        st.success("Files tile seamlessly with no gaps or overlaps.")

# =====================================================================
# 5. DUPLICATE DETECTION
# =====================================================================
with st.expander("5️⃣ Duplicate Detection (Cleaned Dataset)", expanded=True):
    st.markdown(
        "Exact duplicate rows in the cleaned dataset. A high duplicate count "
        "may indicate overlapping source files or data entry issues."
    )

    dup_mask = df_clean.duplicated(keep=False)
    n_dup_rows = dup_mask.sum()
    n_dup_groups = df_clean[dup_mask].duplicated(keep="first").sum()

    d1, d2 = st.columns(2)
    d1.metric("Rows involved in duplicates", f"{n_dup_rows:,}")
    d2.metric("Duplicate groups (extra copies)", f"{n_dup_groups:,}")

    if n_dup_rows > 0:
        st.markdown("**Top duplicated records** (showing up to 20):")
        dup_counts = (
            df_clean.groupby(list(df_clean.columns))
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        dup_counts = dup_counts[dup_counts["count"] > 1].head(20)
        st.dataframe(dup_counts, width='stretch', hide_index=True)
    else:
        st.success("No exact duplicate rows found in the cleaned dataset.")

# =====================================================================
# 6. FLAT_MODEL CASING INCONSISTENCY (raw files)
# =====================================================================
with st.expander("6️⃣ flat_model Casing Inconsistency (Raw Files)", expanded=True):
    st.markdown(
        "The `flat_model` column uses **ALL CAPS** in older files (e.g. `IMPROVED`) "
        "but **Title Case** in newer files (e.g. `Improved`). The cleaning pipeline "
        "normalises these to Title Case. Below is a comparison across raw eras."
    )

    casing_rows = []
    for label in RAW_LABELS:
        rdf = raw_dict[label]
        if "flat_model" in rdf.columns:
            unique_models = sorted(rdf["flat_model"].dropna().unique())
            # Determine predominant casing
            sample = rdf["flat_model"].dropna().iloc[:100]
            upper_pct = (sample == sample.str.upper()).mean() * 100
            casing_rows.append({
                "Source File": label,
                "Unique Values": len(unique_models),
                "Examples": ", ".join(unique_models[:6]),
                "ALL-CAPS %": f"{upper_pct:.0f}%",
            })

    casing_df = pd.DataFrame(casing_rows)
    st.dataframe(casing_df, width='stretch', hide_index=True)

    # Show a cross-file comparison of a specific model
    st.markdown("**Example — 'Improved' model across eras:**")
    improved_rows = []
    for label in RAW_LABELS:
        rdf = raw_dict[label]
        if "flat_model" in rdf.columns:
            matches = rdf["flat_model"].dropna().str.upper().eq("IMPROVED")
            if matches.any():
                raw_val = rdf.loc[matches.idxmax(), "flat_model"]
                improved_rows.append({"Source": label, "Raw Value": raw_val})
    if improved_rows:
        st.dataframe(
            pd.DataFrame(improved_rows), width='stretch', hide_index=True
        )

# =====================================================================
# 7. REMAINING_LEASE FORMAT COMPARISON
# =====================================================================
with st.expander("7️⃣ remaining_lease Format Comparison", expanded=True):
    st.markdown(
        "The `remaining_lease` column exists only in the **2015-2016** and "
        "**2017-present** files, but uses different formats:\n"
        "- **2015-2016**: integer (e.g. `70`)\n"
        "- **2017+**: string (e.g. `61 years 04 months`)\n\n"
        "The cleaning pipeline uses `parse_remaining_lease()` to normalise both "
        "to a decimal year value."
    )

    format_rows = []
    for label in RAW_LABELS:
        rdf = raw_dict[label]
        if "remaining_lease" in rdf.columns:
            sample_vals = rdf["remaining_lease"].dropna().head(5).tolist()
            dtypes = rdf["remaining_lease"].dtype
            # Determine format
            first_val = str(rdf["remaining_lease"].dropna().iloc[0])
            is_numeric = first_val.replace(".", "").isdigit()
            fmt = "Numeric (integer)" if is_numeric else "String (years + months)"
            format_rows.append({
                "Source File": label,
                "Dtype": str(dtypes),
                "Format": fmt,
                "Sample Values": str(sample_vals),
            })
        else:
            format_rows.append({
                "Source File": label,
                "Dtype": "—",
                "Format": "Column absent",
                "Sample Values": "—",
            })

    format_df = pd.DataFrame(format_rows)
    st.dataframe(format_df, width='stretch', hide_index=True)

    # Demo the parser
    st.markdown("**`parse_remaining_lease()` examples:**")
    demo_vals = ["70", "61 years 04 months", "92 years", "55 years 11 months"]
    demo_results = pd.DataFrame({
        "Raw Value": demo_vals,
        "Parsed (decimal years)": [parse_remaining_lease(v) for v in demo_vals],
    })
    st.dataframe(demo_results, width='stretch', hide_index=True)

# =====================================================================
# 8. OUTLIER SUMMARY
# =====================================================================
with st.expander("8️⃣ Outlier Summary (IQR Method)", expanded=True):
    st.markdown(
        "For each numeric column, outliers are defined as values below "
        "`Q1 − 1.5 × IQR` or above `Q3 + 1.5 × IQR`. These are not necessarily "
        "errors — they may be genuine extremes (e.g. very large penthouses)."
    )

    numeric_cols = ["resale_price", "floor_area_sqm", "price_per_sqm", "flat_age"]
    outlier_rows = []
    for col in numeric_cols:
        series = df_clean[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((series < lower) | (series > upper)).sum()
        outlier_rows.append({
            "Column": col,
            "Q1": f"{q1:,.1f}",
            "Q3": f"{q3:,.1f}",
            "IQR": f"{iqr:,.1f}",
            "Lower Bound": f"{lower:,.1f}",
            "Upper Bound": f"{upper:,.1f}",
            "Outliers": f"{n_outliers:,}",
            "Outlier %": f"{n_outliers / len(series) * 100:.2f}%",
        })

    outlier_df = pd.DataFrame(outlier_rows)
    st.dataframe(outlier_df, width='stretch', hide_index=True)

    # Box plots
    st.markdown("**Box plots (log-scale where appropriate):**")
    box_col1, box_col2 = st.columns(2)

    with box_col1:
        fig_bp1 = px.box(
            df_clean, y="resale_price",
            title="Resale Price ($)",
            labels={"resale_price": "Resale Price ($)"},
        )
        fig_bp1.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bp1, width='stretch')

        fig_bp3 = px.box(
            df_clean, y="price_per_sqm",
            title="Price per sqm ($)",
            labels={"price_per_sqm": "Price per sqm ($)"},
        )
        fig_bp3.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bp3, width='stretch')

    with box_col2:
        fig_bp2 = px.box(
            df_clean, y="floor_area_sqm",
            title="Floor Area (sqm)",
            labels={"floor_area_sqm": "Floor Area (sqm)"},
        )
        fig_bp2.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bp2, width='stretch')

        fig_bp4 = px.box(
            df_clean, y="flat_age",
            title="Flat Age (years)",
            labels={"flat_age": "Flat Age (years)"},
        )
        fig_bp4.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bp4, width='stretch')

# =====================================================================
# 9. FLAT_TYPE VALUE INVENTORY
# =====================================================================
with st.expander("9️⃣ flat_type Value Inventory", expanded=True):
    st.markdown(
        "All unique `flat_type` values across the **raw** files and the "
        "**cleaned** dataset, with transaction counts. Variants like "
        "`MULTI-GENERATION` vs `MULTI GENERATION` are shown if they exist in the raw data."
    )

    # Raw file inventory
    st.markdown("**Raw files — unique flat_type values:**")
    raw_ft_rows = []
    for label in RAW_LABELS:
        rdf = raw_dict[label]
        if "flat_type" in rdf.columns:
            for val in sorted(rdf["flat_type"].dropna().unique()):
                cnt = (rdf["flat_type"] == val).sum()
                raw_ft_rows.append({
                    "Source File": label,
                    "flat_type": val,
                    "Count": cnt,
                })
    raw_ft_df = pd.DataFrame(raw_ft_rows)

    # Pivot for compact display
    raw_ft_pivot = raw_ft_df.pivot_table(
        index="flat_type", columns="Source File", values="Count",
        aggfunc="sum", fill_value=0,
    )
    # Reorder columns to match RAW_LABELS
    raw_ft_pivot = raw_ft_pivot.reindex(
        columns=[l for l in RAW_LABELS if l in raw_ft_pivot.columns]
    )
    raw_ft_pivot["Total (raw)"] = raw_ft_pivot.sum(axis=1)
    raw_ft_pivot = raw_ft_pivot.sort_values("Total (raw)", ascending=False)
    st.dataframe(raw_ft_pivot, width='stretch')

    # Cleaned dataset inventory
    st.markdown("**Cleaned dataset — flat_type distribution:**")
    clean_ft = (
        df_clean["flat_type"]
        .value_counts()
        .reset_index()
    )
    clean_ft.columns = ["flat_type", "Count"]
    clean_ft["Share %"] = (clean_ft["Count"] / clean_ft["Count"].sum() * 100).round(2)

    fig_ft = px.bar(
        clean_ft,
        x="Count",
        y="flat_type",
        orientation="h",
        text="Count",
        color="Share %",
        color_continuous_scale="Viridis",
        title="Transactions by flat_type (cleaned dataset)",
    )
    fig_ft.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_ft.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=True,
    )
    st.plotly_chart(fig_ft, width='stretch')

    st.dataframe(clean_ft, width='stretch', hide_index=True)

# =====================================================================
# FOOTER
# =====================================================================
st.divider()
st.caption(
    "Data Quality page — part of the HDB Resale EDA suite. "
    "Source: data.gov.sg HDB Resale Flat Prices."
)
