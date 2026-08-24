"""
Page 12 -- Fair Value Model
============================
Smart Money: hedonic regression model to estimate fair value,
identify mispriced blocks, and find comparable transactions.

Tabs:
  1. Fair Value Model    -- model performance, coefficients, pred vs actual
  2. Fair Value Lookup   -- user inputs a flat, get estimate + interval
  3. Mispricing Detector -- block-level over/under-pricing map & tables
  4. Comparable Engine   -- weighted similarity search
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from eda_helpers import (
    load_clean, load_mrt, fmt_price, TOWN_CENTROIDS,
    load_condo_clean, DISTRICT_CENTROIDS, floor_range_mid,
)

# --------------------------------------------------------------------------
# page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fair Value Model",
    page_icon="\U0001f4d0",
    layout="wide",
)

st.title("\U0001f4d0 Fair Value & Mispricing")
st.caption(
    "Smart Money: hedonic regression to estimate fair value, "
    "identify mispriced blocks, and find comparable transactions."
)

# --------------------------------------------------------------------------
# HDB / Private mode toggle
# --------------------------------------------------------------------------
mode = st.radio(
    "Property type",
    ["\U0001f3d8\ufe0f HDB Resale", "\U0001f3e2 Private (Condo)"],
    horizontal=True,
    key="fv_mode",
)
st.divider()

# --------------------------------------------------------------------------
# constants & paths
# --------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
COORDS_CSV = os.path.join(DATA_DIR, "address_coords.csv")

NUMERIC_FEATS = [
    "floor_area_sqm",
    "storey_mid",
    "remaining_lease_yrs",
    "flat_age",
    "year",
    "month_num",
]
CAT_FEATS = ["town", "flat_type"]
ALL_FEATS = NUMERIC_FEATS + CAT_FEATS
TARGET = "resale_price"

# --------------------------------------------------------------------------
# data loaders (cached)
# --------------------------------------------------------------------------


@st.cache_data
def load_5yr():
    """Load last 5 years of resale transactions with engineered features."""
    df = load_clean()
    df["month"] = pd.to_datetime(df["month"])
    df["month_num"] = df["month"].dt.month
    cutoff = df["month"].max() - pd.DateOffset(years=5)
    return df[df["month"] >= cutoff].copy()


@st.cache_data
def load_24m():
    """Load last 24 months of resale transactions with engineered features."""
    df = load_clean()
    df["month"] = pd.to_datetime(df["month"])
    df["month_num"] = df["month"].dt.month
    cutoff = df["month"].max() - pd.DateOffset(months=24)
    return df[df["month"] >= cutoff].copy()


@st.cache_data
def load_coords():
    """Load block-level geocoding coordinates."""
    if os.path.exists(COORDS_CSV):
        c = pd.read_csv(COORDS_CSV)
        c["block"] = c["block"].astype(str)
        return c
    return pd.DataFrame(columns=["block", "street_name", "lat", "lon"])


@st.cache_data
def get_town_list():
    return sorted(load_5yr()["town"].dropna().unique().tolist())


@st.cache_data
def get_flat_type_list():
    return sorted(load_5yr()["flat_type"].dropna().unique().tolist())


@st.cache_data
def prepare_detect_base():
    """
    Prepare 24-month data for residual analysis.
    Returns a clean DataFrame with block_str column.
    No model predictions (those depend on the model stored in session_state).
    """
    df24 = load_24m()
    df24 = df24.dropna(
        subset=ALL_FEATS + [TARGET, "block", "street_name", "price_per_sqm"]
    ).copy()
    df24["block_str"] = df24["block"].astype(str)
    return df24.reset_index(drop=True)


# --------------------------------------------------------------------------
# helper: haversine distance (km)
# --------------------------------------------------------------------------


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# --------------------------------------------------------------------------
# model builder
# --------------------------------------------------------------------------


def _make_ohe():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATS),
            ("cat", _make_ohe(), CAT_FEATS),
        ]
    )


def train_models_impl():
    """Train Ridge and Gradient Boosting on last-5yr data. Returns result dict."""
    df5 = load_5yr()
    df_model = df5.dropna(subset=ALL_FEATS + [TARGET]).copy()
    df_model = df_model.sort_values("month").reset_index(drop=True)

    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx].copy()
    test_df = df_model.iloc[split_idx:].copy()

    X_train = train_df[ALL_FEATS]
    y_train = train_df[TARGET].values
    X_test = test_df[ALL_FEATS]
    y_test = test_df[TARGET].values

    # -- Ridge -------------------------------------------------------
    ridge_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", Ridge(alpha=10.0)),
    ])
    ridge_pipe.fit(X_train, y_train)
    ridge_train_pred = ridge_pipe.predict(X_train)
    ridge_resid_std = float(np.std(y_train - ridge_train_pred))
    ridge_test_pred = ridge_pipe.predict(X_test)

    ridge_metrics = {
        "MAE": float(mean_absolute_error(y_test, ridge_test_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, ridge_test_pred))),
        "R2": float(r2_score(y_test, ridge_test_pred)),
    }

    # Ridge feature names + coefficients
    prep = ridge_pipe.named_steps["prep"]
    cat_enc = prep.named_transformers_["cat"]
    cat_names = cat_enc.get_feature_names_out(CAT_FEATS).tolist()
    feature_names = NUMERIC_FEATS + cat_names
    coefs = ridge_pipe.named_steps["model"].coef_
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coef": coefs})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )

    # -- Gradient Boosting -------------------------------------------
    gb_pipe = Pipeline([
        ("prep", build_preprocessor()),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
        ),
    ])
    gb_pipe.fit(X_train, y_train)
    gb_train_pred = gb_pipe.predict(X_train)
    gb_resid_std = float(np.std(y_train - gb_train_pred))
    gb_test_pred = gb_pipe.predict(X_test)

    gb_metrics = {
        "MAE": float(mean_absolute_error(y_test, gb_test_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, gb_test_pred))),
        "R2": float(r2_score(y_test, gb_test_pred)),
    }

    return {
        "ridge_pipe": ridge_pipe,
        "gb_pipe": gb_pipe,
        "ridge_resid_std": ridge_resid_std,
        "gb_resid_std": gb_resid_std,
        "ridge_metrics": ridge_metrics,
        "gb_metrics": gb_metrics,
        "coef_df": coef_df,
        "y_test": y_test,
        "ridge_test_pred": ridge_test_pred,
        "gb_test_pred": gb_test_pred,
    }


def get_models():
    """Retrieve already-trained models from session_state, or train on first call."""
    if "fv_models" not in st.session_state:
        with st.spinner(
            "\u23f3 Training Fair Value models (first load only, ~30\u201360 s)..."
        ):
            st.session_state["fv_models"] = train_models_impl()
    return st.session_state["fv_models"]


def get_detect_df(models):
    """
    Retrieve or compute residual-augmented 24-month DataFrame.
    Stored in session_state to avoid re-predicting on every render.
    """
    if "fv_detect_df" not in st.session_state:
        df_base = prepare_detect_base()
        preds = models["ridge_pipe"].predict(df_base[ALL_FEATS])
        df_aug = df_base.copy()
        df_aug["predicted_price"] = preds
        df_aug["predicted_psm"] = preds / df_aug["floor_area_sqm"]
        df_aug["residual"] = df_aug[TARGET] - preds
        df_aug["std_residual"] = df_aug["residual"] / models["ridge_resid_std"]
        st.session_state["fv_detect_df"] = df_aug
    return st.session_state["fv_detect_df"]


# --------------------------------------------------------------------------
# coefficient label helper
# --------------------------------------------------------------------------

_FEAT_LABELS = {
    "floor_area_sqm": "Floor area (per sqm)",
    "storey_mid": "Storey (per floor up)",
    "remaining_lease_yrs": "Remaining lease (per yr)",
    "flat_age": "Flat age (per yr)",
    "year": "Year (time trend)",
    "month_num": "Month seasonality",
}


def _coef_label(feature_name):
    if feature_name in _FEAT_LABELS:
        return _FEAT_LABELS[feature_name]
    if feature_name.startswith("town_"):
        return "Town: " + feature_name[5:].replace("_", " ").title()
    if feature_name.startswith("flat_type_"):
        return "Type: " + feature_name[10:].replace("_", " ").title()
    return feature_name


# ==========================================================================
# PRIVATE CONDO helpers
# ==========================================================================

PRIV_NUMERIC_FEATS = ["area_sqm", "floor_mid", "contract_year", "contract_month"]
PRIV_CAT_FEATS = ["district", "property_type_broad", "tenure_clean"]
PRIV_ALL_FEATS = PRIV_NUMERIC_FEATS + PRIV_CAT_FEATS
PRIV_TARGET = "price_psm"


@st.cache_data
def load_priv_3yr():
    """Load last 3 years of private Condo/EC transactions with engineered features."""
    df = load_condo_clean()
    if df.empty:
        return df
    df = df[df["property_type_broad"].isin(["Condo/Apartment", "EC"])].copy()
    df = df.dropna(subset=["price_psm", "area_sqm"])
    df["floor_mid"] = df["floor_range"].apply(floor_range_mid)
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=3)
    df = df[df["contract_date"] >= cutoff].copy()
    return df.reset_index(drop=True)


@st.cache_data
def load_priv_all():
    """Load all available private Condo/EC transactions (no date filter)."""
    df = load_condo_clean()
    if df.empty:
        return df
    return df[df["property_type_broad"].isin(["Condo/Apartment", "EC"])].copy()


def build_priv_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", PRIV_NUMERIC_FEATS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), PRIV_CAT_FEATS),
        ]
    )


def train_priv_model_impl():
    """Train GradientBoostingRegressor on last-3yr private condo data. Returns result dict."""
    df = load_priv_3yr()
    if df.empty:
        return None
    df_model = df.dropna(subset=PRIV_ALL_FEATS + [PRIV_TARGET]).copy()
    # Convert district to str for OHE (avoids Int64 nullable-int issues)
    df_model["district"] = df_model["district"].astype(str)
    df_model = df_model.sort_values("contract_date").reset_index(drop=True)

    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]

    X_train = train_df[PRIV_ALL_FEATS]
    y_train = train_df[PRIV_TARGET].values
    X_test = test_df[PRIV_ALL_FEATS]
    y_test = test_df[PRIV_TARGET].values

    gb_pipe = Pipeline([
        ("prep", build_priv_preprocessor()),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
        ),
    ])
    gb_pipe.fit(X_train, y_train)
    gb_test_pred = gb_pipe.predict(X_test)

    return {
        "gb_pipe": gb_pipe,
        "metrics": {
            "R2": float(r2_score(y_test, gb_test_pred)),
            "MAE": float(mean_absolute_error(y_test, gb_test_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, gb_test_pred))),
        },
        "y_test": y_test,
        "gb_test_pred": gb_test_pred,
    }


def get_priv_model():
    """Retrieve or train the private condo fair value model from session_state."""
    if "fv_priv_models" not in st.session_state:
        with st.spinner(
            "\u23f3 Training Private Condo Fair Value model (first load only, ~30\u201360 s)..."
        ):
            st.session_state["fv_priv_models"] = train_priv_model_impl()
    return st.session_state["fv_priv_models"]


# ==========================================================================
# TABS
# ==========================================================================
tab_model, tab_lookup, tab_detect, tab_comps, tab_distress, tab_quality = st.tabs([
    "\U0001f4d0 Fair Value Model",
    "\U0001f3af Fair Value Lookup",
    "\U0001f6a8 Mispricing Detector",
    "\U0001f52c Comparable Engine",
    "\U0001f9bb Distressed Sale Proxy",
    "\U0001f4ca Transaction Quality",
])


# ==========================================================================
# TAB 1: Fair Value Model
# ==========================================================================
with tab_model:
    if mode == "\U0001f3d8\ufe0f HDB Resale":
        st.subheader("Hedonic Regression \u2014 Model Performance")
        st.caption(
            "Trained on the **last 5 years** of transactions using a "
            "chronological 80/20 train/test split (not random)."
        )

        models = get_models()

        # ── metrics table ─────────────────────────────────────────────────
        st.markdown("#### Performance on held-out test set")
        metrics_tbl = pd.DataFrame(
            {
                "Metric": ["MAE ($)", "RMSE ($)", "R\u00b2"],
                "Ridge (alpha=10)": [
                    f"${models['ridge_metrics']['MAE']:,.0f}",
                    f"${models['ridge_metrics']['RMSE']:,.0f}",
                    f"{models['ridge_metrics']['R2']:.4f}",
                ],
                "Gradient Boosting (300 trees)": [
                    f"${models['gb_metrics']['MAE']:,.0f}",
                    f"${models['gb_metrics']['RMSE']:,.0f}",
                    f"{models['gb_metrics']['R2']:.4f}",
                ],
            }
        ).set_index("Metric")
        st.dataframe(metrics_tbl, use_container_width=True)

        ridge_r_std = models["ridge_resid_std"]
        st.info(
            f"Ridge residual std (training set): **${ridge_r_std:,.0f}** \u2014 "
            f"so the 90% prediction interval is \u00b1**${1.645 * ridge_r_std:,.0f}** "
            "around each estimate."
        )

        st.divider()

        # ── coefficient chart ─────────────────────────────────────────────
        st.markdown("#### What drives price? (Ridge coefficients, top 20 features)")
        st.caption(
            "Numeric features: $/unit. Town/type dummies: premium or discount vs. "
            "model baseline. Green = price driver; red = price reducer."
        )

        coef_df = models["coef_df"].head(20).copy()
        coef_df["label"] = coef_df["feature"].apply(_coef_label)
        coef_df["bar_color"] = np.where(coef_df["coef"] >= 0, "#16a34a", "#dc2626")

        fig_coef = go.Figure(
            go.Bar(
                x=coef_df["coef"],
                y=coef_df["label"],
                orientation="h",
                marker_color=coef_df["bar_color"].tolist(),
                text=coef_df["coef"].apply(lambda v: f"${v:+,.0f}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Coefficient: $%{x:+,.0f}<extra></extra>",
            )
        )
        fig_coef.update_layout(
            xaxis_title="Coefficient ($)",
            yaxis=dict(autorange="reversed"),
            height=max(420, len(coef_df) * 28),
            margin=dict(t=20, l=240, r=80),
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        # natural-language insight for flat_age
        age_row = models["coef_df"][models["coef_df"]["feature"] == "flat_age"]
        if len(age_row) > 0:
            age_coef = float(age_row.iloc[0]["coef"])
            if age_coef < 0:
                st.info(
                    f"\U0001f4c9 **Each additional year of flat age** reduces the estimated "
                    f"price by **{fmt_price(abs(age_coef))}** (all else equal)."
                )

        lease_row = models["coef_df"][
            models["coef_df"]["feature"] == "remaining_lease_yrs"
        ]
        if len(lease_row) > 0:
            lease_coef = float(lease_row.iloc[0]["coef"])
            if lease_coef > 0:
                st.info(
                    f"\U0001f4c8 **Each additional year of remaining lease** adds "
                    f"**{fmt_price(lease_coef)}** to the estimated price (all else equal)."
                )

        st.divider()

        # ── predicted vs actual ───────────────────────────────────────────
        st.markdown("#### Predicted vs Actual \u2014 test set sample (n=2,000)")

        y_test_arr = models["y_test"]
        ridge_pred_arr = models["ridge_test_pred"]
        gb_pred_arr = models["gb_test_pred"]

        rng = np.random.default_rng(seed=42)
        n_samp = min(2000, len(y_test_arr))
        idx_samp = rng.choice(len(y_test_arr), n_samp, replace=False)

        sdf = pd.DataFrame(
            {
                "Actual ($)": y_test_arr[idx_samp],
                "Ridge Predicted ($)": ridge_pred_arr[idx_samp],
                "GB Predicted ($)": gb_pred_arr[idx_samp],
            }
        )
        axis_max = float(
            max(sdf["Actual ($)"].max(), sdf["Ridge Predicted ($)"].max(),
                sdf["GB Predicted ($)"].max())
        )

        col_l, col_r = st.columns(2)
        for col, model_label, pred_col in [
            (col_l, "Ridge Regression", "Ridge Predicted ($)"),
            (col_r, "Gradient Boosting", "GB Predicted ($)"),
        ]:
            fig_sc = px.scatter(
                sdf,
                x="Actual ($)",
                y=pred_col,
                opacity=0.35,
                color_discrete_sequence=["#3b82f6"],
                title=model_label,
                labels={pred_col: "Predicted ($)"},
            )
            fig_sc.add_trace(
                go.Scatter(
                    x=[0, axis_max],
                    y=[0, axis_max],
                    mode="lines",
                    name="Perfect prediction",
                    line=dict(color="#94a3b8", dash="dash", width=1.5),
                    showlegend=False,
                )
            )
            fig_sc.update_layout(margin=dict(t=50))
            with col:
                st.plotly_chart(fig_sc, use_container_width=True)

        st.caption(
            "\u2139\ufe0f **Model limitations:** Trained on last 5 years only. "
            "Prediction interval = \u00b11.645\u00d7 residual std (approx. 90% confidence). "
            "Does **not** account for: renovation quality, exact unit facing, view, "
            "corner/point block status, or floor-level variation within the storey range."
        )

    else:
        # ── PRIVATE: Model Performance ────────────────────────────────────
        st.subheader("Private Condo \u2014 GradientBoosting Model Performance")
        st.warning(
            "\u26a0\ufe0f **Private model trained on ~3 years of data (2021\u20132026).** "
            "Fewer years than the HDB model (35 years) \u2014 predictions carry wider uncertainty."
        )
        st.caption(
            "Trained on the **last 3 years** of Condo/Apartment and EC transactions "
            "using a chronological 80/20 train/test split. Target: price per sqm (PSM)."
        )

        priv_models = get_priv_model()
        if priv_models is None:
            st.error(
                "Private condo data not available. "
                "Run `fetch_data.py` then `combine_clean_condo.py` first."
            )
        else:
            m = priv_models["metrics"]
            st.markdown("#### Performance on held-out test set (price PSM, $/sqm)")
            metrics_tbl_p = pd.DataFrame(
                {
                    "Metric": ["R\u00b2", "MAE ($/sqm)", "RMSE ($/sqm)"],
                    "Gradient Boosting (300 trees)": [
                        f"{m['R2']:.4f}",
                        f"${m['MAE']:,.0f}",
                        f"${m['RMSE']:,.0f}",
                    ],
                }
            ).set_index("Metric")
            st.dataframe(metrics_tbl_p, use_container_width=True)

            st.divider()

            st.markdown("#### Predicted vs Actual PSM \u2014 test set sample (n=2,000)")
            y_test_p = priv_models["y_test"]
            gb_pred_p = priv_models["gb_test_pred"]

            rng_p = np.random.default_rng(seed=42)
            n_samp_p = min(2000, len(y_test_p))
            idx_samp_p = rng_p.choice(len(y_test_p), n_samp_p, replace=False)

            sdf_p = pd.DataFrame(
                {
                    "Actual PSM ($/sqm)": y_test_p[idx_samp_p],
                    "Predicted PSM ($/sqm)": gb_pred_p[idx_samp_p],
                }
            )
            axis_max_p = float(
                max(sdf_p["Actual PSM ($/sqm)"].max(),
                    sdf_p["Predicted PSM ($/sqm)"].max())
            )
            fig_sc_p = px.scatter(
                sdf_p,
                x="Actual PSM ($/sqm)",
                y="Predicted PSM ($/sqm)",
                opacity=0.35,
                color_discrete_sequence=["#3b82f6"],
                title="Gradient Boosting \u2014 Predicted vs Actual PSM",
            )
            fig_sc_p.add_trace(
                go.Scatter(
                    x=[0, axis_max_p],
                    y=[0, axis_max_p],
                    mode="lines",
                    name="Perfect prediction",
                    line=dict(color="#94a3b8", dash="dash", width=1.5),
                    showlegend=False,
                )
            )
            fig_sc_p.update_layout(margin=dict(t=50))
            st.plotly_chart(fig_sc_p, use_container_width=True)

            st.caption(
                "\u2139\ufe0f Model trained on Condo/Apartment and EC only. "
                "Features: area_sqm, floor_mid (band midpoint), contract_year, "
                "contract_month, district, property_type_broad, tenure_clean. "
                "Does **not** capture: unit facing, view, renovation, developer premium."
            )


# ==========================================================================
# TAB 2: Fair Value Lookup
# ==========================================================================
with tab_lookup:
    if mode == "\U0001f3d8\ufe0f HDB Resale":
        st.subheader("Fair Value Lookup")
        st.caption(
            "Enter a flat's details to get a hedonic fair value estimate "
            "with a 90% prediction interval."
        )

        models = get_models()
        towns = get_town_list()
        flat_types = get_flat_type_list()
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month

        with st.form("fv_lookup_form"):
            c1, c2 = st.columns(2)
            with c1:
                sel_town = st.selectbox("Town", towns, key="lkp_town")
                sel_ft = st.selectbox("Flat type", flat_types, key="lkp_ft")
                sel_area = st.number_input(
                    "Floor area (sqm)",
                    min_value=20.0, max_value=300.0,
                    value=90.0, step=1.0, key="lkp_area",
                )
            with c2:
                sel_storey = st.number_input(
                    "Storey mid",
                    min_value=1, max_value=50,
                    value=10, step=1, key="lkp_storey",
                )
                sel_lease = st.number_input(
                    "Remaining lease (years)",
                    min_value=1.0, max_value=99.0,
                    value=75.0, step=0.5, key="lkp_lease",
                )
            submitted_lkp = st.form_submit_button(
                "\U0001f4d0 Estimate Fair Value", type="primary"
            )

        if submitted_lkp:
            flat_age_implied = max(0, 99 - int(round(sel_lease)))

            input_row = pd.DataFrame([{
                "floor_area_sqm": float(sel_area),
                "storey_mid": float(sel_storey),
                "remaining_lease_yrs": float(sel_lease),
                "flat_age": float(flat_age_implied),
                "year": float(current_year),
                "month_num": float(current_month),
                "town": sel_town,
                "flat_type": sel_ft,
            }])

            ridge_est = float(models["ridge_pipe"].predict(input_row)[0])
            gb_est = float(models["gb_pipe"].predict(input_row)[0])
            r_std = models["ridge_resid_std"]
            ridge_lo = ridge_est - 1.645 * r_std
            ridge_hi = ridge_est + 1.645 * r_std

            # recent median for this town + type
            df5 = load_5yr()
            subset = df5[
                (df5["town"] == sel_town) & (df5["flat_type"] == sel_ft)
            ]
            recent_median = (
                float(subset["resale_price"].median()) if len(subset) > 0 else None
            )

            st.markdown("---")
            st.markdown("#### Estimated Fair Value")
            k1, k2, k3 = st.columns(3)
            k1.metric(
                "Ridge estimate",
                fmt_price(ridge_est),
                help="Interpretable linear model with town/type one-hot encoding.",
            )
            k2.metric(
                "Gradient Boosting estimate",
                fmt_price(gb_est),
                help="Non-linear ensemble \u2014 typically more accurate but less interpretable.",
            )
            k3.metric(
                "90% interval (Ridge)",
                f"{fmt_price(max(0.0, ridge_lo))} \u2013 {fmt_price(ridge_hi)}",
                help=(
                    "\u00b11.645\u00d7 residual std on training set. "
                    "Approximate 90% confidence interval."
                ),
            )

            if recent_median is not None:
                pct_diff = (ridge_est / recent_median - 1) * 100
                direction = "above" if pct_diff >= 0 else "below"
                st.info(
                    f"**Recent median ({sel_town} / {sel_ft}): {fmt_price(recent_median)}** "
                    f"({len(subset):,} transactions in last 5 years) \u2014 "
                    f"this estimate is **{abs(pct_diff):.1f}% {direction}** the recent median."
                )
            else:
                st.info(
                    "No recent transactions found for this town/type combination "
                    "in the last 5 years."
                )

            # ── top price drivers ─────────────────────────────────────────
            st.markdown("#### Top 5 Ridge price drivers for this flat")
            st.caption(
                "Numeric: coefficient \u00d7 input value. "
                "Town/type: one-hot dummy coefficient."
            )

            coef_df_full = models["coef_df"].copy()

            # Active one-hot columns for this flat
            active_town_col = f"town_{sel_town}"
            active_ft_col = f"flat_type_{sel_ft}"

            num_df = (
                coef_df_full[coef_df_full["feature"].isin(NUMERIC_FEATS)]
                .copy()
            )
            cat_active_df = coef_df_full[
                coef_df_full["feature"].isin([active_town_col, active_ft_col])
            ].copy()

            drivers = pd.concat([num_df, cat_active_df], ignore_index=True)
            drivers = (
                drivers
                .assign(abs_coef=lambda d: d["coef"].abs())
                .sort_values("abs_coef", ascending=False)
                .head(5)
                .drop(columns="abs_coef")
            )

            input_values = {
                "floor_area_sqm": sel_area,
                "storey_mid": float(sel_storey),
                "remaining_lease_yrs": sel_lease,
                "flat_age": float(flat_age_implied),
                "year": float(current_year),
                "month_num": float(current_month),
                active_town_col: 1.0,
                active_ft_col: 1.0,
            }

            def _driver_desc(row):
                f = row["feature"]
                c = row["coef"]
                v = input_values.get(f, 1.0)
                contrib = c * v
                lbl = _coef_label(f)
                if f in NUMERIC_FEATS:
                    return f"{lbl}: {v:g} \u00d7 ${c:+,.0f} = **${contrib:+,.0f}**"
                return f"{lbl}: **${c:+,.0f}** (dummy premium/discount)"

            for _, row in drivers.iterrows():
                desc = _driver_desc(row)
                icon = "\U0001f7e2" if row["coef"] >= 0 else "\U0001f534"
                st.markdown(f"{icon} {desc}")

            st.warning(
                "\U0001f4ca **DATA CONFIDENCE: Medium.** "
                "This model cannot capture: renovation quality, interior condition, "
                "exact unit facing, view obstruction, corner/point block premium, "
                "proximity to specific amenities or noise sources, past incidents, "
                "or recent policy changes not yet reflected in training data. "
                "Use as a starting range \u2014 not a definitive valuation."
            )

    else:
        # ── PRIVATE: Fair Value Lookup ────────────────────────────────────
        st.subheader("Private Condo \u2014 Fair Value Lookup")
        st.caption(
            "Enter a condo/EC unit\u2019s details to get an estimated fair value PSM "
            "with an 80% prediction interval (\u00b115% simplified approximation)."
        )

        priv_models = get_priv_model()
        if priv_models is None:
            st.error(
                "Private condo data not available. "
                "Run `fetch_data.py` then `combine_clean_condo.py` first."
            )
        else:
            df_priv = load_priv_3yr()
            dist_options_lkp = sorted(
                [int(d) for d in df_priv["district"].dropna().unique()]
            )
            floor_options_lkp = sorted(df_priv["floor_range"].dropna().unique().tolist())
            tenure_options_lkp = sorted(df_priv["tenure_clean"].dropna().unique().tolist())

            with st.form("fv_priv_lookup_form"):
                c1, c2 = st.columns(2)
                with c1:
                    sel_district = st.selectbox(
                        "District",
                        [f"D{d:02d}" for d in dist_options_lkp],
                        key="fv_priv_lkp_district",
                    )
                    sel_ptb = st.selectbox(
                        "Property type",
                        ["Condo/Apartment", "EC"],
                        key="fv_priv_lkp_ptb",
                    )
                    sel_tenure = st.selectbox(
                        "Tenure",
                        tenure_options_lkp,
                        key="fv_priv_lkp_tenure",
                    )
                with c2:
                    sel_floor = st.selectbox(
                        "Floor range",
                        floor_options_lkp,
                        key="fv_priv_lkp_floor",
                    )
                    sel_area_p = st.number_input(
                        "Area (sqm)",
                        min_value=20.0, max_value=1000.0,
                        value=100.0, step=5.0,
                        key="fv_priv_lkp_area",
                    )
                submitted_priv_lkp = st.form_submit_button(
                    "\U0001f3e2 Estimate Fair Value", type="primary"
                )

            if submitted_priv_lkp:
                dist_num = int(sel_district[1:])
                floor_mid_val = floor_range_mid(sel_floor)
                if math.isnan(float(floor_mid_val)):
                    floor_mid_val = 5.0
                cy = datetime.datetime.now().year
                cm = datetime.datetime.now().month

                input_row_p = pd.DataFrame([{
                    "area_sqm": float(sel_area_p),
                    "floor_mid": float(floor_mid_val),
                    "contract_year": float(cy),
                    "contract_month": float(cm),
                    "district": str(dist_num),
                    "property_type_broad": sel_ptb,
                    "tenure_clean": sel_tenure,
                }])

                pred_psm = float(priv_models["gb_pipe"].predict(input_row_p)[0])
                pred_total = pred_psm * sel_area_p
                lo_80 = pred_psm * 0.85
                hi_80 = pred_psm * 1.15

                # District median PSM from recent data
                dist_subset = df_priv[
                    df_priv["district"].astype(str) == str(dist_num)
                ]
                dist_med_psm = (
                    float(dist_subset["price_psm"].median())
                    if len(dist_subset) > 0 else None
                )

                st.markdown("---")
                st.markdown("#### Estimated Fair Value")
                k1, k2, k3 = st.columns(3)
                k1.metric("Predicted PSM", f"${pred_psm:,.0f}/sqm")
                k2.metric("Predicted Total Price", f"${pred_total:,.0f}")
                k3.metric(
                    "80% Prediction Interval",
                    f"${lo_80:,.0f} \u2013 ${hi_80:,.0f} /sqm",
                    help="\u00b115% of predicted PSM as a simplified interval (GBM has no native PI).",
                )

                if dist_med_psm is not None:
                    pct_diff_p = (pred_psm / dist_med_psm - 1) * 100
                    direction_p = "above" if pct_diff_p >= 0 else "below"
                    st.info(
                        f"**District D{dist_num:02d} median PSM (last 3 yrs): "
                        f"${dist_med_psm:,.0f}/sqm** "
                        f"({len(dist_subset):,} transactions) \u2014 "
                        f"this estimate is **{abs(pct_diff_p):.1f}% {direction_p}** "
                        "the district median."
                    )

                st.warning(
                    "\U0001f4ca **DATA CONFIDENCE: Medium.** "
                    "\u00b115% interval is a simplified approximation \u2014 "
                    "GradientBoosting does not natively produce prediction intervals. "
                    "Cannot capture: renovation quality, exact floor level within band, "
                    "unit facing, view, or developer reputation premium. "
                    "Use as a starting range \u2014 not a definitive valuation."
                )


# ==========================================================================
# TAB 3: Mispricing Detector
# ==========================================================================
with tab_detect:
    if mode == "\U0001f3d8\ufe0f HDB Resale":
        st.subheader("Mispricing Detector \u2014 Block Level (last 24 months)")
        st.caption(
            "Ridge model residuals grouped at block+street level. "
            "Blocks with \u2265 5 transactions flagged if median standardised residual < \u20131 or > +1."
        )

        st.warning(
            "\U0001f6a8 **Important caveat:** Never call something \u2018undervalued\u2019 "
            "just because it is cheap. A block may be below the model estimate for legitimate "
            "structural reasons: lower floors, older flat model, proximity to noise or "
            "industrial areas, past incidents, or simply fewer renovated units in that "
            "cluster. Always verify on the ground. The model residual is a "
            "**starting signal for further investigation, not a confirmed mispricing.**"
        )

        models = get_models()
        df24_resid = get_detect_df(models)
        coords_df = load_coords()

        # ── block-level aggregation ───────────────────────────────────────
        block_grp = (
            df24_resid.groupby(["block_str", "street_name", "town"])
            .agg(
                txn_count=("resale_price", "size"),
                median_std_residual=("std_residual", "median"),
                median_actual_psm=("price_per_sqm", "median"),
                median_predicted_psm=("predicted_psm", "median"),
            )
            .reset_index()
        )
        block_grp = block_grp[block_grp["txn_count"] >= 5].copy()
        block_grp["discount_premium_pct"] = (
            (block_grp["median_actual_psm"] / block_grp["median_predicted_psm"] - 1)
            * 100
        )

        # merge coords
        merged = block_grp.merge(
            coords_df.rename(columns={"block": "block_str"}),
            on=["block_str", "street_name"],
            how="left",
        )
        blocks_geo = merged.dropna(subset=["lat", "lon"]).copy()

        # ── pydeck scatter map ────────────────────────────────────────────
        if len(blocks_geo) > 0:
            st.markdown("#### \U0001f5fa\ufe0f Map: median standardised residual by block")
            st.caption(
                "**Blue** = actual below model (potentially underpriced); "
                "**Red** = actual above model (potentially overpriced). "
                "Colour intensity reflects magnitude."
            )

            # vectorised colour computation: clamp to [-3, 3]
            sr = blocks_geo["median_std_residual"].clip(-3, 3).values
            # negative -> blue channel; positive -> red channel
            blue_mask = sr < 0
            intensity = (np.abs(sr) / 3.0 * 200).astype(int)

            r_ch = np.where(blue_mask, 30, 220)
            g_ch = np.where(blue_mask, 80 + intensity, 80 + (200 - intensity))
            b_ch = np.where(blue_mask, 200 + np.minimum(intensity, 55), 30)
            g_ch = np.clip(g_ch, 0, 255)
            b_ch = np.clip(b_ch, 0, 255)

            blocks_geo = blocks_geo.copy()
            blocks_geo["r"] = r_ch
            blocks_geo["g"] = g_ch.astype(int)
            blocks_geo["b"] = b_ch.astype(int)
            blocks_geo["a"] = 180

            blocks_geo["tip"] = (
                blocks_geo["block_str"].astype(str)
                + " "
                + blocks_geo["street_name"]
                + " | Town: "
                + blocks_geo["town"]
                + " | Std Residual: "
                + blocks_geo["median_std_residual"].round(2).astype(str)
                + " | Txns: "
                + blocks_geo["txn_count"].astype(str)
            )

            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=blocks_geo,
                get_position=["lon", "lat"],
                get_fill_color=["r", "g", "b", "a"],
                get_radius=90,
                pickable=True,
                auto_highlight=True,
            )
            view = pdk.ViewState(
                latitude=1.3521, longitude=103.8198, zoom=11, pitch=0
            )
            deck = pdk.Deck(
                layers=[scatter_layer],
                initial_view_state=view,
                tooltip={"text": "{tip}"},
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            )
            st.pydeck_chart(deck)
            st.caption(
                f"Showing {len(blocks_geo):,} geocoded blocks "
                f"(out of {len(block_grp):,} blocks with \u22655 transactions)."
            )
        else:
            st.info(
                "Geocoded coordinates not available for any blocks in the last "
                "24 months. Check that address_coords.csv is populated. "
                "Tables below still show all qualifying blocks."
            )

        st.divider()

        # ── tables ────────────────────────────────────────────────────────
        _DISP_COLS = {
            "block_str": "Block",
            "street_name": "Street",
            "town": "Town",
            "txn_count": "Transactions",
            "median_actual_psm": "Median Actual PSM ($)",
            "median_predicted_psm": "Median Predicted PSM ($)",
            "discount_premium_pct": "Discount/Premium %",
            "median_std_residual": "Std Residual",
        }

        def _render_block_table(df_blk):
            if len(df_blk) == 0:
                return None
            return (
                df_blk[list(_DISP_COLS.keys())]
                .rename(columns=_DISP_COLS)
                .style.format({
                    "Median Actual PSM ($)": "${:,.0f}",
                    "Median Predicted PSM ($)": "${:,.0f}",
                    "Discount/Premium %": "{:+.1f}%",
                    "Std Residual": "{:+.2f}",
                })
            )

        underpriced = (
            block_grp[block_grp["median_std_residual"] < -1]
            .sort_values("median_std_residual")
            .head(20)
        )
        overpriced = (
            block_grp[block_grp["median_std_residual"] > 1]
            .sort_values("median_std_residual", ascending=False)
            .head(20)
        )

        col_under, col_over = st.columns(2)

        with col_under:
            st.markdown(
                f"#### \U0001f7e6 Potentially Underpriced Blocks "
                f"(std residual < \u20131) \u2014 top {len(underpriced)}"
            )
            styled = _render_block_table(underpriced)
            if styled is not None:
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "No blocks qualify \u2014 need \u22655 transactions and "
                    "std residual < \u20131."
                )

        with col_over:
            st.markdown(
                f"#### \U0001f7e5 Potentially Overpriced Blocks "
                f"(std residual > +1) \u2014 top {len(overpriced)}"
            )
            styled = _render_block_table(overpriced)
            if styled is not None:
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "No blocks qualify \u2014 need \u22655 transactions and "
                    "std residual > +1."
                )

    else:
        # ── PRIVATE: Mispricing Detector ──────────────────────────────────
        st.subheader("Private Condo \u2014 Mispricing Detector (Project Level)")
        st.caption(
            "Compares actual median PSM vs model-predicted PSM at project level "
            "to flag potentially over- or under-priced projects."
        )
        st.warning(
            "\u26a0\ufe0f **Important caveat:** A project below model predictions may have "
            "legitimate reasons (older buildings, fewer facilities, less desirable facing). "
            "Use as a starting signal \u2014 always do on-ground research before concluding mispricing."
        )

        priv_models = get_priv_model()
        if priv_models is None:
            st.error(
                "Private condo data not available. "
                "Run `fetch_data.py` then `combine_clean_condo.py` first."
            )
        else:
            df_priv_det = load_priv_3yr()
            df_for_pred = df_priv_det.dropna(
                subset=PRIV_ALL_FEATS + [PRIV_TARGET]
            ).copy()
            df_for_pred["district"] = df_for_pred["district"].astype(str)
            df_for_pred["predicted_psm"] = priv_models["gb_pipe"].predict(
                df_for_pred[PRIV_ALL_FEATS]
            )

            proj_grp = (
                df_for_pred.groupby("project")
                .agg(
                    txn_count=(PRIV_TARGET, "size"),
                    actual_med_psm=(PRIV_TARGET, "median"),
                    predicted_med_psm=("predicted_psm", "median"),
                    district=("district", "first"),
                )
                .reset_index()
            )
            proj_grp = proj_grp[proj_grp["txn_count"] >= 3].copy()
            proj_grp["pct_diff"] = (
                (proj_grp["actual_med_psm"] / proj_grp["predicted_med_psm"] - 1) * 100
            )

            underpriced_proj = proj_grp.nsmallest(10, "pct_diff")
            overpriced_proj = proj_grp.nlargest(10, "pct_diff")

            col_u, col_o = st.columns(2)

            _PROJ_COLS = {
                "project": "Project", "district": "District",
                "txn_count": "Transactions",
                "actual_med_psm": "Actual PSM ($)",
                "predicted_med_psm": "Predicted PSM ($)",
                "pct_diff": "% Deviation",
            }
            _PROJ_FMT = {
                "Actual PSM ($)": "${:,.0f}",
                "Predicted PSM ($)": "${:,.0f}",
                "% Deviation": "{:+.1f}%",
            }

            with col_u:
                st.markdown("#### \U0001f7e6 Top 10 Most Underpriced Projects")
                st.caption("Actual median PSM \u2264 predicted by most")
                st.dataframe(
                    underpriced_proj[list(_PROJ_COLS)].rename(columns=_PROJ_COLS)
                    .style.format(_PROJ_FMT),
                    use_container_width=True, hide_index=True,
                )

            with col_o:
                st.markdown("#### \U0001f7e5 Top 10 Most Overpriced Projects")
                st.caption("Actual median PSM \u2265 predicted by most")
                st.dataframe(
                    overpriced_proj[list(_PROJ_COLS)].rename(columns=_PROJ_COLS)
                    .style.format(_PROJ_FMT),
                    use_container_width=True, hide_index=True,
                )

            st.divider()
            st.markdown("#### Top 20 Projects by % Deviation from Model")
            top20_dev = (
                proj_grp
                .assign(_abs_pct=proj_grp["pct_diff"].abs())
                .nlargest(20, "_abs_pct")
                .drop(columns="_abs_pct")
                .sort_values("pct_diff")
            )
            fig_bar_p = px.bar(
                top20_dev,
                x="pct_diff",
                y="project",
                orientation="h",
                color="pct_diff",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                labels={
                    "pct_diff": "% Deviation from Model",
                    "project": "Project",
                },
                title="Project-level % deviation: actual PSM vs predicted PSM",
            )
            fig_bar_p.update_layout(
                height=600,
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_bar_p, use_container_width=True)


# ==========================================================================
# TAB 4: Comparable Engine
# ==========================================================================
with tab_comps:
    if mode == "\U0001f3d8\ufe0f HDB Resale":
        st.subheader("Comparable Engine")
        st.caption(
            "Weighted similarity search over the last 5 years. "
            "Implied value = weighted median of top-20 comparable prices."
        )

        towns = get_town_list()
        flat_types = get_flat_type_list()
        current_year = datetime.datetime.now().year

        with st.form("comp_engine_form"):
            c1, c2 = st.columns(2)
            with c1:
                comp_town = st.selectbox("Town", towns, key="comp_town")
                comp_ft = st.selectbox("Flat type", flat_types, key="comp_ft")
                comp_area = st.number_input(
                    "Floor area (sqm)",
                    min_value=20.0, max_value=300.0,
                    value=90.0, step=1.0, key="comp_area",
                )
            with c2:
                comp_storey = st.number_input(
                    "Storey mid",
                    min_value=1, max_value=50,
                    value=10, step=1, key="comp_storey",
                )
                comp_lease = st.number_input(
                    "Remaining lease (years)",
                    min_value=1.0, max_value=99.0,
                    value=75.0, step=0.5, key="comp_lease",
                )
            submitted_comp = st.form_submit_button(
                "\U0001f52c Find Comparables", type="primary"
            )

        if submitted_comp:
            coords_df = load_coords()
            df5 = load_5yr()

            # restrict candidates to same town + flat_type for relevance & speed
            required_cols = [
                "floor_area_sqm", "storey_mid", "remaining_lease_yrs",
                "resale_price", "price_per_sqm", "block", "street_name", "year",
                "month",
            ]
            candidates = df5[
                (df5["town"] == comp_town) & (df5["flat_type"] == comp_ft)
            ].dropna(subset=required_cols).copy()

            if len(candidates) == 0:
                st.warning(
                    "No transactions found for this town/flat type combination "
                    "in the last 5 years."
                )
            else:
                candidates["block_str"] = candidates["block"].astype(str)

                # merge coords onto candidates (vectorised)
                candidates = candidates.merge(
                    coords_df.rename(columns={"block": "block_str"}),
                    on=["block_str", "street_name"],
                    how="left",
                )

                # target lat/lon: block-level if available; else town centroid
                t_centroid = TOWN_CENTROIDS.get(comp_town, (None, None))
                t_lat, t_lon = t_centroid

                # ── vectorised similarity score ───────────────────────────
                target_area = float(comp_area)
                target_storey = float(comp_storey)
                target_lease = float(comp_lease)

                s_area = (
                    (candidates["floor_area_sqm"] - target_area).abs()
                    / max(target_area, 1.0)
                    * 0.35
                )
                s_storey = (
                    (candidates["storey_mid"] - target_storey).abs()
                    / 10.0
                    * 0.20
                )
                s_lease = (
                    (candidates["remaining_lease_yrs"] - target_lease).abs()
                    / 10.0
                    * 0.20
                )
                s_time = (
                    (current_year - candidates["year"]).clip(lower=0)
                    / 3.0
                    * 0.10
                )

                if t_lat is not None and t_lon is not None:
                    has_coords = (
                        candidates["lat"].notna() & candidates["lon"].notna()
                    )
                    # haversine vectorised
                    lat1r = np.radians(t_lat)
                    lon1r = np.radians(t_lon)
                    lat2r = np.where(
                        has_coords,
                        np.radians(candidates["lat"].fillna(t_lat).values),
                        lat1r,
                    )
                    lon2r = np.where(
                        has_coords,
                        np.radians(candidates["lon"].fillna(t_lon).values),
                        lon1r,
                    )
                    dphi = lat2r - lat1r
                    dlam = lon2r - lon1r
                    a_hav = (
                        np.sin(dphi / 2) ** 2
                        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlam / 2) ** 2
                    )
                    dist_km = 6371.0 * 2 * np.arctan2(
                        np.sqrt(a_hav), np.sqrt(1 - a_hav)
                    )
                    # fallback 0.5 km where no coords
                    dist_km = np.where(has_coords, dist_km, 0.5)
                else:
                    dist_km = np.full(len(candidates), 0.5)

                s_geo = dist_km * 0.15
                candidates["similarity_score"] = (
                    s_area.values + s_storey.values + s_lease.values
                    + s_geo + s_time.values
                )

                top20 = candidates.nsmallest(20, "similarity_score").copy()

                # ── weighted median (weight = 1/similarity_score) ─────────
                top20["_weight"] = 1.0 / top20["similarity_score"].clip(lower=1e-6)
                sorted_comps = top20.sort_values("resale_price").reset_index(drop=True)
                cumw = sorted_comps["_weight"].cumsum()
                half_w = sorted_comps["_weight"].sum() / 2.0
                wmed_idx = int((cumw >= half_w).idxmax())
                implied_value = float(sorted_comps.loc[wmed_idx, "resale_price"])

                p25 = float(top20["resale_price"].quantile(0.25))
                p75 = float(top20["resale_price"].quantile(0.75))
                p25_psm = float(top20["price_per_sqm"].quantile(0.25))
                p75_psm = float(top20["price_per_sqm"].quantile(0.75))

                # ── result cards ─────────────────────────────────────────
                st.markdown("---")
                st.markdown("#### Implied Fair Value from Comparables")
                k1, k2, k3 = st.columns(3)
                k1.metric(
                    "Implied value (weighted median)",
                    fmt_price(implied_value),
                    help="Weighted median of top-20 comparable prices (weight = 1/similarity).",
                )
                k2.metric("25th percentile", fmt_price(p25))
                k3.metric("75th percentile", fmt_price(p75))

                st.info(
                    f"**Confidence:** Based on **{len(top20)} comparables** in "
                    f"**{comp_town}** ({comp_ft}). "
                    f"Implied value range: **{fmt_price(p25)} \u2013 {fmt_price(p75)}**. "
                    f"PSM range: **${p25_psm:,.0f} \u2013 ${p75_psm:,.0f}**."
                )

                # ── comparables table ─────────────────────────────────────
                st.markdown("#### Top 20 Comparable Transactions")
                disp = top20[[
                    "block_str", "street_name", "storey_mid",
                    "floor_area_sqm", "remaining_lease_yrs",
                    "resale_price", "price_per_sqm",
                    "month", "similarity_score",
                ]].copy()
                disp["month"] = pd.to_datetime(disp["month"]).dt.strftime("%Y-%m")
                disp = disp.rename(columns={
                    "block_str": "Block",
                    "street_name": "Street",
                    "storey_mid": "Storey",
                    "floor_area_sqm": "Area (sqm)",
                    "remaining_lease_yrs": "Lease (yrs)",
                    "resale_price": "Price ($)",
                    "price_per_sqm": "PSM ($)",
                    "month": "Date",
                    "similarity_score": "Similarity Score",
                })
                st.dataframe(
                    disp.style.format({
                        "Price ($)": "${:,.0f}",
                        "PSM ($)": "${:,.0f}",
                        "Storey": "{:.0f}",
                        "Area (sqm)": "{:.0f}",
                        "Lease (yrs)": "{:.1f}",
                        "Similarity Score": "{:.4f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "**Similarity Score:** lower = more similar. "
                    "Components: area difference (35%), storey (20%), lease (20%), "
                    "geographic distance (15%), transaction recency (10%)."
                )

    else:
        # ── PRIVATE: Comparable Engine ────────────────────────────────────
        st.subheader("Private Condo \u2014 Comparable Engine")
        st.caption(
            "Weighted similarity search over the last 3 years of Condo/EC transactions. "
            "Similarity: area 35%, floor midpoint 20%, tenure match 20%, recency 25%."
        )

        df_priv_comp = load_priv_3yr()
        if df_priv_comp.empty:
            st.error(
                "Private condo data not available. "
                "Run `fetch_data.py` then `combine_clean_condo.py` first."
            )
        else:
            dist_options_comp = sorted(
                [int(d) for d in df_priv_comp["district"].dropna().unique()]
            )
            floor_options_comp = sorted(
                df_priv_comp["floor_range"].dropna().unique().tolist()
            )
            tenure_options_comp = sorted(
                df_priv_comp["tenure_clean"].dropna().unique().tolist()
            )

            with st.form("fv_priv_comp_form"):
                c1, c2 = st.columns(2)
                with c1:
                    comp_district_p = st.selectbox(
                        "District",
                        [f"D{d:02d}" for d in dist_options_comp],
                        key="fv_priv_comp_district",
                    )
                    comp_ptb_p = st.selectbox(
                        "Property type",
                        ["Condo/Apartment", "EC"],
                        key="fv_priv_comp_ptb",
                    )
                    comp_tenure_p = st.selectbox(
                        "Tenure",
                        tenure_options_comp,
                        key="fv_priv_comp_tenure",
                    )
                with c2:
                    comp_floor_p = st.selectbox(
                        "Floor range",
                        floor_options_comp,
                        key="fv_priv_comp_floor",
                    )
                    comp_area_p = st.number_input(
                        "Area (sqm) \u2014 will search \u00b120 sqm",
                        min_value=20.0, max_value=1000.0,
                        value=100.0, step=5.0,
                        key="fv_priv_comp_area",
                    )
                submitted_priv_comp = st.form_submit_button(
                    "\U0001f50d Find Comparables", type="primary"
                )

            if submitted_priv_comp:
                comp_dist_num = int(comp_district_p[1:])
                comp_floor_mid_val = floor_range_mid(comp_floor_p)
                if math.isnan(float(comp_floor_mid_val)):
                    comp_floor_mid_val = 5.0

                # Filter to district, with ±20 sqm area window
                candidates_p = df_priv_comp[
                    df_priv_comp["district"].astype(str) == str(comp_dist_num)
                ].dropna(subset=["area_sqm", "price_psm", "price", "contract_date"]).copy()

                area_filtered = candidates_p[
                    (candidates_p["area_sqm"] >= comp_area_p - 20) &
                    (candidates_p["area_sqm"] <= comp_area_p + 20)
                ]
                # Fall back to full district if too few results
                if len(area_filtered) >= 5:
                    candidates_p = area_filtered.copy()

                if candidates_p.empty:
                    st.warning(
                        f"No transactions found for District D{comp_dist_num:02d}. "
                        "Try a different district."
                    )
                else:
                    candidates_p["floor_mid_c"] = candidates_p["floor_range"].apply(
                        floor_range_mid
                    )
                    today_p = pd.Timestamp.today()
                    candidates_p["days_since"] = (
                        today_p - candidates_p["contract_date"]
                    ).dt.days.clip(lower=0)

                    # Vectorised similarity scores
                    s_area_p = (
                        (candidates_p["area_sqm"] - comp_area_p).abs()
                        / max(float(comp_area_p), 1.0)
                        * 0.35
                    )
                    s_floor_p = (
                        (
                            candidates_p["floor_mid_c"].fillna(float(comp_floor_mid_val))
                            - float(comp_floor_mid_val)
                        ).abs()
                        / 10.0
                    ).clip(0, 1) * 0.20
                    s_tenure_p = candidates_p["tenure_clean"].apply(
                        lambda t: 0.0 if t == comp_tenure_p else 0.5
                    ) * 0.20
                    s_recency_p = (
                        candidates_p["days_since"] / (3 * 365)
                    ).clip(0, 1) * 0.25

                    candidates_p["similarity_score"] = (
                        s_area_p.values
                        + s_floor_p.values
                        + s_tenure_p.values
                        + s_recency_p.values
                    )

                    top20_p = candidates_p.nsmallest(20, "similarity_score").copy()

                    st.markdown("---")
                    st.markdown("#### Top 20 Comparable Transactions")
                    disp_p = top20_p[[
                        "project", "floor_range", "area_sqm",
                        "price", "price_psm", "contract_date", "similarity_score",
                    ]].copy()
                    disp_p["contract_date"] = disp_p["contract_date"].dt.strftime("%Y-%m-%d")
                    disp_p = disp_p.rename(columns={
                        "project": "Project",
                        "floor_range": "Floor Range",
                        "area_sqm": "Area (sqm)",
                        "price": "Price ($)",
                        "price_psm": "PSM ($/sqm)",
                        "contract_date": "Date",
                        "similarity_score": "Similarity Score",
                    })
                    st.dataframe(
                        disp_p.style.format({
                            "Price ($)": "${:,.0f}",
                            "PSM ($/sqm)": "${:,.0f}",
                            "Area (sqm)": "{:.0f}",
                            "Similarity Score": "{:.4f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "**Similarity Score:** lower = more similar. "
                        "Components: area difference (35%), floor midpoint (20%), "
                        "tenure match (20%), recency (25%)."
                    )

                    csv_bytes_p = disp_p.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "\u2b07\ufe0f Export Comparables CSV",
                        data=csv_bytes_p,
                        file_name=f"priv_comps_D{comp_dist_num:02d}.csv",
                        mime="text/csv",
                        key="fv_priv_comp_export",
                    )


# --------------------------------------------------------------------------
# footer
# --------------------------------------------------------------------------
st.divider()
st.caption(
    "\U0001f4cb **Reading guide:** Fair value estimates carry model uncertainty. "
    "The Ridge model offers interpretable coefficients and prediction intervals; "
    "Gradient Boosting maximises accuracy. Cross-reference the Comparable Engine "
    "(Tab 4) and on-ground research before any property decision. "
    "Model residuals in Tab 3 are investigative signals only \u2014 "
    "always validate structural reasons before concluding mispricing."
)


# ==========================================================================
# TAB 5: Distressed Sale Proxy
# ==========================================================================
with tab_distress:
    if mode == "\U0001f3d8\ufe0f HDB Resale":
        st.subheader("\U0001f9bb Distressed Sale Proxy (I10)")
        st.warning(
            "**Important caveat:** A short holding period has many innocent explanations "
            "(job relocation, divorce, inheritance, change of household composition). "
            "This is a probabilistic proxy \u2014 NOT a confirmed distressed sale indicator. "
            "Never present this as confirmed distress without independent verification."
        )
        st.caption(
            "Flags blocks/streets where an unusually high proportion of resale transactions "
            "occurred with a short holding period (<3 years from lease commence date as a proxy). "
            "Historically, motivated sellers sometimes accept slightly below-market prices. "
            "Data confidence: Low-Medium."
        )

        @st.cache_data
        def compute_distress_proxy(_df):
            recent = _df[_df["year"] >= _df["year"].max() - 5].copy()
            recent["short_hold"] = recent["flat_age"] < 6  # flat_age < 6 years as proxy for short hold
            block_stats = (recent.groupby(["block", "street_name", "town", "flat_type"])
                           .agg(
                               total=("resale_price", "count"),
                               short_hold_count=("short_hold", "sum"),
                               median_psm=("price_per_sqm", "median"),
                           ).reset_index())
            block_stats = block_stats[block_stats["total"] >= 5]
            block_stats["short_hold_pct"] = block_stats["short_hold_count"] / block_stats["total"] * 100
            return block_stats.sort_values("short_hold_pct", ascending=False)

        _dist_full = load_clean()
        dist_df = compute_distress_proxy(_dist_full)

        dist_col1, dist_col2 = st.columns([1, 3])
        with dist_col1:
            dist_town = st.selectbox("Filter by town", ["All"] + sorted(_dist_full["town"].dropna().unique()), key="dist_town")
            dist_flat = st.selectbox("Flat type", ["All"] + sorted(_dist_full["flat_type"].dropna().unique()), key="dist_flat")
            dist_min_pct = st.slider("Min short-hold %", 10, 80, 30, key="dist_min_pct")

        filt_dist = dist_df.copy()
        if dist_town != "All":
            filt_dist = filt_dist[filt_dist["town"] == dist_town]
        if dist_flat != "All":
            filt_dist = filt_dist[filt_dist["flat_type"] == dist_flat]
        filt_dist = filt_dist[filt_dist["short_hold_pct"] >= dist_min_pct]

        with dist_col2:
            if len(filt_dist) > 0:
                fig_dist = px.bar(
                    filt_dist.head(30),
                    x=filt_dist.head(30).apply(lambda r: f"Blk {r['block']} {r['street_name'][:20]}", axis=1),
                    y="short_hold_pct",
                    color="median_psm",
                    color_continuous_scale="RdYlGn_r",
                    labels={"x": "Block / Street", "short_hold_pct": "Short-hold transactions (%)"},
                    title="Top 30 blocks by short-hold transaction rate (last 5 years)",
                )
                fig_dist.update_layout(xaxis_tickangle=-45, height=450)
                st.plotly_chart(fig_dist, use_container_width=True)

                st.dataframe(
                    filt_dist[["block", "street_name", "town", "flat_type", "short_hold_pct", "total", "median_psm"]]
                    .rename(columns={"block": "Block", "street_name": "Street", "town": "Town",
                                      "flat_type": "Type", "short_hold_pct": "Short-Hold %",
                                      "total": "Transactions", "median_psm": "Median PSM ($)"})
                    .head(50),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No blocks match the current filters.")

        st.caption(
            "Definition: \u2018Short-hold\u2019 = flat_age < 6 years at time of resale transaction. "
            "This is a proxy \u2014 it catches genuine short holds but also early-lease flats "
            "regardless of how long the seller actually owned them."
        )

    else:
        # ── PRIVATE: Distressed Sale Proxy (Sub-Sales) ────────────────────
        st.subheader("\U0001f3e2 Private Condo \u2014 Sub-Sale Proxy (Distressed Sale Signal)")
        st.info(
            "**Sub-Sales** occur when a buyer who purchased from developer resells before "
            "project completion \u2014 often a signal of financial pressure or portfolio rebalancing. "
            "A high sub-sale proportion in a district may indicate buyer stress."
        )
        st.warning(
            "\u26a0\ufe0f Sub-sales may also reflect portfolio optimisation or legitimate financial "
            "planning. This is a proxy indicator \u2014 not confirmation of distress. "
            "Never present this as confirmed distress without independent verification."
        )

        df_priv_dist = load_priv_all()
        if df_priv_dist.empty:
            st.error(
                "Private condo data not available. "
                "Run `fetch_data.py` then `combine_clean_condo.py` first."
            )
        else:
            total_txns = len(df_priv_dist)
            subsale_df_p = df_priv_dist[
                df_priv_dist["type_of_sale"] == "Sub Sale"
            ].copy()
            n_subsales = len(subsale_df_p)

            st.markdown(
                f"**Total Condo/EC transactions:** {total_txns:,} \u00a0|\u00a0 "
                f"**Sub-sales:** {n_subsales:,} "
                f"({n_subsales / total_txns * 100:.1f}% of total)"
            )

            # Sub-sale % by district
            total_by_dist = (
                df_priv_dist.groupby("district").size().rename("total")
            )
            subsale_by_dist = (
                subsale_df_p.groupby("district").size().rename("sub_sales")
            )
            dist_summary_p = (
                pd.concat([total_by_dist, subsale_by_dist], axis=1)
                .fillna(0)
                .reset_index()
            )
            dist_summary_p["sub_sale_pct"] = (
                dist_summary_p["sub_sales"] / dist_summary_p["total"] * 100
            )
            dist_summary_p = dist_summary_p.sort_values("sub_sale_pct", ascending=False)
            dist_summary_p["district_label"] = dist_summary_p["district"].apply(
                lambda d: (
                    f"D{int(d):02d} \u2013 {DISTRICT_CENTROIDS.get(int(d), ('?',))[0]}"
                    if pd.notna(d) else "Unknown"
                )
            )

            fig_sub_p = px.bar(
                dist_summary_p,
                x="district_label",
                y="sub_sale_pct",
                color="sub_sale_pct",
                color_continuous_scale="Reds",
                labels={
                    "district_label": "District",
                    "sub_sale_pct": "Sub-Sale %",
                },
                title="Sub-Sale % by District (all available data)",
            )
            fig_sub_p.update_layout(
                xaxis_tickangle=-45, height=450, coloraxis_showscale=False
            )
            st.plotly_chart(fig_sub_p, use_container_width=True)

            st.divider()
            st.markdown("#### Most Recent 50 Sub-Sales")
            recent_subsales_p = (
                subsale_df_p
                .sort_values("contract_date", ascending=False)
                .head(50)[["project", "district", "price_psm", "area_sqm", "contract_date"]]
                .copy()
            )
            recent_subsales_p["contract_date"] = (
                recent_subsales_p["contract_date"].dt.strftime("%Y-%m-%d")
            )
            recent_subsales_p["district"] = recent_subsales_p["district"].apply(
                lambda d: f"D{int(d):02d}" if pd.notna(d) else "N/A"
            )
            st.dataframe(
                recent_subsales_p.rename(columns={
                    "project": "Project", "district": "District",
                    "price_psm": "PSM ($/sqm)", "area_sqm": "Area (sqm)",
                    "contract_date": "Date",
                }).style.format({
                    "PSM ($/sqm)": "${:,.0f}",
                    "Area (sqm)": "{:.0f}",
                }),
                use_container_width=True, hide_index=True,
            )


# ==========================================================================
# TAB 6: Transaction Quality Score (A5)
# ==========================================================================
with tab_quality:
    if mode == "\U0001f3d8\ufe0f HDB Resale":
        st.subheader("\U0001f4ca Transaction Quality Score (A5)")
        st.markdown(
            "Score how **reliable** a set of comparables is before using them for valuation. "
            "Each candidate transaction is rated across 5 dimensions \u2014 "
            "low-quality comps can make fair-value estimates unreliable."
        )
        st.info(
            "**Methodology:** Five quality dimensions, each scored 0\u201310:\n"
            "1. **Recency** \u2014 how recent is the transaction (newer = better)\n"
            "2. **Physical similarity** \u2014 same flat type, storey within \u00b13, area within \u00b120 sqm\n"
            "3. **Location** \u2014 same street scores 10; same town 6; otherwise 2\n"
            "4. **Lease similarity** \u2014 remaining lease within \u00b110 years of target\n"
            "5. **Price normality** \u2014 standardised residual from fair-value model (outliers score lower)\n\n"
            "**Composite** = simple average. \u2265 7 = good comp; 5\u20137 = usable with caution; < 5 = weak."
        )

        st.markdown("#### Define your target flat")
        models_q = get_models()
        detect_q  = get_detect_df(models_q)

        towns_q = get_town_list()
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            tgt_town = st.selectbox("Town", towns_q, key="q_town")
            tgt_ft   = st.selectbox("Flat type", get_flat_type_list(), key="q_ft")
        with col_q2:
            tgt_area    = st.number_input("Floor area (sqm)", 20.0, 300.0, 90.0, 1.0, key="q_area")
            tgt_storey  = st.number_input("Storey mid",       1, 50,      10,   1,   key="q_storey")
        with col_q3:
            tgt_lease   = st.number_input("Remaining lease (yrs)", 1.0, 99.0, 75.0, 0.5, key="q_lease")
            tgt_street  = st.text_input("Street name (optional, for location scoring)",
                                        value="", key="q_street",
                                        placeholder="e.g. ANG MO KIO AVE 3")

        if st.button("\U0001f4ca Score Comparables", key="q_run", type="primary"):
            # Pull 24-month data for that town
            pool = detect_q[detect_q["town"] == tgt_town].copy()
            if pool.empty:
                st.warning(f"No transactions found for {tgt_town} in the last 24 months.")
            else:
                tgt_street_upper = tgt_street.strip().upper()

                # ── Score each dimension ──────────────────────────────────────────
                # 1. Recency
                pool["month_dt"] = pd.to_datetime(pool["month"])
                max_dt = pool["month_dt"].max()
                pool["days_since"] = (max_dt - pool["month_dt"]).dt.days
                pool["s_recency"] = (10 * (1 - pool["days_since"] / 730)).clip(0, 10)

                # 2. Physical similarity
                ft_match   = (pool["flat_type"] == tgt_ft).astype(float) * 3.5
                stor_match = (abs(pool["storey_mid"] - tgt_storey) <= 3).astype(float) * 3.5
                area_match = (abs(pool["floor_area_sqm"] - tgt_area) <= 20).astype(float) * 3.0
                pool["s_physical"] = (ft_match + stor_match + area_match).clip(0, 10)

                # 3. Location
                def _loc_score(row):
                    if tgt_street_upper and row["street_name"].upper() == tgt_street_upper:
                        return 10.0
                    return 6.0  # same town (already filtered)
                pool["s_location"] = pool.apply(_loc_score, axis=1)

                # 4. Lease similarity
                pool["s_lease"] = (10 * (1 - (pool["remaining_lease_yrs"] - tgt_lease).abs() / 40)).clip(0, 10)

                # 5. Price normality (from fair-value residuals)
                if "std_residual" in pool.columns:
                    pool["s_residual"] = (10 * (1 - (pool["std_residual"].abs().clip(0, 3) / 3))).clip(0, 10)
                else:
                    pool["s_residual"] = 5.0

                # Composite
                pool["Quality Score"] = (
                    pool[["s_recency", "s_physical", "s_location", "s_lease", "s_residual"]].mean(axis=1)
                ).round(1)

                pool = pool.sort_values("Quality Score", ascending=False).reset_index(drop=True)

                median_q = pool["Quality Score"].median()
                if median_q < 5:
                    st.error(
                        f"\u26a0\ufe0f **Low comparable quality (median score {median_q:.1f}/10).** "
                        "Valuations using these comps carry elevated uncertainty. "
                        "Consider narrowing to a closer street or different storey range."
                    )
                elif median_q < 7:
                    st.warning(f"\U0001f7e1 **Moderate comparable quality (median {median_q:.1f}/10).** Use with caution.")
                else:
                    st.success(f"\U0001f7e2 **Good comparable quality (median {median_q:.1f}/10).**")

                # Display table
                disp_cols = [
                    "block", "street_name", "month", "flat_type", "storey_mid",
                    "floor_area_sqm", "remaining_lease_yrs", "resale_price", "price_per_sqm",
                    "s_recency", "s_physical", "s_location", "s_lease", "s_residual", "Quality Score",
                ]
                disp_cols = [c for c in disp_cols if c in pool.columns]
                rename_map = {
                    "block": "Block", "street_name": "Street", "month": "Month",
                    "flat_type": "Type", "storey_mid": "Storey", "floor_area_sqm": "Area (sqm)",
                    "remaining_lease_yrs": "Lease (yrs)", "resale_price": "Price ($)",
                    "price_per_sqm": "PSM ($/sqm)", "s_recency": "Recency",
                    "s_physical": "Physical", "s_location": "Location",
                    "s_lease": "Lease Sim", "s_residual": "Normality",
                }
                st.dataframe(
                    pool[disp_cols].rename(columns=rename_map).head(200),
                    use_container_width=True, hide_index=True,
                )

                # Radar: Top-10 vs All
                top10 = pool.head(10)
                dims  = ["s_recency", "s_physical", "s_location", "s_lease", "s_residual"]
                dim_labels = ["Recency", "Physical", "Location", "Lease", "Normality"]

                fig_radar = go.Figure()
                for label, subset in [("Top-10 comps", top10), ("All comps", pool)]:
                    vals = [subset[d].mean() for d in dims]
                    vals_closed = vals + [vals[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals_closed,
                        theta=dim_labels + [dim_labels[0]],
                        fill="toself",
                        name=label,
                        opacity=0.6,
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 10])),
                    title="Quality Radar: Top-10 vs All comparables",
                    height=420,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                st.caption(
                    "\U0001f4ca **DATA CONFIDENCE: High** for the quality score computation itself. "
                    "A high quality score means the comparables are internally consistent \u2014 "
                    "it does not guarantee the resulting valuation is correct, "
                    "since unit-level attributes (renovation, facing, view) are not captured."
                )

    else:
        # ── PRIVATE: Transaction Quality / Data Coverage ──────────────────
        st.subheader("\U0001f4ca Private Condo \u2014 Transaction Data Coverage")
        st.caption(
            "Data quality and coverage statistics for the private condo/EC dataset."
        )

        df_condo_cov = load_priv_all()
        if df_condo_cov.empty:
            st.error(
                "Private condo data not available. "
                "Run `fetch_data.py` then `combine_clean_condo.py` first."
            )
        else:
            total_cov = len(df_condo_cov)
            date_min_cov = df_condo_cov["contract_date"].min()
            date_max_cov = df_condo_cov["contract_date"].max()
            pct_geo_cov = df_condo_cov["lat"].notna().mean() * 100
            n_projects_cov = df_condo_cov["project"].nunique()
            n_districts_cov = df_condo_cov["district"].nunique()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Rows", f"{total_cov:,}")
            c2.metric(
                "Date Range",
                f"{date_min_cov.strftime('%Y-%m')} \u2192 {date_max_cov.strftime('%Y-%m')}",
            )
            c3.metric("% With Lat/Lon", f"{pct_geo_cov:.1f}%")
            c4.metric("Unique Projects", f"{n_projects_cov:,}")
            c5.metric("Districts Covered", f"{n_districts_cov}")

            st.divider()

            # Monthly transaction count
            st.markdown("#### Monthly Transaction Count")
            monthly_cov = (
                df_condo_cov
                .groupby(["contract_year", "contract_month"])
                .size()
                .reset_index(name="count")
            )
            monthly_cov["date"] = pd.to_datetime(
                monthly_cov["contract_year"].astype(str)
                + "-"
                + monthly_cov["contract_month"].astype(str).str.zfill(2)
            )
            monthly_cov = monthly_cov.sort_values("date")
            fig_monthly_cov = px.bar(
                monthly_cov,
                x="date",
                y="count",
                labels={"date": "Month", "count": "Transactions"},
                title="Monthly Private Condo/EC Transaction Count",
            )
            fig_monthly_cov.update_layout(height=350)
            st.plotly_chart(fig_monthly_cov, use_container_width=True)

            st.divider()

            # District × Year heatmap
            st.markdown("#### Transaction Coverage by District \u00d7 Year")
            df_condo_cov_h = df_condo_cov.copy()
            df_condo_cov_h["district_int"] = pd.to_numeric(
                df_condo_cov_h["district"], errors="coerce"
            )
            heat_df = (
                df_condo_cov_h.dropna(subset=["district_int"])
                .groupby(["district_int", "contract_year"])
                .size()
                .reset_index(name="count")
            )
            heat_df["district_label"] = heat_df["district_int"].apply(
                lambda d: f"D{int(d):02d}"
            )
            heat_pivot = (
                heat_df
                .pivot(index="district_label", columns="contract_year", values="count")
                .fillna(0)
                .sort_index()
            )
            fig_heat = px.imshow(
                heat_pivot,
                labels={"x": "Year", "y": "District", "color": "Transactions"},
                title="Transaction Count by District and Year",
                color_continuous_scale="Blues",
                aspect="auto",
                text_auto=True,
            )
            fig_heat.update_layout(height=600)
            st.plotly_chart(fig_heat, use_container_width=True)

            st.caption(
                "\U0001f4cb **DATA CONFIDENCE: High** for coverage statistics. "
                "Lat/lon coverage is ~79% overall (landed has no coordinates). "
                "Sub-sale and new-sale transactions are included in coverage counts."
            )
