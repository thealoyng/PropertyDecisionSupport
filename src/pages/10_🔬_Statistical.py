"""
Page 10 — Statistical Analysis
================================
Statistical tests and correlation analysis for Singapore HDB resale data.
Includes correlation matrix, pair plots, ANOVA, normality tests,
Levene's test, Random Forest feature importance, and VIF analysis.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from eda_helpers import load_clean

# ── page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Statistical Analysis",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Statistical Analysis")
st.caption(
    "Rigorous statistical tests and correlation analysis — ANOVA, normality "
    "diagnostics, multicollinearity checks, and machine-learning-based "
    "feature importance for Singapore HDB resale transactions."
)

# ── load data ────────────────────────────────────────────────────
df = load_clean()

# ── numeric columns used throughout ──────────────────────────────
NUMERIC_COLS = [
    "resale_price", "floor_area_sqm", "storey_mid",
    "flat_age", "remaining_lease_yrs", "price_per_sqm",
    "lease_commence_date", "year",
]
KEY_SCATTER_COLS = [
    "resale_price", "floor_area_sqm", "storey_mid",
    "remaining_lease_yrs", "price_per_sqm",
]
FEATURE_COLS = [
    "floor_area_sqm", "storey_mid", "flat_age",
    "remaining_lease_yrs", "year",
]

# ══════════════════════════════════════════════════════════════════
#  Cached heavy computations
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def compute_correlation_matrix(_df):
    """Pearson correlation matrix for numeric columns."""
    return _df[NUMERIC_COLS].corr()


@st.cache_data
def compute_anova_by_group(_df, group_col):
    """One-way ANOVA of price_per_sqm across levels of *group_col*."""
    groups = [g["price_per_sqm"].dropna().values for _, g in _df.groupby(group_col)]
    groups = [g for g in groups if len(g) >= 2]
    f_stat, p_val = stats.f_oneway(*groups)

    grand_mean = _df["price_per_sqm"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((_df["price_per_sqm"] - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return f_stat, p_val, eta_sq


@st.cache_data
def compute_normality(_df, sample_size=5000):
    """Shapiro-Wilk on raw and log resale_price; return stats + Q-Q data."""
    sample = _df["resale_price"].dropna().sample(
        n=min(sample_size, len(_df)), random_state=42
    )
    log_sample = np.log(sample)

    sw_raw_stat, sw_raw_p = stats.shapiro(sample)
    sw_log_stat, sw_log_p = stats.shapiro(log_sample)

    # Q-Q data (theoretical normal quantiles vs sorted observed)
    sorted_raw = np.sort(sample.values)
    sorted_log = np.sort(log_sample.values)
    n = len(sorted_raw)
    theoretical_q = stats.norm.ppf(
        (np.arange(1, n + 1) - 0.5) / n
    )

    return {
        "sw_raw_stat": sw_raw_stat,
        "sw_raw_p": sw_raw_p,
        "sw_log_stat": sw_log_stat,
        "sw_log_p": sw_log_p,
        "theoretical_q": theoretical_q,
        "sorted_raw": sorted_raw,
        "sorted_log": sorted_log,
    }


@st.cache_data
def compute_levene(_df):
    """Levene's test for equal variances of price_per_sqm across towns."""
    groups = [
        g["price_per_sqm"].dropna().values
        for _, g in _df.groupby("town")
    ]
    groups = [g for g in groups if len(g) >= 2]
    stat, p_val = stats.levene(*groups)
    return stat, p_val


@st.cache_data
def compute_rf_importance(_df, max_rows=50_000):
    """Random Forest feature importances (sampled for speed)."""
    sample = _df.dropna(subset=FEATURE_COLS + ["resale_price", "flat_type", "town"])
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=42)

    # Encode categoricals
    le_flat = LabelEncoder()
    le_town = LabelEncoder()
    sample = sample.copy()
    sample["flat_type_enc"] = le_flat.fit_transform(sample["flat_type"])
    sample["town_enc"] = le_town.fit_transform(sample["town"])

    feature_names = FEATURE_COLS + ["flat_type_enc", "town_enc"]
    X = sample[feature_names].values
    y = sample["resale_price"].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=True)

    return imp, rf.score(X, y)


@st.cache_data
def compute_vif(_df):
    """Variance Inflation Factors for numeric features."""
    cols = FEATURE_COLS  # avoid target leakage — no resale_price / price_per_sqm
    sub = _df[cols].dropna()
    if len(sub) > 100_000:
        sub = sub.sample(n=100_000, random_state=42)

    from statsmodels.tools import add_constant
    X = add_constant(sub)

    vif_data = pd.DataFrame({
        "Feature": cols,
        "VIF": [
            variance_inflation_factor(X.values, i + 1)  # +1 skips const
            for i in range(len(cols))
        ],
    })
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_data


# ══════════════════════════════════════════════════════════════════
#  Run all computations up front (cached) for the summary
# ══════════════════════════════════════════════════════════════════
corr_matrix = compute_correlation_matrix(df)
f_town, p_town, eta_town = compute_anova_by_group(df, "town")
f_flat, p_flat, eta_flat = compute_anova_by_group(df, "flat_type")
norm_results = compute_normality(df)
lev_stat, lev_p = compute_levene(df)
rf_imp, rf_r2 = compute_rf_importance(df)
vif_df = compute_vif(df)

# ══════════════════════════════════════════════════════════════════
#  Key Statistical Findings (data-driven summary)
# ══════════════════════════════════════════════════════════════════
st.subheader("Key Statistical Findings")

top_corr_pair = (
    corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )
    .stack()
    .abs()
    .idxmax()
)
top_corr_val = corr_matrix.loc[top_corr_pair[0], top_corr_pair[1]]

top_feature = rf_imp.iloc[-1]
max_vif_row = vif_df.iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Strongest correlation",
    f"{top_corr_val:+.3f}",
    f"{top_corr_pair[0]} vs {top_corr_pair[1]}",
    help="Highest absolute Pearson correlation among numeric features.",
)
k2.metric(
    "ANOVA: town effect",
    f"F = {f_town:,.1f}",
    f"eta-sq = {eta_town:.3f}",
    help="One-way ANOVA F-statistic for price_per_sqm across towns.",
)
k3.metric(
    "Top RF feature",
    top_feature["feature"],
    f"imp = {top_feature['importance']:.3f}",
    help="Most important feature from Random Forest model.",
)
k4.metric(
    "Highest VIF",
    f"{max_vif_row['VIF']:.1f}",
    max_vif_row["Feature"],
    help="Largest Variance Inflation Factor among predictor features.",
)

st.divider()

# ══════════════════════════════════════════════════════════════════
#  1. Correlation Matrix Heatmap
# ══════════════════════════════════════════════════════════════════
with st.expander("1. Correlation Matrix Heatmap", expanded=True):
    st.markdown(
        """
        **What it tests:** Pearson correlation measures the *linear* relationship
        between each pair of numeric variables (range -1 to +1).

        **How to read:** Values near +1 indicate strong positive linear association;
        near -1, strong negative; near 0, weak or no linear relationship.
        Diagonal is always 1.0 (a variable correlates perfectly with itself).
        """
    )

    fig_corr = px.imshow(
        corr_matrix.round(2),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="equal",
        labels=dict(color="Pearson r"),
    )
    fig_corr.update_layout(
        title="Pearson Correlation Matrix — Numeric Features",
        margin=dict(t=60, l=10, r=10, b=10),
        width=800,
        height=700,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Highlight top 5 pairs
    upper = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
        .stack()
        .reset_index()
    )
    upper.columns = ["Var 1", "Var 2", "Pearson r"]
    upper["Abs r"] = upper["Pearson r"].abs()
    top5 = upper.nlargest(5, "Abs r").drop(columns="Abs r").reset_index(drop=True)
    st.markdown("**Top 5 strongest correlations:**")
    st.dataframe(top5, hide_index=True, use_container_width=False)

# ══════════════════════════════════════════════════════════════════
#  2. Pair Plot (Scatter Matrix)
# ══════════════════════════════════════════════════════════════════
with st.expander("2. Pair Plot (Scatter Matrix)", expanded=False):
    st.markdown(
        """
        **What it shows:** A scatter matrix plots every pair of key variables
        against each other, coloured by flat type. Useful for spotting
        non-linear relationships, clusters, and outliers that the correlation
        coefficient alone may miss.

        **Performance note:** Sampled to 5,000 points to keep the chart responsive.
        """
    )

    sample_pair = df[KEY_SCATTER_COLS + ["flat_type"]].dropna().sample(
        n=min(5000, len(df)), random_state=42
    )

    fig_pair = px.scatter_matrix(
        sample_pair,
        dimensions=KEY_SCATTER_COLS,
        color="flat_type",
        opacity=0.4,
        labels={c: c.replace("_", " ") for c in KEY_SCATTER_COLS},
    )
    fig_pair.update_traces(diagonal_visible=False, marker=dict(size=3))
    fig_pair.update_layout(
        title="Scatter Matrix — Key Variables (sampled 5k)",
        height=900,
        margin=dict(t=60),
    )
    st.plotly_chart(fig_pair, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  3. ANOVA — Price by Town
# ══════════════════════════════════════════════════════════════════
with st.expander("3. ANOVA: Price per sqm by Town", expanded=False):
    st.markdown(
        """
        **What it tests:** One-way ANOVA tests whether the *mean* price per sqm
        differs significantly across towns. The null hypothesis is that all
        town means are equal.

        **Assumptions:** Independent observations, approximately normal
        distributions within each group, and homogeneous variances
        (checked via Levene's test below).

        **Effect size:** Eta-squared (eta-sq) quantifies the proportion of total variance
        in price explained by town membership. Guidelines: small ~0.01,
        medium ~0.06, large ~0.14+.
        """
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("F-statistic", f"{f_town:,.2f}")
    col_b.metric("p-value", f"{p_town:.2e}" if p_town < 0.001 else f"{p_town:.4f}")
    col_c.metric("Eta-squared (effect size)", f"{eta_town:.4f}")

    if p_town < 0.05:
        effect_label = (
            "large" if eta_town >= 0.14
            else "medium" if eta_town >= 0.06
            else "small"
        )
        st.success(
            f"**Result:** Statistically significant (p < 0.05). Town explains "
            f"~{eta_town * 100:.1f}% of the variance in price per sqm "
            f"({effect_label} effect size)."
        )
    else:
        st.info(
            "**Result:** Not statistically significant (p >= 0.05). "
            "No evidence that town means differ."
        )

    # Box plot for visual reference
    town_order = (
        df.groupby("town")["price_per_sqm"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig_anova_town = px.box(
        df,
        x="town",
        y="price_per_sqm",
        color="town",
        category_orders={"town": town_order},
        labels={"price_per_sqm": "Price per sqm ($)", "town": "Town"},
    )
    fig_anova_town.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        margin=dict(t=30, b=120),
        height=500,
    )
    st.plotly_chart(fig_anova_town, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  4. ANOVA — Price by Flat Type
# ══════════════════════════════════════════════════════════════════
with st.expander("4. ANOVA: Price per sqm by Flat Type", expanded=False):
    st.markdown(
        """
        **What it tests:** Same one-way ANOVA as above, but grouping by
        flat type (e.g., 3 ROOM, 4 ROOM, 5 ROOM, EXECUTIVE).

        **Interpretation:** A significant result means at least one flat-type
        group has a different mean price per sqm.
        """
    )

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("F-statistic", f"{f_flat:,.2f}")
    col_e.metric("p-value", f"{p_flat:.2e}" if p_flat < 0.001 else f"{p_flat:.4f}")
    col_f.metric("Eta-squared (effect size)", f"{eta_flat:.4f}")

    if p_flat < 0.05:
        effect_label = (
            "large" if eta_flat >= 0.14
            else "medium" if eta_flat >= 0.06
            else "small"
        )
        st.success(
            f"**Result:** Statistically significant (p < 0.05). Flat type explains "
            f"~{eta_flat * 100:.1f}% of the variance in price per sqm "
            f"({effect_label} effect size)."
        )
    else:
        st.info(
            "**Result:** Not statistically significant (p >= 0.05). "
            "No evidence that flat-type means differ."
        )

    flat_order = (
        df.groupby("flat_type")["price_per_sqm"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig_anova_flat = px.box(
        df,
        x="flat_type",
        y="price_per_sqm",
        color="flat_type",
        category_orders={"flat_type": flat_order},
        labels={"price_per_sqm": "Price per sqm ($)", "flat_type": "Flat Type"},
    )
    fig_anova_flat.update_layout(
        showlegend=False,
        margin=dict(t=30),
        height=450,
    )
    st.plotly_chart(fig_anova_flat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  5. Distribution Normality Tests
# ══════════════════════════════════════════════════════════════════
with st.expander("5. Distribution Normality Tests", expanded=False):
    st.markdown(
        """
        **What it tests:** The Shapiro-Wilk test checks whether a sample
        comes from a normal distribution. The null hypothesis is that the
        data *are* normally distributed; a small p-value rejects normality.

        **Why it matters:** Many parametric tests (t-tests, ANOVA, linear
        regression) assume normally distributed residuals. If the raw variable
        is skewed, a log-transform often helps.

        **Q-Q plot:** Quantile-Quantile plots compare observed quantiles
        against theoretical normal quantiles. Points falling on the diagonal
        indicate normality; systematic curvature indicates skew.
        """
    )

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.markdown("**Raw resale_price**")
        st.metric("Shapiro-Wilk W", f"{norm_results['sw_raw_stat']:.6f}")
        st.metric("p-value", f"{norm_results['sw_raw_p']:.2e}")
        if norm_results["sw_raw_p"] < 0.05:
            st.warning("Rejects normality (p < 0.05).")
        else:
            st.success("Consistent with normality (p >= 0.05).")

    with col_n2:
        st.markdown("**log(resale_price)**")
        st.metric("Shapiro-Wilk W", f"{norm_results['sw_log_stat']:.6f}")
        st.metric("p-value", f"{norm_results['sw_log_p']:.2e}")
        if norm_results["sw_log_p"] < 0.05:
            st.warning("Rejects normality (p < 0.05).")
        else:
            st.success("Consistent with normality (p >= 0.05).")

    # Interpretation
    if norm_results["sw_log_stat"] > norm_results["sw_raw_stat"]:
        st.info(
            "The log-transform **improves** normality "
            f"(W increases from {norm_results['sw_raw_stat']:.6f} to "
            f"{norm_results['sw_log_stat']:.6f}). Consider using "
            "log(resale_price) in parametric models."
        )
    else:
        st.info(
            "The log-transform does **not** noticeably improve normality. "
            "Other transformations (Box-Cox) may be worth exploring."
        )

    # Q-Q Plots
    st.markdown("#### Q-Q Plots")
    qq_col1, qq_col2 = st.columns(2)

    with qq_col1:
        fig_qq_raw = go.Figure()
        fig_qq_raw.add_trace(go.Scatter(
            x=norm_results["theoretical_q"],
            y=norm_results["sorted_raw"],
            mode="markers",
            marker=dict(size=2, color="#2563eb", opacity=0.5),
            name="Observed",
        ))
        # Reference line
        x_range = [norm_results["theoretical_q"].min(), norm_results["theoretical_q"].max()]
        raw_mean = norm_results["sorted_raw"].mean()
        raw_std = norm_results["sorted_raw"].std()
        fig_qq_raw.add_trace(go.Scatter(
            x=x_range,
            y=[raw_mean + raw_std * x for x in x_range],
            mode="lines",
            line=dict(color="#dc2626", dash="dash"),
            name="Reference line",
        ))
        fig_qq_raw.update_layout(
            title="Q-Q Plot — Raw resale_price",
            xaxis_title="Theoretical quantiles",
            yaxis_title="Observed quantiles",
            height=400,
            margin=dict(t=40),
            showlegend=True,
        )
        st.plotly_chart(fig_qq_raw, use_container_width=True)

    with qq_col2:
        fig_qq_log = go.Figure()
        fig_qq_log.add_trace(go.Scatter(
            x=norm_results["theoretical_q"],
            y=norm_results["sorted_log"],
            mode="markers",
            marker=dict(size=2, color="#16a34a", opacity=0.5),
            name="Observed",
        ))
        log_mean = norm_results["sorted_log"].mean()
        log_std = norm_results["sorted_log"].std()
        fig_qq_log.add_trace(go.Scatter(
            x=x_range,
            y=[log_mean + log_std * x for x in x_range],
            mode="lines",
            line=dict(color="#dc2626", dash="dash"),
            name="Reference line",
        ))
        fig_qq_log.update_layout(
            title="Q-Q Plot — log(resale_price)",
            xaxis_title="Theoretical quantiles",
            yaxis_title="Observed quantiles",
            height=400,
            margin=dict(t=40),
            showlegend=True,
        )
        st.plotly_chart(fig_qq_log, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  6. Levene's Test (Variance Homogeneity)
# ══════════════════════════════════════════════════════════════════
with st.expander("6. Levene's Test — Variance Homogeneity across Towns", expanded=False):
    st.markdown(
        """
        **What it tests:** Levene's test checks whether the *variances* of
        price per sqm are equal across all town groups. Homogeneity of
        variance (homoscedasticity) is a key assumption of one-way ANOVA.

        **Null hypothesis:** All groups have equal variance.

        **How to read:** A significant result (p < 0.05) means variances
        differ across towns, which may violate the ANOVA assumption.
        In that case, consider Welch's ANOVA or non-parametric alternatives
        (e.g., Kruskal-Wallis).
        """
    )

    col_l1, col_l2 = st.columns(2)
    col_l1.metric("Levene test statistic", f"{lev_stat:,.2f}")
    col_l2.metric("p-value", f"{lev_p:.2e}" if lev_p < 0.001 else f"{lev_p:.4f}")

    if lev_p < 0.05:
        st.warning(
            "**Result:** Variances are **not** equal across towns (p < 0.05). "
            "The homogeneity-of-variance assumption for ANOVA is violated. "
            "Welch's ANOVA or Kruskal-Wallis would be more appropriate."
        )
    else:
        st.success(
            "**Result:** No evidence of unequal variances (p >= 0.05). "
            "The ANOVA assumption of homoscedasticity is satisfied."
        )

# ══════════════════════════════════════════════════════════════════
#  7. Feature Importance (Random Forest)
# ══════════════════════════════════════════════════════════════════
with st.expander("7. Feature Importance — Random Forest", expanded=True):
    st.markdown(
        f"""
        **What it shows:** A Random Forest regressor (100 trees) trained on a
        sample of up to 50,000 rows to predict `resale_price`. Feature
        importances reflect each variable's contribution to reducing
        prediction error (mean decrease in impurity).

        **Features used:** `{', '.join(FEATURE_COLS)}`, plus label-encoded
        `flat_type` and `town`.

        **Model R-squared on training sample:** {rf_r2:.4f}
        """
    )

    fig_rf = px.bar(
        rf_imp,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Feature Importance", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Blues",
    )
    fig_rf.update_layout(
        title="Random Forest Feature Importances",
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=450,
        margin=dict(t=50, l=10),
    )
    st.plotly_chart(fig_rf, use_container_width=True)

    st.caption(
        "Note: Importances are based on mean decrease in impurity (Gini). "
        "They can overweight high-cardinality features (e.g., encoded town). "
        "Permutation importance provides an alternative perspective."
    )

# ══════════════════════════════════════════════════════════════════
#  8. VIF (Variance Inflation Factor)
# ══════════════════════════════════════════════════════════════════
with st.expander("8. Variance Inflation Factor (VIF)", expanded=False):
    st.markdown(
        """
        **What it tests:** VIF quantifies how much a feature's regression
        coefficient variance is inflated due to collinearity with other
        predictors. It is computed for the numeric predictor features only
        (excluding the target).

        **How to read:**
        - **VIF = 1:** No collinearity.
        - **VIF 1-5:** Moderate, generally acceptable.
        - **VIF 5-10:** Concerning — the feature shares substantial variance
          with others.
        - **VIF > 10:** Severe multicollinearity — consider dropping or
          combining features.
        """
    )

    def vif_flag(v):
        if v > 10:
            return "Severe"
        elif v > 5:
            return "Concerning"
        elif v > 1:
            return "Acceptable"
        return "None"

    vif_display = vif_df.copy()
    vif_display["Interpretation"] = vif_display["VIF"].apply(vif_flag)
    vif_display["VIF"] = vif_display["VIF"].round(2)

    st.dataframe(
        vif_display.style.map(
            lambda v: (
                "background-color: #fecaca" if v == "Severe"
                else "background-color: #fed7aa" if v == "Concerning"
                else ""
            ),
            subset=["Interpretation"],
        ),
        hide_index=True,
        use_container_width=False,
    )

    severe = vif_display[vif_display["Interpretation"] == "Severe"]
    concerning = vif_display[vif_display["Interpretation"] == "Concerning"]

    if len(severe) > 0:
        st.error(
            f"**Severe multicollinearity** detected in: "
            f"{', '.join(severe['Feature'].tolist())}. "
            "Consider removing or combining these features in regression models."
        )
    elif len(concerning) > 0:
        st.warning(
            f"**Moderate multicollinearity** in: "
            f"{', '.join(concerning['Feature'].tolist())}. "
            "Monitor but not necessarily problematic for tree-based models."
        )
    else:
        st.success("All VIF values are below 5 — no multicollinearity concerns.")

# ── footer ───────────────────────────────────────────────────────
st.divider()
st.caption(
    "Statistical Analysis | Page 10 of the HDB Resale EDA suite. "
    "Tests use scipy, statsmodels, and scikit-learn. "
    "Cached computations ensure snappy re-runs."
)
