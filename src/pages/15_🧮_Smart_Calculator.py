"""
Page 15 – Smart Money Parametric Calculator Suite
===================================================
Five rules-based calculators for Singapore HDB decision-making.
No external data required – all logic is hard-coded from HDB/CPF/IRAS rules
as of August 2026.

Tabs:
  1. 🏠 Mortgage Affordability
  2. 🏦 CPF Usage Eligibility
  3. 🎁 Grant Eligibility
  4. 💰 ABSD Calculator
  5. 🏗️ HDB Upgrade Pathway
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Calculator",
    page_icon="🧮",
    layout="wide",
)

DISCLAIMER = (
    "⚠️ **Rules verified as of Aug 2026.** HDB/CPF/IRAS policies change — "
    "always verify at [hdb.gov.sg](https://www.hdb.gov.sg), "
    "[cpf.gov.sg](https://www.cpf.gov.sg), "
    "[iras.gov.sg](https://www.iras.gov.sg) before transacting."
)

st.title("🧮 Smart Money Calculator Suite")
st.caption(
    "Rules-based parametric calculators for Singapore HDB buyers, upgraders, and investors. "
    "No live data — all computations follow published HDB / CPF / IRAS guidelines."
)

# ── helper functions ───────────────────────────────────────────────────────────

def fmt_price(val: float) -> str:
    """Format as $X,XXX."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"${val:,.0f}"


def monthly_payment(principal: float, annual_rate_pct: float, tenure_years: int) -> float:
    """Standard annuity monthly payment formula."""
    if principal <= 0 or tenure_years <= 0:
        return 0.0
    r = annual_rate_pct / 100 / 12
    n = tenure_years * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def compute_bsd(price: float) -> float:
    """Buyer Stamp Duty — Singapore 2024 progressive rates."""
    bsd = 0.0
    brackets = [
        (180_000, 0.01),
        (180_000, 0.02),
        (640_000, 0.03),
        (500_000, 0.04),
        (1_500_000, 0.05),
        (float("inf"), 0.06),
    ]
    remaining = price
    for band, rate in brackets:
        if remaining <= 0:
            break
        taxable = min(remaining, band)
        bsd += taxable * rate
        remaining -= taxable
    return bsd


# ══════════════════════════════════════════════════════════════════════════════
#  Build tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🏠 Mortgage Affordability",
        "🏦 CPF Usage Eligibility",
        "🎁 Grant Eligibility",
        "💰 ABSD Calculator",
        "🏗️ HDB Upgrade Pathway",
        "📋 BTO Payment Breakdown",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – MORTGAGE AFFORDABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏠 Mortgage Affordability Calculator")
    st.info(DISCLAIMER)

    with st.form("form_mortgage"):
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Income & Savings")
            income_self = st.number_input(
                "Your gross monthly income ($)",
                min_value=0, value=5_000, step=100,
                help="Include all sources: salary, allowances, etc."
            )
            income_partner = st.number_input(
                "Partner gross monthly income ($) — optional",
                min_value=0, value=0, step=100,
            )
            cash_savings = st.number_input(
                "Cash savings ($)",
                min_value=0, value=50_000, step=1_000,
            )
            cpf_oa = st.number_input(
                "CPF Ordinary Account balance ($)",
                min_value=0, value=30_000, step=1_000,
            )
            existing_debt = st.number_input(
                "Existing monthly debt obligations ($) — for TDSR",
                min_value=0, value=0, step=100,
                help="Car loans, personal loans, credit card minimums, etc."
            )

        with col_r:
            st.subheader("Loan Parameters")
            youngest_age = st.number_input(
                "Youngest buyer's age",
                min_value=21, max_value=65, value=30, step=1,
            )
            loan_type = st.radio(
                "Loan type",
                options=["HDB Loan", "Bank Loan"],
                horizontal=True,
            )
            bank_rate = st.slider(
                "Bank loan interest rate (% p.a.)",
                min_value=2.0, max_value=5.5, value=3.5, step=0.1,
                disabled=(loan_type == "HDB Loan"),
                help="Only applicable for bank loans.",
            )

        submitted_mortgage = st.form_submit_button("Calculate Affordability", type="primary")

    if submitted_mortgage:
        combined_income = income_self + income_partner

        # ── Loan parameters ──────────────────────────────────────────────────
        if loan_type == "HDB Loan":
            ltv = 0.80
            rate_pct = 2.6
            max_tenure = min(25, 65 - youngest_age)
        else:
            ltv = 0.75
            rate_pct = bank_rate
            max_tenure = min(30, 65 - youngest_age)

        max_tenure = max(max_tenure, 1)  # safety floor

        # ── MSR constraint (HDB loan only) ───────────────────────────────────
        msr_limit_monthly = combined_income * 0.30  # 30% of gross income

        # ── TDSR constraint (all loans) ──────────────────────────────────────
        tdsr_limit_monthly = combined_income * 0.55  # 55% of gross income
        tdsr_headroom = max(tdsr_limit_monthly - existing_debt, 0)

        if loan_type == "HDB Loan":
            # Binding constraint: lower of MSR or TDSR headroom
            max_monthly_payment = min(msr_limit_monthly, tdsr_headroom)
        else:
            max_monthly_payment = tdsr_headroom

        # ── Back-calculate max loan from payment capacity ────────────────────
        r_monthly = rate_pct / 100 / 12
        n_payments = max_tenure * 12

        if r_monthly > 0 and max_monthly_payment > 0:
            # P = PMT × [(1+r)^n - 1] / [r × (1+r)^n]
            max_loan_from_servicing = (
                max_monthly_payment
                * ((1 + r_monthly) ** n_payments - 1)
                / (r_monthly * (1 + r_monthly) ** n_payments)
            )
        elif max_monthly_payment > 0:
            max_loan_from_servicing = max_monthly_payment * n_payments
        else:
            max_loan_from_servicing = 0.0

        # ── Max property price ───────────────────────────────────────────────
        # Method A: from loan LTV  →  price = loan / LTV
        max_price_from_ltv = max_loan_from_servicing / ltv if ltv > 0 else 0

        # Method B: total funds available  →  cash + CPF + loan
        max_price_from_funds = cash_savings + cpf_oa + max_loan_from_servicing

        max_property_price = min(max_price_from_ltv, max_price_from_funds)
        actual_loan = max_property_price * ltv
        actual_monthly = monthly_payment(actual_loan, rate_pct, max_tenure)
        total_interest = actual_monthly * n_payments - actual_loan

        # Down payment split (CPF first, then cash)
        down_payment_total = max_property_price - actual_loan
        down_cpf = min(cpf_oa, down_payment_total)
        down_cash = max(down_payment_total - down_cpf, 0)

        # ── MSR / TDSR utilisation ───────────────────────────────────────────
        msr_utilised = (actual_monthly / combined_income) if combined_income > 0 else 0
        tdsr_utilised = ((actual_monthly + existing_debt) / combined_income) if combined_income > 0 else 0

        st.divider()
        st.subheader("📊 Affordability Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Max Property Price", fmt_price(max_property_price))
        c2.metric("Max Loan Amount", fmt_price(actual_loan))
        c3.metric("Monthly Repayment", f"${actual_monthly:,.0f}/mth")

        st.success(
            f"You can afford a flat up to **{fmt_price(max_property_price)}**. "
            f"Monthly repayment: **${actual_monthly:,.0f}/month** over **{max_tenure} years** "
            f"at **{rate_pct:.1f}% p.a.**"
        )

        st.subheader("💳 Payment Breakdown")
        breakdown_df = pd.DataFrame(
            {
                "Item": [
                    "Purchase Price",
                    "Down Payment (CPF OA)",
                    "Down Payment (Cash)",
                    "Loan Amount",
                    "Monthly Payment",
                    "Loan Tenure",
                    "Total Interest Paid",
                    "Total Amount Paid",
                ],
                "Amount": [
                    fmt_price(max_property_price),
                    fmt_price(down_cpf),
                    fmt_price(down_cash),
                    fmt_price(actual_loan),
                    f"${actual_monthly:,.0f}/mth",
                    f"{max_tenure} years",
                    fmt_price(total_interest),
                    fmt_price(actual_monthly * n_payments + down_cpf + down_cash),
                ],
            }
        )
        st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

        st.subheader("📏 MSR / TDSR Utilisation")

        col_msr, col_tdsr = st.columns(2)
        with col_msr:
            msr_pct_display = min(msr_utilised, 1.0)
            st.markdown(
                f"**MSR** (limit 30%) — you use **{msr_utilised*100:.1f}%** of income"
            )
            st.progress(msr_pct_display)
            if loan_type != "HDB Loan":
                st.caption("MSR applies to HDB loans only — shown for reference.")

        with col_tdsr:
            tdsr_pct_display = min(tdsr_utilised, 1.0)
            st.markdown(
                f"**TDSR** (limit 55%) — you use **{tdsr_utilised*100:.1f}%** of income"
            )
            st.progress(tdsr_pct_display)

        st.caption(
            "ℹ️ This is a simplified estimate. Actual eligibility depends on HDB/bank "
            "assessment, credit score, existing property ownership, and other factors."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – CPF USAGE ELIGIBILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🏦 CPF Usage Eligibility for HDB Flats")
    st.info(DISCLAIMER)

    with st.form("form_cpf"):
        col_l, col_r = st.columns(2)

        with col_l:
            cpf_age = st.slider(
                "Youngest buyer's age",
                min_value=21, max_value=65, value=30, step=1,
            )
            remaining_lease = st.slider(
                "Flat's remaining lease (years)",
                min_value=1, max_value=99, value=75, step=1,
            )

        with col_r:
            purchase_price_cpf = st.number_input(
                "Property purchase price ($)",
                min_value=0, value=500_000, step=10_000,
            )

        submitted_cpf = st.form_submit_button("Check CPF Eligibility", type="primary")

    if submitted_cpf:
        coverage_age = cpf_age + remaining_lease  # age when lease expires

        # ── CPF rules ────────────────────────────────────────────────────────
        if remaining_lease >= 20 and coverage_age >= 95:
            cpf_status = "full"
            cpf_limit = purchase_price_cpf  # up to Valuation Limit = purchase price
        elif coverage_age >= 80 and remaining_lease >= 20:
            cpf_status = "prorated"
            prorated_fraction = (95 - cpf_age) / remaining_lease
            cpf_limit = prorated_fraction * purchase_price_cpf
        else:
            cpf_status = "none"
            cpf_limit = 0.0

        # ── HDB loan eligibility ─────────────────────────────────────────────
        hdb_loan_ok = remaining_lease >= 20 and coverage_age >= 95

        st.divider()
        st.subheader("📊 CPF & Loan Eligibility Results")

        if cpf_status == "full":
            st.success(f"✅ **Full CPF Usage** — you may use CPF OA up to {fmt_price(cpf_limit)} (Valuation Limit).")
        elif cpf_status == "prorated":
            st.warning(
                f"⚠️ **Prorated CPF Usage** — CPF OA capped at **{fmt_price(cpf_limit)}** "
                f"({(cpf_limit/purchase_price_cpf*100):.1f}% of purchase price).\n\n"
                f"Formula: (95 − {cpf_age}) / {remaining_lease} × {fmt_price(purchase_price_cpf)}"
            )
        else:
            st.error(
                "❌ **No CPF Usage** — remaining lease is too short "
                f"(buyer age {cpf_age} + remaining lease {remaining_lease} = {coverage_age} < 80, "
                f"or remaining lease < 20 years)."
            )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Remaining Lease", f"{remaining_lease} yrs")
        col_b.metric("Buyer Age at Lease Expiry", f"{coverage_age}")
        col_c.metric("CPF Usage Limit", fmt_price(cpf_limit))

        if hdb_loan_ok:
            st.success("✅ **HDB Loan Eligible** — remaining lease covers youngest buyer to age 95.")
        else:
            st.error(
                f"❌ **HDB Loan Not Eligible** — requires coverage to age 95; "
                f"your flat covers to age {coverage_age}."
            )

        # ── Timeline visualisation ───────────────────────────────────────────
        st.subheader("📅 Lease & Age Timeline")

        fig = go.Figure()

        # Background bar — buyer lifespan to 95
        fig.add_trace(
            go.Bar(
                x=[95 - cpf_age],
                y=["Timeline"],
                base=[cpf_age],
                orientation="h",
                marker_color="lightblue",
                name="Buyer span to 95",
                hovertemplate="Age %{base} → 95<extra></extra>",
            )
        )

        # Lease bar
        fig.add_trace(
            go.Bar(
                x=[remaining_lease],
                y=["Timeline"],
                base=[cpf_age],
                orientation="h",
                marker_color="steelblue" if cpf_status == "full" else
                             "orange" if cpf_status == "prorated" else "salmon",
                name=f"Flat lease ({remaining_lease} yrs)",
                hovertemplate=f"Lease covers age {cpf_age} → {coverage_age}<extra></extra>",
            )
        )

        # Markers
        for x_val, label, color in [
            (cpf_age, f"Current age ({cpf_age})", "navy"),
            (coverage_age, f"Lease expiry (age {coverage_age})", "red"),
            (95, "Age 95 threshold", "green"),
        ]:
            fig.add_vline(x=x_val, line_color=color, line_dash="dash", annotation_text=label,
                          annotation_position="top")

        fig.update_layout(
            barmode="overlay",
            xaxis_title="Age",
            yaxis_showticklabels=False,
            height=200,
            legend=dict(orientation="h", y=-0.3),
            margin=dict(l=10, r=10, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "ℹ️ Bank loans have their own lease requirements — typically they require the flat "
            "lease to cover the loan tenure and youngest borrower until age 65."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – GRANT ELIGIBILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🎁 HDB Grant Eligibility Calculator")
    st.info(DISCLAIMER)

    with st.form("form_grants"):
        col_l, col_r = st.columns(2)

        with col_l:
            buyer_status = st.radio(
                "Buyer status",
                options=["First-timer", "Second-timer"],
                horizontal=True,
            )
            citizenship = st.selectbox(
                "Citizenship profile",
                options=["SC/SC couple", "SC/PR couple", "Single SC", "Foreigner"],
            )
            combined_income_grant = st.number_input(
                "Combined monthly household income ($)",
                min_value=0, value=6_000, step=100,
            )

        with col_r:
            buy_mode = st.radio(
                "Buying mode",
                options=["BTO", "Resale"],
                horizontal=True,
            )
            proximity = st.selectbox(
                "Proximity to parents / child",
                options=["None", "Same town", "Within 4 km", "Same block / development"],
            )
            in_2rm_rental = st.checkbox(
                "Currently living in 2-room HDB rental flat",
                value=False,
            )
            grant_property_price = st.number_input(
                "Indicative property price ($) — for effective net price",
                min_value=0, value=500_000, step=10_000,
            )

        submitted_grants = st.form_submit_button("Check Grant Eligibility", type="primary")

    if submitted_grants:
        is_first = buyer_status == "First-timer"
        is_resale = buy_mode == "Resale"
        is_sc_sc = citizenship == "SC/SC couple"
        is_sc_pr = citizenship == "SC/PR couple"
        is_single = citizenship == "Single SC"
        is_foreigner = citizenship == "Foreigner"

        grants = []  # list of (name, eligible: bool, amount: float, note: str)

        # ── Enhanced Housing Grant (EHG) ─────────────────────────────────────
        # Simplified linear scale: $80k at ≤$1,500 income, reduces by $5k per $500,
        # floor $5k at $9,000, zero above $9,000.
        if is_first and not is_foreigner:
            if combined_income_grant <= 9_000:
                # tiers in $500 steps from 1500 to 9000
                steps_above_1500 = max(0, combined_income_grant - 1_500)
                tier = math.floor(steps_above_1500 / 500)
                ehg_amount = max(80_000 - tier * 5_000, 5_000)
            else:
                ehg_amount = 0

            if ehg_amount > 0:
                grants.append((
                    "Enhanced Housing Grant (EHG)",
                    True,
                    ehg_amount,
                    "First-timer SC. Must stay ≥ 5 years.",
                ))
            else:
                grants.append((
                    "Enhanced Housing Grant (EHG)",
                    False,
                    0,
                    "Income exceeds $9,000 ceiling.",
                ))
        else:
            grants.append((
                "Enhanced Housing Grant (EHG)",
                False,
                0,
                "Only for first-timer SC/PR buyers. Not applicable.",
            ))

        # ── Family Grant (Resale only) ────────────────────────────────────────
        if is_first and is_resale and (is_sc_sc or is_sc_pr):
            fg_amount = 50_000 if is_sc_sc else 40_000
            grants.append((
                "Family Grant",
                True,
                fg_amount,
                f"{'SC+SC' if is_sc_sc else 'SC+PR'} buying first resale flat.",
            ))
        else:
            grants.append((
                "Family Grant",
                False,
                0,
                "Only for SC/SC or SC/PR couples buying first resale flat.",
            ))

        # ── Proximity Housing Grant (PHG) ─────────────────────────────────────
        if is_resale and proximity != "None":
            if proximity in ("Same block / development", "Within 4 km"):
                phg_amount = 30_000
            else:  # Same town
                phg_amount = 20_000
            grants.append((
                "Proximity Housing Grant (PHG)",
                True,
                phg_amount,
                f"Living {proximity.lower()} of parents/child — resale.",
            ))
        else:
            grants.append((
                "Proximity Housing Grant (PHG)",
                False,
                0,
                "Only for resale purchases and if within proximity of parents/child.",
            ))

        # ── Singles Grant ─────────────────────────────────────────────────────
        if is_first and is_single and is_resale:
            grants.append((
                "Singles Grant",
                True,
                25_000,
                "Eligible single SC buying resale (2-room or larger).",
            ))
        else:
            grants.append((
                "Singles Grant",
                False,
                0,
                "Only for single SC buying resale HDB (first-timer).",
            ))

        # ── Step-Up CPF Housing Grant ─────────────────────────────────────────
        if not is_first and in_2rm_rental and (is_sc_sc or is_sc_pr or is_single):
            grants.append((
                "Step-Up CPF Housing Grant",
                True,
                15_000,
                "Second-timer currently in 2-room HDB rental.",
            ))
        else:
            grants.append((
                "Step-Up CPF Housing Grant",
                False,
                0,
                "Only for second-timers currently in 2-room rental flat.",
            ))

        # ── Results ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Grant Summary")

        grant_df = pd.DataFrame(
            grants, columns=["Grant", "Eligible?", "Amount ($)", "Notes"]
        )
        grant_df["Eligible?"] = grant_df["Eligible?"].map(
            {True: "✅ Yes", False: "❌ No"}
        )
        grant_df["Amount ($)"] = grant_df["Amount ($)"].apply(
            lambda x: fmt_price(x) if x > 0 else "—"
        )
        st.dataframe(grant_df, hide_index=True, use_container_width=True)

        total_grants = sum(g[2] for g in grants if g[1])
        effective_price = max(grant_property_price - total_grants, 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Grants", fmt_price(total_grants))
        c2.metric("Property Price", fmt_price(grant_property_price))
        c3.metric("Effective Net Price", fmt_price(effective_price))

        st.caption(
            "ℹ️ Grant eligibility is indicative. Final eligibility determined by HDB during "
            "application. Some grants cannot be combined (e.g. EHG + Step-Up)."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – ABSD CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("💰 ABSD (Additional Buyer Stamp Duty) Calculator")
    st.info(DISCLAIMER)

    with st.form("form_absd"):
        col_l, col_r = st.columns(2)

        with col_l:
            buyer_type_absd = st.selectbox(
                "Buyer type",
                options=["Singapore Citizen (SC)", "Singapore PR", "Foreigner", "Entity"],
            )
            existing_properties = st.number_input(
                "Number of existing properties owned",
                min_value=0, max_value=10, value=0, step=1,
                help="0 = buying first property, 1 = buying second, etc.",
            )

        with col_r:
            purchase_price_absd = st.number_input(
                "Property purchase price ($)",
                min_value=0, value=1_000_000, step=10_000,
            )
            joint_with_sc_spouse = st.checkbox(
                "Jointly purchased with SC spouse? (potential ABSD remission)",
                value=False,
                help="Under certain conditions, SC/PR couples may claim ABSD remission — verify with IRAS.",
            )

        submitted_absd = st.form_submit_button("Calculate ABSD & Stamp Duties", type="primary")

    if submitted_absd:
        # ── ABSD rates ────────────────────────────────────────────────────────
        prop_count = existing_properties + 1  # which property this purchase represents

        if buyer_type_absd == "Singapore Citizen (SC)":
            if prop_count == 1:
                absd_rate = 0.00
            elif prop_count == 2:
                absd_rate = 0.20
            else:
                absd_rate = 0.30
        elif buyer_type_absd == "Singapore PR":
            if prop_count == 1:
                absd_rate = 0.05
            elif prop_count == 2:
                absd_rate = 0.30
            else:
                absd_rate = 0.35
        elif buyer_type_absd == "Foreigner":
            absd_rate = 0.60
        else:  # Entity
            absd_rate = 0.65

        absd_amount = purchase_price_absd * absd_rate
        bsd_amount = compute_bsd(purchase_price_absd)
        total_cost = purchase_price_absd + bsd_amount + absd_amount

        st.divider()
        st.subheader("📊 Stamp Duty Breakdown")

        c1, c2, c3 = st.columns(3)
        c1.metric("ABSD Rate", f"{absd_rate*100:.0f}%")
        c2.metric("ABSD Amount", fmt_price(absd_amount))
        c3.metric("Total Acquisition Cost", fmt_price(total_cost))

        breakdown_absd_df = pd.DataFrame(
            {
                "Item": [
                    "Purchase Price",
                    f"Buyer Stamp Duty (BSD)",
                    f"Additional Buyer Stamp Duty (ABSD @ {absd_rate*100:.0f}%)",
                    "Total Acquisition Cost",
                ],
                "Amount": [
                    fmt_price(purchase_price_absd),
                    fmt_price(bsd_amount),
                    fmt_price(absd_amount),
                    fmt_price(total_cost),
                ],
            }
        )
        st.dataframe(breakdown_absd_df, hide_index=True, use_container_width=True)

        # BSD tiers breakdown
        with st.expander("BSD Progressive Tier Breakdown"):
            bsd_tiers = []
            brackets = [
                ("First $180,000", 180_000, 0.01),
                ("Next $180,000", 180_000, 0.02),
                ("Next $640,000", 640_000, 0.03),
                ("Next $500,000", 500_000, 0.04),
                ("Next $1,500,000", 1_500_000, 0.05),
                ("Remainder", float("inf"), 0.06),
            ]
            remaining = purchase_price_absd
            for label, band, rate in brackets:
                if remaining <= 0:
                    break
                taxable = min(remaining, band)
                bsd_tiers.append({
                    "Band": label,
                    "Rate": f"{rate*100:.0f}%",
                    "Taxable Amount": fmt_price(taxable),
                    "BSD on Band": fmt_price(taxable * rate),
                })
                remaining -= taxable
            st.dataframe(pd.DataFrame(bsd_tiers), hide_index=True, use_container_width=True)

        if joint_with_sc_spouse:
            st.info(
                "💡 **Joint purchase with SC spouse:** If you are a PR or foreigner buying "
                "jointly with an SC spouse, you may apply for ABSD remission under specific "
                "conditions. The property must be for owner occupation and you must sell any "
                "existing properties within 6 months. Verify with IRAS."
            )

        if existing_properties >= 1 and "Citizen" in buyer_type_absd:
            st.warning(
                "⏳ **15-month Wait-Out Period:** Singapore Citizens who own private residential "
                "property must wait 15 months after selling it before they can purchase an HDB "
                "resale flat (rule introduced Sep 2022). This does not apply to buying private."
            )

        if absd_rate == 0:
            st.success("🎉 No ABSD payable — you are an SC buying your first property!")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – HDB UPGRADE PATHWAY
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🏗️ HDB → Private Condo Upgrade Pathway")
    st.info(DISCLAIMER)

    with st.form("form_upgrade"):
        st.subheader("Current HDB Flat Details")
        col_l, col_r = st.columns(2)

        with col_l:
            flat_market_value = st.number_input(
                "Current flat estimated market value ($)",
                min_value=0, value=600_000, step=10_000,
            )
            outstanding_loan = st.number_input(
                "Outstanding HDB loan balance ($)",
                min_value=0, value=100_000, step=5_000,
            )
            cpf_used_flat = st.number_input(
                "CPF OA used for current flat ($)",
                min_value=0, value=150_000, step=5_000,
            )

        with col_r:
            years_held = st.slider(
                "Years held",
                min_value=1, max_value=30, value=10, step=1,
            )
            cpf_accrual_rate = st.slider(
                "CPF accrued interest rate (% p.a.)",
                min_value=2.5, max_value=3.5, value=2.5, step=0.05,
                help="CPF OA rate; currently 2.5% p.a. (floor rate).",
            )
            external_cash = st.number_input(
                "Cash savings outside CPF ($)",
                min_value=0, value=100_000, step=10_000,
            )

        st.subheader("Target Private Condo")
        col_c, col_d = st.columns(2)

        with col_c:
            condo_price = st.number_input(
                "Target condo price ($)",
                min_value=0, value=1_500_000, step=50_000,
            )
            upgrader_age = st.number_input(
                "Your age (youngest buyer)",
                min_value=21, max_value=65, value=38, step=1,
            )
            condo_loan_rate = st.slider(
                "Bank loan interest rate for condo (% p.a.)",
                min_value=2.0, max_value=5.5, value=3.5, step=0.1,
            )

        with col_d:
            buyer_type_upgrade = st.selectbox(
                "Buyer citizenship",
                options=["Singapore Citizen (SC)", "Singapore PR", "Foreigner"],
                key="upgrade_buyer_type",
            )
            sell_first = st.checkbox(
                "Plan to sell current flat before buying condo?",
                value=False,
                help="If YES, ABSD is based on owning 0 properties (after sale). "
                     "If NO, you still own current flat when buying condo.",
            )

        submitted_upgrade = st.form_submit_button("Model Upgrade Scenario", type="primary")

    if submitted_upgrade:
        # ── Step 1: CPF accrued interest ──────────────────────────────────────
        cpf_accrued_interest = cpf_used_flat * (
            (1 + cpf_accrual_rate / 100) ** years_held - 1
        )
        net_cpf_refund = cpf_used_flat + cpf_accrued_interest  # back to CPF OA after sale

        # ── Step 2: Agent commission on flat sale ─────────────────────────────
        agent_commission_sale = flat_market_value * 0.02

        # ── Step 3: Net cash from flat sale ───────────────────────────────────
        net_cash_from_sale = (
            flat_market_value
            - outstanding_loan
            - net_cpf_refund
            - agent_commission_sale
        )
        net_cash_from_sale = max(net_cash_from_sale, 0)  # cannot go negative in practice

        # ── Step 4: Total resources available ────────────────────────────────
        # After selling flat:  cash-in-hand + CPF refund back to OA + external cash
        total_resources = net_cash_from_sale + net_cpf_refund + external_cash

        # ── Step 5: ABSD on condo ────────────────────────────────────────────
        if sell_first:
            # Already sold flat → buying with 0 existing properties
            existing_at_condo_buy = 0
        else:
            # Still owns flat → this is 2nd property
            existing_at_condo_buy = 1

        condo_prop_count = existing_at_condo_buy + 1

        if buyer_type_upgrade == "Singapore Citizen (SC)":
            if condo_prop_count == 1:
                condo_absd_rate = 0.00
            elif condo_prop_count == 2:
                condo_absd_rate = 0.20
            else:
                condo_absd_rate = 0.30
        elif buyer_type_upgrade == "Singapore PR":
            if condo_prop_count == 1:
                condo_absd_rate = 0.05
            elif condo_prop_count == 2:
                condo_absd_rate = 0.30
            else:
                condo_absd_rate = 0.35
        else:
            condo_absd_rate = 0.60

        condo_absd = condo_price * condo_absd_rate
        condo_bsd = compute_bsd(condo_price)
        agent_condo = condo_price * 0.01  # ~1% buyer agent (optional estimate)

        # ── Step 6: Condo financing ───────────────────────────────────────────
        condo_ltv = 0.75
        condo_down_payment = condo_price * (1 - condo_ltv)  # 25%
        condo_loan_amount = condo_price * condo_ltv
        condo_tenure = max(min(30, 65 - upgrader_age), 1)
        condo_monthly = monthly_payment(condo_loan_amount, condo_loan_rate, condo_tenure)
        condo_total_interest = condo_monthly * condo_tenure * 12 - condo_loan_amount

        # ── Step 7: Upfront costs & shortfall ────────────────────────────────
        upfront_needed = condo_down_payment + condo_absd + condo_bsd + agent_condo
        surplus_or_shortfall = total_resources - upfront_needed

        # ── Display ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Upgrade Scenario — Step-by-Step Breakdown")

        st.markdown("**Flat Sale Proceeds**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Flat Market Value", fmt_price(flat_market_value))
        col2.metric("Less: Outstanding Loan", fmt_price(outstanding_loan))
        col3.metric("Less: CPF Refund (incl. accrued interest)", fmt_price(net_cpf_refund))
        col4.metric("Less: Agent Commission (2%)", fmt_price(agent_commission_sale))

        st.markdown("**Net Cash & Resources After Sale**")
        col5, col6, col7 = st.columns(3)
        col5.metric("Net Cash from Flat Sale", fmt_price(net_cash_from_sale))
        col6.metric("CPF OA Refund (reusable)", fmt_price(net_cpf_refund))
        col7.metric("External Cash Savings", fmt_price(external_cash))

        st.markdown("**Condo Purchase Costs**")
        col8, col9, col10, col11 = st.columns(4)
        col8.metric("Down Payment (25%)", fmt_price(condo_down_payment))
        col9.metric(f"ABSD ({condo_absd_rate*100:.0f}%)", fmt_price(condo_absd))
        col10.metric("BSD", fmt_price(condo_bsd))
        col11.metric("Est. Agent (1%)", fmt_price(agent_condo))

        st.markdown("**Condo Financing**")
        col12, col13, col14 = st.columns(3)
        col12.metric("Loan Amount (75% LTV)", fmt_price(condo_loan_amount))
        col13.metric("Monthly Mortgage", f"${condo_monthly:,.0f}/mth")
        col14.metric(f"Tenure ({condo_tenure} yrs)", f"@ {condo_loan_rate:.1f}% p.a.")

        st.divider()
        st.subheader("🏁 Upgrade Verdict")

        if surplus_or_shortfall >= 0:
            st.success(
                f"✅ **You have a surplus of {fmt_price(surplus_or_shortfall)}** to complete "
                f"this upgrade after covering down payment, ABSD, BSD, and agent fees."
            )
        else:
            st.error(
                f"❌ **You need an additional {fmt_price(abs(surplus_or_shortfall))}** to "
                f"complete this upgrade. Consider increasing savings, reducing target price, "
                f"or choosing a smaller condo."
            )

        if not sell_first and condo_absd > 0:
            st.warning(
                f"⏳ **Timeline tip:** If you sell your flat *before* buying the condo, "
                f"you could avoid ABSD of **{fmt_price(condo_absd)}** (you would be buying "
                f"as if owning 0 properties). This requires temporary accommodation between "
                f"sale and purchase. The 15-month wait-out period for private property does "
                f"not apply when you are selling an HDB to buy private."
            )

        # ── Waterfall chart ───────────────────────────────────────────────────
        st.subheader("📉 Cash Flow Waterfall")

        wf_labels = [
            "Flat Sale",
            "− Outstanding Loan",
            "− CPF Refund",
            "− Sale Commission",
            "+ External Cash",
            "− Condo Down Pmt",
            "− ABSD",
            "− BSD",
            "− Condo Agent",
            "Net Position",
        ]
        wf_values = [
            flat_market_value,
            -outstanding_loan,
            -net_cpf_refund,
            -agent_commission_sale,
            external_cash,
            -condo_down_payment,
            -condo_absd,
            -condo_bsd,
            -agent_condo,
        ]
        # Cumulative for waterfall
        running = 0
        measures = []
        y_vals = []
        for v in wf_values:
            measures.append("relative")
            y_vals.append(v)
            running += v
        measures.append("total")
        y_vals.append(running)

        fig_wf = go.Figure(
            go.Waterfall(
                name="Cash Flow",
                orientation="v",
                measure=measures,
                x=wf_labels,
                y=y_vals,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#2ecc71"}},
                decreasing={"marker": {"color": "#e74c3c"}},
                totals={"marker": {"color": "#3498db"}},
            )
        )
        fig_wf.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="$ SGD",
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        st.caption(
            "ℹ️ This is a simplified estimate. Actual cash flow depends on exact CPF account "
            "balance, bank valuation, loan approval, and timing of sale vs purchase. "
            "CPF accrued interest is compounded annually at the OA floor rate."
        )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — BTO PAYMENT BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("📋 BTO Payment Breakdown")
    st.info(DISCLAIMER)
    st.write(
        "See how a BTO purchase is paid across its stages — from booking fee to "
        "monthly loan instalments. This is a guide, not financial advice."
    )

    pb_c1, pb_c2 = st.columns(2)
    with pb_c1:
        pb_price = st.number_input(
            "Flat price ($)", 100_000, 1_500_000, 450_000, step=10_000, key="pb_price"
        )
        pb_flat_type = st.selectbox(
            "Flat type (sets the option fee)",
            ["2-room Flexi", "3-room", "4-room", "5-room / 3Gen / Executive"],
            index=2, key="pb_flat_type",
        )
        pb_loan_type = st.radio(
            "Loan type", ["HDB loan (75%)", "Bank loan (75%)"], key="pb_loan_type"
        )
    with pb_c2:
        pb_staggered = st.checkbox(
            "Use Staggered Downpayment Scheme (eligible first-timers)", key="pb_staggered"
        )
        pb_tenure = st.slider("Loan tenure (years)", 10, 25, 25, key="pb_tenure")
        pb_rate = st.slider(
            "Interest rate (% p.a.)", 2.0, 4.5,
            2.6 if "HDB" in pb_loan_type else 3.5, step=0.1, key="pb_rate"
        )

    pb_has_ehg = st.checkbox("I qualify for the Enhanced CPF Housing Grant (EHG)", key="pb_ehg")
    pb_grant = 0
    if pb_has_ehg:
        pb_g1, pb_g2 = st.columns(2)
        with pb_g1:
            pb_household = st.radio(
                "Household type", ["Family", "Single"], horizontal=True, key="pb_household"
            )
        pb_max_grant = 120_000 if pb_household == "Family" else 60_000
        with pb_g2:
            pb_grant = st.number_input(
                "Your EHG amount ($)", min_value=0, max_value=pb_max_grant,
                value=0, step=500, key="pb_grant_amt",
            )

    # Option fees per flat type
    pb_option_fees = {
        "2-room Flexi": 500, "3-room": 1000,
        "4-room": 2000, "5-room / 3Gen / Executive": 2000,
    }
    pb_option_fee = pb_option_fees[pb_flat_type]

    # Downpayment split
    if pb_staggered:
        pb_signing_pct, pb_keys_pct = 0.05, 0.20
    else:
        pb_signing_pct, pb_keys_pct = 0.10, 0.15

    pb_signing_dp = pb_price * pb_signing_pct
    pb_keys_dp = pb_price * pb_keys_pct
    pb_loan_amount = max(pb_price * 0.75 - pb_grant, 0)

    pb_bsd = compute_bsd(pb_price)
    pb_mortgage_stamp = min(round(pb_loan_amount * 0.004), 500)
    pb_survey_fees = {
        "2-room Flexi": 163.50, "3-room": 218.30,
        "4-room": 272.85, "5-room / 3Gen / Executive": 354.25,
    }
    pb_survey_fee = round(pb_survey_fees[pb_flat_type])
    pb_legal_fee = 500
    pb_admin_fees = 90
    pb_total_fees = pb_bsd + pb_mortgage_stamp + pb_survey_fee + pb_legal_fee + pb_admin_fees

    pb_r = pb_rate / 100 / 12
    pb_n = pb_tenure * 12
    pb_monthly = pb_loan_amount * pb_r / (1 - (1 + pb_r) ** -pb_n) if pb_r > 0 else pb_loan_amount / pb_n
    pb_min_cash = pb_price * 0.05 if "Bank" in pb_loan_type else 0

    st.divider()
    st.markdown("#### Your payment timeline")
    pb_stages = pd.DataFrame([
        {"Stage": "1. At booking",
         "What you pay": "Option fee",
         "Amount": pb_option_fee,
         "When": "When you select your flat"},
        {"Stage": "2. Signing Agreement for Lease",
         "What you pay": f"Downpayment ({int(pb_signing_pct*100)}%) − option fee + fees & stamp duties",
         "Amount": pb_signing_dp - pb_option_fee + pb_total_fees,
         "When": "~4–6 months after booking"},
        {"Stage": "3. Key collection",
         "What you pay": f"Remaining downpayment ({int(pb_keys_pct*100)}%)",
         "Amount": pb_keys_dp,
         "When": "~2.5–4 years later (after construction)"},
        {"Stage": "4. After key collection",
         "What you pay": f"Loan instalments (${pb_loan_amount:,.0f} financed over {pb_tenure} yrs)",
         "Amount": pb_monthly,
         "When": "Monthly, until loan is repaid"},
    ])
    pb_disp = pb_stages.copy()
    pb_disp["Amount"] = pb_disp["Amount"].apply(lambda x: f"${x:,.0f}")
    pb_disp.loc[3, "Amount"] += " / month"
    st.table(pb_disp.set_index("Stage"))

    st.markdown("#### Fees & stamp duties (due at signing)")
    pb_fees = pd.DataFrame([
        {"Item": "Buyer's Stamp Duty (BSD)",                "Amount": pb_bsd},
        {"Item": "Mortgage stamp duty (0.4% of loan, max $500)", "Amount": pb_mortgage_stamp},
        {"Item": "Survey fee",                              "Amount": pb_survey_fee},
        {"Item": "Conveyancing / legal fee (approx.)",      "Amount": pb_legal_fee},
        {"Item": "Caveat & title admin fees",               "Amount": pb_admin_fees},
        {"Item": "Total fees & stamp duties",               "Amount": pb_total_fees},
    ])
    pb_fees_disp = pb_fees.copy()
    pb_fees_disp["Amount"] = pb_fees_disp["Amount"].apply(lambda x: f"${x:,.0f}")
    st.table(pb_fees_disp.set_index("Item"))

    pb_upfront = pb_option_fee + (pb_signing_dp - pb_option_fee + pb_total_fees) + pb_keys_dp
    pm1, pm2, pm3 = st.columns(3)
    pm1.metric("Total downpayment (25%)", f"${pb_price * 0.25:,.0f}")
    pm2.metric("Total upfront (incl. fees)", f"${pb_upfront:,.0f}")
    pm3.metric("Monthly instalment", f"${pb_monthly:,.0f}")

    if pb_grant > 0:
        st.success(
            f"🎁 EHG of **${pb_grant:,.0f}** applied — it reduces the amount financed "
            f"to **${pb_loan_amount:,.0f}**, lowering your monthly instalment."
        )
    if pb_min_cash > 0:
        st.info(
            f"💵 Bank loan: at least **${pb_min_cash:,.0f}** (5% of price) must be paid in cash."
        )
    else:
        st.info("💡 HDB loan: the full 25% downpayment can be paid with CPF OA, cash, or a mix.")

    st.caption(
        "Estimates only. BSD uses standard residential rates; legal/survey fees are approximate. "
        "Always verify with HDB and your bank."
    )
