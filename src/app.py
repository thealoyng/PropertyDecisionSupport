"""
BTO Buddy — Phase 3: Streamlit App
===================================
The web app, with four tabs:
  1. Trends dashboard  — historical price/volume charts
  2. Location map      — BTO projects colour-coded by classification
  3. Payment breakdown — staged BTO payment timeline + grants
  4. Forecast          — Prophet time-series price projection

Run locally:
    pip install streamlit pandas joblib plotly folium streamlit-folium
    streamlit run app.py

Expected files (adjust paths if your layout differs):
    models/price_model.joblib
    data/resale_clean_1990_present.csv
"""
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------------
# Config + data loading (cached so it only loads once per session)
# ---------------------------------------------------------------
st.set_page_config(page_title="BTO Buddy", page_icon="🏠", layout="wide")

DATA_PATH = "data/resale_clean_1990_present.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["month"])
    return df


df = load_data()

# Derive dropdown options from the data so they always match what the
# model was trained on (avoids "unknown category" surprises).
TOWNS = sorted(df["town"].dropna().unique())
FLAT_TYPES = sorted(df["flat_type"].dropna().unique())

st.title("🏠 BTO Buddy")
st.caption("Make sense of the HDB resale market — prices, trends, locations, and affordability.")

tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Trends", "🗺️ Location map", "🧮 Payment breakdown", "🔮 Forecast"]
)

# ===============================================================
# TAB 2 — TRENDS DASHBOARD
# ===============================================================
with tab2:
    st.subheader("Resale price trends")

    sel_towns = st.multiselect("Towns to compare", TOWNS,
                               default=[t for t in ["PUNGGOL", "BISHAN", "WOODLANDS"] if t in TOWNS])
    sel_type = st.selectbox("Flat type", FLAT_TYPES,
                            index=FLAT_TYPES.index("4 ROOM") if "4 ROOM" in FLAT_TYPES else 0,
                            key="trend_type")

    if sel_towns:
        sub = df[(df["town"].isin(sel_towns)) & (df["flat_type"] == sel_type)].copy()
        sub["quarter"] = sub["month"].dt.to_period("Q").dt.to_timestamp()
        trend = sub.groupby(["quarter", "town"])["resale_price"].median().reset_index()
        fig = px.line(trend, x="quarter", y="resale_price", color="town",
                      labels={"resale_price": "Median price ($)", "quarter": ""},
                      title=f"Median {sel_type} price over time")
        st.plotly_chart(fig, use_container_width=True)

    # Volume by year
    vol = df.groupby(df["month"].dt.year).size().reset_index(name="transactions")
    vol.columns = ["year", "transactions"]
    fig2 = px.bar(vol, x="year", y="transactions", title="Resale transactions per year")
    st.plotly_chart(fig2, use_container_width=True)

# ===============================================================
# TAB 3 — LOCATION MAP (BTO classification)
# ===============================================================
with tab3:
    st.subheader("BTO projects by classification")
    st.write("Each pin is an individual project. Classification is assigned per project, "
             "not per town — a single town can contain Standard, Plus, and Prime projects.")

    # Load real projects from data/bto_projects.csv (regenerate coords with
    # src/geocode_bto.py). Falls back to a tiny inline set if the file is absent.
    # The _mtime arg busts the cache automatically whenever the CSV is edited,
    # so you don't need to manually clear the cache after updating the data.
    @st.cache_data
    def load_projects(_mtime):
        try:
            p = pd.read_csv("data/bto_projects.csv")
            p = p.rename(columns={"classification": "cls"})
            return p
        except Exception:
            return pd.DataFrame([
                {"name": "Berlayar Residences", "town": "Bukit Merah",
                 "lat": 1.2735, "lon": 103.8095, "cls": "Prime", "status": "Launched"},
                {"name": "Woodlands North Coast", "town": "Woodlands",
                 "lat": 1.4480, "lon": 103.7860, "cls": "Standard", "status": "Upcoming"},
            ])

    try:
        _csv_mtime = os.path.getmtime("data/bto_projects.csv")
    except OSError:
        _csv_mtime = 0
    bto_projects = load_projects(_csv_mtime)
    colors = {"Standard": "green", "Plus": "orange", "Prime": "red"}

    # Future developments from URA Draft Master Plan 2025 (separate layer).
    @st.cache_data
    def load_future(_mtime):
        try:
            return pd.read_csv("data/future_developments.csv")
        except Exception:
            return pd.DataFrame()

    try:
        _fut_mtime = os.path.getmtime("data/future_developments.csv")
    except OSError:
        _fut_mtime = 0
    future = load_future(_fut_mtime)

    # MRT stations layer
    @st.cache_data
    def load_mrt(_mtime):
        try:
            return pd.read_csv("data/mrt_stations.csv")
        except Exception:
            return pd.DataFrame()

    try:
        _mrt_mtime = os.path.getmtime("data/mrt_stations.csv")
    except OSError:
        _mrt_mtime = 0
    mrt_df = load_mrt(_mrt_mtime)

    LINE_COLORS = {
        "NSL": "#D42E12",   # red
        "EWL": "#009645",   # green
        "CCL": "#FA9E0D",   # orange
        "NEL": "#9B26AF",   # purple
        "DTL": "#005EC4",   # dark blue
        "TEL": "#9D5B25",   # brown
    }
    LINE_NAMES = {
        "NSL": "North–South", "EWL": "East–West",
        "CCL": "Circle",      "NEL": "North East",
        "DTL": "Downtown",    "TEL": "Thomson–East Coast",
    }

    # chronological batch order (only show those present in the data)
    BATCH_ORDER = ["Oct 2024", "Feb 2025", "Jul 2025", "Oct 2025",
                   "Feb 2026", "Jun 2026", "Oct 2026"]
    batches_present = [b for b in BATCH_ORDER
                       if "batch" in bto_projects.columns and b in set(bto_projects["batch"])]

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        show = st.multiselect("Classification",
                              ["Standard", "Plus", "Prime"],
                              default=["Standard", "Plus", "Prime"])
    with fcol2:
        show_batches = st.multiselect("Launch exercise", batches_present,
                                      default=batches_present)

    mrt_col1, mrt_col2 = st.columns([1, 2])
    with mrt_col1:
        show_mrt = st.checkbox("Show MRT stations", value=True)
    with mrt_col2:
        if show_mrt and len(mrt_df):
            show_lines = st.multiselect(
                "Lines", list(LINE_COLORS.keys()),
                default=list(LINE_COLORS.keys()),
                format_func=lambda k: f"{k} ({LINE_NAMES[k]})",
                key="mrt_lines")
        else:
            show_lines = list(LINE_COLORS.keys())

    show_future = st.checkbox(
        "Show URA Master Plan 2025 future growth areas", value=True,
        help="Planned housing neighbourhoods and business nodes from the "
             "Draft Master Plan 2025 — shown as purple markers.")

    pts = bto_projects[bto_projects["cls"].isin(show)]
    if "batch" in pts.columns and show_batches:
        pts = pts[pts["batch"].isin(show_batches)]

    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11, tiles="cartodbpositron")
    for _, p in pts.iterrows():
        batch = p.get("batch", "")
        units = p.get("units", "")
        swt = str(p.get("swt", "")).strip().lower() == "yes"
        conf = p.get("conf", "")

        lines = [f"<b>{p['name']}</b>", f"{p['town']}",
                 f"{p['cls']}" + (f" · {batch}" if batch else "")]
        if pd.notna(units) and str(units).strip() not in ("", "nan"):
            lines.append(f"{int(float(units)):,} units")
        if swt:
            lines.append("Shorter Waiting Time")
        if conf:
            lines.append(f"<i>Classification: {conf.lower()}</i>")

        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=9, color=colors.get(p["cls"], "gray"), fill=True, fill_opacity=0.85,
            popup="<br>".join(lines),
            tooltip=f"{p['name']} ({p['cls']}, {batch})" if batch
                    else f"{p['name']} ({p['cls']})",
        ).add_to(m)

    # Future developments layer (URA Draft Master Plan 2025) — distinct markers
    if show_future and len(future):
        for _, d in future.iterrows():
            dtype = d.get("type", "")
            homes = d.get("homes", "")
            mrt = d.get("nearby_mrt", "")
            amen = d.get("amenities", "")
            horizon = d.get("horizon", "")

            flines = [f"<b>{d['name']}</b>", f"<i>Future {str(dtype).lower()}</i>"]
            if pd.notna(homes) and str(homes).strip() not in ("", "nan"):
                flines.append(f"Planned homes: {homes}")
            if pd.notna(mrt) and str(mrt).strip() not in ("", "nan"):
                flines.append(f"Near MRT: {mrt}")
            if pd.notna(amen) and str(amen).strip() not in ("", "nan"):
                flines.append(f"Amenities: {amen}")
            if pd.notna(horizon) and str(horizon).strip() not in ("", "nan"):
                flines.append(f"Timeline: {horizon}")

            icon = "briefcase" if str(dtype).lower() == "business node" else "home"
            folium.Marker(
                location=[d["lat"], d["lon"]],
                popup=folium.Popup("<br>".join(flines), max_width=280),
                tooltip=f"{d['name']} (future {str(dtype).lower()})",
                icon=folium.Icon(color="purple", icon=icon, prefix="fa"),
            ).add_to(m)

    # MRT stations layer — small circles colour-coded by line
    if show_mrt and len(mrt_df):
        mrt_filtered = mrt_df[mrt_df["line"].isin(show_lines)]
        for _, s in mrt_filtered.iterrows():
            color = LINE_COLORS.get(s["line"], "#888888")
            all_lines = str(s.get("lines", s["line"]))
            line_label = " · ".join(
                f"{l} ({LINE_NAMES.get(l, l)})" for l in all_lines.split(",")
            )
            folium.CircleMarker(
                location=[s["lat"], s["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.9,
                weight=1.5,
                popup=folium.Popup(
                    f"<b>{s['name']}</b><br>{line_label}", max_width=220),
                tooltip=s["name"],
            ).add_to(m)

    st_folium(m, use_container_width=True, height=480)

    # legend
    l1, l2, l3 = st.columns(3)
    l1.markdown("🟢 **Standard** — largest supply, fewest restrictions")
    l2.markdown("🟠 **Plus** — choicer location, more subsidy")
    l3.markdown("🔴 **Prime** — central, most subsidy, 10-yr MOP")
    if show_future and len(future):
        st.markdown("🟣 **Purple pins** — future growth areas from the URA Draft "
                    "Master Plan 2025 (planned housing & business nodes, not yet built)")
    if show_mrt and len(mrt_df):
        st.markdown(
            "**MRT lines:** "
            "<span style='color:#D42E12'>●</span> NSL &nbsp;"
            "<span style='color:#009645'>●</span> EWL &nbsp;"
            "<span style='color:#FA9E0D'>●</span> CCL &nbsp;"
            "<span style='color:#9B26AF'>●</span> NEL &nbsp;"
            "<span style='color:#005EC4'>●</span> DTL &nbsp;"
            "<span style='color:#9D5B25'>●</span> TEL",
            unsafe_allow_html=True)

    st.caption("BTO projects compiled from HDB sales-launch announcements (Oct 2025, "
               "Feb 2026) and previews (Jun 2026); 'confirmed' classifications are from "
               "HDB/reported data, 'estimated' ones pending HDB's official Table 1. "
               "Future growth areas are from URA's Draft Master Plan 2025 — indicative "
               "and subject to change. Coordinates are approximate unless refined via "
               "OneMap (src/geocode_bto.py). Verify at hdb.gov.sg and ura.gov.sg.")

# ===============================================================
# TAB 4 — AFFORDABILITY CALCULATOR
# ===============================================================
with tab4:
    st.subheader("Payment breakdown")
    st.write("See how a BTO purchase is paid across its stages — from booking to "
             "monthly instalments. This is a guide, not financial advice.")

    c1, c2 = st.columns(2)
    with c1:
        price = st.number_input("Flat price ($)", 100_000, 1_500_000, 450_000, step=10_000)
        bto_flat_type = st.selectbox(
            "Flat type (sets the option fee)",
            ["2-room Flexi", "3-room", "4-room", "5-room / 3Gen / Executive"],
            index=2)
        loan_type = st.radio("Loan type", ["HDB loan (75%)", "Bank loan (75%)"])
    with c2:
        staggered = st.checkbox("Use Staggered Downpayment Scheme (eligible first-timers)")
        tenure = st.slider("Loan tenure (years)", 10, 25, 25)
        rate = st.slider("Interest rate (% p.a.)", 2.0, 4.5,
                         2.6 if loan_type.startswith("HDB") else 3.5, step=0.1)

    # --- EHG (Enhanced CPF Housing Grant) ---
    has_ehg = st.checkbox("I qualify for the Enhanced CPF Housing Grant (EHG)")
    grant = 0
    if has_ehg:
        g1, g2 = st.columns(2)
        with g1:
            household = st.radio("Household type", ["Family", "Single"], horizontal=True)
        max_grant = 120_000 if household == "Family" else 60_000
        with g2:
            grant = st.number_input("Your EHG amount ($)", min_value=0,
                                    max_value=max_grant, value=0, step=500,
                                    help="Enter the exact amount from your HDB Flat "
                                         "Eligibility (HFE) letter, or your best estimate.")
        st.caption(f"EHG is income-tiered (lower income = higher grant), up to "
                   f"${max_grant:,} for a {household.lower()} household — income ceiling "
                   f"{'$9,000' if household == 'Family' else '$4,500'}/month. "
                   "The exact amount is confirmed by HDB in your HFE letter. "
                   "It's credited to CPF OA and reduces the amount you finance.")

    # --- option fee (paid at booking, forms part of the downpayment) ---
    option_fees = {"2-room Flexi": 500, "3-room": 1000,
                   "4-room": 2000, "5-room / 3Gen / Executive": 2000}
    option_fee = option_fees[bto_flat_type]

    # --- downpayment split ---
    # HDB loan: 25% total. Standard 10% at signing + 15% at keys;
    #           staggered 5% at signing + 20% at keys.
    # Bank loan: 25% total, of which 5% must be cash; here we show the split.
    if staggered:
        signing_pct, keys_pct = 0.05, 0.20
    else:
        signing_pct, keys_pct = 0.10, 0.15

    signing_dp = price * signing_pct
    keys_dp = price * keys_pct
    # EHG goes into CPF OA and offsets the amount financed (not the cash portion).
    loan_amount = max(price * 0.75 - grant, 0)

    # --- Buyer's Stamp Duty (residential, current tiered rates) ---
    def buyer_stamp_duty(p):
        tiers = [(180_000, 0.01), (180_000, 0.02), (640_000, 0.03),
                 (500_000, 0.04), (1_500_000, 0.05), (float("inf"), 0.06)]
        duty, remaining = 0, p
        for band, rate_ in tiers:
            taxed = min(remaining, band)
            duty += taxed * rate_
            remaining -= taxed
            if remaining <= 0:
                break
        return round(duty)

    bsd = buyer_stamp_duty(price)

    # --- other fees & stamp duties ---
    # Mortgage / Deed stamp duty: 0.4% of the loan amount, capped at $500.
    mortgage_stamp = min(round(loan_amount * 0.004), 500)

    # Survey fee — varies by flat type (approximate HDB schedule).
    survey_fees = {"2-room Flexi": 163.50, "3-room": 218.30,
                   "4-room": 272.85, "5-room / 3Gen / Executive": 354.25}
    survey_fee = round(survey_fees[bto_flat_type])

    # Conveyancing / legal fee (approximate if HDB acts as solicitor).
    legal_fee = 500

    # Caveat registration + title search (small fixed admin fees).
    admin_fees = 90

    total_fees = bsd + mortgage_stamp + survey_fee + legal_fee + admin_fees

    # --- monthly instalment on the 75% loan ---
    r = rate / 100 / 12
    n = tenure * 12
    monthly = loan_amount * r / (1 - (1 + r) ** -n)

    # cash requirement note for bank loans (min 5% must be cash)
    min_cash = price * 0.05 if loan_type.startswith("Bank") else 0

    st.divider()

    # --- stage-by-stage table ---
    st.markdown("#### Your payment timeline")

    stages = pd.DataFrame([
        {"Stage": "1. At booking",
         "What you pay": "Option fee",
         "Amount": option_fee,
         "When": "When you select your flat"},
        {"Stage": "2. Signing Agreement for Lease",
         "What you pay": f"Downpayment ({int(signing_pct*100)}%) − option fee, "
                         f"+ fees & stamp duties (see below)",
         "Amount": signing_dp - option_fee + total_fees,
         "When": "~4–6 months after booking"},
        {"Stage": "3. Key collection",
         "What you pay": f"Remaining downpayment ({int(keys_pct*100)}%)",
         "Amount": keys_dp,
         "When": "~2.5–4 years later (after construction)"},
        {"Stage": "4. After key collection",
         "What you pay": (f"Loan instalments (${loan_amount:,.0f} financed over "
                          f"{tenure} yrs)"),
         "Amount": monthly,
         "When": "Monthly, until loan is repaid"},
    ])
    disp = stages.copy()
    disp["Amount"] = disp["Amount"].apply(lambda x: f"${x:,.0f}")
    disp.loc[3, "Amount"] += " / month"
    st.table(disp.set_index("Stage"))

    # --- fees & stamp duties breakdown (paid at signing) ---
    st.markdown("#### Fees & stamp duties (due at signing)")
    fees = pd.DataFrame([
        {"Item": "Buyer's Stamp Duty (BSD)", "Amount": bsd},
        {"Item": "Mortgage stamp duty (0.4% of loan, max $500)", "Amount": mortgage_stamp},
        {"Item": "Survey fee", "Amount": survey_fee},
        {"Item": "Conveyancing / legal fee (approx.)", "Amount": legal_fee},
        {"Item": "Caveat & title admin fees", "Amount": admin_fees},
        {"Item": "Total fees & stamp duties", "Amount": total_fees},
    ])
    fees_disp = fees.copy()
    fees_disp["Amount"] = fees_disp["Amount"].apply(lambda x: f"${x:,.0f}")
    st.table(fees_disp.set_index("Item"))

    # --- summary metrics ---
    upfront = option_fee + (signing_dp - option_fee + total_fees) + keys_dp
    m1, m2, m3 = st.columns(3)
    m1.metric("Total downpayment (25%)", f"${price*0.25:,.0f}")
    m2.metric("Total upfront (incl. fees)", f"${upfront:,.0f}")
    m3.metric("Monthly instalment", f"${monthly:,.0f}")

    if grant > 0:
        st.success(f"🎁 EHG of **${grant:,.0f}** applied — it reduces the amount financed "
                   f"to **${loan_amount:,.0f}**, lowering your monthly instalment. "
                   "(Credited to CPF OA; can't be used for the cash portion.)")

    if min_cash > 0:
        st.info(f"💵 Bank loan: at least **${min_cash:,.0f}** (5% of price) "
                f"must be paid in cash — the rest of the downpayment can be CPF or cash.")
    else:
        st.info("💡 HDB loan: the full 25% downpayment can be paid with CPF OA, "
                "cash, or a mix.")

    st.caption("Estimates only. Buyer's Stamp Duty uses standard residential rates; "
               "legal/survey fees are approximate. The EHG amount shown is what you enter — "
               "the actual grant is income-tiered and confirmed by HDB. Always verify "
               "figures with HDB and your bank.")

# ===============================================================
# TAB 5 — PRICE FORECAST (Prophet)
# ===============================================================
with tab5:
    st.subheader("Price forecast")
    st.write("A time-series forecast of how prices in a town might trend over the next "
             "few years — a different question from the estimator (which values a flat "
             "today). Useful for Plus/Prime flats with a 10-year MOP.")

    f1, f2, f3 = st.columns(3)
    with f1:
        f_town = st.selectbox("Town", TOWNS,
                              index=TOWNS.index("PUNGGOL") if "PUNGGOL" in TOWNS else 0,
                              key="fc_town")
    with f2:
        f_type = st.selectbox("Flat type", FLAT_TYPES,
                              index=FLAT_TYPES.index("4 ROOM") if "4 ROOM" in FLAT_TYPES else 0,
                              key="fc_type")
    with f3:
        years_ahead = st.slider("Years to forecast", 1, 15, 10)

    if st.button("Run forecast", type="primary"):
        # Prophet is heavy; import only when needed and warn if not installed.
        try:
            from prophet import Prophet
        except ImportError:
            st.error("Prophet isn't installed. Run `pip install prophet` and restart "
                     "the app to use this tab.")
            st.stop()

        sub = df[(df["town"] == f_town) & (df["flat_type"] == f_type)].copy()
        ts = (sub.groupby(sub["month"].dt.to_period("M"))["resale_price"]
                 .median().reset_index())
        ts["month"] = ts["month"].dt.to_timestamp()
        ts.columns = ["ds", "y"]
        ts = ts.dropna()

        if len(ts) < 36:
            st.info("Not enough history for this town + flat type to forecast reliably. "
                    "Try a more common combination.")
        else:
            with st.spinner("Fitting the model and projecting forward..."):
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                            daily_seasonality=False, changepoint_prior_scale=0.1)
                m.fit(ts)
                future = m.make_future_dataframe(periods=years_ahead * 12, freq="MS")
                fc = m.predict(future)

            # chart: history + forecast + uncertainty band
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_scatter(x=ts["ds"], y=ts["y"], mode="lines",
                            name="Actual", line=dict(color="#5F5E5A"))
            fig.add_scatter(x=fc["ds"], y=fc["yhat"], mode="lines",
                            name="Forecast", line=dict(color="#378ADD"))
            fig.add_scatter(x=list(fc["ds"]) + list(fc["ds"][::-1]),
                            y=list(fc["yhat_upper"]) + list(fc["yhat_lower"][::-1]),
                            fill="toself", fillcolor="rgba(55,138,221,0.15)",
                            line=dict(color="rgba(0,0,0,0)"), name="Uncertainty",
                            showlegend=True)
            fig.update_layout(height=420, margin=dict(t=10, b=0, l=0, r=0),
                              yaxis_title="Median price ($)")
            st.plotly_chart(fig, use_container_width=True)

            # a couple of headline numbers
            last_actual_year = ts["ds"].max().year
            fut = fc[fc["ds"].dt.year == last_actual_year + years_ahead]
            if len(fut):
                row = fut.iloc[-1]
                st.metric(f"Projected median price in {last_actual_year + years_ahead}",
                          f"${row['yhat']:,.0f}",
                          help="Mid-estimate; the band shows the likely range.")
                st.caption(f"Likely range: ${row['yhat_lower']:,.0f} – "
                           f"${row['yhat_upper']:,.0f}. This projects the recent trend "
                           "forward and can't foresee policy shocks or downturns — treat "
                           "it as a trend guide, not a guarantee.")

st.divider()
st.caption("Data: HDB / data.gov.sg (Open Data Licence). Built as a portfolio project.")