# PropertyDecisionSupport — Master Smart Money Roadmap (Consolidated)

> This document merges everything so far into one exhaustive plan: (1) the original public-HDB investor/agent/homebuyer brainstorm, (2) the institutional-grade professional-firm layer (REITs, PE, developer balance sheets), (3) the private-property track, (4) the rigorous data-governance/confidence framework from the uploaded roadmap, and (5) the filtered ChatGPT use-case list (90% kept, ~10% dropped as noted). Nothing from prior rounds is dropped — everything is re-tiered by actual money-making edge, and every item states its data source and how confident you should be in it.

---

## 0. Purpose and north star

Transform PropertyDecisionSupport from a descriptive HDB dashboard into a **decision-support and property-finding intelligence system** for three audiences:

1. **Personal investing** — mispricing, catalysts, downside, liquidity, exit economics
2. **Professional property agents** — client matching, accurate pricing, negotiation leverage, retention
3. **Institutional-adjacent use** — the kind of signal REITs, property PE, and family offices actually trade on, built from public data you can legitimately access

**North star statement:** Given a buyer/investor and available market evidence, identify the best properties or areas, explain *why* they're attractive, quantify the risks and assumptions, and show what evidence would change the conclusion. The product should never pretend to know more than the data supports — every analysis carries a visible confidence rating and data-source note.

The real goal is not "we know a secret only 5% of people know." It's: **systematically combine public and semi-public information into decision-relevant signals that are hard to compute by hand, while being unusually transparent about uncertainty.** That is a more defensible edge than a mystique-driven "insider" pitch, and it's also what actually holds up under professional scrutiny from agents and serious investors.

---

## 1. Data governance principle (applies to every single item below)

Every analysis in this plan must, when built, show:

- **Data confidence:** High / Medium / Low / Experimental
- **Primary datasets used**
- **Data source / provider**
- **Coverage period**
- **Geographic granularity**
- **Variables actually used**
- **Assumptions made**
- **Proxy variables used (if any)**
- **Known gaps**
- **How gaps are currently filled**
- **Whether scraping/licensing is required, and which**
- **Whether it can run today or needs new data**
- **What future data would materially improve it**
- **Last refresh date / method version**

Never hide a proxy behind a polished-looking score. A user should always be able to click an analysis and answer: *"Where did this number come from, and how much should I trust it?"* This isn't bureaucracy — it's what separates a tool professional agents and serious investors will actually trust from one they'll dismiss after the first time a number looks off.

---

## 2. Data inventory

### 2.1 Already available (high confidence)
- **HDB resale transactions** — ~978k rows, Jan 1990–May 2026, town/flat-type/block/street/storey/floor-area/model/lease/price. High confidence for transaction-price analysis. Caveats: pre-2012 uses approval date not registration date; older remaining-lease is back-computed; no listing duration, seller motivation, renovation quality, or exact unit condition.
- **OneMap block coordinates** — ~9,972 geocoded block/street pairs. High confidence for location work.

### 2.2 Already available, supplementary (medium confidence, manually compiled)
- **BTO projects** — medium confidence until systematically maintained via API rather than manual compilation
- **MRT stations** — medium/high depending on update cadence
- **URA Master Plan / future developments** — medium confidence; implementation timing can shift

### 2.3 Not yet available — the gaps that matter most

**Gap A — Amenities.** No comprehensive, continuously updated amenity database (schools, kindergartens/childcare, hawker centres, supermarkets, malls, hospitals, polyclinics, parks, sports facilities, libraries, community clubs, markets, employment/commercial nodes). Primary sources: OneMap, data.gov.sg, LTA, MOE, NParks, URA, HDB. Build `amenities.csv` from these government/open sources. Proxy where walking distance is unavailable: straight-line distance first, network-routed walking distance later. **Prefer APIs/open data over scraping** — scrape only where no structured source exists and terms permit it.

**Gap B — Rental data.** Needed for yield, buy-vs-rent, and rent-price divergence work (see Priority section below).

**Gap C — Private residential transaction data.** Needed for HDB-vs-condo comparisons and the entire private-property track.

**Gap D — Listing-level data (asking price, days-on-market, price cuts).** The single hardest, most valuable gap — see Section 8.

---

## 3. Data architecture upgrade

Move from a flat pile of CSVs to a small property intelligence data layer:

```text
data/
  resale/
  rental/
  private_property/
  amenities/
  transport/
  schools/
  planning/
  supply/
  demographics/
  finance/
  derived/
  metadata/

data_dictionary.yml
analysis_registry.yml
source_registry.yml
```

Every analysis references its data dependencies in `analysis_registry.yml`, and every source is logged in `source_registry.yml` with:

```yaml
source:
  name:
  provider:
  url:
  access_method:
  refresh_frequency:
  last_refresh:
  coverage:
  license_notes:
  fields:
  known_limitations:
```

This registry is the audit trail for the whole product and is what makes the confidence labels credible rather than decorative.

---

## 4. PHASE 1 — Valuation Intelligence (build first — least new data required)

### A1. Comparable Property Engine
Find genuinely comparable transactions for a target flat using weighted comparable selection (physical similarity, temporal proximity, geographic proximity, lease similarity, flat-model similarity, storey similarity) rather than naive town medians. Data: HDB resale only. Gap: exact unit attributes (renovation, facing, view, corner/point block) unavailable — proxy with block/storey-band/model/lease/location. Confidence: High for broad valuation, Medium for exact-unit valuation. No scraping needed.

### A2. Fair Value Model
Predict what a flat *should* transact for. Start interpretable (hedonic regression), compare against gradient boosting / random forest / spatial or hierarchical models later. Critical: never rely on a single model — always output a **predicted fair value plus a prediction interval**, not a false-precision point estimate. Data: HDB + derived spatial variables. No scraping needed for V1.

### A3. Mispricing / Residual Detector
`Mispricing % = Actual Price / Fair Value − 1`, plus standardized residual and model uncertainty. **Important rule: never call something "undervalued" just because it's cheap** — require sufficient comparable count, model confidence, a persistent (not one-off) signal, and no obvious structural explanation (e.g. it's cheap because it's a 1st-storey unit facing a rubbish chute, not because it's mispriced). Confidence: Medium initially, High only after backtesting (see Section 12).

### A4. Block Desirability Fingerprint
Price premium/discount, transaction velocity, price resilience, recovery after downturns, storey premium, lease discount, model mix, MRT/neighbourhood premium, transaction variance — a behavioural signature per block relative to comparable blocks. Confidence: Medium (true buyer preference is latent; this is observed *behaviour*, a proxy for it).

### A5. Transaction Quality / Comparable Reliability Score
Prevents bad comparables from contaminating valuation — scores each candidate comparable on recency, distance, physical similarity, residual unusualness, and sample size. Confidence: High/Medium.

### A6. Undervalued Area Finder *(from ChatGPT list — kept, buildable now)*
Blocks anomalously cheap vs. their 1km neighbourhood median over the last 2 years. Method: spatial join using `address_coords.csv` + haversine distance on the resale dataset. This is effectively a simpler, faster-to-ship version of A3 at the *area* rather than *unit* level — build this first as a quick win, then let A2/A3 supersede it with proper modelling.

### A7. Cohort-Based True Lease Depreciation *(fixes existing Page 7 flaw)*
Rewrite the existing lease-depreciation chart, which is confounded by era effects (high-remaining-lease flats sold cheaply in the 1990s vs. low-remaining-lease flats sold at today's high prices, all pooled into one misleading LOWESS line). Fix: group by `lease_commence_date` decade and track PSM as remaining lease ticks down *within each cohort*, isolating true depreciation from era/inflation effects. High priority — this is a data-integrity fix to something already shipped, not a new feature, and it's currently giving users a wrong answer.

### A8. Floor Premium Validator *(ChatGPT — kept)*
Is the asking/actual floor premium justified by the data, or inflated? Can build now from existing storey-range field.

### A9. Flat Model Price Differential *(ChatGPT — kept)*
DBSS vs. Model A vs. Improved etc. — systematic price differences by flat model, controlling for other variables. Can build now.

### A10. PSM vs. Floor Area Scatter *(ChatGPT — kept)*
Catches non-linearly overpriced small flats (small units often carry a PSM premium that isn't always justified). Can build now.

---

## 5. PHASE 2 — Location & Amenity Intelligence

### B1. Comprehensive Amenity Database
The missing location layer (see Gap A above). Fields: amenity type, name, coordinates, opening status, source, last updated, confidence. High confidence for government facilities, medium for private amenities. Scrape only categories with no structured source and where permitted.

### B2. Walkability / Walk-My-Commute *(ChatGPT B7 — kept)*
MRT proximity to a specific workplace address. V1 proxy: straight-line distance (Medium confidence). V2: actual walking-network routing (High confidence).

### B3. Accessibility Score + Change-in-Accessibility
Number of MRT stations reachable, travel time to employment centres, school/amenity/interchange access — and critically, **the change in this score over time**, not just the static level. A currently-mediocre location that's improving may be a better opportunity than one that's already fully priced for its convenience.

### B4. Future Accessibility Catalyst
Locations whose accessibility will materially improve — future MRT, roads, employment/commercial nodes. Proxy: official planning status + estimated completion windows (implementation timing can slip, so label Medium confidence).

### B5. School Proximity Premium *(ChatGPT B8 — kept)*
1km priority-zone premium quantification, distinct from general amenity scoring since school zones drive genuinely large price effects in Singapore.

### B6. Lifestyle-Weighted Town Matching *(ChatGPT C3 — kept, homebuyer-facing)*
Combine B1–B5 into a personalized town/neighbourhood ranking given a family's specific priorities.

---

## 6. PHASE 3 — Supply, Competition & Catalysts

### C1. BTO Competition / Cannibalisation Model
Is future BTO supply negative (resale competition), neutral, or positive (neighbourhood improvement) for a given area? Inputs: BTO location/type/flat-mix/expected supply/distance/completion timing. Proxy: historical behaviour around *past* BTO completions. Medium/Experimental confidence initially — but this is the same underlying mechanism as "BTO Completion Shadow Effect" from the institutional layer (Section 9, I5) and the two should share one model.

### C2. MOP Unlock Calendar *(carried forward — one of the highest-leverage items in the whole plan)*
Using `lease_commence_date` + the 5-year MOP rule, compute exactly which blocks/towns cross resale-eligibility thresholds in the next 3–24 months. A wave of newly-MOP'd units is a predictable, mechanical supply shock that tends to soften nearby prices 6–12 months later. Zero new data needed — buildable today. This is still one of the two or three best ROI items in this entire document.

### C3. Supply Pressure Score
Aggregate future competing supply (BTO + EC + private + planned developments) around a block. Combine URA planning data with HDB launch information. Scraping potentially required for project-level updates if APIs prove insufficient — treat as last resort.

### C4. Catalyst Score
Aggregate positive catalysts (MRT, employment, commercial, parks, schools, infrastructure) against negative ones (competing supply, construction disruption, congestion, undesirable land-use change). This is an evidence-aggregation model, explicitly not a price forecast — label it as such.

### C5. En-Bloc / SERS Watch List (public housing)
Blocks matching historical SERS profiles (age, plot ratio, location value). Should be upgraded from pure pattern-matching to incorporate actual official SERS/en-bloc announcement history where available (see institutional layer, I7 for the private-property equivalent, which has cleaner data).

---

## 7. PHASE 4 — Market Regime & Contrarian Intelligence

### D1. Market Regime Detector
Identify whether current conditions resemble historical regimes using price growth, volume, price-volume relationship, town/flat-type divergence, and policy-event overlay. High confidence descriptively, Medium confidence predictively — regimes rhyme, they don't repeat exactly.

### D2. Price-vs-Fundamentals Divergence
Flag places where price has moved differently from underlying indicators — e.g. price +15% while rents +3%, volume falling, supply rising, accessibility unchanged. Before rental data is integrated, proxy with volume/accessibility/supply only (Experimental); after rental integration, Medium/High confidence. This is the single-family-home-market equivalent of what hedge funds call "growth without fundamentals" — genuinely useful and currently absent from every retail tool.

### D3. Stigma Persistence Detector *(refines original "Stigmatised Flat Detector," A11/A10 in earlier lists)*
Track persistent block-level residuals over many years to distinguish structural stigma from temporary mispricing — categories: permanent discount, temporary shock, recovery, emerging discount, emerging premium. This is a materially better version of the original "stigmatised flat" idea because it's grounded in a multi-year residual pattern rather than folklore/rumor, which also makes it defensible if ever questioned.

### D4. Reversion / Recovery Detector
Find assets that historically recover after unusual discounts, via historical event studies. Critical rule: historical reversion is evidence, not a guarantee — never present it as a promise.

### D5. Entry/Exit Seasonality *(ChatGPT A5 — kept, but correctly demoted)*
Best historical quarter to buy or sell. Real but modest edge — cheap to add once D1–D4 exist, not worth building standalone first.

### D6. Price Floor by Cooling-Measure Era *(kept)*
How deep was each cooling-measure-triggered dip, how long was the recovery. Useful context, feeds directly into D4.

---

## 8. PHASE 5 — Rental & Investment Economics

### E1. HDB Rental Database
Integrate HDB rental transactions from data.gov.sg (median rent by town + flat type, quarterly). Outputs: median rent, rent/sqm, rental growth, block/town yield, rent-price divergence (feeds D2). High confidence once integrated. **Do not scrape PropertyGuru/99.co for this — official aggregate town-level rental data is sufficient for strategic decisions and the ToS risk isn't worth it for data this replaceable.**

### E2. Gross Rental Yield + Buy-vs-Rent Calculator *(ChatGPT A4 — kept, high priority)*
`Annual Gross Rent / Purchase Price`, explicitly labeled as ignoring vacancy, maintenance, property tax, financing cost, and transaction costs (see E3 for the fuller version). Includes an interactive buy-vs-rent breakeven calculator for personal decision-making.

### E3. Net Yield / IRR Engine
Full internal-rate-of-return model incorporating financing cost, CPF usage, transaction costs (BSD, agent fees), maintenance, vacancy assumption, and exit costs — this is what actually determines whether a "good deal" on paper is a good deal after real costs. This is the difference between a retail yield number and how a professional actually underwrites a deal.

### E4. Inflation-Adjusted Real Returns *(ChatGPT A14 — kept, demoted to this phase)*
Are buyers actually beating CPI or just keeping up in nominal terms? Useful framing, modest standalone differentiation — build once E1–E3 exist.

### E5. Mortgage Affordability Simulator *(ChatGPT A7/B4/C1 — kept, merges investor+agent+homebuyer versions)*
Parametric — no new data needed. Income/CPF/cash → max affordable price, given MSR/TDSR rules, current rates, and loan tenure limits.

### E6. Grant Eligibility Estimator (EHG, PHG) *(ChatGPT B9 — kept)*
Parametric, rules-based, no new data.

### E7. ABSD / Overseas Buyer & Upgrade Cost Calculator *(ChatGPT A9/B10 — kept)*
HDB equity → condo upgrade path, ABSD cost calculation by buyer profile (citizen/PR/foreigner/entity), parametric.

---

## 9. PHASE 6 — Institutional-Grade Layer (the actual professional-firm playbook)

> This is what makes the plan genuinely "top 5%" rather than just "a nicer retail dashboard." REITs, property PE funds, and family offices analyze **capital flows, balance sheets, and structural mispricings** that show up in filings and permits *before* they show up in transaction data. Almost none of this requires scraping — it requires reading public filings systematically, which is exactly the kind of boring-but-high-value work a tool can automate.

### I1. Listing-to-Transaction Gap Tracker (asking price decay)
Track live listings (asking price, days-on-market, price revisions) against eventual transacted price — the single most-used signal at property funds, since it's forward-looking where transaction data is backward-looking. **Data reality:** no public bulk API for 99.co/PropertyGuru listings exists. The professional path is licensing this from **SRX Property** (which already sells exactly this aggregated product to institutions) or a direct portal partnership — not scraping, which violates ToS and is fragile. Treat "start an SRX/REALIS conversation" as a standing action item, not a blocker.

### I2. Developer Land Bank & Balance Sheet Model
Track GLS (Government Land Sales) tender results, unsold unit inventory per project (URA publishes this monthly — an underused, fully public, institutional-grade dataset), and listed-developer financing exposure via SGX filings. A developer sitting on unsold inventory near a refinancing date is a structurally forced seller, visible months ahead in public filings. 100% free, official data — GLS results, URA developer sales/take-up data, SGXNet filings for CapitaLand/CDL/UOL etc.

### I3. REIT Cap Rate & NOI Benchmark Layer
Track implied cap rates and NOI yield across SGX-listed REITs with residential/retail/mixed exposure, to benchmark private residential/commercial asking prices against listed real-estate valuations. When private asking prices imply cap rates meaningfully below comparable REIT valuations, that's a textbook overpricing signal institutions act on directly. 100% free, official SGXNet filings and valuer reports — just requires systematically reading them, which nobody retail-facing currently does.

### I4. Capital Flow / Buyer Entity-Type Tracking
Track ABSD-relevant buyer profile shifts (entity purchases, nationality mix where disclosed) by district over time, to detect where institutional or foreign capital is rotating in/out — capital flow precedes price. Data: URA REALIS caveats (paid, official — worth pricing out directly, it's literally the industry-standard tool) for granular signal; IRAS aggregate ABSD collection stats (free) for a coarser macro proxy. Granular buyer identity is not public for privacy reasons regardless of budget — this module works at the aggregate/statistical level only.

### I5. Permit & Construction Pipeline Leading Indicator
Extends C1/BTO-shadow modelling upstream: BCA construction permit data and URA planning-permission approvals as a 2–4 year leading indicator of supply, earlier than BTO launch announcements or GLS tenders alone. Fully public — BCA published data, URA e-Info Development Control portal (a public records lookup system designed for querying, not a walled listings site).

### I6. Cross-Asset Arbitrage: HDB Upgrader → Private Demand Bridge
**The single most differentiated item in this whole plan.** Model MOP-cleared HDB unlock volume (C2) by town as a leading indicator for private mass-market condo demand in adjacent districts. Nobody else has both HDB *and* private transaction data while treating them as a connected feeder pipeline rather than separate silos — this is a genuinely proprietary insight, buildable entirely from data you already have or will have once private transaction data is integrated (Section 10). Build this immediately once both datasets exist.

### I7. En-Bloc Feasibility Screener (private)
The private-property, data-clean version of C5: screens condos for genuine en-bloc economics — plot ratio utilization vs. URA Master Plan zoning allowance, land value uplift potential, unit count vs. the 80%/90% consent thresholds by building age. All inputs public (URA Master Plan GIS, condo unit counts, building age).

### I8. New Launch vs. Resale Premium Decay Curve
Tracks the price gap between new-launch and comparable resale in the same precinct over time since TOP — this decays on a historically predictable curve, and deviations flag over/under-priced launches. Requires URA private transaction data with new-sale/resale flagging (sourced under Section 10).

### I9. Mortgagee Sale / Distressed Private Listings
The private-market, *cleaner* version of a distressed-sale detector: mortgagee sale and auction listings are **publicly and legitimately disclosed** by banks and auction houses (Colliers, Knight Frank, Edmund Tie publish auction results), unlike HDB where distress must be inferred. Genuinely better data than any HDB proxy for the same concept.

### I10. Distressed / Forced-Sale Proxy (HDB) *(carried forward from earlier round, with a caution)*
Flag unusually short holding periods (<3 years, excluding known MOP-flip patterns) clustered by block/street, as a proxy for divorce/estate/financial-distress sales. **Caution, stated explicitly in the UI:** label this a probabilistic proxy, never a confirmed status — short holding periods have many innocent explanations, and overclaiming here is the fastest way to lose credibility with professional agents.

### I11. Rate / SORA Sensitivity by Price Band
Historical price elasticity to SORA/mortgage-rate moves, segmented by price band and town — where leverage risk concentrates, and where rate-cut cycles have historically produced the sharpest catch-up rallies. Fold into I3's cap-rate framework rather than building standalone. Needs one small new public dataset (historical SORA/SIBOR series).

---

## 10. PHASE 7 — Private Property Track (full build-out)

### F1. Private Transaction Data Pipeline
`src/fetch_data.py` pulling URA private residential transaction data from data.gov.sg / URA API, standardized into `data/condo_clean.csv` with the same rigor as the HDB pipeline (schema notes, geocoding, derived columns).

### F2. HDB vs. Condo Comparison Page
PSM comparison, price-gap trend over time, depreciation comparison, transaction volume — same town, matched hold period. This was Priority 1 (parallel) in the ChatGPT list and remains high priority, but sits behind I6 in sequencing since I6 is free and more differentiated.

### F3. HDB vs. Condo ROI Comparison, Same Hold Period *(ChatGPT A3 — kept)*
Full return comparison (not just PSM) — capital appreciation + yield − costs, matched by town and holding period.

### F4. Upgrade Pathway Model *(ChatGPT A9 — kept, ties to E7)*
HDB equity extraction → condo down payment → ABSD cost → break-even analysis for the classic HDB-to-condo upgrade decision.

---

## 11. PHASE 8 — Agent-Specific Tools (full sales-funnel structure)

Restructured around the actual sales funnel: **Qualify → Shortlist → Price check → Negotiate → Retain.**

### Qualify
- **G1. CPF/HDB Loan Eligibility Checker** *(ChatGPT B4 — kept)* — lease + buyer age, parametric
- **G2. Grant Eligibility Estimator** — same as E6, surfaced in agent workflow
- **G3. Overseas Buyer / ABSD Advisory** *(ChatGPT B10 — kept)* — same as E7, agent-facing framing

### Shortlist
- **G4. Client Fit Shortlister** *(ChatGPT B2 — kept, can build now)* — given budget + type + preferences, rank towns by value
- **G5. Town Comparison Report** *(ChatGPT B5 — kept)* — side-by-side stats for 2–3 towns, partly exists already
- **G6. BTO vs. Resale Decision Tool** *(ChatGPT B8/C5 — kept)* — wait time vs. price premium tradeoff, needs BTO supply timing data

### Price check
- **G7. Comps Finder** *(ChatGPT B1 — kept, can build now)* — given a specific block, surface recent similar sales; this is A1 exposed as an agent-facing tool
- **G8. "Is This Price Fair?"** *(ChatGPT B3/C2 — kept, can build now)* — vs. recent comps, percentile rank; this is A3 exposed as a fast agent-facing check
- **G9. Floor Premium Analyser** *(ChatGPT B6 — kept)* — same underlying logic as A8, agent-facing

### Negotiate
- **G10. Negotiation Leverage Report** *(ChatGPT B11 — kept)* — recent volume + price trajectory for a specific flat, feeds into a client-facing talking-points summary
- **G11. Market Heat Report** *(ChatGPT B7 in agent list — kept)* — which towns are active this quarter, partly exists in Market Dynamics
- **G12. Liquidity / Velocity / Block Trend Toolkit** *(carried forward)* — full negotiation data pack combining A4 (block fingerprint), D1 (regime), and transaction velocity
- **G13. MRT Proximity Premium** *(ChatGPT B12 — kept, can build now)* — $/sqm premium within 500m

### Retain
- **G14. Portfolio Tracker** *(carried forward)* — ongoing valuation tracking for past clients, natural re-engagement trigger
- **G15. Upgrade Readiness Alert** *(carried forward)* — flags when a past client's equity position crosses an upgrade threshold, generating a natural outreach moment

### The single highest-ROI agent deliverable
**G16. One-Click Comps Report** — enter an address, generate a shareable, client-ready PDF (recent comps, percentile pricing, floor premium, PSM benchmark) in under 30 seconds. Agent workflows don't die from lack of analysis, they die from lack of a fast client-facing artifact. This should outrank most standalone analytical pages in build priority — it's a packaging exercise on top of G7/G8/G9, not a new data problem.

---

## 12. Homebuyer-Specific Framing (non-investor, non-agent)

Lower priority for a money-making product but worth keeping since it drives adoption/usage which feeds agent trust:

- **H1. "Can I Afford It?"** *(ChatGPT C1 — kept)* — same engine as E5, homebuyer-framed
- **H2. "Is This Price Fair?"** *(ChatGPT C2 — same as G8)*
- **H3. "Which Neighbourhood For My Family?"** *(ChatGPT C3 — same as B6)*
- **H4. "Buy Now or Wait?"** *(ChatGPT C4 — kept)* — market timing signal, same engine as D1
- **H5. "BTO or Resale?"** *(ChatGPT C5 — same as G6)*
- **H6. Remaining Lease / CPF Interactive Explainer** *(ChatGPT C6 — kept)* — policy explainer + calculator, genuinely useful and currently missing
- **H7. Renovation Budget Estimator** *(kept, low priority)* — useful but not a differentiator; build only on request

---

## 13. Dropped from the ChatGPT list (the ~10% left out, and why)

Being explicit about what was cut, since the instruction was to keep 90% and justify the rest:

- **Generic "capital appreciation ranking by town/type"** as a standalone item — dropped as a separate build because it's already fully subsumed by D1 (Market Regime Detector) and the existing Temporal Trends tab; building it separately would just duplicate an existing page under a new name.
- **Nothing else was dropped outright** — everything else in the ChatGPT list was either kept as-is, merged into a more rigorous version of the same idea (e.g. "stigmatised flat detector" → D3's cohort-residual version, "en-bloc watch list" → C5/I7), or re-sequenced to a later tier. The merges are noted inline above so nothing is silently lost.

---

## 14. Data sourcing summary table

| Source | Data | Access path | Confidence if used |
|---|---|---|---|
| data.gov.sg | HDB resale, aggregate stats, rental stats | Free API | High |
| URA (private transactions) | Condo/landed transactions | Free API/portal | High |
| URA REALIS | Caveats, buyer type/nationality flags | **Paid institutional subscription** | High if paid |
| URA (developer sales/take-up) | Unsold unit inventory by project | Free, published monthly | High |
| GLS tender results | Land bids, developer, breakeven estimates | Free, official | High |
| SGXNet / SGX StockFacts | REIT NOI, cap rates, developer financials | Free, public filings | High |
| BCA | Construction permits | Free, published | High |
| URA e-Info / Development Control | Planning permission approvals | Free, public lookup | High |
| IRAS | Aggregate ABSD stats | Free, periodic | Medium (aggregate only) |
| Auction houses (Colliers, KF, Edmund Tie) | Mortgagee sale / auction results | Public press releases/results pages | High |
| OneMap / LTA / MOE / NParks | Amenities, transport network | Free, structured | High |
| 99.co / PropertyGuru | Live listings, asking price, days-on-market | **No public API** — licensing (SRX) or partnership required | High if licensed, do not scrape |
| SRX Property | Aggregated listing + transaction analytics | Commercial data product | High if paid |

**Standing rule:** the highest-value pieces of this entire plan (I2, I3, I5, I6, C2, plus the existing HDB/URA base) run entirely on free, official public data — no scraping needed. The one genuinely hard piece (I1, live listings) is exactly what the industry itself licenses through SRX rather than scrapes, because listing portals actively enforce ToS and a licensed feed is durable where a scrape breaks the moment the HTML changes. Treat starting a URA REALIS / SRX conversation as a standing, parallel-track action item — not a blocker to shipping everything else.

---

## 15. Backtesting requirement — non-negotiable before calling anything "smart" or "contrarian"

Before any strategy is presented to a user as an edge, test it historically:

**Example strategy:** Buy flats that were >8% below model fair value, with improving accessibility, low future supply, and strong historical liquidity. Then ask: *did these properties outperform comparable properties over the next 1, 3, and 5 years?* Do this without leaking future information, using rolling historical train/test windows.

**Avoid look-ahead bias.** When evaluating a 2018 opportunity, the model must only use information that would have been available in 2018 — no future transactions, future planning announcements, future MRT completions not yet known, or future prices. Otherwise the "smart strategy" is an illusion that will fail live.

---

## 16. Scoring philosophy

Never build one opaque "AI score." Use decomposed, inspectable components:

- **Valuation** — how cheap/expensive vs. fair value
- **Fundamentals** — how strong is the underlying location/property
- **Catalyst** — what may change (positive/negative)
- **Supply risk** — what future competition exists
- **Liquidity** — how easy is exit likely to be
- **Lease risk** — how does remaining lease affect future buyer pool/value
- **Confidence** — how reliable is the underlying evidence

An **Overall Opportunity Score** aggregates these, but every component must be inspectable by the user — never a black box.

### Investment Thesis output (per candidate property/area)
Why is it interesting? Why might the market be wrong? What is the market probably pricing in? What could change the price? What could go wrong? Who will buy it from me later? What is estimated fair value and downside? What is the expected holding period? What data is missing? How confident are we? — this is materially more useful than a bare "BUY" signal and is what makes the tool feel like a real analyst's output rather than a stock screener.

---

## 17. North-star screens (eventual product surface)

### Investor: "Find Opportunities"
Filters: budget, town/island-wide, flat type, minimum remaining lease, expected holding period, risk tolerance, minimum liquidity, investment vs. own-stay, catalyst preference. Results table: Property | Fair Value | Discount | Catalyst | Liquidity | Risk | Confidence. Clicking opens the full Investment Thesis (Section 16).

### Agent: "Find the Best Property for This Client"
Client inputs: budget, CPF/cash split, workplace, schools, household composition, flat requirements, risk tolerance, intended holding period. System returns: best lifestyle match, best value match, best investment match, best contrarian match, safest exit, cheapest acceptable substitute — each with stated reasoning, not a bare ranking.

---

## 18. Recommended build order (final, consolidated)

**What can run today, essentially zero new data:**
A1–A5, A6–A10, C2 (MOP calendar), D1, D3, D4, E5–E7 (parametric calculators), G1, G2, G4, G5, G7, G8, G9, G13 — this is a large amount of genuinely valuable product already buildable from what exists.

**Sprint 1 — Data foundations**
Data registry, analysis registry, source registry, confidence framework, amenity schema.

**Sprint 2 — Valuation core**
A1 Comparable engine → A2 Fair value model (with uncertainty intervals) → A3 Mispricing detector → A7 Lease-depreciation fix (data integrity, do this early since it's currently wrong).

**Sprint 3 — Micro-market intelligence**
A4 Block fingerprint, D3 Stigma persistence, D4 Recovery detector, A8–A10 (floor/model/PSM premiums).

**Sprint 4 — Location intelligence**
B1 Amenities, B2 Walkability, B3 Accessibility + change, B4 Future catalyst, B5 School premium.

**Sprint 5 — Supply & catalysts**
C1 BTO cannibalisation, C2 MOP calendar (ship immediately, it's free and high-signal), C3 Supply pressure, C4 Catalyst score, I5 Permit pipeline.

**Sprint 6 — Investment economics**
E1 Rental integration, E2 Gross yield, E3 IRR engine, E4 Real returns, F1–F4 private-property track.

**Sprint 7 — Institutional layer**
I2 Developer balance sheets, I3 REIT cap-rate benchmarking, I6 HDB→private bridge (build as soon as F1 lands — this is the most proprietary single feature in the plan), I7–I9 private en-bloc/launch-decay/distressed tools.

**Sprint 8 — Contrarian/agent product**
D2 Divergence detector, full opportunity scoring (Section 16), G-series agent toolkit culminating in G16 one-click comps report, client matching, portfolio/upgrade retention tools (G14–G15).

**Parallel, ongoing, not blocking anything above:**
Pursue URA REALIS and SRX Property data-licensing conversations — this unlocks I1, I4 properly, and improves I8. Start this now given realistic lead time on data/commercial deals.

---

## 19. Defer until better data exists — do not fake precision here

These should have, at most, clearly-labeled V1 proxy versions, never presented as precise:

1. True days-on-market
2. Seller motivation detection
3. Seller urgency prediction
4. Precise negotiation-outcome probability
5. Listing-to-sale conversion probability
6. True buyer-demand intensity
7. Real-time listing arbitrage

All of these depend on listing-level data (Section 8/9, I1) that either needs licensing or doesn't exist publicly. Build the interface and calculation framework to support them later without depending on them now.

---

## 20. Final strategic principle

The product evolves through three levels:

- **Level 1 — What happened?** Descriptive analytics (this is most of what exists today).
- **Level 2 — What is this property worth?** Valuation and comparables (Phase 1).
- **Level 3 — What should I do?** Risk-adjusted opportunity, catalysts, downside, exit, and client fit (Phases 4, 6, 8).

Level 3 is the real product, and it's what turns this from "a nice HDB dashboard" into something a serious investor or professional agent would actually pay for and trust with real decisions.
