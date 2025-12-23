# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# TANAKA-STYLE SCORECARD (Plotly Edition) – FULL + FIXED
# Requires: streamlit, yfinance, pandas, numpy, plotly
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS (Light KPI cards, no black blocks)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tanaka-Style Scorecard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }

      /* Light professional metric cards */
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }
      div[data-testid="stMetric"] > label {
        color: #6b7280 !important;
        font-weight: 500 !important;
      }
      div[data-testid="stMetric"] span {
        color: #111827 !important;
        font-weight: 650 !important;
      }

      .small-note { color: rgba(17,24,39,0.70); font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SLEEVES = ["Platform", "Biotech/Pharma", "Minerals/Energy", "Financials", "Other"]

BASE_WEIGHTS = {
    "Platform": {"growth": 0.20, "quality": 0.22, "valuation": 0.18, "momentum": 0.10, "convexity": 0.08, "risk": 0.10, "gap": 0.12},
    "Biotech/Pharma": {"growth": 0.14, "quality": 0.10, "valuation": 0.10, "momentum": 0.08, "convexity": 0.22, "risk": 0.10, "gap": 0.26},
    "Minerals/Energy": {"growth": 0.10, "quality": 0.08, "valuation": 0.14, "momentum": 0.08, "convexity": 0.20, "risk": 0.16, "gap": 0.24},
    "Financials": {"growth": 0.12, "quality": 0.18, "valuation": 0.22, "momentum": 0.10, "convexity": 0.06, "risk": 0.18, "gap": 0.14},
    "Other": {"growth": 0.16, "quality": 0.16, "valuation": 0.16, "momentum": 0.10, "convexity": 0.12, "risk": 0.14, "gap": 0.16},
}

DEFAULT_TOP10 = pd.DataFrame(
    [
        {"ticker": "CORT", "name": "Corcept Therapeutics", "sleeve": "Biotech/Pharma", "weight": 13.2},
        {"ticker": "NUVB", "name": "Nuvation Bio", "sleeve": "Biotech/Pharma", "weight": 10.7},
        {"ticker": "NVDA", "name": "NVIDIA", "sleeve": "Platform", "weight": 7.9},
        {"ticker": "UAMY", "name": "US Antimony", "sleeve": "Minerals/Energy", "weight": 6.3},
        {"ticker": "SYM", "name": "Symbotic", "sleeve": "Platform", "weight": 6.0},
        {"ticker": "AAPL", "name": "Apple", "sleeve": "Platform", "weight": 5.8},
        {"ticker": "NXE", "name": "NexGen Energy", "sleeve": "Minerals/Energy", "weight": 5.4},
        {"ticker": "CPRX", "name": "Catalyst Pharma", "sleeve": "Biotech/Pharma", "weight": 3.9},
        {"ticker": "AMAT", "name": "Applied Materials", "sleeve": "Platform", "weight": 3.1},
        {"ticker": "SF", "name": "Stifel Financial", "sleeve": "Financials", "weight": 3.1},
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        if isinstance(x, str):
            x = x.strip().replace(",", "")
            if x == "" or x.lower() in {"none", "nan", "na", "n/a"}:
                return np.nan
            return float(x)
        return float(x)
    except Exception:
        return np.nan

def z_to_01(x, xmin, xmax):
    if np.isnan(x): return np.nan
    if xmax == xmin: return 0.5
    return float(np.clip((x - xmin) / (xmax - xmin), 0.0, 1.0))

def inv_to_01(x, xmin, xmax):
    v = z_to_01(x, xmin, xmax)
    return np.nan if np.isnan(v) else 1.0 - v

def nanmean(vals):
    a = np.array(vals, dtype=float)
    return np.nan if np.all(np.isnan(a)) else float(np.nanmean(a))

def clean_forward_pe(x):
    x = safe_float(x)
    return np.nan if (np.isnan(x) or x <= 0) else x

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker: str):
    t = yf.Ticker(ticker)
    try:
        return t.get_info() or {}
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hist(ticker: str, period="2y"):
    t = yf.Ticker(ticker)
    try:
        h = t.history(period=period, auto_adjust=True)
        return h if h is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(ticker: str):
    t = yf.Ticker(ticker)
    try:
        inc = t.income_stmt if t.income_stmt is not None else pd.DataFrame()
    except Exception:
        inc = pd.DataFrame()
    try:
        cf = t.cashflow if t.cashflow is not None else pd.DataFrame()
    except Exception:
        cf = pd.DataFrame()
    try:
        bs = t.balance_sheet if t.balance_sheet is not None else pd.DataFrame()
    except Exception:
        bs = pd.DataFrame()
    return inc, cf, bs

def calc_mom_vol(hist: pd.DataFrame):
    if hist is None or hist.empty or "Close" not in hist.columns:
        return np.nan, np.nan
    c = hist["Close"].dropna()
    if len(c) < 60:
        return np.nan, np.nan
    k = min(126, len(c) - 1)
    mom = (c.iloc[-1] / c.iloc[-1 - k] - 1) if k > 0 else np.nan
    r = c.pct_change().dropna()
    vol = float(np.std(r) * np.sqrt(252)) if len(r) >= 60 else np.nan
    return float(mom), vol

def fcf_yield(info):
    fcf = safe_float(info.get("freeCashflow"))
    mcap = safe_float(info.get("marketCap"))
    if np.isnan(fcf) or np.isnan(mcap) or mcap <= 0:
        return np.nan
    return fcf / mcap

def cash_runway_months(bs: pd.DataFrame, cf: pd.DataFrame):
    try:
        cash = np.nan
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            for cand in ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash"]:
                if cand in bs.index:
                    cash = safe_float(bs.loc[cand].iloc[0])
                    break
        ocf = np.nan
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            for cand in ["Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow"]:
                if cand in cf.index:
                    ocf = safe_float(cf.loc[cand].iloc[0])
                    break
        if not np.isnan(cash) and not np.isnan(ocf) and ocf < 0:
            return (cash / abs(ocf)) * 12.0
    except Exception:
        pass
    return np.nan

def try_cagr_from_income_stmt(inc: pd.DataFrame, row_name_candidates, years=4):
    try:
        if inc is None or inc.empty:
            return np.nan
        row = None
        for cand in row_name_candidates:
            if cand in inc.index:
                row = cand
                break
        if row is None:
            return np.nan
        s = inc.loc[row].dropna().astype(float)
        if len(s) < 3:
            return np.nan
        n = min(years, len(s))
        start = s.iloc[n - 1]
        end = s.iloc[0]
        if start <= 0 or end <= 0:
            return np.nan
        return (end / start) ** (1 / (n - 1)) - 1
    except Exception:
        return np.nan

# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────
def score_growth(vals):
    eps = vals.get("eps_cagr_3y", np.nan)
    rev = vals.get("rev_cagr_3y", np.nan)
    s = nanmean([z_to_01(eps, -0.20, 0.40), z_to_01(rev, -0.10, 0.30)])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_quality(vals):
    roe = vals.get("roe", np.nan)
    opm = vals.get("oper_margin", np.nan)
    s = nanmean([z_to_01(roe, -0.10, 0.30), z_to_01(opm, -0.10, 0.35)])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_valuation(vals):
    fpe = vals.get("forward_pe", np.nan)
    tpe = vals.get("trailing_pe", np.nan)
    peg = vals.get("peg", np.nan)
    fcfy = vals.get("fcf_yield", np.nan)
    s = nanmean([inv_to_01(fpe, 5, 60), inv_to_01(tpe, 5, 60), inv_to_01(peg, 0.5, 3.0), z_to_01(fcfy, -0.02, 0.08)])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_momentum(vals):
    mom = vals.get("mom_6m", np.nan)
    s = z_to_01(mom, -0.40, 0.60)
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_convexity(vals, sleeve):
    vol = vals.get("vol_1y", np.nan)
    mcap = vals.get("mktcap", np.nan)
    s_vol = z_to_01(vol, 0.15, 0.90)
    s_size = np.nan
    if not np.isnan(mcap) and mcap > 0:
        s_size = inv_to_01(np.log10(mcap), 9.0, 12.0)  # 1B..1T
    base = {"Platform": 0.35, "Biotech/Pharma": 0.70, "Minerals/Energy": 0.70, "Financials": 0.25, "Other": 0.45}.get(sleeve, 0.45)
    s = nanmean([s_vol, s_size, base])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_risk(vals, sleeve):
    vol = vals.get("vol_1y", np.nan)
    nde = vals.get("net_debt_to_ebitda", np.nan)
    runway = vals.get("cash_runway_months", np.nan)

    vol_score = inv_to_01(vol, 0.15, 0.90)
    if sleeve in ["Biotech/Pharma", "Minerals/Energy"] and not np.isnan(vol_score):
        vol_score = 0.6 * vol_score + 0.4 * 0.5  # soften towards neutral

    nde_score = inv_to_01(nde, -1.0, 6.0)
    runway_score = z_to_01(runway, 0.0, 36.0)

    s = nanmean([vol_score, nde_score, runway_score])
    if np.isnan(s):
        return np.nan
    risk = float(np.clip(s * 100, 0, 100))

    # KO-cap runway < 6 months
    if not np.isnan(runway) and runway < 6:
        risk = min(risk, 35.0)

    return risk

def score_expectation_gap(vals):
    eps = vals.get("eps_cagr_3y", np.nan)
    rev = vals.get("rev_cagr_3y", np.nan)
    mom = vals.get("mom_6m", np.nan)

    expected = nanmean([eps, rev])
    fpe = vals.get("forward_pe", np.nan)

    implied = (1.0 / fpe) if (not np.isnan(fpe) and fpe > 0) else 0.0
    mom_tilt = 0.25 * mom if not np.isnan(mom) else 0.0

    gap = (expected if not np.isnan(expected) else 0.0) - implied + mom_tilt
    s = z_to_01(gap, -0.10, 0.30)
    return float(np.clip(s * 100, 0, 100)), expected, implied, gap

def compute_total_score(row: pd.Series):
    sleeve = row.get("sleeve", "Other")
    weights = dict(BASE_WEIGHTS.get(sleeve, BASE_WEIGHTS["Other"]))

    vals = {
        "eps_cagr_3y": row.get("eps_cagr_3y", np.nan),
        "rev_cagr_3y": row.get("rev_cagr_3y", np.nan),
        "roe": row.get("roe", np.nan),
        "oper_margin": row.get("oper_margin", np.nan),
        "forward_pe": row.get("forward_pe", np.nan),
        "trailing_pe": row.get("trailing_pe", np.nan),
        "peg": row.get("peg", np.nan),
        "fcf_yield": row.get("fcf_yield", np.nan),
        "mom_6m": row.get("mom_6m", np.nan),
        "vol_1y": row.get("vol_1y", np.nan),
        "mktcap": row.get("mktcap", np.nan),
        "net_debt_to_ebitda": row.get("net_debt_to_ebitda", np.nan),
        "cash_runway_months": row.get("cash_runway_months", np.nan),
    }

    subs = {}
    subs["growth"] = score_growth(vals)
    subs["quality"] = score_quality(vals)
    subs["valuation"] = score_valuation(vals)
    subs["momentum"] = score_momentum(vals)
    subs["convexity"] = score_convexity(vals, sleeve)
    subs["risk"] = score_risk(vals, sleeve)

    gap_score, exp_g, impl_g, gap_raw = score_expectation_gap(vals)
    subs["gap"] = gap_score

    # Sleeve adjustment: reduce risk weight and boost convexity for Biotech/Minerals
    if sleeve in ["Biotech/Pharma", "Minerals/Energy"]:
        weights["risk"] *= 0.60
        weights["convexity"] *= 1.15
        ssum = sum(weights.values())
        weights = {k: v / ssum for k, v in weights.items()}

    avail = {k: v for k, v in subs.items() if not np.isnan(v)}
    if not avail:
        return np.nan, subs, exp_g, impl_g, gap_raw

    wsum, wtot = 0.0, 0.0
    for k, v in subs.items():
        if np.isnan(v):
            continue
        wsum += weights.get(k, 0.0) * v
        wtot += weights.get(k, 0.0)
    total = wsum / (wtot if wtot > 0 else 1.0)
    return float(np.clip(total, 0, 100)), subs, exp_g, impl_g, gap_raw

def build_row(ticker: str, sleeve: str, weight: float, name_override: str = ""):
    info = fetch_info(ticker)
    hist = fetch_hist(ticker, "2y")
    inc, cf, bs = fetch_financials(ticker)

    mom, vol = calc_mom_vol(hist)

    row = {
        "ticker": ticker.upper().strip(),
        "name": (info.get("shortName") or info.get("longName") or ""),
        "sleeve": sleeve,
        "weight": weight,
        "price": safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
        "mktcap": safe_float(info.get("marketCap")),
        "trailing_pe": safe_float(info.get("trailingPE")),
        "forward_pe": clean_forward_pe(info.get("forwardPE")),
        "peg": safe_float(info.get("pegRatio")),
        "ps": safe_float(info.get("priceToSalesTrailing12Months")),
        "pb": safe_float(info.get("priceToBook")),
        "roe": safe_float(info.get("returnOnEquity")),
        "oper_margin": safe_float(info.get("operatingMargins")),
        "net_debt_to_ebitda": safe_float(info.get("netDebtToEBITDA")),
        "fcf_yield": fcf_yield(info),
        "rev_cagr_3y": np.nan,
        "eps_cagr_3y": np.nan,
        "cash_runway_months": cash_runway_months(bs, cf),
        "mom_6m": mom,
        "vol_1y": vol,
    }

    row["rev_cagr_3y"] = try_cagr_from_income_stmt(inc, ["Total Revenue", "TotalRevenue", "Total revenue"], years=4)
    row["eps_cagr_3y"] = try_cagr_from_income_stmt(inc, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"], years=4)

    if name_override.strip():
        row["name"] = name_override.strip()

    total, subs, exp_g, impl_g, gap_raw = compute_total_score(pd.Series(row))
    row["tanaka_score"] = total
    row["expected_growth"] = exp_g
    row["implied_growth"] = impl_g
    row["expectation_gap"] = gap_raw

    for k, v in subs.items():
        row[f"score_{k}"] = v

    return row

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Tanaka-Style Scorecard")
auto_fetch = st.sidebar.toggle("Auto-fetch (yfinance)", value=True)
normalize_weights = st.sidebar.toggle("Normalize weights to 100%", value=True)
refresh = st.sidebar.button("Refresh Data")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="small-note">Plotly version: Donut, Heatmap, Radar, Scatters. Fixes: neg. P/E filtered, sleeve-aware risk, runway KO-cap, expectation-gap.</div>',
    unsafe_allow_html=True,
)

if "portfolio" not in st.session_state or refresh:
    st.session_state["portfolio"] = DEFAULT_TOP10.copy()

# ─────────────────────────────────────────────────────────────────────────────
# UI: INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 TANAKA-Style Portfolio Scorecard")
st.caption("Interaktives, institutionelles KPI- & Scoring-Dashboard (Plotly Edition).")

st.subheader("1) Portfolio-Input (editierbar)")
edited = st.data_editor(
    st.session_state["portfolio"],
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "name": st.column_config.TextColumn("Name", width="medium"),
        "sleeve": st.column_config.SelectboxColumn("Sleeve", options=SLEEVES, width="medium"),
        "weight": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
    },
    hide_index=True,
)
st.session_state["portfolio"] = edited

df_in = st.session_state["portfolio"].copy()
df_in["ticker"] = df_in["ticker"].astype(str).str.upper().str.strip()
df_in = df_in[df_in["ticker"].str.len() > 0].reset_index(drop=True)

if df_in.empty:
    st.warning("Bitte mindestens einen Ticker eintragen.")
    st.stop()

# Normalize weights
w = df_in["weight"].apply(safe_float).fillna(0.0).values
ws = float(np.sum(w))
if normalize_weights and ws > 0:
    df_in["weight"] = (w / ws) * 100.0

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("2) KPIs & Tanaka-Score")

rows = []
with st.spinner("Loading fundamentals & computing scores …"):
    for _, r in df_in.iterrows():
        tkr = r["ticker"]
        sleeve = r.get("sleeve", "Other")
        if sleeve not in SLEEVES:
            sleeve = "Other"
        wt = safe_float(r.get("weight", np.nan))
        nm = r.get("name", "")
        if auto_fetch:
            rows.append(build_row(tkr, sleeve, wt, name_override=nm))
        else:
            rows.append({"ticker": tkr, "name": nm, "sleeve": sleeve, "weight": wt})

df = pd.DataFrame(rows)

needed_cols = [
    "ticker","name","sleeve","weight","price","mktcap",
    "forward_pe","trailing_pe","peg","ps","pb","fcf_yield",
    "rev_cagr_3y","eps_cagr_3y","oper_margin","roe",
    "mom_6m","vol_1y","net_debt_to_ebitda","cash_runway_months",
    "expected_growth","implied_growth","expectation_gap",
    "tanaka_score","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap"
]
for c in needed_cols:
    if c not in df.columns:
        df[c] = np.nan

df["weight_dec"] = df["weight"].apply(safe_float) / 100.0
port_score = float(np.nansum(df["tanaka_score"] * df["weight_dec"])) if df["tanaka_score"].notna().any() else np.nan

m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Portfolio Tanaka Score (wtd.)", f"{port_score:.1f}" if not np.isnan(port_score) else "—")
m2.metric("Names", f"{len(df)}")
top_sleeve = df.groupby("sleeve")["weight"].sum().sort_values(ascending=False).index[0]
m3.metric("Top Sleeve", top_sleeve)
coverage = int(df["tanaka_score"].notna().sum())
m4.metric("Score Coverage", f"{coverage}/{len(df)}")

st.dataframe(df[needed_cols].sort_values("weight", ascending=False), use_container_width=True, hide_index=True)

st.download_button(
    "Download KPI Table (CSV)",
    df[needed_cols].to_csv(index=False).encode("utf-8"),
    file_name="tanaka_kpi_scorecard.csv",
    mime="text/csv",
)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("3) Charts (Plotly)")

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    sleeve_w = df.groupby("sleeve", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
    fig = px.pie(sleeve_w, names="sleeve", values="weight", hole=0.55, title="Sleeve Allocation (%)")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    d = df.copy()
    d["wtd_contrib"] = d["tanaka_score"] * d["weight_dec"]
    d = d.sort_values("wtd_contrib", ascending=False)
    fig = px.bar(
        d,
        x="wtd_contrib",
        y="ticker",
        orientation="h",
        title="Weighted Score Contribution (Score × Weight)",
        hover_data=["name", "sleeve", "tanaka_score", "weight"],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns([1.25, 0.75], gap="large")

with c3:
    g = df["eps_cagr_3y"].where(df["eps_cagr_3y"].notna(), df["rev_cagr_3y"])
    scatter = df.copy()
    scatter["growth_proxy"] = g
    fig = px.scatter(
        scatter,
        x="forward_pe",
        y="growth_proxy",
        size="weight",
        color="sleeve",
        hover_name="ticker",
        hover_data={"name": True, "tanaka_score": True, "weight": True, "forward_pe": True, "growth_proxy": True},
        title="Valuation vs Growth (proxy) — 'Undervalued Growth' Map",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.write("**Radar: Sub-scores**")
    pick = st.selectbox("Select ticker", df["ticker"].tolist(), index=0)
    rr = df[df["ticker"] == pick].iloc[0]

    cats = ["Growth", "Quality", "Valuation", "Momentum", "Convexity", "Risk", "Gap"]
    vals = [
        safe_float(rr.get("score_growth", np.nan)),
        safe_float(rr.get("score_quality", np.nan)),
        safe_float(rr.get("score_valuation", np.nan)),
        safe_float(rr.get("score_momentum", np.nan)),
        safe_float(rr.get("score_convexity", np.nan)),
        safe_float(rr.get("score_risk", np.nan)),
        safe_float(rr.get("score_gap", np.nan)),
    ]
    cats2 = cats + [cats[0]]
    vals2 = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals2, theta=cats2, fill="toself", name=pick))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# Expectation vs Implied Growth
st.markdown("---")
st.subheader("4) Expectation-Gap Overlay")

eg = df.copy()
fig = px.scatter(
    eg,
    x="implied_growth",
    y="expected_growth",
    size="weight",
    color="sleeve",
    hover_name="ticker",
    hover_data={"name": True, "tanaka_score": True, "expectation_gap": True},
    title="Expected vs Implied Growth (Tanaka Expectation-Gap Overlay)",
)
fig.add_shape(type="line", x0=0, y0=0, x1=0.30, y1=0.30, line=dict(dash="dash"))
fig.update_xaxes(tickformat=".0%", range=[0, 0.30])
fig.update_yaxes(tickformat=".0%", range=[-0.10, 0.40])
fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# Heatmap
st.markdown("---")
st.subheader("5) Score Heatmap (0–100)")

heat = df[["ticker", "score_growth", "score_quality", "score_valuation", "score_momentum", "score_convexity", "score_risk", "score_gap", "tanaka_score"]].set_index("ticker")
fig = px.imshow(heat.T, aspect="auto", title="Sub-scores and Total Score (0–100)")
fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption("Research/education dashboard. Fundamentals coverage varies by yfinance; missing values are normal.")
