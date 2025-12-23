# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# TANAKA-STYLE SCORECARD (Top-10 / Any Portfolio) – FULL VERSION (FIXED)
#
# Fixes & improvements included:
# 1) NO Plotly dependency (matplotlib only) → runs on minimal Streamlit Cloud
# 2) Professional LIGHT KPI cards (no black blocks) + clean layout
# 3) Negative forward P/E handling: forward_pe <= 0 → NaN (prevents nonsense scoring)
# 4) Volatility treatment: for Biotech/Minerals, vol is NOT double-penalized
#    - Risk weight reduced, Convexity weight increased (sleeve-aware)
# 5) Cash runway is KO-style cap, not linear to zero:
#    - if runway < 6 months → risk score capped (not annihilated)
# 6) Expectation-Gap overlay (Tanaka-style):
#    - Implied growth proxy ≈ 1 / forward_pe
#    - Expected growth proxy from EPS/Rev + momentum
#    - Gap score added into total (0–20) via 'score_gap'
# 7) Professional charts:
#    - Sleeve allocation donut
#    - Weighted score contribution bars
#    - Valuation vs Growth scatter
#    - EXPECTATION vs IMPLIED growth scatter
#    - Heatmap-like table of subscores
#
# Install:
#   pip install streamlit yfinance pandas numpy matplotlib
#
# Run:
#   streamlit run streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS (LIGHT, professional)
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
      .section-title { margin-top: 0.4rem; margin-bottom: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SLEEVES = ["Platform", "Biotech/Pharma", "Minerals/Energy", "Financials", "Other"]

# Base weights sum to 1.0
BASE_WEIGHTS = {
    "Platform": {"growth": 0.20, "quality": 0.22, "valuation": 0.18, "momentum": 0.10, "convexity": 0.08, "risk": 0.10, "gap": 0.12},
    "Biotech/Pharma": {"growth": 0.14, "quality": 0.10, "valuation": 0.10, "momentum": 0.08, "convexity": 0.22, "risk": 0.10, "gap": 0.26},
    "Minerals/Energy": {"growth": 0.10, "quality": 0.08, "valuation": 0.14, "momentum": 0.08, "convexity": 0.20, "risk": 0.16, "gap": 0.24},
    "Financials": {"growth": 0.12, "quality": 0.18, "valuation": 0.22, "momentum": 0.10, "convexity": 0.06, "risk": 0.18, "gap": 0.14},
    "Other": {"growth": 0.16, "quality": 0.16, "valuation": 0.16, "momentum": 0.10, "convexity": 0.12, "risk": 0.14, "gap": 0.16},
}

# Top-10 from your factsheet (editable in UI)
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
    if np.isnan(x):
        return np.nan
    if xmax == xmin:
        return 0.5
    return float(np.clip((x - xmin) / (xmax - xmin), 0.0, 1.0))


def inv_to_01(x, xmin, xmax):
    v = z_to_01(x, xmin, xmax)
    if np.isnan(v):
        return np.nan
    return 1.0 - v


def nanmean(vals):
    a = np.array(vals, dtype=float)
    return np.nan if np.all(np.isnan(a)) else float(np.nanmean(a))


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
    k = min(126, len(c) - 1)  # ~6m
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
    # runway ~ cash / |operating cashflow| * 12
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
    # yfinance: rows are metrics, columns are periods; take up to 'years' points
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
def clean_forward_pe(x):
    x = safe_float(x)
    # Critical fix: negative or zero forward P/E is meaningless -> treat as missing
    if np.isnan(x) or x <= 0:
        return np.nan
    return x


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

    s = nanmean(
        [
            inv_to_01(fpe, 5, 60),
            inv_to_01(tpe, 5, 60),
            inv_to_01(peg, 0.5, 3.0),
            z_to_01(fcfy, -0.02, 0.08),
        ]
    )
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
        # smaller -> more optionality
        s_size = inv_to_01(np.log10(mcap), 9.0, 12.0)  # 1B..1T
    base = {"Platform": 0.35, "Biotech/Pharma": 0.70, "Minerals/Energy": 0.70, "Financials": 0.25, "Other": 0.45}.get(
        sleeve, 0.45
    )
    s = nanmean([s_vol, s_size, base])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))


def score_risk(vals, sleeve):
    # Sleeve-aware: for Biotech/Minerals, volatility is not penalized as hard (it's part of convexity)
    vol = vals.get("vol_1y", np.nan)
    nde = vals.get("net_debt_to_ebitda", np.nan)
    runway = vals.get("cash_runway_months", np.nan)

    # vol: lower better (but softened for biotech/minerals)
    vol_score = inv_to_01(vol, 0.15, 0.90)
    if sleeve in ["Biotech/Pharma", "Minerals/Energy"] and not np.isnan(vol_score):
        vol_score = 0.6 * vol_score + 0.4 * 0.5  # pull towards neutral

    nde_score = inv_to_01(nde, -1.0, 6.0)
    runway_score = z_to_01(runway, 0.0, 36.0)

    s = nanmean([vol_score, nde_score, runway_score])
    if np.isnan(s):
        return np.nan

    risk = float(np.clip(s * 100, 0, 100))

    # KO-style runway: if runway < 6 months, cap risk score (not annihilate)
    if not np.isnan(runway) and runway < 6:
        risk = min(risk, 35.0)

    return risk


def score_expectation_gap(vals):
    # Expected growth proxy: blend of EPS CAGR, Rev CAGR, and 6M momentum
    eps = vals.get("eps_cagr_3y", np.nan)
    rev = vals.get("rev_cagr_3y", np.nan)
    mom = vals.get("mom_6m", np.nan)

    expected = nanmean([eps, rev])  # if one missing, uses the other
    if np.isnan(expected):
        expected = np.nan

    # Implied growth proxy: ~ 1 / forward P/E (very rough, but useful)
    fpe = vals.get("forward_pe", np.nan)
    implied = np.nan
    if not np.isnan(fpe) and fpe > 0:
        implied = 1.0 / fpe

    # If implied missing (no earnings), assume market-implied ~0 (common for pre-profit)
    if np.isnan(implied):
        implied = 0.0

    # Add momentum tilt (revision proxy)
    mom_tilt = 0.25 * mom if not np.isnan(mom) else 0.0

    # GAP: expected - implied + momentum tilt
    gap = (expected if not np.isnan(expected) else 0.0) - implied + mom_tilt

    # Map gap into 0..100.
    # Range: -10%..+30% gap -> 0..1
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

    # sleeve-aware: for Biotech/Minerals, reduce risk weight, boost convexity weight a bit
    if sleeve in ["Biotech/Pharma", "Minerals/Energy"]:
        weights["risk"] *= 0.60
        weights["convexity"] *= 1.15
        # renormalize
        ssum = sum(weights.values())
        weights = {k: v / ssum for k, v in weights.items()}

    # Weighted sum with NaN handling: renormalize on available subs
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

    fpe = clean_forward_pe(info.get("forwardPE"))
    tpe = safe_float(info.get("trailingPE"))

    row = {
        "ticker": ticker.upper().strip(),
        "name": (info.get("shortName") or info.get("longName") or ""),
        "sleeve": sleeve,
        "weight": weight,
        "price": safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
        "mktcap": safe_float(info.get("marketCap")),
        "trailing_pe": tpe,
        "forward_pe": fpe,
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

    # Optional: derive revenue/EPS CAGR from income statement when available
    row["rev_cagr_3y"] = try_cagr_from_income_stmt(inc, ["Total Revenue", "TotalRevenue", "Total revenue"], years=4)
    row["eps_cagr_3y"] = try_cagr_from_income_stmt(inc, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"], years=4)

    if name_override.strip():
        row["name"] = name_override.strip()

    total, subs, exp_g, impl_g, gap_raw = compute_total_score(pd.Series(row))
    row["tanaka_score"] = total
    row["expected_growth"] = exp_g
    row["implied_growth"] = impl_g
    row["expectation_gap"] = gap_raw

    # store subscores
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
st.sidebar.markdown('<div class="small-note">Fixes: negative P/E handling, sleeve-aware risk/convexity, runway KO-cap, expectation-gap overlay.</div>', unsafe_allow_html=True)

if "portfolio" not in st.session_state or refresh:
    st.session_state["portfolio"] = DEFAULT_TOP10.copy()

# ─────────────────────────────────────────────────────────────────────────────
# UI: INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 TANAKA-Style Portfolio Scorecard")
st.caption("Professionelles KPI- & Scoring-Dashboard (Tanaka-nah). Läuft stabil ohne Plotly.")

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

# normalize weights
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
            try:
                rows.append(build_row(tkr, sleeve, wt, name_override=nm))
            except Exception:
                rows.append({"ticker": tkr, "name": nm, "sleeve": sleeve, "weight": wt})
        else:
            rows.append({"ticker": tkr, "name": nm, "sleeve": sleeve, "weight": wt})

df = pd.DataFrame(rows)

# Ensure columns
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

# KPIs top cards
m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Portfolio Tanaka Score (wtd.)", f"{port_score:.1f}" if not np.isnan(port_score) else "—")
m2.metric("Names", f"{len(df)}")
top_sleeve = df.groupby("sleeve")["weight"].sum().sort_values(ascending=False).index[0]
m3.metric("Top Sleeve", top_sleeve)
coverage = int(df["tanaka_score"].notna().sum())
m4.metric("Score Coverage", f"{coverage}/{len(df)}")

# Table formatting for display
def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x*100:.1f}%"
def fmt_num(x, d=2):
    return "—" if pd.isna(x) else f"{float(x):.{d}f}"
def fmt_money(x):
    if pd.isna(x): return "—"
    x = float(x)
    if x >= 1e12: return f"${x/1e12:.2f}T"
    if x >= 1e9: return f"${x/1e9:.1f}B"
    if x >= 1e6: return f"${x/1e6:.1f}M"
    return f"${x:,.0f}"

df_view = df[needed_cols].copy()
df_view["weight"] = df_view["weight"].apply(lambda x: "—" if pd.isna(x) else f"{x:.1f}%")
df_view["price"] = df_view["price"].apply(lambda x: fmt_num(x, 2))
df_view["mktcap"] = df_view["mktcap"].apply(fmt_money)
df_view["forward_pe"] = df_view["forward_pe"].apply(lambda x: fmt_num(x, 1))
df_view["trailing_pe"] = df_view["trailing_pe"].apply(lambda x: fmt_num(x, 1))
df_view["peg"] = df_view["peg"].apply(lambda x: fmt_num(x, 2))
df_view["ps"] = df_view["ps"].apply(lambda x: fmt_num(x, 2))
df_view["pb"] = df_view["pb"].apply(lambda x: fmt_num(x, 2))
df_view["fcf_yield"] = df_view["fcf_yield"].apply(fmt_pct)
df_view["rev_cagr_3y"] = df_view["rev_cagr_3y"].apply(fmt_pct)
df_view["eps_cagr_3y"] = df_view["eps_cagr_3y"].apply(fmt_pct)
df_view["oper_margin"] = df_view["oper_margin"].apply(fmt_pct)
df_view["roe"] = df_view["roe"].apply(fmt_pct)
df_view["mom_6m"] = df_view["mom_6m"].apply(fmt_pct)
df_view["vol_1y"] = df_view["vol_1y"].apply(fmt_pct)
df_view["net_debt_to_ebitda"] = df_view["net_debt_to_ebitda"].apply(lambda x: fmt_num(x, 2))
df_view["cash_runway_months"] = df_view["cash_runway_months"].apply(lambda x: "—" if pd.isna(x) else f"{float(x):.0f}")
df_view["expected_growth"] = df_view["expected_growth"].apply(fmt_pct)
df_view["implied_growth"] = df_view["implied_growth"].apply(fmt_pct)
df_view["expectation_gap"] = df_view["expectation_gap"].apply(fmt_pct)

for c in ["tanaka_score","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap"]:
    df_view[c] = df_view[c].apply(lambda x: "—" if pd.isna(x) else f"{float(x):.0f}")

# Sort by weight or score
sort_opt = st.selectbox("Sort table by", ["weight (desc)", "tanaka_score (desc)"], index=0)
if sort_opt.startswith("weight"):
    df_show = df_view.sort_values(by="weight", ascending=False)
else:
    # Need numeric for sorting; use original df
    tmp = df.copy()
    tmp["_sort"] = tmp["tanaka_score"].fillna(-1e9)
    tmp = tmp.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    df_show = tmp[needed_cols].copy()
    # reformat again quickly
    df_show = df_show.merge(df_view[["ticker","tanaka_score"]], on="ticker", suffixes=("","_v"))
    # keep view already formatted:
    df_show = df_view.loc[df_view["ticker"].isin(tmp["ticker"])].set_index("ticker").loc[tmp["ticker"]].reset_index()

st.dataframe(df_show, use_container_width=True, hide_index=True)

st.download_button(
    "Download KPI Table (CSV)",
    df[needed_cols].to_csv(index=False).encode("utf-8"),
    file_name="tanaka_kpi_scorecard.csv",
    mime="text/csv",
)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("3) Charts")

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.write("**Sleeve Allocation (%)**")
    sleeve_w = df.groupby("sleeve")["weight"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.pie(sleeve_w.values, labels=sleeve_w.index, autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.45))
    ax.axis("equal")
    st.pyplot(fig, clear_figure=True)

with c2:
    st.write("**Weighted Score Contribution (Score × Weight)**")
    d = df.copy()
    d["wtd_contrib"] = d["tanaka_score"] * d["weight_dec"]
    d = d.sort_values("wtd_contrib", ascending=True)
    fig, ax = plt.subplots()
    ax.barh(d["ticker"], d["wtd_contrib"])
    ax.set_xlabel("Contribution")
    ax.grid(True, axis="x", alpha=0.25)
    st.pyplot(fig, clear_figure=True)

# Valuation vs Growth
st.write("**Valuation vs Growth (proxy)**")
g = df["eps_cagr_3y"].copy()
g = g.where(~g.isna(), df["rev_cagr_3y"])
x = df["forward_pe"].astype(float)
y = g.astype(float)
sizes = np.clip(df["weight"].astype(float).fillna(1.0).values, 1.0, 30.0) * 22

fig, ax = plt.subplots()
ax.scatter(x, y, s=sizes, alpha=0.75)
for i, t in enumerate(df["ticker"].tolist()):
    if pd.isna(x.iloc[i]) or pd.isna(y.iloc[i]):
        continue
    ax.annotate(t, (x.iloc[i], y.iloc[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Forward P/E (<=0 filtered)")
ax.set_ylabel("Growth proxy (EPS CAGR; fallback Rev CAGR)")
ax.grid(True, alpha=0.25)
st.pyplot(fig, clear_figure=True)

# Expectation vs Implied Growth (Tanaka overlay)
st.write("**Expectation vs Implied Growth (Tanaka Expectation-Gap Overlay)**")
ex = df["expected_growth"].astype(float)
im = df["implied_growth"].astype(float)

fig, ax = plt.subplots()
ax.scatter(im, ex, s=sizes, alpha=0.75)
for i, t in enumerate(df["ticker"].tolist()):
    if pd.isna(im.iloc[i]) or pd.isna(ex.iloc[i]):
        continue
    ax.annotate(t, (im.iloc[i], ex.iloc[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.plot([0, 0.30], [0, 0.30], linestyle="--", linewidth=1)  # parity line
ax.set_xlim(0, 0.30)
ax.set_ylim(-0.10, 0.40)
ax.set_xlabel("Implied growth proxy (~1/Forward P/E)")
ax.set_ylabel("Expected growth proxy (EPS/Rev blend)")
ax.grid(True, alpha=0.25)
st.pyplot(fig, clear_figure=True)

# Heatmap-like view using styled table (no seaborn; keep it stable)
st.markdown("---")
st.subheader("4) Subscore Matrix (0–100)")

score_cols = ["score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap","tanaka_score"]
mat = df[["ticker"] + score_cols].copy().set_index("ticker")
st.dataframe(mat.round(0), use_container_width=True)

st.caption("Dashboard: Research/education use. Data coverage varies by yfinance; missing KPIs are expected.")
