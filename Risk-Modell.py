# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# TANAKA-STYLE SCORECARD (Top-10 / Any Portfolio)
# Professional Streamlit dashboard with:
# - Portfolio table (editable)
# - Auto fundamentals via Yahoo Finance (yfinance) + robust fallbacks
# - Tanaka-style KPI score (0–100) with sleeve-specific weighting
# - Clean charts: Heatmap, Radar, Scatter (Valuation vs Growth), Waterfall-like bars
# - Downloads (CSV)
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
#
# Run:
#   streamlit run streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────

import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# UI CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tanaka-Style Scorecard",
    page_icon="📈",
    layout="wide",
)

# Minimal CSS polish (no crazy theming; stays professional)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] { background: #0b1220; border: 1px solid rgba(255,255,255,0.08);
        padding: 10px 12px; border-radius: 14px; }
      div[data-testid="stMetric"] > label { color: rgba(255,255,255,0.75) !important; }
      .small-note { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# TANAKA-STYLE KPI MODEL (reconstructed)
# ─────────────────────────────────────────────────────────────────────────────

SLEEVES = ["Platform", "Biotech/Pharma", "Minerals/Energy", "Financials", "Other"]

# Sleeve-specific weights (sum to 1.0)
# Goal: mimic how Tanaka would likely emphasize different KPIs per “cluster”
SLEEVE_WEIGHTS = {
    "Platform": {
        "growth": 0.22,      # EPS/Rev growth, inflection proxies
        "quality": 0.24,     # margins, ROE/ROIC proxy
        "valuation": 0.18,   # forward P/E, PEG, FCF yield
        "momentum": 0.12,    # revisions proxy (fallback: price momentum)
        "convexity": 0.10,   # optionality (lower for platform)
        "risk": 0.14,        # balance sheet + drawdown sensitivity
    },
    "Biotech/Pharma": {
        "growth": 0.18,
        "quality": 0.14,
        "valuation": 0.12,
        "momentum": 0.10,
        "convexity": 0.30,   # catalysts dominate
        "risk": 0.16,        # cash runway / leverage / vol
    },
    "Minerals/Energy": {
        "growth": 0.12,
        "quality": 0.10,
        "valuation": 0.16,
        "momentum": 0.10,
        "convexity": 0.28,   # scarcity/regime optionality
        "risk": 0.24,        # cycle + balance sheet risk
    },
    "Financials": {
        "growth": 0.16,
        "quality": 0.22,
        "valuation": 0.24,
        "momentum": 0.12,
        "convexity": 0.06,
        "risk": 0.20,
    },
    "Other": {
        "growth": 0.18,
        "quality": 0.18,
        "valuation": 0.18,
        "momentum": 0.14,
        "convexity": 0.14,
        "risk": 0.18,
    },
}

# KPI inputs we try to fetch (many may be missing; we degrade gracefully)
KPI_COLUMNS = [
    "ticker",
    "name",
    "sleeve",
    "weight",
    "price",
    "mktcap",
    "trailing_pe",
    "forward_pe",
    "peg",
    "ps",
    "pb",
    "fcf_yield",
    "roe",
    "oper_margin",
    "rev_cagr_3y",
    "eps_cagr_3y",
    "mom_6m",
    "vol_1y",
    "net_debt_to_ebitda",
    "cash_runway_months",
    "tanaka_score",
    "score_growth",
    "score_quality",
    "score_valuation",
    "score_momentum",
    "score_convexity",
    "score_risk",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: robust numeric parsing & scoring functions
# ─────────────────────────────────────────────────────────────────────────────

def safe_float(x) -> float:
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


def clip01(v: float) -> float:
    if np.isnan(v):
        return np.nan
    return float(np.clip(v, 0.0, 1.0))


def z_to_01(z: float, zmin: float, zmax: float) -> float:
    """Map metric to 0..1 with linear clamp."""
    if np.isnan(z):
        return np.nan
    if zmax == zmin:
        return 0.5
    return float(np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0))


def inv_to_01(z: float, zmin: float, zmax: float) -> float:
    """Inverse mapping: smaller is better."""
    v = z_to_01(z, zmin, zmax)
    if np.isnan(v):
        return np.nan
    return 1.0 - v


def nanmean_safe(vals: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def score_bucket(values: Dict[str, float], bucket: str, sleeve: str) -> float:
    """
    Convert raw KPIs into 0..100 sub-scores per bucket.
    Uses heuristic ranges typical in public equities.
    """
    # Growth proxies
    if bucket == "growth":
        # prefer eps_cagr_3y, rev_cagr_3y
        eps = values.get("eps_cagr_3y", np.nan)
        rev = values.get("rev_cagr_3y", np.nan)
        # Map: -20%..+40% -> 0..1
        s_eps = z_to_01(eps, -0.20, 0.40)
        s_rev = z_to_01(rev, -0.10, 0.30)
        s = nanmean_safe([s_eps, s_rev])
        return float(np.clip(s * 100.0, 0, 100))

    # Quality proxies
    if bucket == "quality":
        roe = values.get("roe", np.nan)
        opm = values.get("oper_margin", np.nan)
        # ROE: -10%..+30%
        s_roe = z_to_01(roe, -0.10, 0.30)
        # Op margin: -10%..+35%
        s_opm = z_to_01(opm, -0.10, 0.35)
        s = nanmean_safe([s_roe, s_opm])
        return float(np.clip(s * 100.0, 0, 100))

    # Valuation proxies (lower multiples better; higher FCF yield better)
    if bucket == "valuation":
        fpe = values.get("forward_pe", np.nan)
        tpe = values.get("trailing_pe", np.nan)
        peg = values.get("peg", np.nan)
        fcfy = values.get("fcf_yield", np.nan)

        # PE: 5..60 (lower better)
        s_fpe = inv_to_01(fpe, 5, 60)
        s_tpe = inv_to_01(tpe, 5, 60)
        # PEG: 0.5..3.0 (lower better; NaN is common)
        s_peg = inv_to_01(peg, 0.5, 3.0)
        # FCF yield: -2%..+8% (higher better)
        s_fcfy = z_to_01(fcfy, -0.02, 0.08)

        s = nanmean_safe([s_fpe, s_tpe, s_peg, s_fcfy])
        return float(np.clip(s * 100.0, 0, 100))

    # Momentum (proxy for revisions when we don’t have analyst revisions)
    if bucket == "momentum":
        mom = values.get("mom_6m", np.nan)
        # -40%..+60%
        s_mom = z_to_01(mom, -0.40, 0.60)
        return float(np.clip(s_mom * 100.0, 0, 100))

    # Convexity – in real life this is qualitative (catalysts, optionality).
    # Here we approximate with (volatility + sleeve bias + “small cap optionality”)
    if bucket == "convexity":
        vol = values.get("vol_1y", np.nan)
        mktcap = values.get("mktcap", np.nan)

        # vol: 15%..90% -> higher = more convex optionality (not always “good”, but matches sleeve logic)
        s_vol = z_to_01(vol, 0.15, 0.90)
        # mktcap: smaller tends to have more optionality (inverse). Use $1B..$1T
        if not np.isnan(mktcap):
            s_size = inv_to_01(np.log10(max(mktcap, 1.0)), 9.0, 12.0)  # log10(1B)=9, log10(1T)=12
        else:
            s_size = np.nan

        # Sleeve bias: biotech/minerals get a base bump, platform gets lower
        base = {
            "Platform": 0.35,
            "Biotech/Pharma": 0.65,
            "Minerals/Energy": 0.65,
            "Financials": 0.25,
            "Other": 0.45,
        }.get(sleeve, 0.45)

        s = nanmean_safe([s_vol, s_size, base])
        return float(np.clip(s * 100.0, 0, 100))

    # Risk (lower vol/leverage better; higher cash runway better)
    if bucket == "risk":
        vol = values.get("vol_1y", np.nan)
        nde = values.get("net_debt_to_ebitda", np.nan)
        runway = values.get("cash_runway_months", np.nan)

        # vol: 15%..90 (lower better)
        s_vol = inv_to_01(vol, 0.15, 0.90)
        # net debt/EBITDA: -1..6 (lower better; negative is net cash)
        s_nde = inv_to_01(nde, -1.0, 6.0)
        # runway: 0..36 months (higher better)
        s_runway = z_to_01(runway, 0.0, 36.0)

        s = nanmean_safe([s_vol, s_nde, s_runway])
        return float(np.clip(s * 100.0, 0, 100))

    return np.nan


def compute_tanaka_score(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    sleeve = row.get("sleeve", "Other")
    w = SLEEVE_WEIGHTS.get(sleeve, SLEEVE_WEIGHTS["Other"])

    values = {
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
    for b in ["growth", "quality", "valuation", "momentum", "convexity", "risk"]:
        subs[b] = score_bucket(values, b, sleeve)

    # Weighted sum; ignore NaNs by re-normalizing weights of available subscores
    available = {k: v for k, v in subs.items() if not np.isnan(v)}
    if len(available) == 0:
        return np.nan, subs

    wsum = 0.0
    wtot = 0.0
    for k, v in subs.items():
        if np.isnan(v):
            continue
        wsum += w.get(k, 0.0) * v
        wtot += w.get(k, 0.0)

    score = wsum / (wtot if wtot > 0 else 1.0)
    return float(np.clip(score, 0, 100)), subs


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH (robust) – yfinance with fallbacks
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_info(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    try:
        info = t.get_info()
    except Exception:
        info = {}
    return info or {}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period=period, auto_adjust=True)
    except Exception:
        hist = pd.DataFrame()
    if hist is None:
        hist = pd.DataFrame()
    return hist


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_financials(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def cagr(series: pd.Series) -> float:
    """CAGR from first to last, assuming annual spacing in yfinance financials."""
    if series is None or len(series) < 2:
        return np.nan
    vals = series.dropna().astype(float)
    if len(vals) < 2:
        return np.nan
    start = vals.iloc[-1]  # yfinance often has columns as dates, but in DF rows; we handle in caller
    end = vals.iloc[0]
    if start <= 0 or end <= 0:
        return np.nan
    n = len(vals) - 1
    return (end / start) ** (1 / n) - 1


def calc_momentum_vol(hist: pd.DataFrame) -> Tuple[float, float]:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return np.nan, np.nan
    close = hist["Close"].dropna()
    if len(close) < 50:
        return np.nan, np.nan

    # 6M momentum ~ 126 trading days
    k = min(126, len(close) - 1)
    mom_6m = close.iloc[-1] / close.iloc[-1 - k] - 1 if k > 0 else np.nan

    # 1Y vol from daily returns, annualized
    rets = close.pct_change().dropna()
    if len(rets) < 50:
        vol_1y = np.nan
    else:
        vol_1y = float(np.std(rets) * np.sqrt(252))
    return float(mom_6m), vol_1y


def extract_financial_kpis(info: Dict[str, Any], inc: pd.DataFrame, cf: pd.DataFrame, bs: pd.DataFrame) -> Dict[str, float]:
    out = {
        "mktcap": np.nan,
        "trailing_pe": np.nan,
        "forward_pe": np.nan,
        "peg": np.nan,
        "ps": np.nan,
        "pb": np.nan,
        "roe": np.nan,
        "oper_margin": np.nan,
        "net_debt_to_ebitda": np.nan,
        "fcf_yield": np.nan,
        "rev_cagr_3y": np.nan,
        "eps_cagr_3y": np.nan,
        "cash_runway_months": np.nan,
        "price": np.nan,
    }

    # Info-based (often available)
    out["mktcap"] = safe_float(info.get("marketCap"))
    out["trailing_pe"] = safe_float(info.get("trailingPE"))
    out["forward_pe"] = safe_float(info.get("forwardPE"))
    out["peg"] = safe_float(info.get("pegRatio"))
    out["ps"] = safe_float(info.get("priceToSalesTrailing12Months"))
    out["pb"] = safe_float(info.get("priceToBook"))
    out["roe"] = safe_float(info.get("returnOnEquity"))
    out["oper_margin"] = safe_float(info.get("operatingMargins"))
    out["price"] = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

    # Net debt / EBITDA proxy if present
    nde = info.get("netDebtToEBITDA")
    out["net_debt_to_ebitda"] = safe_float(nde)

    # FCF yield approximation: freeCashflow / marketCap (info sometimes has freeCashflow)
    fcf = safe_float(info.get("freeCashflow"))
    if not np.isnan(fcf) and not np.isnan(out["mktcap"]) and out["mktcap"] > 0:
        out["fcf_yield"] = fcf / out["mktcap"]
    else:
        out["fcf_yield"] = np.nan

    # Revenue CAGR (3y) from income statement if possible
    # yfinance income_stmt: index rows are metrics, columns are periods (dates)
    try:
        if isinstance(inc, pd.DataFrame) and not inc.empty:
            # Possible row names: "Total Revenue" or "TotalRevenue"
            rev_row = None
            for cand in ["Total Revenue", "TotalRevenue", "Total revenue"]:
                if cand in inc.index:
                    rev_row = cand
                    break
            if rev_row:
                rev_series = inc.loc[rev_row].dropna()
                # Need at least 3-4 points to compute 3Y-ish CAGR; we approximate using available points
                if len(rev_series) >= 3:
                    out["rev_cagr_3y"] = cagr(rev_series.iloc[:4])  # use up to 4 annual points
    except Exception:
        pass

    # EPS CAGR (3y) – yfinance often lacks EPS time series in income_stmt reliably.
    # Try 'Diluted EPS' or 'Basic EPS' rows if available.
    try:
        if isinstance(inc, pd.DataFrame) and not inc.empty:
            eps_row = None
            for cand in ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"]:
                if cand in inc.index:
                    eps_row = cand
                    break
            if eps_row:
                eps_series = inc.loc[eps_row].dropna()
                if len(eps_series) >= 3:
                    out["eps_cagr_3y"] = cagr(eps_series.iloc[:4])
    except Exception:
        pass

    # Cash runway (months) for pre-profit names: cash / |operating cashflow|
    # If cashflow has "Total Cash From Operating Activities" (or similar)
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
            out["cash_runway_months"] = (cash / abs(ocf)) * 12.0
    except Exception:
        pass

    return out


def build_row(ticker: str, sleeve: str, weight: float) -> Dict[str, Any]:
    info = fetch_info(ticker)
    hist = fetch_history(ticker, period="2y")
    inc, cf, bs = fetch_financials(ticker)

    kpis = extract_financial_kpis(info, inc, cf, bs)
    mom_6m, vol_1y = calc_momentum_vol(hist)

    name = info.get("shortName") or info.get("longName") or ""

    row = {
        "ticker": ticker.upper().strip(),
        "name": name,
        "sleeve": sleeve,
        "weight": weight,
        "price": kpis.get("price", np.nan),
        "mktcap": kpis.get("mktcap", np.nan),
        "trailing_pe": kpis.get("trailing_pe", np.nan),
        "forward_pe": kpis.get("forward_pe", np.nan),
        "peg": kpis.get("peg", np.nan),
        "ps": kpis.get("ps", np.nan),
        "pb": kpis.get("pb", np.nan),
        "fcf_yield": kpis.get("fcf_yield", np.nan),
        "roe": kpis.get("roe", np.nan),
        "oper_margin": kpis.get("oper_margin", np.nan),
        "rev_cagr_3y": kpis.get("rev_cagr_3y", np.nan),
        "eps_cagr_3y": kpis.get("eps_cagr_3y", np.nan),
        "mom_6m": mom_6m,
        "vol_1y": vol_1y,
        "net_debt_to_ebitda": kpis.get("net_debt_to_ebitda", np.nan),
        "cash_runway_months": kpis.get("cash_runway_months", np.nan),
    }

    score, subs = compute_tanaka_score(pd.Series(row))
    row["tanaka_score"] = score
    row["score_growth"] = subs.get("growth", np.nan)
    row["score_quality"] = subs.get("quality", np.nan)
    row["score_valuation"] = subs.get("valuation", np.nan)
    row["score_momentum"] = subs.get("momentum", np.nan)
    row["score_convexity"] = subs.get("convexity", np.nan)
    row["score_risk"] = subs.get("risk", np.nan)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Tanaka-Style Scorecard")
st.sidebar.caption("Top-10 Mapping • KPI Dashboard • 0–100 Score")

preset = st.sidebar.selectbox(
    "Preset",
    ["Tanaka Top-10 (Nov 2025)", "Blank (manual)"],
    index=0,
)

auto_fetch = st.sidebar.toggle("Auto-fetch fundamentals (yfinance)", value=True)
normalize_weights = st.sidebar.toggle("Normalize weights to 100%", value=True)
refresh = st.sidebar.button("Refresh data")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="small-note">If a KPI is missing, the model degrades gracefully and re-weights available inputs.</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT TOP-10 (from your factsheet, weights can be edited)
# ─────────────────────────────────────────────────────────────────────────────
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

if "portfolio" not in st.session_state or refresh:
    if preset == "Tanaka Top-10 (Nov 2025)":
        st.session_state["portfolio"] = DEFAULT_TOP10.copy()
    else:
        st.session_state["portfolio"] = pd.DataFrame(
            [{"ticker": "", "name": "", "sleeve": "Other", "weight": 0.0} for _ in range(10)]
        )

portfolio = st.session_state["portfolio"]

# Editor
st.title("📈 TANAKA-Style Portfolio Scorecard")
st.caption("Professionelles KPI-Dashboard: Screening, Scoring, Charts, Exposures – robust auch bei fehlenden Daten.")

lead_left, lead_right = st.columns([1.1, 0.9], gap="large")

with lead_left:
    st.subheader("1) Portfolio-Input (editierbar)")
    edited = st.data_editor(
        portfolio,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "name": st.column_config.TextColumn("Name", width="medium"),
            "sleeve": st.column_config.SelectboxColumn("Sleeve", options=SLEEVES, width="medium"),
            "weight": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
        },
        hide_index=True,
        key="portfolio_editor",
    )
    st.session_state["portfolio"] = edited

with lead_right:
    st.subheader("2) Modell-Einstellungen")
    st.write("**Sleeve-Gewichtungen** (Tanaka-nah):")
    sleeve_sel = st.selectbox("Preview Sleeve Weights", SLEEVES, index=0)
    st.json(SLEEVE_WEIGHTS.get(sleeve_sel, SLEEVE_WEIGHTS["Other"]), expanded=False)

    st.markdown("---")
    st.write("**Hinweis zur Datenqualität**")
    st.write("yfinance liefert Fundamentals je nach Aktie unterschiedlich vollständig. Fehlende KPIs werden automatisch ignoriert und die Score-Gewichte intern neu-normalisiert.")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD ANALYTICS TABLE
# ─────────────────────────────────────────────────────────────────────────────

df_in = st.session_state["portfolio"].copy()
df_in["ticker"] = df_in["ticker"].astype(str).str.upper().str.strip()
df_in = df_in[df_in["ticker"].str.len() > 0].reset_index(drop=True)

if df_in.empty:
    st.warning("Bitte mindestens einen Ticker eintragen.")
    st.stop()

# Normalize weights
w = df_in["weight"].apply(safe_float).fillna(0.0).values
w_sum = float(np.sum(w))
if normalize_weights and w_sum > 0:
    df_in["weight"] = (w / w_sum) * 100.0

# Fetch / Compute
st.markdown("---")
st.subheader("3) KPIs & Tanaka-Score")

rows = []
with st.spinner("Daten werden geladen und Score berechnet …"):
    for _, r in df_in.iterrows():
        tkr = r["ticker"]
        sleeve = r.get("sleeve", "Other") if r.get("sleeve", "Other") in SLEEVES else "Other"
        wt = safe_float(r.get("weight", np.nan))

        if auto_fetch:
            try:
                row = build_row(tkr, sleeve, wt)
            except Exception:
                # fallback minimal row
                row = {"ticker": tkr, "name": r.get("name", ""), "sleeve": sleeve, "weight": wt}
        else:
            row = {"ticker": tkr, "name": r.get("name", ""), "sleeve": sleeve, "weight": wt}

        # Keep user-entered name if present
        if isinstance(r.get("name", ""), str) and r.get("name", "").strip():
            row["name"] = r.get("name", "").strip()

        rows.append(row)

df = pd.DataFrame(rows)
for col in KPI_COLUMNS:
    if col not in df.columns:
        df[col] = np.nan

# Weighted portfolio score (by weights)
df["weight_dec"] = df["weight"].apply(safe_float) / 100.0
port_score = float(np.nansum(df["tanaka_score"] * df["weight_dec"])) if "tanaka_score" in df.columns else np.nan

# Top metrics
m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Portfolio Tanaka Score (wtd.)", f"{port_score:.1f}" if not np.isnan(port_score) else "—")
m2.metric("Names", f"{len(df)}")
m3.metric("Top Sleeve", df.groupby("sleeve")["weight"].sum().sort_values(ascending=False).index[0])
m4.metric("Cash Runway Coverage", "Auto" if auto_fetch else "Manual")

# Format display
def fmt_pct(x):
    return "—" if np.isnan(x) else f"{x*100:.1f}%"

def fmt_num(x, d=2):
    return "—" if np.isnan(x) else f"{x:.{d}f}"

def fmt_money(x):
    return "—" if np.isnan(x) else f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.1f}M"

display_cols = [
    "ticker","name","sleeve","weight","price","mktcap",
    "forward_pe","trailing_pe","peg","ps","pb","fcf_yield",
    "rev_cagr_3y","eps_cagr_3y","oper_margin","roe",
    "mom_6m","vol_1y","net_debt_to_ebitda","cash_runway_months",
    "tanaka_score","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk"
]

df_disp = df[display_cols].copy()

# Human formatting in a separate view (keep numeric df for charts)
df_view = df_disp.copy()
df_view["weight"] = df_view["weight"].apply(lambda x: "—" if np.isnan(x) else f"{x:.1f}%")
df_view["price"] = df_view["price"].apply(lambda x: "—" if np.isnan(x) else f"{x:.2f}")
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
df_view["cash_runway_months"] = df_view["cash_runway_months"].apply(lambda x: "—" if np.isnan(x) else f"{x:.0f}")
for c in ["tanaka_score","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk"]:
    df_view[c] = df_view[c].apply(lambda x: "—" if np.isnan(x) else f"{x:.0f}")

st.dataframe(
    df_view,
    use_container_width=True,
    hide_index=True,
)

# Downloads
csv_bytes = df_disp.to_csv(index=False).encode("utf-8")
st.download_button("Download KPI Table (CSV)", data=csv_bytes, file_name="tanaka_kpi_scorecard.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("4) Charts (professionell & fokussiert)")

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    # Sleeve allocation donut
    sleeve_w = df.groupby("sleeve", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
    fig = px.pie(
        sleeve_w,
        names="sleeve",
        values="weight",
        hole=0.55,
        title="Sleeve Allocation (%)",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Weighted score bar (like contribution)
    df_bar = df.copy()
    df_bar["wtd_score_contrib"] = df_bar["tanaka_score"] * df_bar["weight_dec"]
    df_bar = df_bar.sort_values("wtd_score_contrib", ascending=False)

    fig = px.bar(
        df_bar,
        x="wtd_score_contrib",
        y="ticker",
        orientation="h",
        title="Weighted Score Contribution (Score × Weight)",
        hover_data=["name", "sleeve", "tanaka_score", "weight"],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns([1.25, 0.75], gap="large")

with c3:
    # Valuation vs Growth scatter (Tanaka intuition map)
    # Use eps_cagr_3y if available else rev_cagr_3y
    g = df["eps_cagr_3y"].copy()
    g = g.where(~g.isna(), df["rev_cagr_3y"])
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
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    # Radar for top-1 or selected ticker
    st.write("**Radar: Sub-scores**")
    pick = st.selectbox("Select ticker", df["ticker"].tolist(), index=0)
    rr = df[df["ticker"] == pick].iloc[0]

    cats = ["Growth", "Quality", "Valuation", "Momentum", "Convexity", "Risk"]
    vals = [
        safe_float(rr.get("score_growth", np.nan)),
        safe_float(rr.get("score_quality", np.nan)),
        safe_float(rr.get("score_valuation", np.nan)),
        safe_float(rr.get("score_momentum", np.nan)),
        safe_float(rr.get("score_convexity", np.nan)),
        safe_float(rr.get("score_risk", np.nan)),
    ]
    # Close the loop
    cats2 = cats + [cats[0]]
    vals2 = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals2, theta=cats2, fill="toself", name=pick))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# Heatmap of sub-scores
st.markdown("---")
st.subheader("5) Score Heatmap (Top-Down View)")

heat = df[["ticker", "score_growth", "score_quality", "score_valuation", "score_momentum", "score_convexity", "score_risk", "tanaka_score"]].copy()
heat = heat.set_index("ticker")
fig = px.imshow(
    heat.T,
    aspect="auto",
    title="Sub-scores and Total Score (0–100)",
)
fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ACTION PANEL: “Tanaka-style” watchlist flags
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("6) Action Panel (Tanaka-Style Flags)")

# Simple flags derived from the prospectus logic:
# - Potential TRIM: very high valuation and high score (you'd check Target P/E)
# - Potential BUY / ADD: high score + undervaluation proxies
# - Risk: runway low / leverage high / volatility high
flags = []
for _, r in df.iterrows():
    fpe = safe_float(r.get("forward_pe", np.nan))
    peg = safe_float(r.get("peg", np.nan))
    score = safe_float(r.get("tanaka_score", np.nan))
    vol = safe_float(r.get("vol_1y", np.nan))
    runway = safe_float(r.get("cash_runway_months", np.nan))
    nde = safe_float(r.get("net_debt_to_ebitda", np.nan))

    flag = []
    if not np.isnan(score) and score >= 85:
        flag.append("High Conviction")
    if not np.isnan(fpe) and fpe >= 45 and not np.isnan(score) and score >= 75:
        flag.append("Trim-check (Target P/E?)")
    if (not np.isnan(peg) and peg <= 1.2) and (not np.isnan(score) and score >= 70):
        flag.append("Undervalued-growth candidate")
    if not np.isnan(vol) and vol >= 0.70:
        flag.append("High vol (convex / risk)")
    if not np.isnan(runway) and runway <= 12:
        flag.append("Runway risk (<12m)")
    if not np.isnan(nde) and nde >= 4:
        flag.append("Leverage risk (ND/EBITDA high)")

    flags.append(", ".join(flag) if flag else "—")

df_flags = df[["ticker", "name", "sleeve", "weight", "tanaka_score", "forward_pe", "peg", "vol_1y", "cash_runway_months", "net_debt_to_ebitda"]].copy()
df_flags["flags"] = flags

st.dataframe(
    df_flags.sort_values("tanaka_score", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.caption("Disclaimer: Research/education dashboard. Not investment advice. Data quality depends on vendor coverage.")
