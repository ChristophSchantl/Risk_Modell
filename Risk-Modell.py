# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# SHI Scorecard – Screenshot-Style Sidebar + Weights + Yahoo + Charts
# + Beta/Correlation vs S&P 500 & DAX
# + Action Panel mit farbigen Badges (HTML)
#
# UPDATE (Fallback-Engine):
# - Wenn Yahoo/yfinance Felder fehlen (None/NaN), werden zentrale KPIs robust
#   aus Income/CF/Balance Sheet approximiert:
#     * trailing_pe, ps, pb, fcf_yield, oper_margin, roe, net_debt_to_ebitda
# - forward_pe / peg bleiben i.d.R. estimate-abhängig (ohne Analysten-Coverage
#   nicht sauber berechenbar). forward_pe wird NICHT “erfunden”.
# ─────────────────────────────────────────────────────────────────────────────

import io
import re
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SHI Scorecard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }

      /* Metric cards */
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }
      div[data-testid="stMetric"] > label { color: #6b7280 !important; font-weight: 500 !important; }
      div[data-testid="stMetric"] span { color: #111827 !important; font-weight: 650 !important; }

      /* HTML tables (for badges) */
      table { width:100%; border-collapse: collapse; }
      thead th {
        background:#f9fafb;
        border-bottom: 1px solid #e5e7eb;
        padding: 10px;
        text-align:left;
        font-weight: 700;
        color:#111827;
        font-size: 0.92rem;
      }
      tbody td {
        border-bottom: 1px solid #eef2f7;
        padding: 10px;
        vertical-align: middle;
        color:#111827;
        font-size: 0.92rem;
      }
      tbody tr:hover { background:#f9fafb; }
      code { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SLEEVES = ["Auto", "Platform", "Biotech/Pharma", "Minerals/Energy", "Financials", "Other"]

BASE_WEIGHTS = {
    "Platform": {"growth": 0.20, "quality": 0.22, "valuation": 0.18, "momentum": 0.10, "convexity": 0.08, "risk": 0.10, "gap": 0.12},
    "Biotech/Pharma": {"growth": 0.14, "quality": 0.10, "valuation": 0.10, "momentum": 0.08, "convexity": 0.22, "risk": 0.10, "gap": 0.26},
    "Minerals/Energy": {"growth": 0.10, "quality": 0.08, "valuation": 0.14, "momentum": 0.08, "convexity": 0.20, "risk": 0.16, "gap": 0.24},
    "Financials": {"growth": 0.12, "quality": 0.18, "valuation": 0.22, "momentum": 0.10, "convexity": 0.06, "risk": 0.18, "gap": 0.14},
    "Other": {"growth": 0.16, "quality": 0.16, "valuation": 0.16, "momentum": 0.10, "convexity": 0.12, "risk": 0.14, "gap": 0.16},
}

DEFAULT_TICKERS = ["LULU", "REI", "SRPT", "CAG", "NVO", "PYPL", "VIXL", "NVDA"]

SHOW_COLS = [
    "ticker","name","sleeve","weight","price","mktcap",
    "forward_pe","trailing_pe","peg","ps","pb","fcf_yield",
    "rev_cagr_3y","eps_cagr_3y","oper_margin","roe",
    "mom_6m","vol_1y","net_debt_to_ebitda","cash_runway_months",
    "expected_growth","implied_growth","expectation_gap",
    "shi_score","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap"
]

REQUIRED_COLS = set(SHOW_COLS + ["weight_dec"])

# ─────────────────────────────────────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        if isinstance(x, str):
            x = x.strip().replace("%", "").replace(",", ".")
            if x == "" or x.lower() in {"none", "nan", "na", "n/a"}:
                return np.nan
            return float(x)
        return float(x)
    except Exception:
        return np.nan

def sanitize_ticker(t: str) -> str:
    t = (t or "").upper().strip()
    return t if re.fullmatch(r"[A-Z0-9\.\-\^]{1,15}", t) else ""

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

def _parse_tickers_any(text: str):
    if not text:
        return []
    raw = text.replace("\n", " ").replace("\t", " ").replace(";", ",").replace("|", ",")
    parts = []
    for chunk in raw.split(","):
        parts.extend(chunk.split())
    tickers = [sanitize_ticker(p.strip()) for p in parts if p.strip()]
    tickers = [t for t in tickers if t]
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def _read_tickers_from_csv(uploaded_file) -> list[str]:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    sep = ";" if text.count(";") > text.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]

    candidates = ["ticker", "symbol", "code", "codes", "ric"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        col = df.columns[0]

    tickers = df[col].astype(str).str.upper().str.strip().tolist()
    tickers = [sanitize_ticker(t) for t in tickers]
    tickers = [t for t in tickers if t]
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def normalize_weights_pct(df):
    w = df["weight"].apply(safe_float).fillna(0.0).values
    s = float(np.sum(w))
    if s <= 0:
        df["weight"] = 0.0
        return df
    df["weight"] = (w / s) * 100.0
    return df

def sleeve_auto_heuristic(info: dict):
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    name = (info.get("shortName") or info.get("longName") or "").lower()
    txt = " ".join([sector, industry, name])
    if any(k in txt for k in ["biotech", "biotechnology", "pharmaceutical", "pharma", "drug", "therapeutics"]):
        return "Biotech/Pharma"
    if any(k in txt for k in ["semiconductor", "software", "internet", "computer", "technology", "cloud", "hardware", "ai"]):
        return "Platform"
    if any(k in txt for k in ["uranium", "mining", "metals", "materials", "oil", "gas", "energy", "coal"]):
        return "Minerals/Energy"
    if any(k in txt for k in ["bank", "financial", "insurance", "capital markets", "asset management"]):
        return "Financials"
    return "Other"

def ensure_required_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["weight", "shi_score", "forward_pe", "trailing_pe", "peg", "ps", "pb", "vol_1y",
              "cash_runway_months", "net_debt_to_ebitda", "oper_margin", "roe", "fcf_yield", "mktcap", "price"]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# FLAGS – Klassifikation + HTML Badges
# ─────────────────────────────────────────────────────────────────────────────
def classify_flags(row):
    out = []
    score = safe_float(row.get("shi_score", np.nan))
    fpe = safe_float(row.get("forward_pe", np.nan))
    peg = safe_float(row.get("peg", np.nan))
    vol = safe_float(row.get("vol_1y", np.nan))
    runway = safe_float(row.get("cash_runway_months", np.nan))
    nde = safe_float(row.get("net_debt_to_ebitda", np.nan))
    exp_g = safe_float(row.get("expected_growth", np.nan))
    impl_g = safe_float(row.get("implied_growth", np.nan))
    gap = safe_float(row.get("expectation_gap", np.nan))

    # Positive
    if not np.isnan(score) and score >= 85:
        out.append(("High Conviction", "positive"))
    if (not np.isnan(peg) and peg <= 1.2) and (not np.isnan(score) and score >= 70):
        out.append(("Undervalued-growth candidate", "positive"))
    if not np.isnan(exp_g) and not np.isnan(impl_g) and (exp_g - impl_g) >= 0.05:
        out.append(("Expectation Gap (exp > implied)", "positive"))
    if not np.isnan(gap) and gap >= 0.10:
        out.append(("Large Gap (>=10%)", "positive"))

    # Neutral
    if not np.isnan(fpe) and fpe >= 45 and not np.isnan(score) and score >= 75:
        out.append(("Trim-check (Target P/E?)", "neutral"))

    # Negative / Risk
    if not np.isnan(vol) and vol >= 0.70:
        out.append(("High vol", "negative"))
    if not np.isnan(runway) and runway <= 12:
        out.append(("Runway risk (<12m)", "negative"))
    if not np.isnan(nde) and nde >= 4:
        out.append(("Leverage risk (ND/EBITDA high)", "negative"))

    return out

def render_flag_badges(flags):
    if not flags:
        return "—"
    parts = []
    for label, kind in flags:
        if kind == "positive":
            color, bg = "#166534", "#dcfce7"   # green
        elif kind == "negative":
            color, bg = "#991b1b", "#fee2e2"   # red
        else:
            color, bg = "#92400e", "#fef3c7"   # amber
        parts.append(
            f'<span style="background:{bg};color:{color};padding:4px 10px;'
            f'border-radius:12px;font-size:0.75rem;font-weight:650;'
            f'margin-right:6px;white-space:nowrap;display:inline-block;line-height:1.4;">'
            f'{label}</span>'
        )
    return "".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# YF FETCH (cached)
# ─────────────────────────────────────────────────────────────────────────────
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
    # yfinance liefert i.d.R. annual statements (letzte 4 Jahre)
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

# ─────────────────────────────────────────────────────────────────────────────
# STATEMENT HELPERS (robust row lookup)
# ─────────────────────────────────────────────────────────────────────────────
def _norm_idx(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _find_row(df: pd.DataFrame, candidates: list[str]):
    if df is None or df.empty:
        return None
    # map normalized index -> original index
    idx_map = { _norm_idx(str(ix)): ix for ix in df.index }
    for cand in candidates:
        key = _norm_idx(cand)
        if key in idx_map:
            return idx_map[key]
    # fallback: contains match
    keys = list(idx_map.keys())
    for cand in candidates:
        key = _norm_idx(cand)
        for k in keys:
            if key and (key in k or k in key):
                return idx_map[k]
    return None

def _latest_value(df: pd.DataFrame, row_name):
    try:
        if df is None or df.empty or row_name is None:
            return np.nan
        s = df.loc[row_name].dropna()
        if s is None or len(s) == 0:
            return np.nan
        # yfinance stellt häufig Spalten als Datums-Objekte bereit; "iloc[0]" ist i.d.R. das jüngste Jahr
        return safe_float(s.iloc[0])
    except Exception:
        return np.nan

def _latest_positive(df: pd.DataFrame, row_candidates: list[str]):
    row = _find_row(df, row_candidates)
    v = _latest_value(df, row)
    return v

# ─────────────────────────────────────────────────────────────────────────────
# MARKET HELPERS
# ─────────────────────────────────────────────────────────────────────────────
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

def cash_runway_months(bs: pd.DataFrame, cf: pd.DataFrame):
    try:
        cash = np.nan
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            cash = _latest_positive(bs, [
                "Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash", "Cash Cash Equivalents And Short Term Investments",
                "CashAndCashEquivalentsAndShortTermInvestments"
            ])

        ocf = np.nan
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            ocf = _latest_positive(cf, [
                "Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow",
                "Net Cash Provided By Operating Activities", "NetCashProvidedByOperatingActivities"
            ])

        if not np.isnan(cash) and not np.isnan(ocf) and ocf < 0:
            return (cash / abs(ocf)) * 12.0
    except Exception:
        pass
    return np.nan

def try_cagr_from_income_stmt(inc: pd.DataFrame, row_name_candidates, years=4):
    try:
        if inc is None or inc.empty:
            return np.nan
        row = _find_row(inc, list(row_name_candidates))
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
# FALLBACK KPI ENGINE (aus Statements)
# ─────────────────────────────────────────────────────────────────────────────
def compute_kpi_fallbacks(info: dict, inc: pd.DataFrame, cf: pd.DataFrame, bs: pd.DataFrame, price: float, mcap: float):
    """
    Liefert KPI-Fallbacks, wenn Yahoo Felder fehlen.
    Wichtig:
      - Statements sind i.d.R. annual -> als "TTM-proxy" (letztes FY) interpretieren.
      - forward_pe / peg sind estimates -> werden nicht künstlich geschätzt.
    """
    out = {}

    price = safe_float(price)
    mcap = safe_float(mcap)

    # Shares (proxy)
    shares = safe_float(info.get("sharesOutstanding"))
    if np.isnan(shares) and (not np.isnan(mcap)) and (not np.isnan(price)) and price > 0:
        shares = mcap / price

    # Income statement proxies
    revenue = _latest_positive(inc, ["Total Revenue", "TotalRevenue", "Revenue"])
    op_income = _latest_positive(inc, ["Operating Income", "OperatingIncome"])
    net_income = _latest_positive(inc, ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"])

    ebitda = _latest_positive(inc, ["EBITDA", "Ebitda"])  # nicht immer vorhanden

    # Cashflow proxies
    ocf = _latest_positive(cf, [
        "Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow",
        "Net Cash Provided By Operating Activities", "NetCashProvidedByOperatingActivities"
    ])
    capex = _latest_positive(cf, ["Capital Expenditures", "CapitalExpenditures"])
    # capex ist häufig negativ; FCF = OCF - capex (bei capex negativ => plus)
    fcf = np.nan
    if not np.isnan(ocf) and not np.isnan(capex):
        fcf = ocf - capex

    # Balance sheet proxies
    cash = _latest_positive(bs, [
        "Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash",
        "Cash Cash Equivalents And Short Term Investments", "CashAndCashEquivalentsAndShortTermInvestments"
    ])
    total_debt = _latest_positive(bs, [
        "Total Debt", "TotalDebt",
        "Long Term Debt", "LongTermDebt",
        "Long Term Debt And Capital Lease Obligation", "LongTermDebtAndCapitalLeaseObligation"
    ])
    # Falls "Total Debt" fehlt, versuche Summe aus LT + ST
    if np.isnan(total_debt):
        lt = _latest_positive(bs, ["Long Term Debt", "LongTermDebt"])
        st = _latest_positive(bs, ["Short Long Term Debt", "ShortLongTermDebt", "Short Term Debt", "ShortTermDebt"])
        if not np.isnan(lt) or not np.isnan(st):
            total_debt = (0.0 if np.isnan(lt) else lt) + (0.0 if np.isnan(st) else st)

    equity = _latest_positive(bs, [
        "Total Stockholder Equity", "TotalStockholderEquity",
        "Stockholders Equity", "StockholdersEquity",
        "Total Equity Gross Minority Interest", "TotalEquityGrossMinorityInterest"
    ])

    # --- trailing P/E (proxy: Net income / shares)
    trailing_pe_fb = np.nan
    if not np.isnan(price) and price > 0 and not np.isnan(net_income) and not np.isnan(shares) and shares > 0:
        eps = net_income / shares
        if eps > 0:
            trailing_pe_fb = price / eps
    out["trailing_pe_fb"] = trailing_pe_fb

    # --- P/S (mcap / revenue)
    ps_fb = np.nan
    if not np.isnan(mcap) and mcap > 0 and not np.isnan(revenue) and revenue > 0:
        ps_fb = mcap / revenue
    out["ps_fb"] = ps_fb

    # --- P/B (mcap / equity)
    pb_fb = np.nan
    if not np.isnan(mcap) and mcap > 0 and not np.isnan(equity) and equity > 0:
        pb_fb = mcap / equity
    out["pb_fb"] = pb_fb

    # --- FCF Yield (fcf / mcap)
    fcf_yield_fb = np.nan
    if not np.isnan(fcf) and not np.isnan(mcap) and mcap > 0:
        fcf_yield_fb = fcf / mcap
    out["fcf_yield_fb"] = fcf_yield_fb

    # --- Operating margin (op_income / revenue)
    oper_margin_fb = np.nan
    if not np.isnan(op_income) and not np.isnan(revenue) and revenue != 0:
        oper_margin_fb = op_income / revenue
    out["oper_margin_fb"] = oper_margin_fb

    # --- ROE (net_income / equity)
    roe_fb = np.nan
    if not np.isnan(net_income) and not np.isnan(equity) and equity != 0:
        roe_fb = net_income / equity
    out["roe_fb"] = roe_fb

    # --- Net debt / EBITDA
    nde_fb = np.nan
    if not np.isnan(total_debt) or not np.isnan(cash):
        net_debt = (0.0 if np.isnan(total_debt) else total_debt) - (0.0 if np.isnan(cash) else cash)
        if not np.isnan(ebitda) and ebitda != 0:
            nde_fb = net_debt / ebitda
    out["net_debt_to_ebitda_fb"] = nde_fb

    return out

# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────
def score_growth(vals):
    s = nanmean([z_to_01(vals.get("eps_cagr_3y", np.nan), -0.20, 0.40),
                 z_to_01(vals.get("rev_cagr_3y", np.nan), -0.10, 0.30)])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_quality(vals):
    s = nanmean([z_to_01(vals.get("roe", np.nan), -0.10, 0.30),
                 z_to_01(vals.get("oper_margin", np.nan), -0.10, 0.35)])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_valuation(vals):
    s = nanmean([inv_to_01(vals.get("forward_pe", np.nan), 5, 60),
                 inv_to_01(vals.get("trailing_pe", np.nan), 5, 60),
                 inv_to_01(vals.get("peg", np.nan), 0.5, 3.0),
                 z_to_01(vals.get("fcf_yield", np.nan), -0.02, 0.08)])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_momentum(vals):
    s = z_to_01(vals.get("mom_6m", np.nan), -0.40, 0.60)
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_convexity(vals, sleeve):
    vol = vals.get("vol_1y", np.nan)
    mcap = vals.get("mktcap", np.nan)
    s_vol = z_to_01(vol, 0.15, 0.90)
    s_size = np.nan
    if not np.isnan(mcap) and mcap > 0:
        s_size = inv_to_01(np.log10(mcap), 9.0, 12.0)
    base = {"Platform": 0.35, "Biotech/Pharma": 0.70, "Minerals/Energy": 0.70, "Financials": 0.25, "Other": 0.45}.get(sleeve, 0.45)
    s = nanmean([s_vol, s_size, base])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_risk(vals, sleeve):
    vol = vals.get("vol_1y", np.nan)
    nde = vals.get("net_debt_to_ebitda", np.nan)
    runway = vals.get("cash_runway_months", np.nan)

    vol_score = inv_to_01(vol, 0.15, 0.90)
    if sleeve in ["Biotech/Pharma", "Minerals/Energy"] and not np.isnan(vol_score):
        vol_score = 0.6 * vol_score + 0.4 * 0.5

    nde_score = inv_to_01(nde, -1.0, 6.0)
    runway_score = z_to_01(runway, 0.0, 36.0)

    s = nanmean([vol_score, nde_score, runway_score])
    if np.isnan(s): return np.nan
    risk = float(np.clip(s * 100, 0, 100))
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

    subs = {
        "growth": score_growth(vals),
        "quality": score_quality(vals),
        "valuation": score_valuation(vals),
        "momentum": score_momentum(vals),
        "convexity": score_convexity(vals, sleeve),
        "risk": score_risk(vals, sleeve),
    }
    gap_score, exp_g, impl_g, gap_raw = score_expectation_gap(vals)
    subs["gap"] = gap_score

    if sleeve in ["Biotech/Pharma", "Minerals/Energy"]:
        weights["risk"] *= 0.60
        weights["convexity"] *= 1.15
        ssum = sum(weights.values())
        weights = {k: v / ssum for k, v in weights.items()}

    wsum, wtot = 0.0, 0.0
    for k, v in subs.items():
        if np.isnan(v):
            continue
        wsum += weights.get(k, 0.0) * v
        wtot += weights.get(k, 0.0)
    if wtot <= 0:
        return np.nan, subs, exp_g, impl_g, gap_raw

    total = wsum / wtot
    return float(np.clip(total, 0, 100)), subs, exp_g, impl_g, gap_raw

# ─────────────────────────────────────────────────────────────────────────────
# BUILD ROW (Yahoo + Fallbacks)
# ─────────────────────────────────────────────────────────────────────────────
def build_row(ticker: str, sleeve_choice: str, weight_pct: float):
    info = fetch_info(ticker)
    hist = fetch_hist(ticker, "2y")
    inc, cf, bs = fetch_financials(ticker)
    mom, vol = calc_mom_vol(hist)

    sleeve = sleeve_choice if sleeve_choice in SLEEVES else "Auto"
    if sleeve == "Auto":
        sleeve = sleeve_auto_heuristic(info)

    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    mktcap = safe_float(info.get("marketCap"))

    # Raw Yahoo fields
    trailing_pe = safe_float(info.get("trailingPE"))
    forward_pe = clean_forward_pe(info.get("forwardPE"))
    peg = safe_float(info.get("pegRatio"))

    ps = safe_float(info.get("priceToSalesTrailing12Months"))
    pb = safe_float(info.get("priceToBook"))
    roe = safe_float(info.get("returnOnEquity"))
    oper_margin = safe_float(info.get("operatingMargins"))
    nde = safe_float(info.get("netDebtToEBITDA"))

    # FCF yield via Yahoo direct
    fcf = safe_float(info.get("freeCashflow"))
    fcf_y = np.nan
    if not np.isnan(fcf) and not np.isnan(mktcap) and mktcap > 0:
        fcf_y = fcf / mktcap

    # Fallbacks aus Statements (nur wenn Yahoo missing/NaN)
    fb = compute_kpi_fallbacks(info, inc, cf, bs, price=price, mcap=mktcap)

    if np.isnan(trailing_pe):
        trailing_pe = fb.get("trailing_pe_fb", np.nan)

    if np.isnan(ps):
        ps = fb.get("ps_fb", np.nan)

    if np.isnan(pb):
        pb = fb.get("pb_fb", np.nan)

    if np.isnan(fcf_y):
        fcf_y = fb.get("fcf_yield_fb", np.nan)

    if np.isnan(oper_margin):
        oper_margin = fb.get("oper_margin_fb", np.nan)

    if np.isnan(roe):
        roe = fb.get("roe_fb", np.nan)

    if np.isnan(nde):
        nde = fb.get("net_debt_to_ebitda_fb", np.nan)

    row = {
        "ticker": ticker.upper().strip(),
        "name": (info.get("shortName") or info.get("longName") or ""),
        "sleeve": sleeve,
        "weight": float(weight_pct),

        "price": price,
        "mktcap": mktcap,

        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,   # bleibt estimate-abhängig
        "peg": peg,                 # bleibt estimate-abhängig

        "ps": ps,
        "pb": pb,
        "roe": roe,
        "oper_margin": oper_margin,
        "net_debt_to_ebitda": nde,

        "fcf_yield": fcf_y,

        "rev_cagr_3y": try_cagr_from_income_stmt(inc, ["Total Revenue", "TotalRevenue", "Revenue"], years=4),
        "eps_cagr_3y": try_cagr_from_income_stmt(inc, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"], years=4),

        "cash_runway_months": cash_runway_months(bs, cf),
        "mom_6m": mom,
        "vol_1y": vol,
    }

    total, subs, exp_g, impl_g, gap_raw = compute_total_score(pd.Series(row))
    row["shi_score"] = total
    row["expected_growth"] = exp_g
    row["implied_growth"] = impl_g
    row["expectation_gap"] = gap_raw
    for k, v in subs.items():
        row[f"score_{k}"] = v

    return row

# ─────────────────────────────────────────────────────────────────────────────
# BETA/CORR PANEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    data = yf.download(tickers=tickers, period=period, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            px_ = data["Close"].copy()
        else:
            px_ = data.xs(data.columns.levels[0][0], axis=1, level=0).copy()
    else:
        if "Close" in data.columns:
            px_ = data[["Close"]].copy()
            px_.columns = [tickers[0]]
        else:
            px_ = data.copy()
    return px_.dropna(how="all")

def compute_beta_corr(asset_ret: pd.Series, bench_ret: pd.Series) -> tuple[float, float]:
    df2 = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if df2.shape[0] < 60:
        return np.nan, np.nan
    a = df2.iloc[:, 0].values
    b = df2.iloc[:, 1].values
    var_b = np.var(b, ddof=1)
    beta = np.cov(a, b, ddof=1)[0, 1] / var_b if var_b > 0 else np.nan
    corr = np.corrcoef(a, b)[0, 1]
    return float(beta), float(corr)

def portfolio_returns_from_prices(px: pd.DataFrame, weights_pct: pd.Series) -> pd.Series:
    rets = px.pct_change().dropna(how="all")
    common = [c for c in rets.columns if c in weights_pct.index]
    if len(common) == 0:
        return pd.Series(dtype=float)
    w = (weights_pct.loc[common] / 100.0).astype(float)
    w = w / w.sum() if w.sum() > 0 else w
    port = (rets[common].mul(w, axis=1)).sum(axis=1)
    port.name = "PORT"
    return port

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR – Screenshot-Style
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("CSV-Dateien")
uploaded = st.sidebar.file_uploader("Drag and drop files here", type=["csv"], accept_multiple_files=False)
manual = st.sidebar.text_input("Weitere Ticker manuell hinzufügen (Komma-getrennt)", value="")

st.sidebar.caption("")
shuffle = st.sidebar.checkbox("Zufällig mischen", value=False)
max_n = st.sidebar.number_input("Max. Anzahl (0 = alle)", min_value=0, value=0, step=1)

tickers = []
if uploaded is not None:
    try:
        tickers.extend(_read_tickers_from_csv(uploaded))
    except Exception as e:
        st.sidebar.error(f"CSV konnte nicht gelesen werden: {e}")

tickers.extend(_parse_tickers_any(manual))

if len(tickers) == 0:
    tickers = DEFAULT_TICKERS.copy()

seen, combined = set(), []
for t in tickers:
    if t and t not in seen:
        combined.append(t)
        seen.add(t)

if shuffle and len(combined) > 1:
    rng = np.random.default_rng(42)
    combined = list(rng.permutation(combined))

if max_n and max_n > 0:
    combined = combined[: int(max_n)]

st.sidebar.caption(f"Gefundene Ticker: {len(combined)}")

selected = st.sidebar.multiselect("Auswahl verfeinern", options=combined, default=combined)

df_out = pd.DataFrame({"ticker": selected})
st.sidebar.download_button(
    "Kombinierte Ticker als CSV",
    data=df_out.to_csv(index=False).encode("utf-8"),
    file_name="combined_tickers.csv",
    mime="text/csv",
)

st.sidebar.markdown("---")
default_sleeve = st.sidebar.selectbox("Default Sleeve", SLEEVES, index=0)
auto_normalize = st.sidebar.toggle("Weights automatisch auf 100% normalisieren", value=True)
auto_fetch = st.sidebar.toggle("Yahoo Finance automatisch laden", value=True)
run = st.sidebar.button("Load / Refresh", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 SHI Scorecard")
st.caption("Ticker rein → Gewicht setzen → Yahoo Pull → Score, Charts, Risk-Panel, Flags.")

if len(selected) == 0:
    st.warning("Keine Ticker selektiert.")
    st.stop()

# Init weights state
if "weights_df" not in st.session_state:
    eq_w = 100.0 / len(selected)
    st.session_state["weights_df"] = pd.DataFrame(
        {"ticker": selected, "weight": [eq_w] * len(selected), "sleeve": [default_sleeve] * len(selected)}
    )

# Sync tickers with selection
old = st.session_state["weights_df"].copy()
old_map_w = dict(zip(old["ticker"], old["weight"]))
old_map_s = dict(zip(old["ticker"], old["sleeve"]))

new_rows = []
for t in selected:
    new_rows.append(
        {"ticker": t, "weight": float(old_map_w.get(t, 100.0 / len(selected))), "sleeve": old_map_s.get(t, default_sleeve)}
    )
st.session_state["weights_df"] = pd.DataFrame(new_rows)

st.subheader("1) Weights (Ticker + Gewicht)")
edited = st.data_editor(
    st.session_state["weights_df"],
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
        "weight": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
        "sleeve": st.column_config.SelectboxColumn("Sleeve", options=SLEEVES, width="medium"),
    },
)

df_in = edited.copy()
df_in["ticker"] = df_in["ticker"].astype(str).apply(sanitize_ticker)
df_in["weight"] = df_in["weight"].apply(safe_float).fillna(0.0)
df_in["sleeve"] = df_in["sleeve"].astype(str).str.strip()
df_in.loc[~df_in["sleeve"].isin(SLEEVES), "sleeve"] = "Auto"
df_in = df_in[df_in["ticker"].astype(str).str.strip() != ""].reset_index(drop=True)

if auto_normalize:
    df_in = normalize_weights_pct(df_in)

st.session_state["weights_df"] = df_in

if not run and "ran_once" not in st.session_state:
    st.info("Gewichte einstellen und links **Load / Refresh** drücken.")
    st.stop()
st.session_state["ran_once"] = True

# ─────────────────────────────────────────────────────────────────────────────
# FETCH + SCORE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("2) KPIs & SHI Score")

rows = []
with st.spinner("Yahoo Finance Daten laden & Score berechnen …"):
    for _, r in df_in.iterrows():
        tkr = r["ticker"]
        wt = float(safe_float(r["weight"]))
        sl = r.get("sleeve", "Auto")
        if not auto_fetch:
            rows.append({"ticker": tkr, "weight": wt, "sleeve": sl, "name": ""})
        else:
            rows.append(build_row(tkr, sl, wt))

df = ensure_required_cols(pd.DataFrame(rows))
df["weight_dec"] = df["weight"].fillna(0.0) / 100.0

port_score = np.nan
if "shi_score" in df.columns and df["shi_score"].notna().any():
    port_score = float(np.nansum(df["shi_score"] * df["weight_dec"]))

if df.empty:
    st.warning("Keine Datenpunkte – prüfe Ticker-Auswahl.")
    st.stop()

m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Portfolio SHI Score (wtd.)", f"{port_score:.1f}" if not np.isnan(port_score) else "—")
m2.metric("Names", f"{len(df)}")

top_sleeve = "—"
if df["sleeve"].notna().any():
    gs = df.groupby("sleeve")["weight"].sum().sort_values(ascending=False)
    if len(gs) > 0:
        top_sleeve = gs.index[0]
m3.metric("Top Sleeve", top_sleeve)

m4.metric("Coverage", f"{int(df['shi_score'].notna().sum())}/{len(df)}" if "shi_score" in df.columns else f"0/{len(df)}")

st.dataframe(df[SHOW_COLS].sort_values("weight", ascending=False), use_container_width=True, hide_index=True)
st.download_button("Download KPI Table (CSV)", df[SHOW_COLS].to_csv(index=False).encode("utf-8"), "shi_scorecard.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("3) Charts")

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    sleeve_w = df.groupby("sleeve", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
    fig = px.pie(sleeve_w, names="sleeve", values="weight", hole=0.55, title="Sleeve Allocation (%)")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    d = df.copy()
    d["wtd_contrib"] = d["shi_score"] * d["weight_dec"]
    d = d.sort_values("wtd_contrib", ascending=False)
    fig = px.bar(
        d, x="wtd_contrib", y="ticker", orientation="h",
        title="Weighted Score Contribution (Score × Weight)",
        hover_data=["name", "sleeve", "shi_score", "weight"],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns([1.25, 0.75], gap="large")
with c3:
    gproxy = df["eps_cagr_3y"].where(df["eps_cagr_3y"].notna(), df["rev_cagr_3y"])
    scatter = df.copy()
    scatter["growth_proxy"] = gproxy
    fig = px.scatter(
        scatter, x="forward_pe", y="growth_proxy",
        size="weight", color="sleeve", hover_name="ticker",
        hover_data={"name": True, "shi_score": True, "weight": True, "forward_pe": True, "growth_proxy": True},
        title="Valuation vs Growth (proxy) — Undervalued Growth Map",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c4:
    pick = st.selectbox("Radar ticker", df["ticker"].tolist(), index=0)
    rr = df[df["ticker"] == pick].iloc[0]
    cats = ["Growth","Quality","Valuation","Momentum","Convexity","Risk","Gap"]
    vals = [
        safe_float(rr.get("score_growth")),
        safe_float(rr.get("score_quality")),
        safe_float(rr.get("score_valuation")),
        safe_float(rr.get("score_momentum")),
        safe_float(rr.get("score_convexity")),
        safe_float(rr.get("score_risk")),
        safe_float(rr.get("score_gap")),
    ]
    cats2 = cats + [cats[0]]
    vals2 = vals + [vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals2, theta=cats2, fill="toself", name=pick))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("4) Expectation-Gap Overlay")

fig = px.scatter(
    df, x="implied_growth", y="expected_growth",
    size="weight", color="sleeve", hover_name="ticker",
    hover_data={"name": True, "shi_score": True, "expectation_gap": True},
    title="Expected vs Implied Growth (Expectation-Gap Overlay)",
)
fig.add_shape(type="line", x0=0, y0=0, x1=0.30, y1=0.30, line=dict(dash="dash"))
fig.update_xaxes(tickformat=".0%", range=[0, 0.30])
fig.update_yaxes(tickformat=".0%", range=[-0.10, 0.40])
fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("5) Heatmap (0–100)")

heat_cols = ["ticker","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap","shi_score"]
heat = df[heat_cols].set_index("ticker")
if heat.dropna(how="all").empty:
    st.info("Heatmap: keine Subscore-Daten (z.B. Yahoo Coverage / auto_fetch).")
else:
    fig = px.imshow(heat.T, aspect="auto", title="Sub-scores & Total Score (0–100)")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5b) Beta / Correlation Panel vs S&P500 & DAX
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("5b) Beta / Correlation vs S&P 500 & DAX")

colA, colB, colC = st.columns([1, 1, 1], gap="large")
with colA:
    lookback = st.selectbox("Lookback", ["6mo", "1y", "2y", "5y"], index=2)
with colB:
    rolling_win = st.selectbox("Rolling Window (Trading Days)", [30, 60, 90, 126], index=1)
with colC:
    use_log = st.toggle("Log Returns", value=False)

bench_sp = "^GSPC"
bench_dax = "^GDAXI"

tickers_list = df["ticker"].astype(str).str.upper().str.strip().tolist()
need = list(dict.fromkeys(tickers_list + [bench_sp, bench_dax]))

with st.spinner("Preisdaten laden (Beta/Korrelation) …"):
    px_all = fetch_prices(need, period=lookback)

if px_all.empty or px_all.shape[0] < 80:
    st.warning("Zu wenig Preisdaten für Beta/Korrelation (oder Yahoo liefert nichts).")
else:
    if use_log:
        ret_all = np.log(px_all).diff().dropna(how="all")
    else:
        ret_all = px_all.pct_change().dropna(how="all")

    w_series = df.set_index("ticker")["weight"].apply(safe_float).fillna(0.0)
    port_ret = portfolio_returns_from_prices(px_all[tickers_list], w_series)

    sp_ret = ret_all[bench_sp].dropna() if bench_sp in ret_all.columns else pd.Series(dtype=float)
    dax_ret = ret_all[bench_dax].dropna() if bench_dax in ret_all.columns else pd.Series(dtype=float)

    tmp = pd.concat([port_ret, sp_ret.rename("SPX"), dax_ret.rename("DAX")], axis=1).dropna()
    if tmp.shape[0] < 60:
        st.warning("Zu wenig überlappende Datenpunkte für saubere Schätzung.")
    else:
        port_beta_sp, port_corr_sp = compute_beta_corr(tmp["PORT"], tmp["SPX"])
        port_beta_dax, port_corr_dax = compute_beta_corr(tmp["PORT"], tmp["DAX"])

        m1, m2, m3, m4 = st.columns(4, gap="large")
        m1.metric("Portfolio Beta vs S&P 500", f"{port_beta_sp:.2f}" if not np.isnan(port_beta_sp) else "—")
        m2.metric("Portfolio Corr vs S&P 500", f"{port_corr_sp:.2f}" if not np.isnan(port_corr_sp) else "—")
        m3.metric("Portfolio Beta vs DAX", f"{port_beta_dax:.2f}" if not np.isnan(port_beta_dax) else "—")
        m4.metric("Portfolio Corr vs DAX", f"{port_corr_dax:.2f}" if not np.isnan(port_corr_dax) else "—")

        rows_b = []
        for t in tickers_list:
            if t not in ret_all.columns:
                continue
            a = ret_all[t].dropna()
            b1 = ret_all[bench_sp].dropna() if bench_sp in ret_all.columns else pd.Series(dtype=float)
            b2 = ret_all[bench_dax].dropna() if bench_dax in ret_all.columns else pd.Series(dtype=float)

            beta_sp, corr_sp = compute_beta_corr(a, b1) if not b1.empty else (np.nan, np.nan)
            beta_dx, corr_dx = compute_beta_corr(a, b2) if not b2.empty else (np.nan, np.nan)

            rows_b.append({
                "ticker": t,
                "weight_%": float(w_series.get(t, 0.0)),
                "beta_spx": beta_sp,
                "corr_spx": corr_sp,
                "beta_dax": beta_dx,
                "corr_dax": corr_dx,
            })

        df_b = pd.DataFrame(rows_b).sort_values("weight_%", ascending=False)
        st.dataframe(df_b, use_container_width=True, hide_index=True)

        roll = tmp.copy()
        roll["corr_spx_roll"] = roll["PORT"].rolling(rolling_win).corr(roll["SPX"])
        roll["corr_dax_roll"] = roll["PORT"].rolling(rolling_win).corr(roll["DAX"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll.index, y=roll["corr_spx_roll"], name=f"Rolling Corr PORT vs SPX ({rolling_win}D)"))
        fig.add_trace(go.Scatter(x=roll.index, y=roll["corr_dax_roll"], name=f"Rolling Corr PORT vs DAX ({rolling_win}D)"))
        fig.update_layout(
            title="Rolling Correlation (Portfolio vs Benchmarks)",
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(range=[-1, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ACTION PANEL – HTML Badges (Grün/Rot/Gelb)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("6) Action Panel (SHI Flags)")

df_flags = df.copy()
df_flags["flag_objects"] = df_flags.apply(classify_flags, axis=1)
df_flags["flags_badges"] = df_flags["flag_objects"].apply(render_flag_badges)
df_flags = df_flags.sort_values("shi_score", ascending=False)

view = df_flags[
    ["ticker", "name", "sleeve", "weight", "shi_score",
     "forward_pe", "peg", "vol_1y", "cash_runway_months", "net_debt_to_ebitda",
     "flags_badges"]
].copy()

st.markdown(view.to_html(escape=False, index=False), unsafe_allow_html=True)

st.caption("Hinweis: Grün = Chance, Rot = Risiko, Gelb = Prozess/Monitoring.")
st.caption("Research dashboard (education). Not investment advice. Yahoo Finance coverage varies; missing values are normal.")
