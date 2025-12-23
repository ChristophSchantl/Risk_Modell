# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Tanaka-Style Scorecard – Screenshot-Style Sidebar + Weights Editor + Auto Yahoo
# Stabilized version: guards + required cols + safe charts
# ─────────────────────────────────────────────────────────────────────────────

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tanaka-Style Scorecard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }
      div[data-testid="stMetric"] > label { color: #6b7280 !important; font-weight: 500 !important; }
      div[data-testid="stMetric"] span { color: #111827 !important; font-weight: 650 !important; }
      .small-note { color: rgba(17,24,39,0.70); font-size: 0.92rem; }
      code { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    table { width:100%; border-collapse: collapse; }
    thead th {
      background:#f9fafb;
      border-bottom: 1px solid #e5e7eb;
      padding: 10px;
      text-align:left;
      font-weight: 700;
      color:#111827;
    }
    tbody td {
      border-bottom: 1px solid #eef2f7;
      padding: 10px;
      vertical-align: middle;
      color:#111827;
    }
    tbody tr:hover { background:#f9fafb; }
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

# Anzeige-Spalten (wichtig: wird auch als Required-Cols verwendet)
SHOW_COLS = [
    "ticker","name","sleeve","weight","price","mktcap",
    "forward_pe","trailing_pe","peg","ps","pb","fcf_yield",
    "rev_cagr_3y","eps_cagr_3y","oper_margin","roe",
    "mom_6m","vol_1y","net_debt_to_ebitda","cash_runway_months",
    "expected_growth","implied_growth","expectation_gap",
    "tanaka_score","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap"
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

# ─────────────────────────────────────────────────────────────────────────────
# FLAGS (Variante A) – Badges
# ─────────────────────────────────────────────────────────────────────────────
def classify_flags(row):
    out = []
    score = safe_float(row.get("tanaka_score", np.nan))
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
            f"""
            <span style="
                background:{bg};
                color:{color};
                padding:4px 10px;
                border-radius:12px;
                font-size:0.75rem;
                font-weight:600;
                margin-right:6px;
                white-space:nowrap;
                display:inline-block;
                line-height:1.4;
            ">{label}</span>
            """
        )
    return "".join(parts)

def ensure_required_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # numeric coercion for critical cols
    for c in ["weight", "tanaka_score", "forward_pe", "peg", "vol_1y", "cash_runway_months", "net_debt_to_ebitda"]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# YF fetch (cached)
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
                    cash = safe_float(bs.loc[cand].iloc[0]); break
        ocf = np.nan
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            for cand in ["Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow"]:
                if cand in cf.index:
                    ocf = safe_float(cf.loc[cand].iloc[0]); break
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
                row = cand; break
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
        if np.isnan(v): continue
        wsum += weights.get(k, 0.0) * v
        wtot += weights.get(k, 0.0)
    if wtot <= 0:
        return np.nan, subs, exp_g, impl_g, gap_raw

    total = wsum / wtot
    return float(np.clip(total, 0, 100)), subs, exp_g, impl_g, gap_raw

def build_row(ticker: str, sleeve_choice: str, weight_pct: float):
    info = fetch_info(ticker)
    hist = fetch_hist(ticker, "2y")
    inc, cf, bs = fetch_financials(ticker)
    mom, vol = calc_mom_vol(hist)

    sleeve = sleeve_choice if sleeve_choice in SLEEVES else "Auto"
    if sleeve == "Auto":
        sleeve = sleeve_auto_heuristic(info)

    row = {
        "ticker": ticker.upper().strip(),
        "name": (info.get("shortName") or info.get("longName") or ""),
        "sleeve": sleeve,
        "weight": float(weight_pct),
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
        "rev_cagr_3y": try_cagr_from_income_stmt(inc, ["Total Revenue", "TotalRevenue", "Total revenue"], years=4),
        "eps_cagr_3y": try_cagr_from_income_stmt(inc, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"], years=4),
        "cash_runway_months": cash_runway_months(bs, cf),
        "mom_6m": mom,
        "vol_1y": vol,
    }

    total, subs, exp_g, impl_g, gap_raw = compute_total_score(pd.Series(row))
    row["tanaka_score"] = total
    row["expected_growth"] = exp_g
    row["implied_growth"] = impl_g
    row["expectation_gap"] = gap_raw
    for k, v in subs.items():
        row[f"score_{k}"] = v
    return row

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR – Screenshot-Flow
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("CSV-Dateien")

uploaded = st.sidebar.file_uploader(
    "Drag and drop files here",
    type=["csv"],
    accept_multiple_files=False
)

manual = st.sidebar.text_input("Weitere Ticker manuell hinzufügen (Komma-getrennt)", value="")

st.sidebar.caption("")  # spacing
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

selected = st.sidebar.multiselect(
    "Auswahl verfeinern",
    options=combined,
    default=combined
)

df_out = pd.DataFrame({"ticker": selected})
st.sidebar.download_button(
    "Kombinierte Ticker als CSV",
    data=df_out.to_csv(index=False).encode("utf-8"),
    file_name="combined_tickers.csv",
    mime="text/csv"
)

st.sidebar.markdown("---")
default_sleeve = st.sidebar.selectbox("Default Sleeve", SLEEVES, index=0)
auto_normalize = st.sidebar.toggle("Weights automatisch auf 100% normalisieren", value=True)
auto_fetch = st.sidebar.toggle("Yahoo Finance automatisch laden", value=True)
run = st.sidebar.button("Load / Refresh", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Tanaka-Style Scorecard")
st.caption("Workflow: Ticker rein → Gewicht setzen → Auto Yahoo Pull → Score + Charts + Action Panel.")

if len(selected) == 0:
    st.warning("Keine Ticker selektiert.")
    st.stop()

# Initialize weights table session state
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

st.subheader("1) Weights (nur Ticker + Gewicht)")
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
df_in["weight"] = df_in["weight"].apply(safe_float).fillna(0.0)
df_in["sleeve"] = df_in["sleeve"].astype(str).str.strip()
df_in.loc[~df_in["sleeve"].isin(SLEEVES), "sleeve"] = "Auto"
df_in["ticker"] = df_in["ticker"].astype(str).apply(sanitize_ticker)
df_in = df_in[df_in["ticker"].astype(str).str.strip() != ""].reset_index(drop=True)

if auto_normalize:
    df_in = normalize_weights_pct(df_in)

st.session_state["weights_df"] = df_in

if not run and "ran_once" not in st.session_state:
    st.info("Stell die Weights ein und klicke links auf **Load / Refresh**.")
    st.stop()
st.session_state["ran_once"] = True

# ─────────────────────────────────────────────────────────────────────────────
# FETCH + SCORE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("2) KPIs & Tanaka Score")

rows = []
with st.spinner("Pulling Yahoo Finance fundamentals & computing scores …"):
    for _, r in df_in.iterrows():
        tkr = r["ticker"]
        wt = float(safe_float(r["weight"]))
        sl = r.get("sleeve", "Auto")
        if not auto_fetch:
            # Minimal row; rest is filled by ensure_required_cols()
            rows.append({"ticker": tkr, "weight": wt, "sleeve": sl, "name": ""})
        else:
            rows.append(build_row(tkr, sl, wt))

df = pd.DataFrame(rows)
df = ensure_required_cols(df)

df["weight_dec"] = df["weight"].apply(safe_float).fillna(0.0) / 100.0

# Guard: wenn kein Score (z.B. auto_fetch aus), bleibt port_score NaN statt Crash
if "tanaka_score" in df.columns and df["tanaka_score"].notna().any():
    port_score = float(np.nansum(df["tanaka_score"] * df["weight_dec"]))
else:
    port_score = np.nan

# Guard: leeres df
if df.empty:
    st.warning("Keine Datenpunkte – prüfe Ticker / CSV / Auswahl.")
    st.stop()

# Metrics
m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Portfolio Tanaka Score (wtd.)", f"{port_score:.1f}" if not np.isnan(port_score) else "—")
m2.metric("Names", f"{len(df)}")

if df["sleeve"].notna().any():
    top_sleeve = df.groupby("sleeve")["weight"].sum().sort_values(ascending=False).index[0]
else:
    top_sleeve = "—"
m3.metric("Top Sleeve", top_sleeve)

m4.metric("Coverage", f"{int(df['tanaka_score'].notna().sum())}/{len(df)}" if "tanaka_score" in df.columns else f"0/{len(df)}")

# Table
st.dataframe(df[SHOW_COLS].sort_values("weight", ascending=False), use_container_width=True, hide_index=True)
st.download_button("Download KPI Table (CSV)", df[SHOW_COLS].to_csv(index=False).encode("utf-8"), "tanaka_scorecard.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS (guarded)
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
    d["wtd_contrib"] = d["tanaka_score"] * d["weight_dec"]
    d = d.sort_values("wtd_contrib", ascending=False)
    fig = px.bar(
        d, x="wtd_contrib", y="ticker", orientation="h",
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
        scatter, x="forward_pe", y="growth_proxy",
        size="weight", color="sleeve", hover_name="ticker",
        hover_data={"name": True, "tanaka_score": True, "weight": True, "forward_pe": True, "growth_proxy": True},
        title="Valuation vs Growth (proxy) — 'Undervalued Growth' Map",
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
    hover_data={"name": True, "tanaka_score": True, "expectation_gap": True},
    title="Expected vs Implied Growth (Tanaka Expectation-Gap Overlay)",
)
fig.add_shape(type="line", x0=0, y0=0, x1=0.30, y1=0.30, line=dict(dash="dash"))
fig.update_xaxes(tickformat=".0%", range=[0, 0.30])
fig.update_yaxes(tickformat=".0%", range=[-0.10, 0.40])
fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("5) Heatmap (0–100)")

try:
    heat = df[["ticker","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap","tanaka_score"]].set_index("ticker")
    # Guard: wenn komplett NaN, dann keine Heatmap rendern
    if heat.dropna(how="all").empty:
        st.info("Heatmap: keine Subscore-Daten (z.B. auto_fetch aus oder Yahoo Coverage).")
    else:
        fig = px.imshow(heat.T, aspect="auto", title="Sub-scores and Total Score (0–100)")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Heatmap konnte nicht gerendert werden: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ACTION PANEL (Badges)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("6) Action Panel (Tanaka-Style Flags)")

df_flags = df.copy()
df_flags["flag_objects"] = df_flags.apply(classify_flags, axis=1)
df_flags["flags_badges"] = df_flags["flag_objects"].apply(render_flag_badges)

df_flags = df_flags.sort_values("tanaka_score", ascending=False)

view = df_flags[
    ["ticker", "name", "sleeve", "weight", "tanaka_score",
     "forward_pe", "peg", "vol_1y", "cash_runway_months", "net_debt_to_ebitda",
     "flags_badges"]
].copy()

st.markdown(
    view.to_html(escape=False, index=False, justify="left"),
    unsafe_allow_html=True
)

st.caption("Hinweis: Grün = Chance, Rot = Risiko, Gelb = Prozess/Monitoring.")
st.caption("Research dashboard (education). Not investment advice. Yahoo Finance coverage varies; missing values are normal.")
