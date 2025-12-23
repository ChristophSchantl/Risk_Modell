# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Tanaka-Style Scorecard – CSV Upload (Ticker/Weight/Sleeve) + Auto Yahoo Finance
#
# CSV format (recommended):
#   ticker,weight,sleeve
#   NVDA,7.9,Platform
#   CORT,13.2,Biotech/Pharma
#
# Minimal CSV:
#   ticker,weight
#   AAPL,10
#   MSFT,15
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
#
# requirements.txt (Streamlit Cloud):
#   streamlit>=1.32
#   pandas>=2.0
#   numpy>=1.26
#   yfinance>=0.2.36
#   plotly>=5.18
# ─────────────────────────────────────────────────────────────────────────────

import io
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS (Light KPI cards)
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

DEFAULT_TICKERS = "CORT\nNUVB\nNVDA\nUAMY\nSYM\nAAPL\nNXE\nCPRX\nAMAT\nSF"
DEFAULT_WEIGHTS = "13.2\n10.7\n7.9\n6.3\n6.0\n5.8\n5.4\n3.9\n3.1\n3.1"

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
    return np.nan if np.isnan(v) else 1.0 - v

def nanmean(vals):
    a = np.array(vals, dtype=float)
    return np.nan if np.all(np.isnan(a)) else float(np.nanmean(a))

def clean_forward_pe(x):
    x = safe_float(x)
    return np.nan if (np.isnan(x) or x <= 0) else x

def parse_tickers(text: str):
    if not text:
        return []
    raw = (
        text.replace("\n", " ")
            .replace("\t", " ")
            .replace(";", ",")
            .replace("|", ",")
    )
    parts = []
    for chunk in raw.split(","):
        parts.extend(chunk.split())
    tickers = [p.strip().upper() for p in parts if p.strip()]
    # dedupe keep order
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def parse_weights(text: str, n: int):
    if n <= 0:
        return np.array([])
    if not text or not text.strip():
        return np.array([1.0 / n] * n)

    raw = text.replace("\n", " ").replace("\t", " ").replace(";", " ").replace(",", " ")
    vals = [safe_float(x) for x in raw.split() if x.strip()]
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) == 0:
        return np.array([1.0 / n] * n)

    if len(vals) < n:
        vals = vals + [vals[-1]] * (n - len(vals))
    vals = np.array(vals[:n], dtype=float)

    s = float(np.sum(vals))
    if s <= 0:
        return np.array([1.0 / n] * n)

    # if looks like percent
    if 80 <= s <= 120:
        vals = vals / 100.0
    else:
        vals = vals / s
    return vals

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

def read_portfolio_csv(uploaded_file) -> pd.DataFrame:
    # robust CSV read: commas/semicolons, decimal commas
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    # guess separator
    sep = ";" if text.count(";") > text.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        # allow 'symbol'
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})
        else:
            raise ValueError("CSV braucht Spalte 'ticker' (alternativ 'symbol').")

    if "weight" not in df.columns:
        raise ValueError("CSV braucht Spalte 'weight' (Gewichtung).")

    if "sleeve" not in df.columns:
        df["sleeve"] = "Auto"

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    # decimal comma support for weight
    df["weight"] = df["weight"].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False).str.strip()
    df["weight"] = df["weight"].apply(safe_float)
    df["sleeve"] = df["sleeve"].astype(str).str.strip()
    df.loc[~df["sleeve"].isin(SLEEVES), "sleeve"] = "Auto"

    df = df[["ticker", "weight", "sleeve"]].copy()
    df = df[df["ticker"].str.len() > 0].reset_index(drop=True)
    return df

def validate_and_fix_portfolio(df_in: pd.DataFrame, *,
                               merge_duplicates: bool,
                               normalize_to_100: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    warnings, errors = [], []
    df = df_in.copy()

    # basic checks
    if df.empty:
        errors.append("Portfolio ist leer.")
        return df, warnings, errors

    if df["ticker"].isna().any() or (df["ticker"].astype(str).str.strip() == "").any():
        errors.append("Es gibt leere Ticker-Zeilen.")
        df = df[df["ticker"].astype(str).str.strip() != ""].copy()

    if df["weight"].isna().any():
        warnings.append("Einige Weights fehlen → werden mit 0 gesetzt.")
        df["weight"] = df["weight"].fillna(0.0)

    if (df["weight"] < 0).any():
        errors.append("Negative Weights gefunden. Bitte korrigieren.")
        return df, warnings, errors

    # duplicates
    dups = df["ticker"][df["ticker"].duplicated()].unique().tolist()
    if len(dups) > 0:
        if merge_duplicates:
            warnings.append(f"Doppelte Ticker zusammengeführt: {', '.join(dups)}")
            df = df.groupby("ticker", as_index=False).agg(
                weight=("weight", "sum"),
                sleeve=("sleeve", "first"),
            )
        else:
            warnings.append(f"Doppelte Ticker gefunden (nicht zusammengeführt): {', '.join(dups)}")

    # weight sum logic
    wsum = float(df["weight"].sum())
    if wsum <= 0:
        errors.append("Summe der Weights ist 0. Bitte Weights setzen.")
        return df, warnings, errors

    # If likely decimals (0..1), convert to %
    if wsum <= 1.5:  # heuristic
        warnings.append("Weights sehen nach Dezimalgewichten aus (0–1). Konvertiere zu %.")
        df["weight"] = df["weight"] * 100.0
        wsum = float(df["weight"].sum())

    if normalize_to_100:
        df["weight"] = df["weight"] / wsum * 100.0
        warnings.append("Weights auf 100% normalisiert.")
    else:
        if not (99.0 <= wsum <= 101.0):
            warnings.append(f"Summe der Weights = {wsum:.2f}% (nicht 100%).")

    # sleeve defaults
    df.loc[df["sleeve"].isna() | (df["sleeve"].astype(str).str.strip() == ""), "sleeve"] = "Auto"
    df.loc[~df["sleeve"].isin(SLEEVES), "sleeve"] = "Auto"

    return df, warnings, errors

# ─────────────────────────────────────────────────────────────────────────────
# YAHOO FINANCE FETCH
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
        vol_score = 0.6 * vol_score + 0.4 * 0.5

    nde_score = inv_to_01(nde, -1.0, 6.0)
    runway_score = z_to_01(runway, 0.0, 36.0)

    s = nanmean([vol_score, nde_score, runway_score])
    if np.isnan(s):
        return np.nan
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

    # sleeve adjustment
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
# SIDEBAR: CSV Upload + Controls
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Portfolio Input")

uploaded = st.sidebar.file_uploader("Upload CSV (ticker, weight, optional sleeve)", type=["csv"])
merge_duplicates = st.sidebar.toggle("Merge duplicate tickers", value=True)
normalize_to_100 = st.sidebar.toggle("Normalize weights to 100%", value=True)
auto_fetch = st.sidebar.toggle("Auto-fetch Yahoo Finance", value=True)
default_sleeve = st.sidebar.selectbox("Default sleeve (fallback)", SLEEVES, index=0)
refresh = st.sidebar.button("Load / Refresh")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="small-note">CSV: columns <b>ticker</b>, <b>weight</b>, optional <b>sleeve</b> (Auto/Platform/Biotech/...).</div>',
    unsafe_allow_html=True,
)

# Fallback manual input
with st.sidebar.expander("Manual input (fallback)", expanded=False):
    tickers_text = st.text_area("Tickers", value=DEFAULT_TICKERS, height=130)
    weights_text = st.text_area("Weights", value=DEFAULT_WEIGHTS, height=130)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD INPUT DF
# ─────────────────────────────────────────────────────────────────────────────
def build_input_df():
    if uploaded is not None:
        df0 = read_portfolio_csv(uploaded)
        # fill sleeve if missing
        df0["sleeve"] = df0["sleeve"].fillna("Auto")
        return df0

    # fallback from textareas
    tickers = parse_tickers(tickers_text)
    if len(tickers) == 0:
        return pd.DataFrame(columns=["ticker", "weight", "sleeve"])
    w_dec = parse_weights(weights_text, len(tickers))  # decimals
    df0 = pd.DataFrame({"ticker": tickers, "weight": w_dec * 100.0, "sleeve": [default_sleeve] * len(tickers)})
    return df0

df_raw = build_input_df()
df_port, warns, errs = validate_and_fix_portfolio(df_raw, merge_duplicates=merge_duplicates, normalize_to_100=normalize_to_100)

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Tanaka-Style Scorecard (CSV Upload + Auto Yahoo Finance)")
st.caption("Du pflegst nur Ticker & Gewichtung (CSV oder manuell). Der Rest wird automatisch gezogen und gescored.")

if errs:
    for e in errs:
        st.error(e)
    st.stop()

if warns:
    for w in warns:
        st.warning(w)

st.subheader("1) Portfolio Setup (editable)")
edited = st.data_editor(
    df_port,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "weight": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
        "sleeve": st.column_config.SelectboxColumn("Sleeve", options=SLEEVES, width="medium"),
    },
    hide_index=True,
    key="portfolio_editor",
)

# Apply same validation after edit
df_port2, warns2, errs2 = validate_and_fix_portfolio(edited, merge_duplicates=merge_duplicates, normalize_to_100=normalize_to_100)
if errs2:
    for e in errs2:
        st.error(e)
    st.stop()
if warns2:
    for w in warns2:
        st.info(w)

# ─────────────────────────────────────────────────────────────────────────────
# FETCH + SCORE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("2) KPIs & Tanaka Score")

rows = []
with st.spinner("Pulling Yahoo Finance data & computing scores …"):
    for _, r in df_port2.iterrows():
        tkr = str(r["ticker"]).upper().strip()
        wt = float(safe_float(r["weight"]))
        sl = str(r.get("sleeve", "Auto")).strip()
        if sl not in SLEEVES:
            sl = "Auto"
        if auto_fetch:
            rows.append(build_row(tkr, sl, wt))
        else:
            rows.append({"ticker": tkr, "weight": wt, "sleeve": sl})

df = pd.DataFrame(rows)
if df.empty:
    st.stop()

df["weight_dec"] = df["weight"].apply(safe_float) / 100.0
port_score = float(np.nansum(df["tanaka_score"] * df["weight_dec"])) if df["tanaka_score"].notna().any() else np.nan

m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Portfolio Tanaka Score (wtd.)", f"{port_score:.1f}" if not np.isnan(port_score) else "—")
m2.metric("Names", f"{len(df)}")
top_sleeve = df.groupby("sleeve")["weight"].sum().sort_values(ascending=False).index[0]
m3.metric("Top Sleeve", top_sleeve)
m4.metric("Coverage", f"{int(df['tanaka_score'].notna().sum())}/{len(df)}")

show_cols = [
    "ticker","name","sleeve","weight","price","mktcap",
    "forward_pe","trailing_pe","peg","ps","pb","fcf_yield",
    "rev_cagr_3y","eps_cagr_3y","oper_margin","roe",
    "mom_6m","vol_1y","net_debt_to_ebitda","cash_runway_months",
    "expected_growth","implied_growth","expectation_gap",
    "tanaka_score","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap"
]
for c in show_cols:
    if c not in df.columns:
        df[c] = np.nan

st.dataframe(df[show_cols].sort_values("weight", ascending=False), use_container_width=True, hide_index=True)
st.download_button("Download KPI Table (CSV)", df[show_cols].to_csv(index=False).encode("utf-8"), "tanaka_scorecard.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS (Plotly)
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
    df,
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

st.markdown("---")
st.subheader("5) Heatmap (0–100)")

heat = df[["ticker","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap","tanaka_score"]].set_index("ticker")
fig = px.imshow(heat.T, aspect="auto", title="Sub-scores and Total Score (0–100)")
fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ACTION PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("6) Action Panel (Tanaka-Style Flags)")

flags = []
for _, r in df.iterrows():
    fpe = safe_float(r.get("forward_pe", np.nan))
    peg = safe_float(r.get("peg", np.nan))
    score = safe_float(r.get("tanaka_score", np.nan))
    vol = safe_float(r.get("vol_1y", np.nan))
    runway = safe_float(r.get("cash_runway_months", np.nan))
    nde = safe_float(r.get("net_debt_to_ebitda", np.nan))
    exp_g = safe_float(r.get("expected_growth", np.nan))
    impl_g = safe_float(r.get("implied_growth", np.nan))
    gap = safe_float(r.get("expectation_gap", np.nan))

    flag = []
    if not np.isnan(score) and score >= 85:
        flag.append("High Conviction")
    if not np.isnan(fpe) and fpe >= 45 and not np.isnan(score) and score >= 75:
        flag.append("Trim-check (Target P/E?)")
    if (not np.isnan(peg) and peg <= 1.2) and (not np.isnan(score) and score >= 70):
        flag.append("Undervalued-growth candidate")
    if not np.isnan(exp_g) and not np.isnan(impl_g) and (exp_g - impl_g) >= 0.05:
        flag.append("Expectation Gap (exp > implied)")
    if not np.isnan(gap) and gap >= 0.10:
        flag.append("Large Gap (>=10%)")
    if not np.isnan(vol) and vol >= 0.70:
        flag.append("High vol")
    if not np.isnan(runway) and runway <= 12:
        flag.append("Runway risk (<12m)")
    if not np.isnan(nde) and nde >= 4:
        flag.append("Leverage risk (ND/EBITDA high)")

    flags.append(", ".join(flag) if flag else "—")

df_flags = df[[
    "ticker","name","sleeve","weight","tanaka_score",
    "forward_pe","peg","vol_1y","cash_runway_months","net_debt_to_ebitda",
    "expected_growth","implied_growth","expectation_gap"
]].copy()
df_flags["flags"] = flags

st.dataframe(df_flags.sort_values("tanaka_score", ascending=False), use_container_width=True, hide_index=True)
st.caption("Research dashboard (education). Not investment advice. Yahoo Finance coverage varies; missing values are normal.")
