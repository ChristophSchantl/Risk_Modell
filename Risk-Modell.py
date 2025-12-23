# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Tanaka-Style Scorecard (Cloud-hardened, pro Plotly, vollständige Version)
# - Sidebar wie Screenshot: CSV Upload + manuelle Ticker + refine + shuffle + max_n
# - Inputs: Ticker + Gewicht + Sleeve
# - Preise: 1x bulk yf.download (stabil auf Streamlit Cloud)
# - Fundamentals: optional (Yahoo blockt oft); Dashboard degradiert sauber
# - KPIs + Tanaka-Proxy Score + Heatmap/Scatter
# - Beta/Corr/R²/Alpha p.a. + TE/AR/IR vs S&P500 & DAX + Rolling Beta/Alpha
# - Action Panel: farbige Badges + Overall (Opportunity / Mixed / Risk)
# ─────────────────────────────────────────────────────────────────────────────

import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# PAGE + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tanaka-Style Scorecard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 14px 16px;
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
      }
      div[data-testid="stMetric"] > label { color: #6b7280 !important; font-weight: 650 !important; }
      div[data-testid="stMetric"] span { color: #111827 !important; font-weight: 800 !important; }

      table { width:100%; border-collapse: collapse; }
      thead th {
        background:#f9fafb;
        border-bottom: 1px solid #e5e7eb;
        padding: 10px;
        text-align:left;
        font-weight: 800;
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
      .muted { color:#6b7280; font-size:0.90rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BENCH_SP = "^GSPC"
BENCH_DAX = "^GDAXI"

DEFAULT_TICKERS = ["LULU", "REI", "SRPT", "CAG", "NVO", "PYPL"]
SLEEVES = ["Auto", "Platform", "Biotech/Pharma", "Minerals/Energy", "Financials", "Other"]

BASE_WEIGHTS = {
    "Platform": {"growth": 0.20, "quality": 0.22, "valuation": 0.18, "momentum": 0.10, "convexity": 0.08, "risk": 0.10, "gap": 0.12},
    "Biotech/Pharma": {"growth": 0.14, "quality": 0.10, "valuation": 0.10, "momentum": 0.08, "convexity": 0.22, "risk": 0.10, "gap": 0.26},
    "Minerals/Energy": {"growth": 0.10, "quality": 0.08, "valuation": 0.14, "momentum": 0.08, "convexity": 0.20, "risk": 0.16, "gap": 0.24},
    "Financials": {"growth": 0.12, "quality": 0.18, "valuation": 0.22, "momentum": 0.10, "convexity": 0.06, "risk": 0.18, "gap": 0.14},
    "Other": {"growth": 0.16, "quality": 0.16, "valuation": 0.16, "momentum": 0.10, "convexity": 0.12, "risk": 0.14, "gap": 0.16},
}

SHOW_COLS = [
    "ticker","name","sleeve","weight",
    "price","mktcap",
    "forward_pe","trailing_pe","peg","ps","pb",
    "rev_cagr_3y","eps_cagr_3y","oper_margin","roe",
    "mom_6m","vol_1y","net_debt_to_ebitda","cash_runway_months",
    "expected_growth","implied_growth","expectation_gap",
    "tanaka_score","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap"
]

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
            s = x.strip().replace("%", "").replace(",", ".")
            if s == "" or s.lower() in {"none", "nan", "na", "n/a"}:
                return np.nan
            return float(s)
        return float(x)
    except Exception:
        return np.nan

def sanitize_ticker(t: str) -> str:
    t = (t or "").upper().strip()
    return t if re.fullmatch(r"[A-Z0-9\.\-\^]{1,15}", t) else ""

def parse_tickers_any(text: str) -> List[str]:
    if not text:
        return []
    raw = text.replace("\n", " ").replace("\t", " ").replace(";", ",").replace("|", ",")
    parts = []
    for chunk in raw.split(","):
        parts.extend(chunk.split())
    tickers = [sanitize_ticker(p.strip()) for p in parts if p.strip()]
    tickers = [t for t in tickers if t]
    return list(dict.fromkeys(tickers))

def read_tickers_from_csv(uploaded_file) -> List[str]:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    sep = ";" if text.count(";") > text.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]
    col = None
    for c in ["ticker", "symbol", "code", "codes", "ric"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        col = df.columns[0]
    tickers = df[col].astype(str).str.upper().str.strip().tolist()
    tickers = [sanitize_ticker(t) for t in tickers]
    tickers = [t for t in tickers if t]
    return list(dict.fromkeys(tickers))

def normalize_weights_pct(df: pd.DataFrame) -> pd.DataFrame:
    w = df["weight"].apply(safe_float).fillna(0.0).values
    s = float(np.sum(w))
    if s <= 0:
        df["weight"] = 0.0
        return df
    df["weight"] = (w / s) * 100.0
    return df

def z01(x, lo, hi):
    if np.isnan(x):
        return np.nan
    if hi == lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def inv01(x, lo, hi):
    v = z01(x, lo, hi)
    return np.nan if np.isnan(v) else 1.0 - v

def nanmean(vals):
    a = np.array(vals, dtype=float)
    return np.nan if np.all(np.isnan(a)) else float(np.nanmean(a))

def clean_pos(x):
    x = safe_float(x)
    return np.nan if (np.isnan(x) or x <= 0) else x

# ─────────────────────────────────────────────────────────────────────────────
# DATA (Cloud-hardened)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices_bulk(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"].copy() if "Close" in data.columns.get_level_values(0) else data.xs(data.columns.levels[0][0], axis=1, level=0).copy()
    else:
        if "Close" in data.columns:
            px = data[["Close"]].copy()
            px.columns = [tickers[0]]
        else:
            px = data.copy()
    px = px.dropna(how="all")
    px.columns = [str(c).upper() for c in px.columns]
    return px

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info_cloud(ticker: str) -> Dict:
    t = yf.Ticker(ticker)
    # get_info kann auf Cloud leer sein → fallback fast_info
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}
    if not info:
        try:
            fi = getattr(t, "fast_info", None)
            if fi:
                info = {
                    "currentPrice": fi.get("last_price"),
                    "regularMarketPrice": fi.get("last_price"),
                    "marketCap": fi.get("market_cap"),
                    "currency": fi.get("currency"),
                }
        except Exception:
            info = {}
    return info or {}

def mom_vol_from_prices(px: pd.Series) -> Tuple[float, float]:
    px = px.dropna()
    if len(px) < 60:
        return np.nan, np.nan
    k = min(126, len(px) - 1)
    mom_6m = (px.iloc[-1] / px.iloc[-1-k] - 1) if k > 0 else np.nan
    r = px.pct_change().dropna()
    vol_1y = float(np.std(r, ddof=1) * np.sqrt(252)) if len(r) >= 60 else np.nan
    return float(mom_6m), float(vol_1y)

def sleeve_auto_heuristic(info: Dict) -> str:
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    name = (info.get("shortName") or info.get("longName") or "").lower()
    txt = " ".join([sector, industry, name])

    if any(k in txt for k in ["biotech", "biotechnology", "pharma", "pharmaceutical", "therapeutics", "drug"]):
        return "Biotech/Pharma"
    if any(k in txt for k in ["semiconductor", "software", "internet", "technology", "cloud", "ai", "platform"]):
        return "Platform"
    if any(k in txt for k in ["uranium", "mining", "metals", "materials", "oil", "gas", "energy"]):
        return "Minerals/Energy"
    if any(k in txt for k in ["bank", "financial", "insurance", "capital markets"]):
        return "Financials"
    return "Other"

# ─────────────────────────────────────────────────────────────────────────────
# SCORE (Tanaka-Proxy; robust bei fehlenden Fundamentals)
# ─────────────────────────────────────────────────────────────────────────────
def score_growth(vals: Dict) -> float:
    s = nanmean([z01(vals.get("eps_cagr_3y", np.nan), -0.20, 0.40),
                 z01(vals.get("rev_cagr_3y", np.nan), -0.10, 0.30)])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_quality(vals: Dict) -> float:
    s = nanmean([z01(vals.get("roe", np.nan), -0.10, 0.30),
                 z01(vals.get("oper_margin", np.nan), -0.10, 0.35)])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_valuation(vals: Dict) -> float:
    s = nanmean([
        inv01(vals.get("forward_pe", np.nan), 5, 60),
        inv01(vals.get("trailing_pe", np.nan), 5, 60),
        inv01(vals.get("peg", np.nan), 0.5, 3.0),
        inv01(vals.get("ps", np.nan), 0.5, 20.0),
        inv01(vals.get("pb", np.nan), 0.2, 15.0),
    ])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_momentum(vals: Dict) -> float:
    s = z01(vals.get("mom_6m", np.nan), -0.40, 0.60)
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_convexity(vals: Dict, sleeve: str) -> float:
    vol = vals.get("vol_1y", np.nan)
    mcap = vals.get("mktcap", np.nan)
    s_vol = z01(vol, 0.15, 0.90)

    s_size = np.nan
    if not np.isnan(mcap) and mcap > 0:
        s_size = inv01(np.log10(mcap), 9.0, 12.0)  # smaller → higher

    base = {"Platform": 0.35, "Biotech/Pharma": 0.70, "Minerals/Energy": 0.70, "Financials": 0.25, "Other": 0.45}.get(sleeve, 0.45)
    s = nanmean([s_vol, s_size, base])
    return np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))

def score_risk(vals: Dict, sleeve: str) -> float:
    vol = vals.get("vol_1y", np.nan)
    nde = vals.get("net_debt_to_ebitda", np.nan)
    runway = vals.get("cash_runway_months", np.nan)

    vol_score = inv01(vol, 0.15, 0.90)
    nde_score = inv01(nde, -1.0, 6.0)
    runway_score = z01(runway, 0.0, 36.0)

    s = nanmean([vol_score, nde_score, runway_score])
    if np.isnan(s):
        return np.nan
    risk = float(np.clip(s * 100, 0, 100))
    if not np.isnan(runway) and runway < 6:
        risk = min(risk, 35.0)
    return risk

def score_gap(vals: Dict) -> Tuple[float, float, float, float]:
    # robust, korrekt benannt: implied_growth = earnings_yield_proxy (1/ForwardPE)
    eps = vals.get("eps_cagr_3y", np.nan)
    rev = vals.get("rev_cagr_3y", np.nan)
    expected = nanmean([eps, rev])  # proxy
    fpe = vals.get("forward_pe", np.nan)
    implied = (1.0 / fpe) if (not np.isnan(fpe) and fpe > 0) else np.nan  # earnings yield proxy
    gap_raw = (expected - implied) if (not np.isnan(expected) and not np.isnan(implied)) else np.nan
    s = z01(gap_raw, -0.10, 0.25)
    return (np.nan if np.isnan(s) else float(np.clip(s * 100, 0, 100))), expected, implied, gap_raw

def compute_total_score(row: pd.Series) -> Tuple[float, Dict, float, float, float]:
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
        "ps": row.get("ps", np.nan),
        "pb": row.get("pb", np.nan),
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
    gap_score, exp_g, impl_g, gap_raw = score_gap(vals)
    subs["gap"] = gap_score

    # sleeve tweak
    if sleeve in ["Biotech/Pharma", "Minerals/Energy"]:
        weights["risk"] *= 0.65
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
# Regression / Risk helpers
# ─────────────────────────────────────────────────────────────────────────────
def portfolio_returns_from_prices(px_close: pd.DataFrame, weights_pct: pd.Series) -> pd.Series:
    rets = px_close.pct_change().dropna(how="all")
    common = [c for c in rets.columns if c in weights_pct.index]
    if not common:
        return pd.Series(dtype=float)
    w = (weights_pct.loc[common] / 100.0).astype(float)
    w = (w / w.sum()) if w.sum() > 0 else w
    port = (rets[common].mul(w, axis=1)).sum(axis=1)
    port.name = "PORT"
    return port

def compute_regression_metrics(asset_ret: pd.Series, bench_ret: pd.Series, ppy: int = 252):
    df2 = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if df2.shape[0] < 80:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    a = df2.iloc[:, 0].values
    b = df2.iloc[:, 1].values
    var_b = np.var(b, ddof=1)
    if var_b <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    cov_ab = np.cov(a, b, ddof=1)[0, 1]
    beta = cov_ab / var_b
    alpha_d = float(np.mean(a) - beta * np.mean(b))
    corr = float(np.corrcoef(a, b)[0, 1])
    r2 = float(corr ** 2) if not np.isnan(corr) else np.nan
    alpha_a = float((1.0 + alpha_d) ** ppy - 1.0)
    return float(beta), corr, r2, alpha_d, alpha_a

def tracking_error_and_ir(asset_ret: pd.Series, bench_ret: pd.Series, ppy: int = 252):
    df2 = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if df2.shape[0] < 80:
        return np.nan, np.nan, np.nan
    active = df2.iloc[:, 0] - df2.iloc[:, 1]
    te = float(np.std(active, ddof=1) * np.sqrt(ppy))
    ar = float(np.mean(active) * ppy)
    ir = float(ar / te) if te > 0 else np.nan
    return te, ar, ir

def rolling_beta_alpha(asset_ret: pd.Series, bench_ret: pd.Series, window: int = 126, ppy: int = 252) -> pd.DataFrame:
    df2 = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if df2.shape[0] < window + 20:
        return pd.DataFrame()
    a = df2.iloc[:, 0]
    b = df2.iloc[:, 1]
    beta = a.rolling(window).cov(b) / b.rolling(window).var()
    alpha_d = a.rolling(window).mean() - beta * b.rolling(window).mean()
    alpha_a = (1.0 + alpha_d) ** ppy - 1.0
    out = pd.DataFrame({"beta": beta, "alpha_annual": alpha_a}).dropna(how="all")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# FLAGS
# ─────────────────────────────────────────────────────────────────────────────
def classify_flags(row: pd.Series):
    out = []
    score = safe_float(row.get("tanaka_score", np.nan))
    fpe = safe_float(row.get("forward_pe", np.nan))
    peg = safe_float(row.get("peg", np.nan))
    vol = safe_float(row.get("vol_1y", np.nan))
    runway = safe_float(row.get("cash_runway_months", np.nan))
    nde = safe_float(row.get("net_debt_to_ebitda", np.nan))
    gap_raw = safe_float(row.get("expectation_gap", np.nan))

    # Positive
    if not np.isnan(score) and score >= 85:
        out.append(("High Conviction", "pos"))
    if (not np.isnan(peg) and peg <= 1.2) and (not np.isnan(score) and score >= 70):
        out.append(("Undervalued-growth", "pos"))
    if not np.isnan(gap_raw) and gap_raw >= 0.08:
        out.append(("Gap (expected > implied)", "pos"))

    # Neutral
    if not np.isnan(fpe) and fpe >= 45 and not np.isnan(score) and score >= 75:
        out.append(("Trim-check (P/E high)", "neu"))

    # Negative
    if not np.isnan(vol) and vol >= 0.70:
        out.append(("High vol", "neg"))
    if not np.isnan(runway) and runway <= 12:
        out.append(("Runway risk (<12m)", "neg"))
    if not np.isnan(nde) and nde >= 4:
        out.append(("Leverage risk (ND/EBITDA)", "neg"))

    return out

def render_badges(flags):
    if not flags:
        return "—", "Mixed"
    pos = any(k == "pos" for _, k in flags)
    neg = any(k == "neg" for _, k in flags)
    if pos and not neg:
        overall = "Opportunity"
    elif neg and not pos:
        overall = "Risk"
    else:
        overall = "Mixed"

    parts = []
    for label, kind in flags:
        if kind == "pos":
            fg, bg = "#166534", "#dcfce7"
        elif kind == "neg":
            fg, bg = "#991b1b", "#fee2e2"
        else:
            fg, bg = "#92400e", "#fef3c7"
        parts.append(
            f'<span style="background:{bg};color:{fg};padding:4px 10px;'
            f'border-radius:12px;font-size:0.75rem;font-weight:800;'
            f'margin-right:6px;white-space:nowrap;display:inline-block;line-height:1.4;">{label}</span>'
        )
    return "".join(parts), overall

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (Screenshot-style)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("CSV-Dateien")
uploaded = st.sidebar.file_uploader("Drag and drop files here", type=["csv"], accept_multiple_files=False)
manual = st.sidebar.text_input("Weitere Ticker manuell hinzufügen (Komma-getrennt)", value="")

shuffle = st.sidebar.checkbox("Zufällig mischen", value=False)
max_n = st.sidebar.number_input("Max. Anzahl (0 = alle)", min_value=0, value=0, step=1)

tickers = []
if uploaded is not None:
    try:
        tickers.extend(read_tickers_from_csv(uploaded))
    except Exception as e:
        st.sidebar.error(f"CSV konnte nicht gelesen werden: {e}")

tickers.extend(parse_tickers_any(manual))
tickers = list(dict.fromkeys([t for t in tickers if t])) or DEFAULT_TICKERS.copy()

if shuffle and len(tickers) > 1:
    rng = np.random.default_rng(42)
    tickers = list(rng.permutation(tickers))

if max_n and max_n > 0:
    tickers = tickers[: int(max_n)]

st.sidebar.caption(f"Gefundene Ticker: {len(tickers)}")
selected = st.sidebar.multiselect("Auswahl verfeinern", options=tickers, default=tickers)

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
use_fundamentals = st.sidebar.toggle("Fundamentals (Yahoo Info) laden", value=True)
debug_yahoo = st.sidebar.toggle("Debug (Yahoo Coverage)", value=False)

period = st.sidebar.selectbox("Preis-Historie (für KPIs)", ["6mo", "1y", "2y", "5y"], index=2)
period_beta = st.sidebar.selectbox("Lookback (Beta/Alpha)", ["6mo", "1y", "2y", "5y"], index=2)
rolling_win = st.sidebar.selectbox("Rolling Window (Trading Days)", [60, 126, 252], index=1)

run = st.sidebar.button("Load / Refresh", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Tanaka-Style Scorecard (Cloud-hardened)")
st.caption("Ticker rein → Gewicht setzen → Scores/Charts → Beta/Alpha Panel → Action Flags.")

if len(selected) == 0:
    st.warning("Keine Ticker selektiert.")
    st.stop()

# init weights state
if "weights_df" not in st.session_state:
    eq_w = 100.0 / len(selected)
    st.session_state["weights_df"] = pd.DataFrame(
        {"ticker": selected, "weight": [eq_w] * len(selected), "sleeve": [default_sleeve] * len(selected)}
    )

# sync to selection
old = st.session_state["weights_df"].copy()
old_map_w = dict(zip(old["ticker"], old["weight"]))
old_map_s = dict(zip(old["ticker"], old["sleeve"]))

new_rows = []
eq_w = 100.0 / len(selected)
for t in selected:
    new_rows.append({"ticker": t, "weight": float(old_map_w.get(t, eq_w)), "sleeve": old_map_s.get(t, default_sleeve)})
st.session_state["weights_df"] = pd.DataFrame(new_rows)

st.subheader("1) Weights (Ticker + Gewicht + Sleeve)")
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

tickers_list = df_in["ticker"].tolist()
need_prices = list(dict.fromkeys([*tickers_list, BENCH_SP, BENCH_DAX]))

# ─────────────────────────────────────────────────────────────────────────────
# 2) KPIs & Score
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("2) KPIs & Tanaka-Proxy Score")

with st.spinner("Preisdaten laden (bulk) …"):
    px_bulk = fetch_prices_bulk(need_prices, period=period)

if px_bulk.empty:
    st.error("Keine Preisdaten von Yahoo. (Rate-limit/Block) → später erneut.")
    st.stop()

blocked = []
rows = []

for _, r in df_in.iterrows():
    tkr = r["ticker"].upper()
    wt = float(r["weight"])
    sleeve_choice = r.get("sleeve", "Auto")

    info = {}
    if use_fundamentals:
        info = fetch_info_cloud(tkr)
        if not info:
            blocked.append(tkr)

    # price from bulk close
    price = np.nan
    if tkr in px_bulk.columns:
        s_px = px_bulk[tkr].dropna()
        if len(s_px) > 0:
            price = float(s_px.iloc[-1])

    mom_6m, vol_1y = (np.nan, np.nan)
    if tkr in px_bulk.columns:
        mom_6m, vol_1y = mom_vol_from_prices(px_bulk[tkr])

    sleeve = sleeve_choice
    if sleeve == "Auto":
        sleeve = sleeve_auto_heuristic(info) if (use_fundamentals and info) else "Other"

    # Fundamentals (best-effort)
    mcap = safe_float(info.get("marketCap")) if info else np.nan
    forward_pe = clean_pos(info.get("forwardPE")) if info else np.nan
    trailing_pe = safe_float(info.get("trailingPE")) if info else np.nan
    peg = safe_float(info.get("pegRatio")) if info else np.nan
    ps = safe_float(info.get("priceToSalesTrailing12Months")) if info else np.nan
    pb = safe_float(info.get("priceToBook")) if info else np.nan
    roe = safe_float(info.get("returnOnEquity")) if info else np.nan
    oper_margin = safe_float(info.get("operatingMargins")) if info else np.nan
    nde = safe_float(info.get("netDebtToEBITDA")) if info else np.nan

    # optional: leave NaN unless you feed them from another provider
    rev_cagr_3y = np.nan
    eps_cagr_3y = np.nan
    cash_runway_months = np.nan

    row = {
        "ticker": tkr,
        "name": (info.get("shortName") or info.get("longName") or "") if info else "",
        "sleeve": sleeve,
        "weight": wt,
        "price": price,
        "mktcap": mcap,
        "forward_pe": forward_pe,
        "trailing_pe": trailing_pe,
        "peg": peg,
        "ps": ps,
        "pb": pb,
        "roe": roe,
        "oper_margin": oper_margin,
        "net_debt_to_ebitda": nde,
        "cash_runway_months": cash_runway_months,
        "rev_cagr_3y": rev_cagr_3y,
        "eps_cagr_3y": eps_cagr_3y,
        "mom_6m": mom_6m,
        "vol_1y": vol_1y,
    }

    total, subs, exp_g, impl_g, gap_raw = compute_total_score(pd.Series(row))
    row["tanaka_score"] = total
    row["expected_growth"] = exp_g
    row["implied_growth"] = impl_g
    row["expectation_gap"] = gap_raw
    for k, v in subs.items():
        row[f"score_{k}"] = v

    rows.append(row)

df = pd.DataFrame(rows)
df["weight_dec"] = df["weight"].fillna(0.0) / 100.0

port_score = np.nan
if df["tanaka_score"].notna().any():
    port_score = float(np.nansum(df["tanaka_score"] * df["weight_dec"]))

top_sleeve = "—"
gs = df.groupby("sleeve")["weight"].sum().sort_values(ascending=False)
if len(gs) > 0:
    top_sleeve = str(gs.index[0])

m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Portfolio Score (wtd.)", f"{port_score:.1f}" if not np.isnan(port_score) else "—")
m2.metric("Names", f"{len(df)}")
m3.metric("Top Sleeve", top_sleeve)
m4.metric("Coverage (Fundamentals)", f"{int(df['forward_pe'].notna().sum())}/{len(df)}" if use_fundamentals else "off")

if debug_yahoo and use_fundamentals and blocked:
    st.warning("Yahoo Info lückenhaft auf Cloud (normal). Ticker ohne Info: " + ", ".join(sorted(set(blocked))))

st.dataframe(df[SHOW_COLS].sort_values("weight", ascending=False), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Charts (pro Plotly)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("3) Charts")

c1, c2 = st.columns(2, gap="large")

with c1:
    sleeve_w = df.groupby("sleeve", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
    fig = px.pie(sleeve_w, values="weight", names="sleeve", title="Sleeve Allocation (%)", hole=0.35)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with c2:
    d = df.copy()
    d["wtd_contrib"] = d["tanaka_score"] * d["weight_dec"]
    d = d.sort_values("wtd_contrib", ascending=False)
    fig = px.bar(d, x="wtd_contrib", y="ticker", orientation="h", title="Weighted Score Contribution (Score × Weight)")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("#### Valuation vs Momentum (Fallback wenn Growth fehlt)")
sc = df.dropna(subset=["forward_pe", "mom_6m"], how="any").copy()
if sc.empty:
    st.info("Zu wenig Daten für Scatter (Forward P/E oder Momentum fehlt).")
else:
    fig = px.scatter(
        sc, x="forward_pe", y="mom_6m", size="weight", color="sleeve",
        hover_data=["ticker","name","tanaka_score"],
        title="Forward P/E vs 6M Momentum"
    )
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Heatmap
st.markdown("#### Heatmap (0–100)")
heat_cols = ["ticker","score_growth","score_quality","score_valuation","score_momentum","score_convexity","score_risk","score_gap","tanaka_score"]
heat = df[heat_cols].set_index("ticker")
heat_long = heat.reset_index().melt(id_vars=["ticker"], var_name="metric", value_name="value").dropna()
if heat_long.empty:
    st.info("Heatmap leer (Scores fehlen → typischerweise wegen fehlender Fundamentals).")
else:
    fig = px.imshow(
        heat.values,
        x=heat.columns,
        y=heat.index,
        aspect="auto",
        title="Score Heatmap",
        zmin=0, zmax=100
    )
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# 4) Beta/Alpha Panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("4) Beta / Correlation / Alpha vs S&P 500 & DAX")

with st.spinner("Preisdaten laden (Beta/Alpha, bulk) …"):
    px_all = fetch_prices_bulk(need_prices, period=period_beta)

if px_all.empty or px_all.shape[0] < 80:
    st.warning("Zu wenig Preisdaten für Beta/Alpha (oder Yahoo liefert nichts).")
    st.stop()

ret_all = px_all.pct_change().dropna(how="all")

w_series = df_in.set_index("ticker")["weight"].apply(safe_float).fillna(0.0)
px_const = px_all[[c for c in px_all.columns if c in tickers_list]].copy()
port_ret = portfolio_returns_from_prices(px_const, w_series)

sp_ret = ret_all[BENCH_SP].dropna() if BENCH_SP in ret_all.columns else pd.Series(dtype=float)
dax_ret = ret_all[BENCH_DAX].dropna() if BENCH_DAX in ret_all.columns else pd.Series(dtype=float)

tmp = pd.concat([port_ret, sp_ret.rename("SPX"), dax_ret.rename("DAX")], axis=1).dropna()
if tmp.shape[0] < 80:
    st.warning("Zu wenig überlappende Datenpunkte.")
    st.stop()

port_beta_sp, port_corr_sp, port_r2_sp, _, port_alpha_a_sp = compute_regression_metrics(tmp["PORT"], tmp["SPX"])
port_beta_dx, port_corr_dx, port_r2_dx, _, port_alpha_a_dx = compute_regression_metrics(tmp["PORT"], tmp["DAX"])

te_sp, ar_sp, ir_sp = tracking_error_and_ir(tmp["PORT"], tmp["SPX"])
te_dx, ar_dx, ir_dx = tracking_error_and_ir(tmp["PORT"], tmp["DAX"])

m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Beta vs S&P 500", f"{port_beta_sp:.2f}" if not np.isnan(port_beta_sp) else "—")
m2.metric("Corr vs S&P 500", f"{port_corr_sp:.2f}" if not np.isnan(port_corr_sp) else "—")
m3.metric("R² vs S&P 500", f"{port_r2_sp:.2f}" if not np.isnan(port_r2_sp) else "—")
m4.metric("Alpha p.a. vs S&P 500", f"{port_alpha_a_sp*100:.1f}%" if not np.isnan(port_alpha_a_sp) else "—")

m5, m6, m7, m8 = st.columns(4, gap="large")
m5.metric("Beta vs DAX", f"{port_beta_dx:.2f}" if not np.isnan(port_beta_dx) else "—")
m6.metric("Corr vs DAX", f"{port_corr_dx:.2f}" if not np.isnan(port_corr_dx) else "—")
m7.metric("R² vs DAX", f"{port_r2_dx:.2f}" if not np.isnan(port_r2_dx) else "—")
m8.metric("Alpha p.a. vs DAX", f"{port_alpha_a_dx*100:.1f}%" if not np.isnan(port_alpha_a_dx) else "—")

t1, t2, t3, t4 = st.columns(4, gap="large")
t1.metric("Tracking Error p.a. vs S&P 500", f"{te_sp*100:.1f}%" if not np.isnan(te_sp) else "—")
t2.metric("Active Return p.a. vs S&P 500", f"{ar_sp*100:.1f}%" if not np.isnan(ar_sp) else "—")
t3.metric("Info Ratio vs S&P 500", f"{ir_sp:.2f}" if not np.isnan(ir_sp) else "—")
t4.metric(" ", " ")

# Position table
rows_b = []
for t in tickers_list:
    if t not in ret_all.columns:
        continue
    a = ret_all[t].dropna()
    b1 = ret_all[BENCH_SP].dropna()
    b2 = ret_all[BENCH_DAX].dropna()

    beta_sp, corr_sp, r2_sp, _, alpha_a_sp = compute_regression_metrics(a, b1)
    te_sp_i, ar_sp_i, ir_sp_i = tracking_error_and_ir(a, b1)

    beta_dx, corr_dx, r2_dx, _, alpha_a_dx = compute_regression_metrics(a, b2)
    te_dx_i, ar_dx_i, ir_dx_i = tracking_error_and_ir(a, b2)

    rows_b.append({
        "ticker": t,
        "weight_%": float(w_series.get(t, 0.0)),
        "beta_spx": beta_sp, "corr_spx": corr_sp, "r2_spx": r2_sp,
        "alpha_pa_spx": alpha_a_sp, "te_pa_spx": te_sp_i, "ir_spx": ir_sp_i,
        "beta_dax": beta_dx, "corr_dax": corr_dx, "r2_dax": r2_dx,
        "alpha_pa_dax": alpha_a_dx, "te_pa_dax": te_dx_i, "ir_dax": ir_dx_i,
    })

df_b = pd.DataFrame(rows_b).sort_values("weight_%", ascending=False)
st.dataframe(df_b, use_container_width=True, hide_index=True)

# Rolling Beta/Alpha (ohne Flächen / ohne rot-schwarz)
st.markdown("#### Rolling Beta & Alpha (Portfolio)")
roll_sp = rolling_beta_alpha(tmp["PORT"], tmp["SPX"], window=int(rolling_win))
roll_dx = rolling_beta_alpha(tmp["PORT"], tmp["DAX"], window=int(rolling_win))

def plot_roll(title: str, r: pd.DataFrame):
    fig = go.Figure()
    if not r.empty:
        fig.add_trace(go.Scatter(x=r.index, y=r["beta"], name="Beta", mode="lines", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=r.index, y=r["alpha_annual"]*100, name="Alpha p.a. (%)", mode="lines",
                                 line=dict(width=2, dash="dash"), yaxis="y2"))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(0,0,0,0.25)", yref="y2")
    fig.update_layout(
        title=title,
        height=430,
        template="plotly_white",
        margin=dict(l=10,r=10,t=45,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="Beta", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis2=dict(title="Alpha p.a. (%)", overlaying="y", side="right", showgrid=False, zeroline=False),
    )
    return fig

cL, cR = st.columns(2, gap="large")
with cL:
    st.plotly_chart(plot_roll(f"S&P 500 ({rolling_win}D)", roll_sp), use_container_width=True, config={"displayModeBar": False})
with cR:
    st.plotly_chart(plot_roll(f"DAX ({rolling_win}D)", roll_dx), use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# 5) Action Panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("5) Action Panel (Tanaka-Style Flags)")

df_flags = df.copy()
df_flags["flag_objects"] = df_flags.apply(classify_flags, axis=1)

badges = []
overall = []
for flags in df_flags["flag_objects"].tolist():
    b, o = render_badges(flags)
    badges.append(b)
    overall.append(o)

df_flags["flags_badges"] = badges
df_flags["overall"] = overall

view = df_flags[
    ["ticker","name","sleeve","weight","tanaka_score","forward_pe","peg","vol_1y",
     "net_debt_to_ebitda","cash_runway_months","overall","flags_badges"]
].copy()

# HTML render (damit Farben wirklich sichtbar sind)
st.markdown(view.to_html(escape=False, index=False), unsafe_allow_html=True)

st.caption("Hinweis: Grün = Opportunity, Rot = Risk, Gelb = Monitoring. Yahoo-Fundamentals sind auf Streamlit Cloud oft lückenhaft; Dashboard degradiert sauber.")
