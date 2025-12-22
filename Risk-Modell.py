# streamlit_app.py
# ------------------------------------------------------------
# Quick Portfolio Risk — VaR/Vol, Beta (S&P500/DAX), robust 2W Forecast
# Robust & Stable Version
#
# Key Fixes:
# - yfinance download normalization: works with Series / DataFrame / MultiIndex
# - Portfolio returns robust (daily weight renorm, no global dropna)
# - Forecast minimum dynamic
# - Stable Streamlit editor state (key + session_state as source of truth)
# - Better diagnostics (missing tickers, history length)
# ------------------------------------------------------------

import json
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

STATE_FILE = "risk_portfolio_state.json"

DEFAULT_SPX = "^GSPC"
DEFAULT_DAX = "^GDAXI"
DEFAULT_FX  = "EURUSD=X"

# ----------------------------
# Helpers
# ----------------------------
def _norm_ticker(x: str) -> str:
    return str(x or "").strip().upper()

def _dedup_keep_order(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        x = _norm_ticker(x)
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def dl_close(tickers: List[str]) -> pd.DataFrame:
    """
    Robust yfinance close loader:
    - Handles Series vs DataFrame
    - Handles MultiIndex columns (common with yf.download)
    - Returns DataFrame indexed by date with columns=tickers
    """
    tickers = _dedup_keep_order(tickers)
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        auto_adjust=True,
        progress=False,
        group_by="column",     # still may return MultiIndex depending on yf version
        threads=True
    )

    if data is None or len(data) == 0:
        return pd.DataFrame()

    # If Series (single ticker edge)
    if isinstance(data, pd.Series):
        # rare edge: if returned series is Close
        return data.to_frame(name=tickers[0]).dropna(how="all")

    # If columns are MultiIndex: typically ("Close", "AAPL") or ("AAPL","Close")
    if isinstance(data.columns, pd.MultiIndex):
        # Case A: first level contains OHLC like "Close"
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
            if isinstance(close, pd.Series):
                close = close.to_frame()
            close.columns = [str(c) for c in close.columns]
            return close.dropna(how="all")

        # Case B: first level contains tickers
        lvl0 = list(map(str, data.columns.get_level_values(0)))
        # try: for each ticker take subcolumn "Close"
        out = {}
        for t in tickers:
            if t in lvl0:
                try:
                    sub = data[t]
                    if isinstance(sub, pd.DataFrame) and "Close" in sub.columns:
                        out[t] = sub["Close"]
                except Exception:
                    pass
        if out:
            close = pd.DataFrame(out)
            return close.dropna(how="all")

        # fallback: attempt to find any column with name close-ish
        # (very defensive)
        flat = data.copy()
        flat.columns = ["|".join(map(str, c)) for c in flat.columns]
        close_cols = [c for c in flat.columns if c.lower().endswith("|close") or c.lower().startswith("close|")]
        if close_cols:
            tmp = flat[close_cols].copy()
            tmp.columns = [c.split("|")[-1] if c.lower().startswith("close|") else c.split("|")[0] for c in close_cols]
            return tmp.dropna(how="all")

        return pd.DataFrame()

    # Standard DataFrame with single-level columns
    if "Close" in data.columns:
        close = data["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        close.columns = [str(c) for c in close.columns]
        return close.dropna(how="all")

    # If already close-like (columns are tickers)
    data.columns = [str(c) for c in data.columns]
    return data.dropna(how="all")


def infer_usd_like(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        if c.endswith("=X") or c.startswith("^"):
            continue
        # heuristic: US tickers usually without "."
        if "." not in c:
            out.append(c)
    return out

def var_hist(pnl: pd.Series, level: float) -> float:
    x = pnl.dropna().to_numpy()
    if x.size < 60:
        return np.nan
    return max(0.0, -np.quantile(x, 1 - level))

def var_param(pnl: pd.Series, level: float) -> float:
    x = pnl.dropna().to_numpy()
    if x.size < 60:
        return np.nan
    mu = float(np.mean(x))
    sig = float(np.std(x, ddof=1))
    if not np.isfinite(sig) or sig <= 0:
        return np.nan
    return max(0.0, norm.ppf(level) * sig - mu)

def ann_vol(rets: pd.Series) -> float:
    x = rets.dropna()
    if len(x) < 60:
        return np.nan
    return float(x.std(ddof=1) * np.sqrt(252))

def beta_alpha(port_rets: pd.Series, mkt_rets: pd.Series) -> Tuple[float, float]:
    df = pd.concat([port_rets, mkt_rets], axis=1).dropna()
    if len(df) < 60:
        return (np.nan, np.nan)
    x = df.iloc[:, 1].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    vx = np.var(x, ddof=1)
    if vx <= 0:
        return (np.nan, np.nan)
    b = np.cov(y, x, ddof=1)[0, 1] / vx
    a = float(np.mean(y) - b * np.mean(x))
    return (float(b), float(a))

def portfolio_returns_robust(rets_pos: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Robust daily weighting:
    - uses only available returns per day
    - renormalizes weights across available assets
    """
    if rets_pos.empty:
        return pd.Series(dtype=float)
    weights = weights.reindex(rets_pos.columns).fillna(0.0)

    avail = rets_pos.notna().astype(float)
    w_eff = avail.mul(weights.values, axis=1)

    w_sum = w_eff.sum(axis=1).replace(0.0, np.nan)
    w_norm = w_eff.div(w_sum, axis=0)

    port = (rets_pos * w_norm).sum(axis=1)
    return port.dropna()

def bootstrap_forecast(returns: pd.Series, horizon_days: int, n_sims: int, seed: int = 42) -> np.ndarray:
    r = returns.dropna().to_numpy()
    min_points = max(60, 10 * int(horizon_days))
    if r.size < min_points:
        return np.array([])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(n_sims, horizon_days))
    sims = r[idx]
    return np.prod(1.0 + sims, axis=1) - 1.0

def fmt_ccy(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:,.0f}"

def fmt_pct(x: float, dp: int = 2) -> str:
    return "n/a" if pd.isna(x) else f"{x*100:.{dp}f}%"

# ----------------------------
# State I/O
# ----------------------------
DEFAULT_PORT = pd.DataFrame([
    {"ticker": "AAPL",    "shares": 50.0, "entry": 180.0, "entry_ccy": "USD", "px_override": np.nan},
    {"ticker": "MSFT",    "shares": 20.0, "entry": 350.0, "entry_ccy": "USD", "px_override": np.nan},
    {"ticker": "VOW3.DE", "shares": 20.0, "entry": 120.0, "entry_ccy": "EUR", "px_override": np.nan},
])
DEFAULT_CASH_EUR = 10_000.0

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["ticker", "shares", "entry", "entry_ccy", "px_override"]:
        if col not in df.columns:
            df[col] = np.nan

    df["ticker"] = df["ticker"].map(_norm_ticker)
    df = df[df["ticker"] != ""].copy()

    df["entry_ccy"] = df["entry_ccy"].fillna("USD")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df["entry"] = pd.to_numeric(df["entry"], errors="coerce").fillna(0.0)
    df["px_override"] = pd.to_numeric(df["px_override"], errors="coerce")
    return df.reset_index(drop=True)

def load_state_file() -> Tuple[pd.DataFrame, float]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            df = _ensure_schema(pd.DataFrame(d.get("portfolio", [])))
            cash = float(d.get("cash_eur", DEFAULT_CASH_EUR))
            return df, cash
        except Exception:
            pass
    return _ensure_schema(DEFAULT_PORT), float(DEFAULT_CASH_EUR)

def save_state_file(df: pd.DataFrame, cash_eur: float) -> None:
    payload = {"portfolio": df.to_dict("records"), "cash_eur": float(cash_eur)}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def merge_tickers_into_df(df: pd.DataFrame, tickers: List[str]) -> Tuple[pd.DataFrame, int]:
    df = _ensure_schema(df)
    tickers = _dedup_keep_order(tickers)
    if not tickers:
        return df, 0

    existing = set(df["ticker"].tolist())
    new_rows = []
    for t in tickers:
        if t and t not in existing:
            new_rows.append({"ticker": t, "shares": 0.0, "entry": 0.0, "entry_ccy": "USD", "px_override": np.nan})
            existing.add(t)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = _ensure_schema(df)
    return df, len(new_rows)

# ----------------------------
# App
# ----------------------------
st.set_page_config(layout="wide")
st.title("Quick Portfolio Risk — VaR, Vol, Beta (S&P 500 / DAX), 2-Wochen-Forecast")

if "df" not in st.session_state or "cash_eur" not in st.session_state:
    df0, cash0 = load_state_file()
    st.session_state.df = df0
    st.session_state.cash_eur = cash0

with st.sidebar:
    st.header("Parameter")
    lookback = st.slider("Lookback (Trading Days)", 252, 1500, 756, step=21)
    var_level = st.selectbox("VaR Level", [0.90, 0.95, 0.975, 0.99], index=1)
    horizon_days = st.selectbox("Forecast Horizon", [5, 10, 15], index=1)
    n_sims = st.slider("Forecast Sims", 500, 20000, 5000, step=500)
    use_fx = st.checkbox("USD → EUR umrechnen (EURUSD=X)", value=True)
    st.session_state.cash_eur = st.number_input("Cash (EUR)", value=float(st.session_state.cash_eur), step=1000.0)

    st.divider()
    st.header("Ticker laden")
    ticker_source = st.radio("Ticker-Eingabe", ["Manuell (Tabelle)", "Liste (Text/CSV)"], index=0)

    uploaded_tickers: List[str] = []
    if ticker_source == "Liste (Text/CSV)":
        st.caption("Ticker kommasepariert oder CSV mit Spalte 'ticker' (oder erste Spalte).")
        tickers_text = st.text_area("Ticker-Liste", placeholder="AAPL, MSFT, NVDA, VOW3.DE")
        upload = st.file_uploader("CSV Upload", type=["csv"])

        if tickers_text and tickers_text.strip():
            uploaded_tickers = [_norm_ticker(t) for t in tickers_text.split(",") if t.strip()]

        if upload is not None:
            try:
                csv = pd.read_csv(upload)
                col = "ticker" if "ticker" in csv.columns else csv.columns[0]
                uploaded_tickers = [_norm_ticker(t) for t in csv[col].dropna().tolist()]
            except Exception as e:
                st.error(f"CSV konnte nicht gelesen werden: {e}")

        apply_list = st.button("➕ Ticker in Tabelle übernehmen", type="secondary")
        if apply_list:
            if not uploaded_tickers:
                st.warning("Keine Ticker gefunden (Textfeld/CSV prüfen).")
            else:
                st.session_state.df, added = merge_tickers_into_df(st.session_state.df, uploaded_tickers)
                st.success(f"{added} Ticker hinzugefügt.")
                st.rerun()

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        save_btn = st.button("💾 Save")
    with colB:
        run = st.button("▶️ Compute", type="primary")

st.subheader("Inputs (editierbar)")
st.session_state.df = _ensure_schema(st.session_state.df)

edited = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    num_rows="dynamic",
    key="portfolio_editor",  # stabilisiert Editor-State
    column_config={
        "ticker": st.column_config.TextColumn("Ticker"),
        "shares": st.column_config.NumberColumn("Shares", step=1.0),
        "entry": st.column_config.NumberColumn("Entry Price", step=0.01),
        "entry_ccy": st.column_config.SelectboxColumn("Entry CCY", options=["EUR", "USD"]),
        "px_override": st.column_config.NumberColumn("Current Price Override (optional)", step=0.01),
    }
)
st.session_state.df = _ensure_schema(edited)

if st.session_state.df["ticker"].duplicated().any():
    st.error("Duplicate tickers – bitte bereinigen.")
    st.stop()

if save_btn:
    save_state_file(st.session_state.df, float(st.session_state.cash_eur))
    st.success("Saved.")

if not run:
    st.stop()

df = st.session_state.df.copy()
cash_eur = float(st.session_state.cash_eur)

# ----------------------------
# Compute Risk
# ----------------------------
tickers_all = df["ticker"].tolist()
tickers_pos = df.loc[df["shares"] != 0, "ticker"].tolist()
if len(tickers_pos) == 0:
    st.warning("Keine Positionen mit Shares ≠ 0. Ich rechne Risk für alle Ticker in der Tabelle.")
    tickers_pos = tickers_all.copy()

needed = _dedup_keep_order(tickers_pos + [DEFAULT_SPX, DEFAULT_DAX] + ([DEFAULT_FX] if use_fx else []))
px_all = dl_close(needed)

if px_all.empty:
    st.error("Keine Preisdaten geladen (yfinance).")
    st.stop()

# FX series
fx = None
if use_fx:
    if DEFAULT_FX not in px_all.columns:
        st.error("EURUSD=X nicht ladbar. Deaktiviere FX oder prüfe Verbindung.")
        st.stop()
    fx = px_all[DEFAULT_FX].dropna()

# remove FX from price table for further steps
px = px_all.drop(columns=[c for c in [DEFAULT_FX] if c in px_all.columns]).copy()
px = px.tail(lookback).dropna(how="all")

have = set(px.columns)
missing = [t for t in tickers_pos if t not in have]
if missing:
    st.warning(f"Fehlende Ticker in Daten (ignoriert): {', '.join(missing)}")

tickers_use = [t for t in tickers_pos if t in have]

if DEFAULT_SPX not in have or DEFAULT_DAX not in have:
    st.error("S&P (^GSPC) oder DAX (^GDAXI) fehlen.")
    st.stop()

if len(tickers_use) == 0:
    st.error("Keine gültigen Positions-Ticker übrig.")
    st.stop()

# Convert USD-like to EUR prices if FX on (no global dropna!)
px_eur = px.copy()
if use_fx and fx is not None and len(fx) > 0:
    usd_cols = infer_usd_like(px_eur.columns.tolist())
    fx_aligned = fx.reindex(px_eur.index).ffill()
    for c in usd_cols:
        if c in px_eur.columns:
            px_eur[c] = px_eur[c] / fx_aligned

# last prices: forward-fill to avoid NaN last row per asset
px_eur_ffill = px_eur.ffill()
last = px_eur_ffill.iloc[-1]

# position table
df2 = df[df["ticker"].isin(tickers_use)].copy()
df2["px_eur"] = df2["ticker"].map(last)

# override current price if provided
override = pd.to_numeric(df2["px_override"], errors="coerce")
df2.loc[override.notna(), "px_eur"] = override[override.notna()]

# defensive: drop rows where price missing (otherwise weights become NaN)
bad_px = df2["px_eur"].isna()
if bad_px.any():
    dropped = df2.loc[bad_px, "ticker"].tolist()
    df2 = df2.loc[~bad_px].copy()
    st.warning(f"Letzte Preise fehlen (Positionen entfernt): {', '.join(dropped)}")

df2["value_eur"] = df2["px_eur"] * df2["shares"]
total_eur = float(df2["value_eur"].sum() + cash_eur)
if total_eur <= 0:
    st.error("Total EUR <= 0 (Shares/Prices/Cash prüfen).")
    st.stop()

df2["weight"] = df2["value_eur"] / total_eur

# entry -> EUR for display
eurusd_last = np.nan
if use_fx and fx is not None and len(fx) > 0:
    eurusd_last = float(fx.iloc[-1])

entry_eur = []
for _, r in df2.iterrows():
    ep = float(r["entry"])
    if r["entry_ccy"] == "USD" and use_fx and np.isfinite(eurusd_last) and eurusd_last > 0:
        ep = ep / eurusd_last
    entry_eur.append(ep)

df2["entry_eur"] = entry_eur
df2["unreal_pnl_eur"] = (df2["px_eur"] - df2["entry_eur"]) * df2["shares"]
df2["unreal_pnl_pct"] = np.where(df2["entry_eur"] > 0, df2["px_eur"] / df2["entry_eur"] - 1.0, np.nan)

# returns (no global dropna)
rets_all = px_eur.pct_change()

# positions returns
rets_pos = rets_all[df2["ticker"].tolist()]  # use only tickers that survived price mapping
spx_rets = rets_all[DEFAULT_SPX].dropna()
dax_rets = rets_all[DEFAULT_DAX].dropna()

# portfolio returns robust
w = df2.set_index("ticker")["weight"]
port_rets = portfolio_returns_robust(rets_pos, w)
port_pnl = port_rets * total_eur

# metrics
vol_ann = ann_vol(port_rets)
var_h = var_hist(port_pnl, var_level)
var_p = var_param(port_pnl, var_level)

b_spx, a_spx = beta_alpha(port_rets, spx_rets)
b_dax, a_dax = beta_alpha(port_rets, dax_rets)

corr = rets_pos.corr()

# forecast
fc = bootstrap_forecast(port_rets, horizon_days=horizon_days, n_sims=n_sims, seed=42)
if fc.size:
    fc_mean = float(np.mean(fc))
    fc_q05 = float(np.quantile(fc, 0.05))
    fc_q95 = float(np.quantile(fc, 0.95))
    fc_pnl_var_like = max(0.0, -fc_q05 * total_eur)
else:
    fc_mean = fc_q05 = fc_q95 = np.nan
    fc_pnl_var_like = np.nan

# ----------------------------
# Output
# ----------------------------
st.subheader("Key Risk Snapshot")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total (EUR)", f"{total_eur:,.0f}")
k2.metric("Unrealized P&L (EUR)", f"{df2['unreal_pnl_eur'].sum():,.0f}")
k3.metric("Ann. Vol (Port)", fmt_pct(vol_ann))
k4.metric(f"VaR Hist {int(var_level*100)}% (1D, EUR)", fmt_ccy(var_h))
k5.metric(f"VaR Param {int(var_level*100)}% (1D, EUR)", fmt_ccy(var_p))
k6.metric("Beta vs S&P / DAX", f"{b_spx:.2f} / {b_dax:.2f}" if np.isfinite(b_spx) and np.isfinite(b_dax) else "n/a")

st.subheader("Positions (EUR)")
show_cols = ["ticker", "shares", "entry_ccy", "entry_eur", "px_eur", "value_eur", "weight", "unreal_pnl_eur", "unreal_pnl_pct"]
st.dataframe(
    df2[show_cols].style.format({
        "entry_eur": "{:,.4f}",
        "px_eur": "{:,.4f}",
        "value_eur": "{:,.0f}",
        "weight": "{:.2%}",
        "unreal_pnl_eur": "{:,.0f}",
        "unreal_pnl_pct": "{:.2%}",
    }),
    use_container_width=True
)

# Charts
c1, c2 = st.columns(2)
with c1:
    st.subheader("Equity (norm) & Drawdown")
    equity = (1.0 + port_rets).cumprod()
    dd = equity / equity.cummax() - 1.0

    fig = plt.figure()
    plt.plot(equity.index, equity.values)
    plt.title("Portfolio Equity (normalized)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(dd.index, dd.values)
    plt.title("Drawdown")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with c2:
    st.subheader("PnL Distribution (1D) + VaR")
    fig = plt.figure()
    x = port_pnl.dropna().to_numpy()
    plt.hist(x, bins=40)
    if np.isfinite(var_h):
        plt.axvline(-var_h, linestyle="--")
    plt.title(f"Daily PnL (EUR) — VaR {int(var_level*100)}% (hist) marked")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Beta Check (Scatter)")
    df_sc = pd.concat([port_rets, spx_rets, dax_rets], axis=1).dropna()
    df_sc.columns = ["port", "spx", "dax"]

    fig = plt.figure()
    plt.scatter(df_sc["spx"], df_sc["port"], s=10)
    plt.title(f"Port vs S&P500 (beta≈{b_spx:.2f})")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(df_sc["dax"], df_sc["port"], s=10)
    plt.title(f"Port vs DAX (beta≈{b_dax:.2f})")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with c4:
    st.subheader("Correlation (Positions)")
    fig = plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Forecast
st.subheader(f"Forecast (Bootstrap) — {horizon_days} Trading Days (~{horizon_days/5:.1f} Wochen)")
min_points = max(60, 10 * int(horizon_days))
if fc.size:
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Expected Return", fmt_pct(fc_mean))
    f2.metric("5% Quantile Return", fmt_pct(fc_q05))
    f3.metric("95% Quantile Return", fmt_pct(fc_q95))
    f4.metric("Loss Proxy (VaR-like, EUR)", fmt_ccy(fc_pnl_var_like))

    fig = plt.figure()
    plt.hist(fc, bins=50)
    plt.title("Forecast Distribution (Horizon Return)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info(f"Zu wenig Historie für Forecast (mind. ~{min_points} Return-Punkte).")

# Diagnostics (very useful in production)
with st.expander("Diagnostics", expanded=False):
    st.write(f"Loaded columns: {len(px_all.columns)}")
    st.write(f"Positions tickers used: {len(df2)}")
    st.write(f"Return points (portfolio): {len(port_rets.dropna())}")
    st.write("Missing tickers (requested but not in prices):", missing if missing else "None")
    st.write("Sample price tail (EUR):")
    st.dataframe(px_eur_ffill.tail(5), use_container_width=True)

# Bench exposure summary
st.subheader("Benchmark Exposure (daily)")
bm1, bm2, bm3, bm4 = st.columns(4)
bm1.metric("Alpha vs S&P (daily)", fmt_pct(a_spx, 3) if np.isfinite(a_spx) else "n/a")
bm2.metric("Alpha vs DAX (daily)", fmt_pct(a_dax, 3) if np.isfinite(a_dax) else "n/a")
bm3.metric("Corr (Port, S&P)", f"{port_rets.corr(spx_rets):.2f}" if len(pd.concat([port_rets, spx_rets], axis=1).dropna()) > 60 else "n/a")
bm4.metric("Corr (Port, DAX)", f"{port_rets.corr(dax_rets):.2f}" if len(pd.concat([port_rets, dax_rets], axis=1).dropna()) > 60 else "n/a")
