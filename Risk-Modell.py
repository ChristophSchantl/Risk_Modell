# streamlit_app.py
# ------------------------------------------------------------
# Quick Portfolio Risk — VaR/Vol, Beta vs S&P500 & DAX, 2W Forecast
# Inputs: ticker, shares, entry, optional manual current price
# Currency: optional USD->EUR conversion via EURUSD=X
# Data: yfinance (Adj Close / auto_adjust)
# ------------------------------------------------------------

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

STATE_FILE = "risk_portfolio_state.json"

# ----------------------------
# Helpers
# ----------------------------
def _norm_ticker(x: str) -> str:
    return str(x or "").strip().upper()

@st.cache_data(ttl=1800)
def dl_close(tickers: List[str]) -> pd.DataFrame:
    px = yf.download(tickers=tickers, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all")
    return px

def infer_usd_like(cols: List[str]) -> List[str]:
    # Heuristik: US Ticker haben i.d.R. kein "."; FX & Indizes ausnehmen
    out = []
    for c in cols:
        if c.endswith("=X") or c.startswith("^"):
            continue
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
    mu, sig = float(np.mean(x)), float(np.std(x, ddof=1))
    return max(0.0, norm.ppf(level) * sig - mu)

def ann_vol(rets: pd.Series) -> float:
    return float(rets.std(ddof=1) * np.sqrt(252))

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

def bootstrap_forecast(returns: pd.Series, horizon_days: int, n_sims: int, seed: int = 42) -> np.ndarray:
    r = returns.dropna().to_numpy()
    if r.size < 120:
        return np.array([])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(n_sims, horizon_days))
    sims = r[idx]
    # compounded return over horizon
    cum = np.prod(1.0 + sims, axis=1) - 1.0
    return cum

def fmt_ccy(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:,.0f}"

def fmt_pct(x: float, dp: int = 2) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x*100:.{dp}f}%"

# ----------------------------
# Defaults
# ----------------------------
DEFAULT_PORT = pd.DataFrame([
    {"ticker": "REI",    "shares": 1000.0, "entry": 0.72,   "entry_ccy": "USD", "px_override": np.nan},
    {"ticker": "PYPL",   "shares": 50.0,   "entry": 62.10,  "entry_ccy": "USD", "px_override": np.nan},
    {"ticker": "NVO",    "shares": 60.0,   "entry": 36.50,  "entry_ccy": "USD", "px_override": np.nan},
    {"ticker": "VOW3.DE","shares": 20.0,   "entry": 120.0,  "entry_ccy": "EUR", "px_override": np.nan},
])
DEFAULT_CASH_EUR = 10_000.0

DEFAULT_SPX = "^GSPC"
DEFAULT_DAX = "^GDAXI"
DEFAULT_FX = "EURUSD=X"

def load_state() -> Tuple[pd.DataFrame, float]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            df = pd.DataFrame(d["portfolio"])
            cash = float(d["cash_eur"])
            return df, cash
        except Exception:
            pass
    return DEFAULT_PORT.copy(), float(DEFAULT_CASH_EUR)

def save_state(df: pd.DataFrame, cash_eur: float) -> None:
    payload = {"portfolio": df.to_dict("records"), "cash_eur": float(cash_eur)}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("Quick Portfolio Risk — VaR, Vol, Beta (S&P 500 / DAX), 2-Wochen-Forecast")

df, cash_eur = load_state()

with st.sidebar:
    st.header("Settings")
    lookback = st.slider("Lookback (Trading Days)", 252, 1500, 756, step=21)
    var_level = st.selectbox("VaR Level", [0.90, 0.95, 0.975, 0.99], index=1)
    horizon_days = st.selectbox("Forecast Horizon", [5, 10, 15], index=1)  # 10 ~ 2 Wochen
    n_sims = st.slider("Forecast Sims", 500, 20000, 5000, step=500)
    use_fx = st.checkbox("USD → EUR umrechnen (EURUSD=X)", value=True)
    cash_eur = st.number_input("Cash (EUR)", value=float(cash_eur), step=1000.0)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("💾 Save Inputs"):
            save_state(df, cash_eur)
            st.success("Saved")
    with col_s2:
        run = st.button("▶️ Compute", type="primary")

st.subheader("Inputs (editierbar)")
df = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker"),
        "shares": st.column_config.NumberColumn("Shares", step=1.0),
        "entry": st.column_config.NumberColumn("Entry Price", step=0.01),
        "entry_ccy": st.column_config.SelectboxColumn("Entry CCY", options=["EUR", "USD"]),
        "px_override": st.column_config.NumberColumn("Current Price Override (optional)", step=0.01),
    }
).copy()

df["ticker"] = df["ticker"].map(_norm_ticker)
df = df[df["ticker"] != ""].dropna(subset=["shares", "entry", "entry_ccy"], how="any").copy()

if df.duplicated("ticker").any():
    st.error("Duplicate tickers – bitte bereinigen.")
    st.stop()

if not run:
    st.stop()

# ----------------------------
# Data & Valuation
# ----------------------------
tickers = df["ticker"].tolist()
needed = list(dict.fromkeys(tickers + [DEFAULT_SPX, DEFAULT_DAX] + ([DEFAULT_FX] if use_fx else [])))
px_all = dl_close(needed)

if px_all.empty:
    st.error("Keine Preisdaten geladen (yfinance).")
    st.stop()

# FX
fx = None
if use_fx:
    if DEFAULT_FX not in px_all.columns:
        st.error("FX Ticker EURUSD=X nicht ladbar. Deaktiviere FX oder prüfe Verbindung.")
        st.stop()
    fx = px_all[DEFAULT_FX].dropna()

# Price table for tickers & benchmarks
px = px_all.drop(columns=[c for c in [DEFAULT_FX] if c in px_all.columns]).copy()
px = px.dropna(how="all")
px = px.tail(lookback)

# enforce availability
have = set(px.columns)
missing = [t for t in tickers if t not in have]
if missing:
    st.warning(f"Fehlende Ticker in Daten (werden ignoriert): {', '.join(missing)}")

tickers_use = [t for t in tickers if t in have]
if DEFAULT_SPX not in have or DEFAULT_DAX not in have:
    st.error("S&P 500 (^GSPC) oder DAX (^GDAXI) fehlen in Daten.")
    st.stop()

if len(tickers_use) == 0:
    st.error("Keine gültigen Ticker übrig.")
    st.stop()

# Convert USD-like tickers to EUR price series if FX on
px_eur = px.copy()
if use_fx and fx is not None:
    usd_cols = infer_usd_like(px_eur.columns.tolist())
    fx_aligned = fx.reindex(px_eur.index).ffill()
    for c in usd_cols:
        if c in px_eur.columns:
            px_eur[c] = px_eur[c] / fx_aligned

px_eur = px_eur.dropna()

# latest prices (EUR where applicable)
last = px_eur.iloc[-1].copy()

df2 = df[df["ticker"].isin(tickers_use)].copy()
df2["px_eur"] = df2["ticker"].map(last)

# manual override of current price (interpret as EUR if FX active; otherwise as native)
override = pd.to_numeric(df2["px_override"], errors="coerce")
df2.loc[override.notna(), "px_eur"] = override[override.notna()]

df2["value_eur"] = df2["px_eur"] * df2["shares"]
total_eur = float(df2["value_eur"].sum() + cash_eur)
df2["weight"] = df2["value_eur"] / total_eur

# entry price in EUR for unrealized P&L
if use_fx and fx is not None:
    eurusd_last = float(px_all[DEFAULT_FX].dropna().iloc[-1])
else:
    eurusd_last = np.nan

entry_eur = []
for _, r in df2.iterrows():
    ep = float(r["entry"])
    if r["entry_ccy"] == "USD" and use_fx and np.isfinite(eurusd_last) and eurusd_last > 0:
        ep = ep / eurusd_last
    entry_eur.append(ep)

df2["entry_eur"] = entry_eur
df2["unreal_pnl_eur"] = (df2["px_eur"] - df2["entry_eur"]) * df2["shares"]
df2["unreal_pnl_pct"] = df2["px_eur"] / df2["entry_eur"] - 1.0

# ----------------------------
# Returns & Risk
# ----------------------------
rets = px_eur.pct_change().dropna()

port_rets = (rets[tickers_use] * df2.set_index("ticker")["weight"]).sum(axis=1)
port_pnl = port_rets * total_eur

spx_rets = rets[DEFAULT_SPX]
dax_rets = rets[DEFAULT_DAX]

b_spx, a_spx = beta_alpha(port_rets, spx_rets)
b_dax, a_dax = beta_alpha(port_rets, dax_rets)

vol_ann = ann_vol(port_rets)
var_h = var_hist(port_pnl, var_level)
var_p = var_param(port_pnl, var_level)

corr = rets[tickers_use].corr()

# 2-week forecast (bootstrap)
fc = bootstrap_forecast(port_rets, horizon_days=horizon_days, n_sims=n_sims, seed=42)
if fc.size:
    fc_mean = float(np.mean(fc))
    fc_q05 = float(np.quantile(fc, 0.05))
    fc_q95 = float(np.quantile(fc, 0.95))
    # translate to EUR
    fc_pnl_mean = fc_mean * total_eur
    fc_pnl_var = max(0.0, -fc_q05 * total_eur)   # 95% one-sided loss proxy
else:
    fc_mean = fc_q05 = fc_q95 = np.nan
    fc_pnl_mean = fc_pnl_var = np.nan

# ----------------------------
# Output: Key Metrics
# ----------------------------
st.subheader("Key Risk Snapshot")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total (EUR)", f"{total_eur:,.0f}")
k2.metric("Unrealized P&L (EUR)", f"{df2['unreal_pnl_eur'].sum():,.0f}")
k3.metric("Ann. Vol (Port)", fmt_pct(vol_ann))
k4.metric(f"VaR Hist {int(var_level*100)}% (1D, EUR)", fmt_ccy(var_h))
k5.metric(f"VaR Param {int(var_level*100)}% (1D, EUR)", fmt_ccy(var_p))
k6.metric("Beta vs S&P / DAX", f"{b_spx:.2f} / {b_dax:.2f}" if np.isfinite(b_spx) and np.isfinite(b_dax) else "n/a")

st.caption("VaR = erwarteter Verlust, der mit dem gewählten Konfidenzniveau an einem Tag nicht überschritten wird (historisch/parametrisch).")

# ----------------------------
# Positions Table
# ----------------------------
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

# ----------------------------
# Charts
# ----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Equity (norm) & Drawdown")
    equity = (1.0 + port_rets).cumprod()
    dd = equity / equity.cummax() - 1.0

    fig = plt.figure()
    plt.plot(equity.index, equity.values, label="Equity")
    plt.title("Portfolio Equity (normalized)")
    plt.tight_layout()
    st.pyplot(fig)

    fig = plt.figure()
    plt.plot(dd.index, dd.values)
    plt.title("Drawdown")
    plt.tight_layout()
    st.pyplot(fig)

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

# ----------------------------
# Beta / Correlation
# ----------------------------
c3, c4 = st.columns(2)

with c3:
    st.subheader("Beta Check (Scatter)")
    fig = plt.figure()
    df_sc = pd.concat([port_rets, spx_rets, dax_rets], axis=1).dropna()
    df_sc.columns = ["port", "spx", "dax"]

    plt.scatter(df_sc["spx"], df_sc["port"], s=10)
    plt.title(f"Port vs S&P500 (beta≈{b_spx:.2f})")
    plt.tight_layout()
    st.pyplot(fig)

    fig = plt.figure()
    plt.scatter(df_sc["dax"], df_sc["port"], s=10)
    plt.title(f"Port vs DAX (beta≈{b_dax:.2f})")
    plt.tight_layout()
    st.pyplot(fig)

with c4:
    st.subheader("Correlation (Positions)")
    fig = plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------
# 2-Week Forecast Summary
# ----------------------------
st.subheader(f"Forecast (Bootstrap) — {horizon_days} Trading Days (~{horizon_days/5:.1f} Wochen)")
if fc.size:
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Expected Return", fmt_pct(fc_mean))
    f2.metric("5% Quantile Return", fmt_pct(fc_q05))
    f3.metric("95% Quantile Return", fmt_pct(fc_q95))
    f4.metric("Loss Proxy (VaR-like, EUR)", fmt_ccy(fc_pnl_var))

    fig = plt.figure()
    plt.hist(fc, bins=50)
    plt.title("Forecast Distribution (Horizon Return)")
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Zu wenig Historie für Forecast (mind. ~120 Return-Punkte empfohlen).")
