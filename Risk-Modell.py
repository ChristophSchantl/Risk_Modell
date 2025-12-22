# streamlit_app.py
# ------------------------------------------------------------
# Quick Portfolio Risk — VaR/Vol, Beta (S&P500/DAX), robust 2W Forecast
# FIXES:
# 1) Ticker-Liste (Text/CSV) wird in session_state gemerged + rerun (UI-stabil)
# 2) Portfolio-Returns robust (kein globales dropna über alle Spalten)
#    -> tägliche Renormalisierung der Gewichte auf verfügbare Returns
# 3) Forecast-Minimum dynamisch (max(60, 10*horizon)) statt hart 120
# ------------------------------------------------------------

import json
import os
from typing import List, Tuple

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

@st.cache_data(ttl=1800)
def dl_close(tickers: List[str]) -> pd.DataFrame:
    px = yf.download(tickers=tickers, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")

def infer_usd_like(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        if c.endswith("=X") or c.startswith("^"):
            continue
        # Heuristik: US tickers meist ohne ".", viele EU tickers mit ".", z.B. VOW3.DE
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
    Robust: nutzt nur verfügbare Returns pro Tag und renormalisiert Gewichte.
    rets_pos: DataFrame (Spalten = tickers)
    weights:  Series index = tickers (target weights, z.B. current weights)
    """
    rets_pos = rets_pos.copy()
    weights = weights.reindex(rets_pos.columns)

    avail = rets_pos.notna().astype(float)
    w_eff = avail.mul(weights.values, axis=1)
    w_sum = w_eff.sum(axis=1).replace(0.0, np.nan)
    w_norm = w_eff.div(w_sum, axis=0)

    port = (rets_pos * w_norm).sum(axis=1)
    return port.dropna()

def bootstrap_forecast(returns: pd.Series, horizon_days: int, n_sims: int, seed: int = 42) -> np.ndarray:
    r = returns.dropna().to_numpy()
    min_points = max(60, 10 * horizon_days)  # dynamisch statt fix 120
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
    {"ticker": "AAPL",   "shares": 50.0, "entry": 180.0, "entry_ccy": "USD", "px_override": np.nan},
    {"ticker": "MSFT",   "shares": 20.0, "entry": 350.0, "entry_ccy": "USD", "px_override": np.nan},
    {"ticker": "VOW3.DE","shares": 20.0, "entry": 120.0, "entry_ccy": "EUR", "px_override": np.nan},
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
            df = _ensure_schema(pd.DataFrame(d["portfolio"]))
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
    tickers = [_norm_ticker(t) for t in (tickers or []) if str(t).strip()]
    if not tickers:
        return df, 0

    existing = set(df["ticker"].dropna().map(_norm_ticker))
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

# init session state once
if "df" not in st.session_state or "cash_eur" not in st.session_state:
    df0, cash0 = load_state_file()
    st.session_state.df = df0
    st.session_state.cash_eur = cash0

# Sidebar
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
        if apply_list and uploaded_tickers:
            st.session_state.df, added = merge_tickers_into_df(st.session_state.df, uploaded_tickers)
            st.success(f"{added} Ticker hinzugefügt.")
            st.rerun()

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        save_btn = st.button("💾 Save")
    with colB:
        run = st.button("▶️ Compute", type="primary")

# Editor
st.subheader("Inputs (editierbar)")
st.session_state.df = _ensure_schema(st.session_state.df)

edited = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    num_rows="dynamic",
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
    save_state_file(st.session_state.df, st.session_state.cash_eur)
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

needed = list(dict.fromkeys(tickers_pos + [DEFAULT_SPX, DEFAULT_DAX] + ([DEFAULT_FX] if use_fx else [])))
px_all = dl_close(needed)

if px_all.empty:
    st.error("Keine Preisdaten geladen (yfinance).")
    st.stop()

# FX series (EURUSD)
fx = None
if use_fx:
    if DEFAULT_FX not in px_all.columns:
        st.error("EURUSD=X nicht ladbar. Deaktiviere FX oder prüfe Verbindung.")
        st.stop()
    fx = px_all[DEFAULT_FX].dropna()

# prices without FX column
drop_fx_cols = [DEFAULT_FX] if DEFAULT_FX in px_all.columns else []
px = px_all.drop(columns=drop_fx_cols).tail(lookback).dropna(how="all")

have = set(px.columns)
missing = [t for t in tickers_pos if t not in have]
if missing:
    st.warning(f"Fehlende Ticker in Daten (ignoriert): {', '.join(missing)}")

tickers_use = [t for t in tickers_pos if t in have]
if DEFAULT_SPX not in have or DEFAULT_DAX not in have:
    st.error("S&P (^GSPC) oder DAX (^GDAXI) fehlen.")
    st.stop()
if len(tickers_use) == 0:
    st.error("Keine gültigen Ticker übrig.")
    st.stop()

# Convert USD-like tickers to EUR price series if FX on (do NOT dropna globally afterwards)
px_eur = px.copy()
if use_fx and fx is not None:
    usd_cols = infer_usd_like(px_eur.columns.tolist())
    fx_aligned = fx.reindex(px_eur.index).ffill()
    for c in usd_cols:
        if c in px_eur.columns:
            px_eur[c] = px_eur[c] / fx_aligned

# last available price per column (use latest row; if some columns NaN at end, ffill)
px_eur_ffill = px_eur.ffill()
last = px_eur_ffill.iloc[-1]

# Position table mapped to last prices (EUR)
df2 = df[df["ticker"].isin(tickers_use)].copy()
df2["px_eur"] = df2["ticker"].map(last)

# override current prices if provided (interpreted as EUR reporting price)
override = pd.to_numeric(df2["px_override"], errors="coerce")
df2.loc[override.notna(), "px_eur"] = override[override.notna()]

df2["value_eur"] = df2["px_eur"] * df2["shares"]
total_eur = float(df2["value_eur"].sum() + cash_eur)
if total_eur <= 0:
    st.error("Total EUR <= 0 (Shares/Prices/Cash prüfen).")
    st.stop()

df2["weight"] = df2["value_eur"] / total_eur

# entry -> EUR for PnL display
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

# Returns (no global dropna!)
rets_all = px_eur.pct_change()

rets_pos = rets_all[tickers_use]
spx_rets = rets_all[DEFAULT_SPX].dropna()
dax_rets = rets_all[DEFAULT_DAX].dropna()

# Portfolio returns robust with daily renormalization
w = df2.set_index("ticker")["weight"]
port_rets = portfolio_returns_robust(rets_pos, w)
port_pnl = port_rets * total_eur

# risk metrics
vol_ann = ann_vol(port_rets)
var_h = var_hist(port_pnl, var_level)
var_p = var_param(port_pnl, var_level)

# beta/alpha (aligned inside beta_alpha)
b_spx, a_spx = beta_alpha(port_rets, spx_rets)
b_dax, a_dax = beta_alpha(port_rets, dax_rets)

# correlation on available position returns (pairwise)
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

# Forecast
st.subheader(f"Forecast (Bootstrap) — {horizon_days} Trading Days (~{horizon_days/5:.1f} Wochen)")
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
else:
    min_points = max(60, 10 * horizon_days)
    st.info(f"Zu wenig Historie für Forecast (mind. ~{min_points} Return-Punkte).")

# Bench exposure summary
st.subheader("Benchmark Exposure (daily)")
bm1, bm2, bm3, bm4 = st.columns(4)
bm1.metric("Alpha vs S&P (daily)", fmt_pct(a_spx, 3) if np.isfinite(a_spx) else "n/a")
bm2.metric("Alpha vs DAX (daily)", fmt_pct(a_dax, 3) if np.isfinite(a_dax) else "n/a")
bm3.metric("Corr (Port, S&P)", f"{port_rets.corr(spx_rets):.2f}" if len(pd.concat([port_rets, spx_rets], axis=1).dropna()) > 60 else "n/a")
bm4.metric("Corr (Port, DAX)", f"{port_rets.corr(dax_rets):.2f}" if len(pd.concat([port_rets, dax_rets], axis=1).dropna()) > 60 else "n/a")
