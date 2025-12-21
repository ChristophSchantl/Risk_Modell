# ===== START: risk_report_app.py =====
# Portfolio Risk & Trim Tool (Yahoo Finance only)
# - Editable inputs + persistence
# - USD/EUR entry handling
# - VaR/ES (hist + param), correlation, risk contributions
# - P&L attribution + risk-adjusted P&L
# - Trim screener + auto-trim optimizer + what-if
# - VaR backtest (Kupiec UC)
# ------------------------------------------------------------

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2


# -----------------------------
# Persistence
# -----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(APP_DIR, "portfolio_inputs.json")

DEFAULT_ROWS = pd.DataFrame(
    [
        {"ticker": "REI", "shares": 17400, "entry": 0.72, "entry_ccy": "USD"},
        {"ticker": "NVO", "shares": 400, "entry": 36.50, "entry_ccy": "USD"},
        {"ticker": "PYPL", "shares": 280, "entry": 62.10, "entry_ccy": "USD"},
        {"ticker": "SRPT", "shares": 950, "entry": 109.30, "entry_ccy": "USD"},
        {"ticker": "LULU", "shares": 80, "entry": 176.00, "entry_ccy": "USD"},
        {"ticker": "CMCSA", "shares": 630, "entry": 42.50, "entry_ccy": "USD"},
        {"ticker": "CAG", "shares": 970, "entry": 29.60, "entry_ccy": "USD"},
        {"ticker": "VIXL.L", "shares": 3000, "entry": 2.31, "entry_ccy": "EUR"},
    ]
)

DEFAULT_CASH_EUR = 40805.85
DEFAULT_BENCH = "^GSPC"
DEFAULT_FX = "EURUSD=X"


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(payload: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ensure_session_state():
    disk = load_state()

    if "portfolio_df" not in st.session_state:
        if disk.get("portfolio_rows"):
            st.session_state.portfolio_df = pd.DataFrame(disk["portfolio_rows"])
        else:
            st.session_state.portfolio_df = DEFAULT_ROWS.copy()

    if "cash_eur" not in st.session_state:
        st.session_state.cash_eur = float(disk.get("cash_eur", DEFAULT_CASH_EUR))

    if "bench" not in st.session_state:
        st.session_state.bench = disk.get("bench", DEFAULT_BENCH)

    if "fx" not in st.session_state:
        st.session_state.fx = disk.get("fx", DEFAULT_FX)


# -----------------------------
# Risk config
# -----------------------------
@dataclass
class RiskConfig:
    lookback_days: int = 756
    levels: Tuple[float, ...] = (0.95, 0.99)
    horizon_days: int = 1
    rf_daily: float = 0.0


# -----------------------------
# Risk math
# -----------------------------
def var_hist(pnl: pd.Series, level: float) -> float:
    q = np.quantile(pnl.dropna().values, 1 - level)
    return max(0.0, -q)


def es_hist(pnl: pd.Series, level: float) -> float:
    x = pnl.dropna().values
    q = np.quantile(x, 1 - level)
    tail = x[x <= q]
    return max(0.0, -tail.mean()) if tail.size else 0.0


def var_param(pnl: pd.Series, level: float) -> float:
    mu = pnl.mean()
    sig = pnl.std(ddof=1)
    z = float(norm.ppf(level))
    return max(0.0, z * sig - mu)


def es_param(pnl: pd.Series, level: float) -> float:
    mu = pnl.mean()
    sig = pnl.std(ddof=1)
    z = float(norm.ppf(level))
    phi = float(norm.pdf(z))
    es = (sig * phi / (1 - level)) - mu
    return max(0.0, es)


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Series]:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()), dd


def annualize_vol(daily_rets: pd.Series) -> float:
    return float(daily_rets.std(ddof=1) * np.sqrt(252))


def annualize_return(daily_rets: pd.Series) -> float:
    return float((1 + daily_rets.mean()) ** 252 - 1)


def sharpe(daily_rets: pd.Series, rf_daily: float = 0.0) -> float:
    ex = daily_rets - rf_daily
    vol = ex.std(ddof=1)
    return float(ex.mean() / vol * np.sqrt(252)) if vol != 0 else np.nan


def sortino(daily_rets: pd.Series, rf_daily: float = 0.0) -> float:
    ex = daily_rets - rf_daily
    downside = ex[ex < 0].std(ddof=1)
    return float(ex.mean() / downside * np.sqrt(252)) if downside != 0 else np.nan


def beta_alpha(port_rets: pd.Series, bench_rets: pd.Series, rf_daily: float = 0.0) -> Tuple[float, float]:
    aligned = pd.concat([port_rets, bench_rets], axis=1).dropna()
    if aligned.shape[0] < 60:
        return np.nan, np.nan
    rp = aligned.iloc[:, 0] - rf_daily
    rb = aligned.iloc[:, 1] - rf_daily
    cov = np.cov(rp, rb, ddof=1)[0, 1]
    varb = np.var(rb, ddof=1)
    b = cov / varb if varb != 0 else np.nan
    a_daily = rp.mean() - b * rb.mean()
    a_ann = (1 + a_daily) ** 252 - 1 if np.isfinite(a_daily) else np.nan
    return float(b), float(a_ann)


# -----------------------------
# Yahoo data
# -----------------------------
@st.cache_data(ttl=3600)
def download_prices(tickers: List[str]) -> pd.DataFrame:
    px = yf.download(tickers=tickers, period="max", auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")


def infer_usd_like(tickers: List[str]) -> List[str]:
    # Plain tickers assumed USD; suffix tickers assumed local.
    usd = []
    for t in tickers:
        if t.endswith("=X"):
            continue
        if "." in t:
            continue
        usd.append(t)
    return usd


def convert_usd_to_eur(px: pd.DataFrame, usd_cols: List[str], eurusd: pd.Series) -> pd.DataFrame:
    out = px.copy()
    fx = eurusd.reindex(out.index).ffill()
    for t in usd_cols:
        if t in out.columns:
            out[t] = out[t] / fx
    return out


# -----------------------------
# Kupiec UC test
# -----------------------------
def kupiec_uc_test(pnl: pd.Series, var_level: float, var_pos: pd.Series) -> Tuple[float, float, int, int]:
    df = pd.concat([pnl, var_pos], axis=1).dropna()
    if df.shape[0] < 120:
        breaches = int(((-df.iloc[:, 0]) > df.iloc[:, 1]).sum())
        return np.nan, np.nan, breaches, int(df.shape[0])

    losses = -df.iloc[:, 0]
    var = df.iloc[:, 1]
    breaches = (losses > var).astype(int)
    x = int(breaches.sum())
    n = int(breaches.shape[0])

    p = 1 - var_level
    phat = x / n if n else 0.0

    eps = 1e-12
    p = float(np.clip(p, eps, 1 - eps))
    phat = float(np.clip(phat, eps, 1 - eps))

    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    lr = -2.0 * (ll0 - ll1)
    pval = float(1 - chi2.cdf(lr, df=1))
    return float(lr), float(pval), x, n


# -----------------------------
# Plot helpers
# -----------------------------
def fig_line(series: pd.Series, title: str, ylabel: str):
    fig = plt.figure()
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.xlabel("Datum")
    plt.ylabel(ylabel)
    plt.tight_layout()
    return fig


def fig_corr_heatmap(corr: pd.DataFrame):
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    plt.title("Korrelationsmatrix")
    plt.colorbar(im)
    plt.tight_layout()
    return fig


def fig_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str):
    fig = plt.figure()
    plt.bar(df[x].values, df[y].values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def fig_hist_pnl_with_var(pnl: pd.Series, var_dict: Dict[float, float]):
    fig = plt.figure()
    plt.hist(pnl.values, bins=60)
    plt.title("Portfolio Tages-PnL Histogramm (EUR) mit VaR-Linien")
    plt.xlabel("PnL (EUR)")
    plt.ylabel("Häufigkeit")
    for lvl, v in var_dict.items():
        plt.axvline(-v, linestyle="--", label=f"VaR {int(lvl*100)}%")
    plt.legend()
    plt.tight_layout()
    return fig


# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Portfolio Risiko-Analyse", layout="wide")
ensure_session_state()

st.title("Portfolio Risiko-Analyse — VaR/ES, Korrelation, P&L, Trim-Optimizer")

with st.sidebar:
    st.header("Inputs")
    st.session_state.cash_eur = st.number_input("Cash (EUR)", value=float(st.session_state.cash_eur), step=100.0)
    st.session_state.bench = st.text_input("Benchmark", value=st.session_state.bench)
    st.session_state.fx = st.text_input("FX (USD pro EUR)", value=st.session_state.fx)

    st.subheader("Risk Settings")
    lookback_days = st.number_input("Lookback (Trading Days)", min_value=252, max_value=3000, value=756, step=21)
    horizon_days = st.selectbox("Horizont (Tage)", options=[1, 10], index=0)
    levels = st.multiselect("Konfidenzlevel", options=[0.90, 0.95, 0.975, 0.99], default=[0.95, 0.99])

    st.subheader("Actions")
    c1, c2, c3 = st.columns(3)
    with c1:
        save_btn = st.button("💾 Speichern")
    with c2:
        reset_btn = st.button("↩️ Reset")
    with c3:
        run_btn = st.button("📈 Berechnen")

st.subheader("Portfolio Inputs (editierbar, persistent)")
df = st.session_state.portfolio_df.copy()
if "entry_ccy" not in df.columns:
    df["entry_ccy"] = "EUR"

edited = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "ticker": st.column_config.TextColumn("ticker"),
        "shares": st.column_config.NumberColumn("shares", min_value=0.0, step=1.0),
        "entry": st.column_config.NumberColumn("entry", min_value=0.0, step=0.01),
        "entry_ccy": st.column_config.SelectboxColumn("entry_ccy", options=["EUR", "USD"]),
    },
)

edited = edited.copy()
edited["ticker"] = edited["ticker"].astype(str).str.strip()
edited = edited[edited["ticker"] != ""].copy()

if edited.duplicated(subset=["ticker"]).any():
    st.error("Doppelte Ticker gefunden. Bitte bereinigen.")
    st.stop()

st.session_state.portfolio_df = edited

if reset_btn:
    st.session_state.portfolio_df = DEFAULT_ROWS.copy()
    st.session_state.cash_eur = DEFAULT_CASH_EUR
    st.session_state.bench = DEFAULT_BENCH
    st.session_state.fx = DEFAULT_FX
    save_state({
        "portfolio_rows": st.session_state.portfolio_df.to_dict(orient="records"),
        "cash_eur": float(st.session_state.cash_eur),
        "bench": st.session_state.bench,
        "fx": st.session_state.fx,
        "saved_at": pd.Timestamp.utcnow().isoformat(),
    })
    st.rerun()

if save_btn:
    save_state({
        "portfolio_rows": st.session_state.portfolio_df.to_dict(orient="records"),
        "cash_eur": float(st.session_state.cash_eur),
        "bench": st.session_state.bench,
        "fx": st.session_state.fx,
        "saved_at": pd.Timestamp.utcnow().isoformat(),
    })
    st.success("Gespeichert.")

if not run_btn:
    st.info("Inputs anpassen → optional Speichern → dann Berechnen.")
    st.stop()

if len(levels) == 0:
    st.error("Bitte mindestens ein Konfidenzlevel wählen.")
    st.stop()

cfg = RiskConfig(
    lookback_days=int(lookback_days),
    levels=tuple(float(x) for x in levels),
    horizon_days=int(horizon_days),
    rf_daily=0.0
)

# -----------------------------
# Build maps
# -----------------------------
rows = st.session_state.portfolio_df.to_dict("records")
holdings = {r["ticker"]: float(r["shares"]) for r in rows}
entry_map = {r["ticker"]: float(r["entry"]) for r in rows}
entry_ccy_map = {r["ticker"]: str(r.get("entry_ccy", "EUR")) for r in rows}

tickers = list(holdings.keys())
bench = st.session_state.bench
fx_ticker = st.session_state.fx
cash_eur = float(st.session_state.cash_eur)

# -----------------------------
# Download & normalize to EUR
# -----------------------------
all_tickers = list(dict.fromkeys(tickers + [bench, fx_ticker]))
px_all = download_prices(all_tickers)

if fx_ticker not in px_all.columns:
    st.error(f"FX-Serie {fx_ticker} konnte nicht geladen werden.")
    st.stop()

eurusd = px_all[fx_ticker].dropna()
px = px_all.drop(columns=[fx_ticker]).copy()

usd_like = infer_usd_like(list(px.columns))
px_eur = convert_usd_to_eur(px, usd_like, eurusd)

rets = px_eur.pct_change().dropna(how="all").tail(cfg.lookback_days)
usable_cols = rets.columns[rets.notna().all()].tolist()
rets = rets[usable_cols].copy()

missing_assets = [t for t in tickers if t not in usable_cols]
if missing_assets:
    st.warning(f"Ignoriert (zu wenig Historie / NA): {missing_assets}")

if bench not in usable_cols:
    st.error(f"Benchmark {bench} hat nicht genug Daten nach Cleaning.")
    st.stop()

assets = [t for t in tickers if t in usable_cols]
if len(assets) == 0:
    st.error("Keine Assets mit ausreichender Historie.")
    st.stop()

last_px = px_eur.reindex(rets.index).iloc[-1]

shares = pd.Series({t: holdings[t] for t in assets}, dtype=float)
pos_values = pd.Series({t: shares[t] * float(last_px[t]) for t in assets}, dtype=float)
total_value = float(pos_values.sum() + cash_eur)
weights = pos_values / total_value

rets_assets = rets[assets]
bench_rets = rets[bench]
port_rets = (rets_assets * weights[assets]).sum(axis=1)

equity = (1 + port_rets).cumprod()
mdd, dd = max_drawdown(equity)

kpi = {
    "Portfolio Value": total_value,
    "Ann. Return": annualize_return(port_rets),
    "Ann. Vol": annualize_vol(port_rets),
    "Sharpe": sharpe(port_rets, cfg.rf_daily),
    "Sortino": sortino(port_rets, cfg.rf_daily),
    "Max Drawdown": mdd,
}
b, a = beta_alpha(port_rets, bench_rets, cfg.rf_daily)
kpi["Beta vs Bench"] = b
kpi["Alpha vs Bench (ann)"] = a

cov = rets_assets.cov()
corr = rets_assets.corr()

pos_pnl = rets_assets.mul(pos_values, axis=1)
port_pnl = port_rets * total_value

# -----------------------------
# VaR/ES (position + portfolio)
# -----------------------------
pos_rows = []
port_rows = []
var_for_hist_plot = {}

for lvl in cfg.levels:
    for t in assets:
        pnl_t = pos_pnl[t]
        pos_rows.append({
            "level": lvl,
            "ticker": t,
            "pos_value_eur": float(pos_values[t]),
            "weight_total": float(weights[t]),
            "VaR_hist_EUR": var_hist(pnl_t, lvl),
            "ES_hist_EUR": es_hist(pnl_t, lvl),
            "VaR_param_EUR": var_param(pnl_t, lvl),
            "ES_param_EUR": es_param(pnl_t, lvl),
        })

    v_hist = var_hist(port_pnl, lvl)
    var_for_hist_plot[lvl] = v_hist
    port_rows.append({
        "level": lvl,
        "total_value_eur": total_value,
        "VaR_port_hist_EUR": v_hist,
        "ES_port_hist_EUR": es_hist(port_pnl, lvl),
        "VaR_port_param_EUR": var_param(port_pnl, lvl),
        "ES_port_param_EUR": es_param(port_pnl, lvl),
    })

pos_risk = pd.DataFrame(pos_rows)
port_risk = pd.DataFrame(port_rows)

if cfg.horizon_days != 1:
    s = float(np.sqrt(cfg.horizon_days))
    for c in ["VaR_hist_EUR", "ES_hist_EUR", "VaR_param_EUR", "ES_param_EUR"]:
        pos_risk[c] *= s
    for c in ["VaR_port_hist_EUR", "ES_port_hist_EUR", "VaR_port_param_EUR", "ES_port_param_EUR"]:
        port_risk[c] *= s
    for k in list(var_for_hist_plot.keys()):
        var_for_hist_plot[k] *= s

# -----------------------------
# Risk Contributions (Euler, parametric)
# -----------------------------
rc_rows = []
for lvl in cfg.levels:
    w = weights[assets]
    Sigma = cov
    wv = w.values.reshape(-1, 1)
    port_var = float(wv.T @ Sigma.values @ wv)
    port_sig = float(np.sqrt(max(port_var, 0.0)))

    z = float(norm.ppf(lvl))
    phi = float(norm.pdf(z))
    k_es = phi / (1 - lvl)

    Sigma_w = Sigma @ w
    mvar_ret = z * (Sigma_w / port_sig) if port_sig > 0 else 0.0 * Sigma_w
    mes_ret = k_es * (Sigma_w / port_sig) if port_sig > 0 else 0.0 * Sigma_w

    rc_var = (w * mvar_ret) * total_value
    rc_es = (w * mes_ret) * total_value

    df_rc = pd.DataFrame({
        "level": lvl,
        "ticker": assets,
        "RC_VaR_param_EUR": rc_var.values,
        "RC_ES_param_EUR": rc_es.values,
    }).sort_values("RC_ES_param_EUR", ascending=False)

    df_rc["RC_VaR_share_%"] = 100 * df_rc["RC_VaR_param_EUR"] / (df_rc["RC_VaR_param_EUR"].sum() or 1)
    df_rc["RC_ES_share_%"] = 100 * df_rc["RC_ES_param_EUR"] / (df_rc["RC_ES_param_EUR"].sum() or 1)

    rc_rows.append(df_rc)

rc = pd.concat(rc_rows, ignore_index=True)

# -----------------------------
# P&L Attribution + Risk-adjusted P&L
# -----------------------------
eurusd_last = float(eurusd.reindex(rets.index).ffill().iloc[-1])

entry_eur = {}
for t in assets:
    ent = float(entry_map.get(t, np.nan))
    ccy = entry_ccy_map.get(t, "EUR")
    entry_eur[t] = ent / eurusd_last if ccy == "USD" else ent

entry_eur = pd.Series(entry_eur, dtype=float)
current_eur = pd.Series({t: float(last_px[t]) for t in assets}, dtype=float)

unreal_pnl_eur = (current_eur - entry_eur) * shares
unreal_pnl_pct = (current_eur / entry_eur - 1.0).replace([np.inf, -np.inf], np.nan)
total_unreal = float(np.nansum(unreal_pnl_eur.values))

rap_level = float(max(cfg.levels))
pos_risk_lvl = pos_risk[pos_risk["level"] == rap_level].set_index("ticker")
rc_lvl = rc[rc["level"] == rap_level].set_index("ticker")

pnl_table = pd.DataFrame({
    "ticker": assets,
    "shares": shares.values,
    "entry_ccy": [entry_ccy_map.get(t, "EUR") for t in assets],
    "entry_px_eur": entry_eur.values,
    "current_px_eur": current_eur.values,
    "pos_value_eur": pos_values.values,
    "weight_total": weights.values,
    "unreal_pnl_eur": unreal_pnl_eur.values,
    "unreal_pnl_pct": unreal_pnl_pct.values,
})

pnl_table["pnl_contrib_%_of_total"] = np.where(total_unreal != 0, 100 * pnl_table["unreal_pnl_eur"] / total_unreal, np.nan)

pnl_table["ES_hist_EUR"] = pnl_table["ticker"].map(pos_risk_lvl["ES_hist_EUR"].to_dict())
pnl_table["PnL_per_ES_hist"] = pnl_table["unreal_pnl_eur"] / pnl_table["ES_hist_EUR"]

pnl_table["RC_ES_param_EUR"] = pnl_table["ticker"].map(rc_lvl["RC_ES_param_EUR"].to_dict())
pnl_table["PnL_per_RC_ES"] = pnl_table["unreal_pnl_eur"] / pnl_table["RC_ES_param_EUR"]

pnl_table = pnl_table.replace([np.inf, -np.inf], np.nan).sort_values("unreal_pnl_eur", ascending=False)

# -----------------------------
# Trim tools: Screener + Auto-Trim + What-if
# -----------------------------
st.subheader("✂️ Trim Tools")

trim_notional = st.number_input("Trim-Notional je Position (EUR)", min_value=1000, max_value=50000, value=10000, step=1000)

trim_rows = []
for t in assets:
    pos_val = float(pos_values[t])
    if pos_val <= trim_notional:
        continue
    rc_before = float(rc_lvl.loc[t, "RC_ES_param_EUR"])
    # linear approx: ΔES ≈ RC * (trim/position_value)
    delta = rc_before * (trim_notional / pos_val)
    trim_rows.append({
        "ticker": t,
        "pos_value_eur": pos_val,
        "trim_eur": float(trim_notional),
        "risk_reduction_eur": float(delta),
        "risk_reduction_per_1k": float(delta / (trim_notional / 1000.0)),
    })

trim_df = pd.DataFrame(trim_rows).sort_values("risk_reduction_per_1k", ascending=False) if trim_rows else pd.DataFrame(
    columns=["ticker", "pos_value_eur", "trim_eur", "risk_reduction_eur", "risk_reduction_per_1k"]
)

cA, cB = st.columns(2)
with cA:
    if len(trim_df):
        st.pyplot(fig_bar(trim_df, "ticker", "risk_reduction_per_1k", "Δ ES pro 1.000 € Trim", "Δ ES (EUR) / 1.000 €"))
    else:
        st.info("Kein Trim möglich (Positionen <= Trim-Notional).")
with cB:
    st.dataframe(trim_df.style.format({
        "pos_value_eur": "{:,.0f}",
        "trim_eur": "{:,.0f}",
        "risk_reduction_eur": "{:,.0f}",
        "risk_reduction_per_1k": "{:,.2f}",
    }), use_container_width=True)

st.subheader("🤖 Auto-Trim Optimizer (Greedy auf Δ ES)")

target_es_reduction = st.number_input("Ziel-Risikoreduktion (Δ ES, EUR)", min_value=1000, max_value=int(total_value * 0.2), value=20000, step=1000)
max_trim_per_pos = st.number_input("Max. Trim je Position (EUR)", min_value=1000, max_value=50000, value=20000, step=1000)

remaining = float(target_es_reduction)
auto_rows = []

for _, r in trim_df.iterrows():
    if remaining <= 0:
        break

    t = r["ticker"]
    pos_val = float(pos_values[t])
    rc_before = float(rc_lvl.loc[t, "RC_ES_param_EUR"])

    trim_cap = min(float(max_trim_per_pos), max(0.0, pos_val - 1.0))
    max_delta = rc_before * (trim_cap / pos_val) if pos_val > 0 else 0.0
    if max_delta <= 0:
        continue

    used_delta = min(max_delta, remaining)
    used_trim = used_delta / max_delta * trim_cap if max_delta > 0 else 0.0

    auto_rows.append({"ticker": t, "trim_eur": float(used_trim), "risk_reduction_eur": float(used_delta)})
    remaining -= used_delta

auto_trim_df = pd.DataFrame(auto_rows) if auto_rows else pd.DataFrame(columns=["ticker", "trim_eur", "risk_reduction_eur"])
total_trim_eur = float(auto_trim_df["trim_eur"].sum()) if len(auto_trim_df) else 0.0
total_delta_es = float(auto_trim_df["risk_reduction_eur"].sum()) if len(auto_trim_df) else 0.0

st.caption(f"Ziel: −{target_es_reduction:,.0f} € | Erreicht: −{total_delta_es:,.0f} € | Gesamt-Trim: {total_trim_eur:,.0f} €")
st.dataframe(auto_trim_df.style.format({"trim_eur": "{:,.0f}", "risk_reduction_eur": "{:,.0f}"}), use_container_width=True)

st.subheader("🔮 What-If (Portfolio ES nach Trims, approx.)")

pos_after = pos_values.copy()
for _, rr in auto_trim_df.iterrows():
    pos_after[rr["ticker"]] = max(0.0, float(pos_after[rr["ticker"]]) - float(rr["trim_eur"]))

cash_after = cash_eur + total_trim_eur
total_after = float(pos_after.sum() + cash_after)
w_after = pos_after / total_after

es_before = float(rc_lvl["RC_ES_param_EUR"].sum())
es_after = 0.0
for t in assets:
    w_b = float(weights[t])
    w_a = float(w_after[t])
    rc_b = float(rc_lvl.loc[t, "RC_ES_param_EUR"])
    rc_a = rc_b * (w_a / w_b) if w_b > 0 else 0.0
    es_after += rc_a

delta_es_sim = es_before - es_after

m1, m2, m3 = st.columns(3)
m1.metric(f"ES vorher (RC, {int(rap_level*100)}%)", f"{es_before:,.0f} €")
m2.metric("ES nach Trim (approx.)", f"{es_after:,.0f} €")
m3.metric("Δ ES", f"{delta_es_sim:,.0f} €")

# -----------------------------
# VaR Backtest (Kupiec)
# -----------------------------
st.subheader("🧪 VaR Backtesting (Kupiec UC)")

bt_level = float(max(cfg.levels))
bt_window = int(min(cfg.lookback_days, 500))
pnl_bt = port_pnl.tail(bt_window).copy()

# rolling historical VaR in the backtest window (expanding)
var_series = []
idx = pnl_bt.index
for i in range(len(pnl_bt)):
    sample = pnl_bt.iloc[: i + 1]
    if len(sample) < 60:
        var_series.append(np.nan)
    else:
        v = var_hist(sample, bt_level)
        if cfg.horizon_days != 1:
            v *= float(np.sqrt(cfg.horizon_days))
        var_series.append(v)

var_series = pd.Series(var_series, index=idx, name="VaR_hist_pos")

lr, pval, breaches, nobs = kupiec_uc_test(pnl_bt, bt_level, var_series)
st.caption(f"Level {int(bt_level*100)}% | Breaches: {breaches}/{nobs} | Kupiec LR={lr:.2f} | p-value={pval:.3f}")

fig_bt = plt.figure()
plt.plot(pnl_bt.index, pnl_bt.values, label="PnL (EUR)")
plt.plot(var_series.index, -var_series.values, label=f"-VaR (hist, {int(bt_level*100)}%)")
plt.title("Backtest: PnL vs -VaR")
plt.legend()
plt.tight_layout()
st.pyplot(fig_bt)

# -----------------------------
# Output sections
# -----------------------------
st.subheader("📌 Key Metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Portfolio Value", f"{kpi['Portfolio Value']:,.0f} €")
c2.metric("Ann. Return", f"{kpi['Ann. Return']*100:.2f}%")
c3.metric("Ann. Vol", f"{kpi['Ann. Vol']*100:.2f}%")
c4.metric("Max Drawdown", f"{kpi['Max Drawdown']*100:.2f}%")
c5.metric("Sharpe", f"{kpi['Sharpe']:.2f}")
c6.metric("Beta vs Bench", f"{kpi['Beta vs Bench']:.2f}")

st.subheader("📈 Equity & Drawdown")
st.pyplot(fig_line(equity, "Equity Curve (normiert)", "Wert (Start=1)"))
st.pyplot(fig_line(dd, "Drawdown", "Drawdown"))

st.subheader("📊 Positions & Weights")
ww = pd.DataFrame({"Value_EUR": pos_values, "Weight_Total": weights}).sort_values("Value_EUR", ascending=False)
st.dataframe(ww.style.format({"Value_EUR": "{:,.0f}", "Weight_Total": "{:.2%}"}), use_container_width=True)

st.subheader("⚠️ VaR & ES (Positionen)")
st.dataframe(pos_risk.style.format({
    "pos_value_eur": "{:,.0f}",
    "weight_total": "{:.2%}",
    "VaR_hist_EUR": "{:,.0f}",
    "ES_hist_EUR": "{:,.0f}",
    "VaR_param_EUR": "{:,.0f}",
    "ES_param_EUR": "{:,.0f}",
}), use_container_width=True)

st.subheader("⚠️ VaR & ES (Portfolio)")
st.dataframe(port_risk.style.format({
    "total_value_eur": "{:,.0f}",
    "VaR_port_hist_EUR": "{:,.0f}",
    "ES_port_hist_EUR": "{:,.0f}",
    "VaR_port_param_EUR": "{:,.0f}",
    "ES_port_param_EUR": "{:,.0f}",
}), use_container_width=True)

st.subheader("📉 PnL Histogramm + VaR")
st.pyplot(fig_hist_pnl_with_var(port_pnl, var_for_hist_plot))

st.subheader("🧩 Korrelation")
st.pyplot(fig_corr_heatmap(corr))

st.subheader("🧮 Risk Contributions (Parametric, Euler)")
for lvl in cfg.levels:
    sub = rc[rc["level"] == lvl].copy()
    cc1, cc2 = st.columns(2)
    with cc1:
        st.pyplot(fig_bar(sub, "ticker", "RC_ES_param_EUR", f"RC ES @ {int(lvl*100)}% (EUR)", "RC ES (EUR)"))
    with cc2:
        st.pyplot(fig_bar(sub, "ticker", "RC_VaR_param_EUR", f"RC VaR @ {int(lvl*100)}% (EUR)", "RC VaR (EUR)"))

st.subheader("💰 P&L Attribution + Risk-Adjusted P&L")
st.caption("Entry wird je Position als EUR oder USD interpretiert (entry_ccy). USD wird über EURUSD=X nach EUR konvertiert.")

st.dataframe(pnl_table.style.format({
    "shares": "{:,.0f}",
    "entry_px_eur": "{:,.4f}",
    "current_px_eur": "{:,.4f}",
    "pos_value_eur": "{:,.0f}",
    "weight_total": "{:.2%}",
    "unreal_pnl_eur": "{:,.0f}",
    "unreal_pnl_pct": "{:.2%}",
    "pnl_contrib_%_of_total": "{:.1f}",
    "ES_hist_EUR": "{:,.0f}",
    "PnL_per_ES_hist": "{:,.4f}",
    "RC_ES_param_EUR": "{:,.0f}",
    "PnL_per_RC_ES": "{:,.4f}",
}), use_container_width=True)

st.pyplot(fig_bar(pnl_table, "ticker", "unreal_pnl_eur", "Unrealized P&L (EUR) pro Position", "EUR"))
st.pyplot(fig_bar(pnl_table, "ticker", "PnL_per_ES_hist", f"PnL / ES_hist @ {int(rap_level*100)}%", "PnL pro ES"))
st.pyplot(fig_bar(pnl_table, "ticker", "PnL_per_RC_ES", f"PnL / RC_ES @ {int(rap_level*100)}%", "PnL pro RC-ES"))

st.subheader("⬇️ Downloads")
st.download_button("Position VaR/ES (CSV)", pos_risk.to_csv(index=False).encode("utf-8"), "position_var_es.csv", "text/csv")
st.download_button("Portfolio VaR/ES (CSV)", port_risk.to_csv(index=False).encode("utf-8"), "portfolio_var_es.csv", "text/csv")
st.download_button("Correlation (CSV)", corr.to_csv().encode("utf-8"), "correlation.csv", "text/csv")
st.download_button("Risk Contributions (CSV)", rc.to_csv(index=False).encode("utf-8"), "risk_contributions.csv", "text/csv")
st.download_button("P&L Attribution (CSV)", pnl_table.to_csv(index=False).encode("utf-8"), "pnl_attribution.csv", "text/csv")

# ===== END: risk_report_app.py =====
